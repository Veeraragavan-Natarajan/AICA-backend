"""One realistic opener + one follow-up for EVERY flow, through the real
ConversationManager and the real Ollama. clean_call.py exercises three flows
in depth; register_eval/safety_eval exercise seven and three respectively.
Between them roughly half the twenty flows have never been run through the
live model and read by a human. This closes that gap.

Openers are the phrases test_prompt_builder.py already asserts route to each
flow (or the same trigger pattern for the six not covered there), so a bad
transcript here is a MODEL quality question, not a router question - the
router is proven elsewhere.

    LLM_TEMPERATURE=0 python -m backend.scripts.flow_sweep
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentClause, ConversationManager
from backend.llm import LlmClient
from backend.settings import ConversationSettings, LlmSettings

# Two turns per flow: the opener (the trigger phrase itself), then one natural
# follow-up answer a caller would actually give next. Not full bookings - the
# question is whether the SECOND turn (the one after the model has picked a
# playbook) is still coherent, on-topic, and in register.
FLOWS: dict[str, list[str]] = {
    "emergency.escalate": [
        "என் அம்மாவுக்கு திடீர்னு நெஞ்சு வலி, மூச்சு வாங்குது!",
        "Velachery, 4th Cross Street, number 9.",
    ],
    "complaint.escalation_angry": [
        "இது மூணாவது தடவை call பண்றது, என் பணம் இன்னும் வரல்.",
        "பேரு சொல்றேன், Suresh. Refund reference என்ன?",
    ],
    "complaint.register": [
        "ரெண்டு மணி நேரம் காக்க வெச்சீங்க, staff மோசமா பேசுனாங்க.",
        "நேத்து morning OP-ல நடந்தது.",
    ],
    "postprocedure.checkin": [
        "நேத்து knee surgery ஆகி வீட்ல இருக்கேன், கொஞ்சம் doubt இருக்கு.",
        "Dressing-ல கொஞ்சம் ஈரமா இருக்கு போல.",
    ],
    "clinical.triage": [
        "மூணு நாளா காய்ச்சல் விடமாட்டேங்குது, உடம்பெல்லாம் வலி.",
        "வயசு 34, என் சொந்த பிரச்சனை தான்.",
    ],
    "prescription.refill": [
        "என் அப்பாவுக்கு tablets தீர்ந்துடுச்சு, refill வேணும்.",
        "MRN ARV-094512, மருந்து பேரு Telmisartan 40.",
    ],
    "medication.query": [
        "மருந்து சாப்பிட்ட பிறகு தூக்கம் வர்றது normal-ஆ?",
        "Metformin தான் சாப்பிடுறேன், காலைல ஒரு tablet.",
    ],
    "lab.result_inquiry": [
        "நேத்து blood test பண்ணேன், report வந்துடுச்சா?",
        "Order number இல்ல, mobile 90045 33218.",
    ],
    "lab.book": [
        "Doctor ஒரு blood test எழுதி கொடுத்திருக்காரு, book பண்ணணும்.",
        "நாளைக்கு காலைல fasting-ல வரலாமா?",
    ],
    "insurance.query": [
        "Gall bladder surgery insurance-ல cover ஆகுமா?",
        "Policy number POL-4521, TPA Star Health.",
    ],
    "billing.query": [
        "பில்-ல ஒரு charge தப்பா இருக்கு.",
        "Bill number ARV-4471, consultation fee ரெண்டு தடவை போட்டிருக்காங்க.",
    ],
    "records.request": [
        "என் அப்பாவோட discharge summary copy வேணும்.",
        "MRN தெரியல, பேரு Muthu, கடைசி admit ஆனது கடந்த மாசம்.",
    ],
    "referral.status": [
        "வேற hospital-க்கு referral letter கேட்டிருந்தேன், status என்ன?",
        "கடந்த வாரம் cardiology-ல கேட்டேன்.",
    ],
    "patient.register": [
        "நான் புதுசா register பண்ணணும், முதல் தடவை வர்றேன்.",
        "பேரு Kalaivani, வயசு 41, மொபைல் 98765 43210.",
    ],
    "appointment.followup": [
        "கடந்த மாசம் surgery ஆனது, follow-up review வர சொன்னாங்க.",
        "MRN ARV-604417, course முடிஞ்சு ரெண்டு வாரமாச்சு.",
    ],
    "appointment.reschedule": [
        "நாளைக்கு appointment இருக்கு, வேற date-க்கு மாத்தணும்.",
        "அடுத்த வெள்ளிக்கிழமை மாலை convenient.",
    ],
    "appointment.cancel": [
        "இந்த வெள்ளிக்கிழமை appointment இருக்கு, cancel பண்ணணும்.",
        "பேரு Ravi, MRN தேவையா?",
    ],
    "appointment.confirm": [
        "நாளைக்கு appointment confirm ஆயிடுச்சான்னு check பண்ணணும்.",
        "பேரு Lakshmi, மொபைல் 99887 66554.",
    ],
    "appointment.book": [
        "எனக்கு ஒரு appointment book பண்ணனும்.",
        "Orthopaedics department, பேரு முருகேசன்.",
    ],
    "info.general": [
        "Visiting hours என்ன, parking இருக்கா?",
        "Wheelchair வேணும், attender ஒருத்தர் கூட வரலாமா?",
    ],
}


async def main() -> None:
    convo = ConversationManager(ConversationSettings())
    convo.load()
    llm = LlmClient(LlmSettings())
    llm.load()

    all_first: list[float] = []
    for intent, turns in FLOWS.items():
        print("=" * 72)
        print(intent)
        print("=" * 72)
        cid = intent
        print(f"AGENT   {convo.start_call(cid, agent_name='Gayathri')}\n")
        for text in turns:
            print(f"CALLER  {text}")
            started = time.perf_counter()
            first = None
            spoken = []
            async for event in convo.stream_utterance(cid, llm, text):
                if isinstance(event, AgentClause):
                    if first is None:
                        first = time.perf_counter() - started
                    spoken.append(event.text)
            whole = time.perf_counter() - started
            if first is not None:
                all_first.append(first)
            print(f"AGENT   {' '.join(spoken)}")
            print(f"        [first {first:.2f}s · turn {whole:.2f}s]\n")
        convo.end_call(cid)

    print("=" * 72)
    print(f"first clause: median {statistics.median(all_first):.2f}s  "
          f"min {min(all_first):.2f}s  max {max(all_first):.2f}s  (n={len(all_first)})")
    print(f"turns over the 2s target: {sum(t > 2.0 for t in all_first)} of {len(all_first)}")


if __name__ == "__main__":
    asyncio.run(main())
