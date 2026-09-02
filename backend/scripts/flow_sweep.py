"""One realistic opener + one follow-up for each of the five served flows,
through the real ConversationManager and the real Ollama.

The routing is proven without a model in backend/test_mvp_scope.py, so a bad
transcript HERE is a model quality question, not a router question. Two extra
rows are not flows at all and are the two things the five-flow MVP has to get
right beyond the happy path: a department that appears in no exemplar, and an
out-of-scope turn that must cost no LLM call and must not derail the call it
interrupts.

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
    "appointment.book": [
        "எனக்கு ஒரு appointment book பண்ணனும்.",
        "Orthopaedics department, பேரு முருகேசன்.",
    ],
    # Not the trigger phrase: "மாத்தி தர முடியுமா" is how a caller actually
    # asks, and it is the wording the widened trigger was added for.
    "appointment.reschedule": [
        "நாளைக்கு appointment இருக்கு, கொஞ்சம் மாத்தி தர முடியுமா?",
        "அடுத்த வெள்ளிக்கிழமை மாலை convenient.",
    ],
    "appointment.cancel": [
        "இந்த வெள்ளிக்கிழமை appointment இருக்கு, cancel பண்ணணும்.",
        "பேரு Ravi, MRN தேவையா?",
    ],
    "info.general": [
        "Visiting hours என்ன, parking இருக்கா?",
        "Wheelchair வேணும், attender ஒருத்தர் கூட வரலாமா?",
    ],
    # A department the exemplars never mention, opened without the word
    # "appointment" - the two things a five-flow MVP has to get right.
    "appointment.book (unseen department)": [
        "என் பையனை குழந்தை doctor-கிட்ட காட்டணும்.",
        "வயசு 6, நாளைக்கு காலைல வர முடியும்.",
    ],
    # Must cost no LLM call at all and must not derail the booking behind it.
    "out of scope (bill, then back to booking)": [
        "என் bill-ல ஒரு charge தப்பா இருக்கு.",
        "சரி, அப்போ Dermatology-ல ஒரு appointment வேணும்.",
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
