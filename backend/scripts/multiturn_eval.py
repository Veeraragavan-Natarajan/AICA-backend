"""Multi-turn call replay: catches the failures a two-turn eval cannot see.

register_eval and flow_sweep both stop after two turns. Every failure the
20260902 transcript showed happens on turn 3 or later:

    AGENT   முருகேசன், குறிச்சுக்கிட்டேன். எந்த நாள் convenient Sir?
    CALLER  வர வே
    AGENT   முருகேசன், குறிச்சுக்கிட்டேன். எந்த நாள் convenient Sir?

- a turn repeated verbatim because the caller's answer did not parse
- a fact re-asked after the caller already gave it
- the register drifting into mostly-English clerical noise as a call goes on

So this replays whole calls and scores those three, reusing register_eval's
per-turn scorer for everything it already checks (script, symbols, length,
one-question-per-turn, fabricated IDs, unbacked action claims).

    LLM_TEMPERATURE=0 python -m backend.scripts.multiturn_eval
    LLM_TEMPERATURE=0 python -m backend.scripts.multiturn_eval appointment.book
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentClause, AgentTurn, ConversationManager
from backend.llm import LlmClient
from backend.scripts.register_eval import score_reply
from backend.settings import ConversationSettings, LlmSettings


@dataclass
class Call:
    name: str
    turns: list[str]
    # Facts the caller states, and the words an agent uses to ask for them
    # again. Keyed fact -> (turn index the caller first gives it, ask pattern).
    # A hit AFTER that turn is a re-ask, which THE LEDGER forbids outright.
    reasks: dict[str, tuple[int, str]] = field(default_factory=dict)


# The failing 20260902 transcript, plus one whole call for each of the other
# four served flows, plus the two cases the five-flow MVP added: a department
# no exemplar mentions, and a decline the caller recovers from.
CALLS: list[Call] = [
    Call(
        "appointment.book — the 20260902 transcript",
        [
            "Cardiology-ல ஒரு appointment book பண்ணணும்.",
            "பேஷண்ட் நான் தான். என் பேரு முருகேசன், வயசு 58.",
            "வர வியாழன் கிழமை.",
            "98407 21534.",
            "சரி, நன்றி.",
        ],
        reasks={
            "department": (0, r"எந்த\s*department|which\s*department|department\s*சொல்"),
            "patient name": (1, r"பேரு\s*சொல்|patient\s*(?:name|பேர)|name\s*சொல்"),
            "preferred day": (2, r"எந்த\s*நாள்|which\s*day|நாள்\s*சொல்|date\s*சொல்"),
            "mobile number": (3, r"mobile\s*number\s*சொல்|number\s*சொல்லுங்க"),
        },
    ),
    Call(
        # A department that appears in no exemplar, opened without the word
        # "appointment" - the two things the five-flow MVP has to get right.
        "appointment.book — an unseen department, no booking verb",
        [
            "என் பையனை குழந்தை doctor-கிட்ட காட்டணும்.",
            "அவன் பேரு Arjun, வயசு 6.",
            "நாளைக்கு காலைல வர முடியும்.",
            "98407 21534.",
        ],
        reasks={
            "department": (0, r"எந்த\s*department|which\s*department|department\s*சொல்"),
            "patient name": (1, r"பேரு\s*சொல்|patient\s*(?:name|பேர)"),
            "preferred day": (2, r"எந்த\s*நாள்|which\s*day|நாள்\s*சொல்"),
        },
    ),
    Call(
        "appointment.reschedule — a day and a part of the day are ONE answer",
        [
            "நாளைக்கு appointment இருக்கு, கொஞ்சம் மாத்தி தர முடியுமா?",
            "என் பேரு Lakshmi.",
            "அடுத்த வெள்ளிக்கிழமை மாலை convenient.",
            "98407 21534.",
        ],
        reasks={
            "patient name": (1, r"பேரு\s*சொல்|patient\s*(?:name|பேர)"),
            "new day": (2, r"எந்த\s*நாள்|which\s*day|எந்த\s*நேரம்|what\s*time|நேரம்\s*சொல்"),
        },
    ),
    Call(
        "appointment.cancel — the day is in the opener",
        [
            "இந்த வெள்ளிக்கிழமை appointment இருக்கு, cancel பண்ணணும்.",
            "பேரு Ravi.",
            "வேணாம், reschedule வேணாம். Cancel பண்ணிடுங்க.",
            "98407 21534.",
        ],
        reasks={
            "appointment day": (0, r"எந்த\s*நாள்|which\s*day|நாள்\s*சொல்"),
            "patient name": (1, r"பேரு\s*சொல்|patient\s*(?:name|பேர)"),
        },
    ),
    Call(
        "info.general — answer what was asked, do not recite the whole sheet",
        [
            "Visiting hours என்ன?",
            "ICU-ல எப்படி?",
            "Parking இருக்கா?",
            "சரி, நன்றி.",
        ],
        reasks={},
    ),
    Call(
        "emergency.escalate — must not loop on the address",
        [
            "என் அம்மாவுக்கு திடீர்னு நெஞ்சு வலி, மூச்சு வாங்குது!",
            "Velachery, 4th Cross Street, number 9.",
            "ஆமா, அவங்க பேசுறாங்க. ரொம்ப வியர்க்குது.",
            "Aspirin கொடுக்கலாமா?",
        ],
        reasks={"address": (1, r"address\s*சொல்|எங்க\s*இருக்|முகவரி")},
    ),
    Call(
        # The scope line must not end the call: the caller hears the five and
        # picks one, and the rest of the booking has to work normally.
        "out of scope — declined, then the caller picks one of the five",
        [
            "Discharge bill-ல ஒரு charge ரெண்டு தடவை போட்டு இருக்கீங்க.",
            "சரி, அப்போ Dermatology-ல ஒரு appointment வேணும்.",
            "என் பேரு Kavitha.",
            "அடுத்த திங்கள் காலைல.",
            "98407 21534.",
        ],
        reasks={
            "patient name": (2, r"பேரு\s*சொல்|patient\s*(?:name|பேர)"),
            "preferred day": (3, r"எந்த\s*நாள்|which\s*day|நாள்\s*சொல்"),
        },
    ),
]


def _norm(text: str) -> str:
    """Compare turns for repetition the way a caller hears them."""
    return " ".join(re.sub(r"[^\w஀-௿\s]", " ", text.casefold()).split())


def _overlap(a: str, b: str) -> float:
    """Jaccard over words. A near-repeat is a repeat to the person on the line."""
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class CallResult:
    name: str
    replies: list[str]
    latencies: list[float]
    # Time to the FIRST clause, which is when TTS starts and therefore when the
    # caller stops hearing silence. The full-turn number below it is the model
    # still writing while audio is already playing.
    first_clause: list[float] = field(default_factory=list)
    repeats: list[str] = field(default_factory=list)
    reasked: list[str] = field(default_factory=list)
    turn_problems: list[str] = field(default_factory=list)
    tamil_ratio: float = 0.0

    @property
    def clean(self) -> bool:
        return not (self.repeats or self.reasked or self.turn_problems)


async def run_call(convo: ConversationManager, llm: LlmClient, call: Call) -> CallResult:
    cid = f"multiturn::{call.name}"
    convo.start_call(cid, agent_name="Gayathri")
    result = CallResult(call.name, [], [])
    tamil_words = total_words = 0

    for index, text in enumerate(call.turns):
        started = time.perf_counter()
        spoken: list[str] = []
        turn_text = ""
        turn_ungrounded: tuple[str, ...] = ()
        turn_claims: tuple[str, ...] = ()
        first = None
        async for event in convo.stream_utterance(cid, llm, text):
            if isinstance(event, AgentClause):
                if first is None:
                    first = time.perf_counter() - started
                spoken.append(event.text)
            elif isinstance(event, AgentTurn):
                turn_text = event.text
                turn_ungrounded = event.ungrounded
                turn_claims = event.unbacked_claims
        elapsed = time.perf_counter() - started
        reply = turn_text or " ".join(spoken)
        result.replies.append(reply)
        result.latencies.append(elapsed)
        result.first_clause.append(elapsed if first is None else first)

        score = score_reply(
            reply,
            list(turn_ungrounded),
            turn_claims,
            caller_said=" ".join(call.turns),
        )
        tamil_words += round(score.tamil_ratio * score.scored_words)
        total_words += score.scored_words
        for problem in score.problems:
            result.turn_problems.append(f"turn {index + 1}: {problem}")

        # Repetition: against every earlier agent turn in THIS call.
        for earlier_index, earlier in enumerate(result.replies[:-1]):
            if _norm(earlier) and _overlap(earlier, reply) >= 0.85:
                result.repeats.append(
                    f"turn {index + 1} repeats turn {earlier_index + 1}: {reply[:70]}"
                )
                break

        # Re-asking: the caller gave this fact on an earlier turn.
        for fact, (given_at, pattern) in call.reasks.items():
            if index > given_at and re.search(pattern, reply, re.IGNORECASE):
                result.reasked.append(f"turn {index + 1}: re-asked {fact}")

    convo.end_call(cid)
    result.tamil_ratio = tamil_words / total_words if total_words else 0.0
    return result


async def main(only: list[str]) -> int:
    convo = ConversationManager(ConversationSettings())
    convo.load()
    llm = LlmClient(LlmSettings())
    llm.load()

    calls = [c for c in CALLS if not only or any(o in c.name for o in only)]
    results = [await run_call(convo, llm, call) for call in calls]

    for call, result in zip(calls, results):
        print("=" * 74)
        print(result.name)
        print("=" * 74)
        for text, reply, latency in zip(call.turns, result.replies, result.latencies):
            print(f"CALLER  {text}")
            print(f"AGENT   {reply}")
            print(f"        [{latency:.2f}s]\n")
        for problem in result.repeats + result.reasked + result.turn_problems:
            print(f"  FAIL  {problem}")
        print(f"  Tamil {result.tamil_ratio:.0%} of words\n")

    every_latency = [t for r in results for t in r.latencies]
    every_latency.sort()
    every_first = sorted(t for r in results for t in r.first_clause)
    clean = sum(r.clean for r in results)
    repeats = sum(len(r.repeats) for r in results)
    reasks = sum(len(r.reasked) for r in results)
    problems = sum(len(r.turn_problems) for r in results)
    overall_tamil = sum(r.tamil_ratio for r in results) / len(results)

    print("=" * 74)
    print(f"clean calls      {clean}/{len(results)}")
    print(f"repeated turns   {repeats}")
    print(f"re-asked facts   {reasks}")
    print(f"turn problems    {problems}")
    print(f"Tamil ratio      {overall_tamil:.0%}  (target 65%)")
    if every_latency:
        median = every_latency[len(every_latency) // 2]
        p95 = every_latency[int(len(every_latency) * 0.95) - 1]
        first_median = every_first[len(every_first) // 2]
        first_p95 = every_first[int(len(every_first) * 0.95) - 1]
        print(f"first clause     median {first_median:.2f}s  p95 {first_p95:.2f}s  (target 1.0-1.5s)")
        print(f"full turn        median {median:.2f}s  p95 {p95:.2f}s")
    print("=" * 74)
    return 0 if repeats == 0 and reasks == 0 else 1


raise SystemExit(asyncio.run(main(sys.argv[1:])))
