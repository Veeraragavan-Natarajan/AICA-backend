"""Mechanically scores whether replies are actually spoken Tamil-English
code-mix, on scenarios that appear NOWHERE in golden/ — new names, new
intents, new phrasing.

Why this exists: "it looks Tamil" is not a check. The two real failure modes
observed on this pipeline were (a) a model that answered in fluent pure
English, and (b) a model that produced Tamil script but semantic gibberish.
Neither is caught by the golden-flow tool-call harness (golden_eval.py), which
only asks whether the right tool fired.

So this scores each reply on the properties golden/main_prompt.txt Sec2 and
Sec3 actually demand, and prints the transcript for a human to read. It is a
FLOOR: passing means nothing is mechanically wrong, not that the Tamil is good.
A native reader still has to read the transcript.

Scenarios are deliberately unseen so a fine-tuned model that merely memorised
the 20 training flows scores badly here - that is the point.

Usage:
    python -m backend.scripts.register_eval
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentTurn, ConversationManager
from backend.llm import LlmClient
from backend.scripts.smoke_llm import SAMPLE_METADATA
from backend.settings import ConversationSettings, LlmSettings

_TAMIL_RE = re.compile(r"[஀-௿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Devanagari / Kannada / Telugu - Sec2 forbids all three outright.
_WRONG_SCRIPT_RE = re.compile(r"[ऀ-ॿಀ-೿ఀ-౿]")
# Sec2 "audio stability": these must never reach a TTS engine. The numbered-list
# arm is anchored to line start on purpose - unanchored, it fires on the period
# after any time or amount ("மாலை 5:00. எது...") and reports a correct turn as
# broken, which is worse than missing a list.
_SPOKEN_SYMBOL_RE = re.compile(r"[*#|•→\[\]{}<>_`]|^\s*\d+[.)]\s", re.MULTILINE)
_WORD_RE = re.compile(r"[\w஀-௿]+")

# Superseded as the primary check by backend/grounding.py, which decides
# "invented" against what this call's tools actually returned rather than
# against a hand-written list of suspicious-looking numbers. Kept as a cheap
# extra net for the placeholder-ish values a model reaches for when it is
# making something up outright.
_FABRICATION_RE = re.compile(r"\b(?:98765\d{5}|1234\d+|ARV-0{3}\d+|APT-0{3}\d+)\b")


@dataclass
class Scenario:
    name: str
    turns: list[str]
    # Intents never exercised by the 20 golden flows, so memorisation cannot help.
    note: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        "dietician referral (unseen intent)",
        [
            "மேடம், sugar patient-க்கு diet plan வேணும். Dietician-ஐ பாக்க முடியுமா?",
            "என் பேரு Kalaiselvi. இந்த வாரம் எப்போ வேணாலும் வரலாம்.",
        ],
    ),
    Scenario(
        "physiotherapy after knee surgery (unseen intent)",
        [
            "Knee replacement ஆயிடுச்சு, physiotherapy session book பண்ணணும்.",
            "வாரத்துக்கு ரெண்டு தடவை வர முடியும்.",
        ],
    ),
    Scenario(
        "travel vaccination (unseen intent)",
        [
            "சார், நான் அடுத்த மாசம் foreign travel போறேன். Vaccination வேணும்.",
            "Yellow fever certificate வேணும்னு சொன்னாங்க.",
        ],
    ),
    Scenario(
        "blood bank enquiry (unseen intent)",
        [
            "B positive blood இருக்கா? எங்க அத்தைக்கு operation.",
            "நாளைக்கு காலைக்கு தேவைப்படும்.",
        ],
    ),
    Scenario(
        "mobile number update (unseen intent)",
        [
            "என்னோட registered mobile number மாத்தணும் மேடம்.",
            "பழைய number தான் இப்போ வேலை செய்யல.",
        ],
    ),
    Scenario(
        "caller speaks English throughout (mirroring rule, Sec2)",
        [
            "Hello, I need to book an appointment with a skin doctor.",
            "Sometime this weekend would be good.",
        ],
        note="Sec2 allows shifting to majority English here, but Tamil address forms should survive.",
    ),
    Scenario(
        "emergency mid-billing (override, Sec6A)",
        [
            "மேடம், bill-ல ஒரு charge சரி இல்லன்னு நினைக்கிறேன்.",
            "ஐயோ மேடம், என் கணவருக்கு திடீர்னு நெஞ்சு வலி, மூச்சு வாங்குது!",
        ],
        note="Turn 2 must abandon billing and take the address first.",
    ),
]


@dataclass
class TurnScore:
    reply: str
    tamil_ratio: float
    has_english: bool
    wrong_script: bool
    symbols: bool
    word_count: int
    question_count: int
    # The words the Tamil ratio was computed over: word_count minus the ones
    # Sec2 forces to be Latin and the ones the caller supplied. See score_reply.
    scored_words: int = 0
    fabricated: list[str] = field(default_factory=list)
    # Actions claimed as done with no tool call behind them - see
    # backend/grounding.py. A turn carrying one is never clean.
    unbacked_claims: tuple[str, ...] = ()

    @property
    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.reply.strip():
            issues.append("EMPTY reply")
            return issues
        if self.scored_words >= _MIN_WORDS_TO_SCORE_RATIO and self.tamil_ratio < 0.35:
            issues.append(f"too little Tamil ({self.tamil_ratio:.0%} of words)")
        # There is deliberately NO per-turn "too much Tamil" check. Sec2's
        # "roughly 65% Tamil / 35% English" is a property of a CALL, not of
        # every sentence in it, and golden/flow_exemplars.json is full of
        # entirely-Tamil agent turns that are exactly right - "கவலைப்படாதீங்க
        # மேடம், நான் கேட்டுட்டு சொல்றேன். குழந்தைக்கு எத்தனை வயசு?" has no
        # English in it and needs none. Flagging those per turn marked correct
        # output as broken, which is worse than missing a genuinely
        # English-free call. The aggregate check in main() is where that rule
        # actually belongs.
        if self.wrong_script:
            issues.append("Devanagari/Kannada/Telugu script present")
        if self.symbols:
            issues.append("unspeakable symbol/bullet in a voice reply")
        if self.word_count > 45:
            issues.append(f"turn too long ({self.word_count} words, limit 40)")
        if self.question_count > 1:
            issues.append(f"{self.question_count} questions in one turn (limit 1)")
        if self.fabricated:
            issues.append(f"fabricated identifier(s): {', '.join(self.fabricated)}")
        for claim in self.unbacked_claims:
            issues.append(f"{claim} - with NO tool call behind it")
        return issues


# Words that are Latin because the register rule SAYS they must be, and that
# therefore say nothing about whether the agent is drifting into English.
#
# Sec2 is explicit on both: "ALL numbers, dates, times, money, IDs in Latin
# digits/English" and "Write `Sir` in Latin script for a male caller". Counting
# them against the Tamil ratio marks the agent down for obeying its own prompt,
# and on a short turn that is the whole score: "98407 21534, குறிச்சுக்கிட்டேன்
# Sir." scored 25% Tamil, of which three of the four words had no choice.
#
# This was measuring the wrong thing, not aiming at the wrong number - the
# 65/35 target is about prose drifting into English, and a phone number is not
# prose. The call-level ratio is unchanged in what it means; it just stops
# being dominated by digits.
_MANDATED_LATIN_RE = re.compile(r"^(?:\d+|sir|madam)$", re.IGNORECASE)

# Below this a per-turn ratio is noise rather than a measurement: "Kavitha Sir.
# Mobile number சொல்லுங்க?" is exactly how a Chennai hospital desk speaks, and
# with three scoreable words no threshold can tell it from a register failure.
# Drift is a property of a stretch of speech, which is what the call-level
# ratio measures and why that one has no floor.
_MIN_WORDS_TO_SCORE_RATIO = 6


def score_reply(
    reply: str,
    ungrounded: tuple[str, ...] = (),
    claims: tuple[str, ...] = (),
    caller_said: str = "",
) -> TurnScore:
    """Score one reply.

    `ungrounded` comes from backend/grounding.py, which unlike the regex below
    has the call's actual tool results in hand and so can tell an ID the agent
    looked up from one it made up - including the ones it copies verbatim out
    of its own few-shot exemplars, which look entirely plausible.

    `caller_said` is everything the caller has said this call. A Latin word the
    agent got FROM the caller - their name, their doctor's name, the department
    they asked for - is the ledger working, not the agent choosing English, so
    it is excluded from the ratio the same way digits are.
    """
    caller_words = {w.casefold() for w in _WORD_RE.findall(caller_said)}
    words = [
        w
        for w in _WORD_RE.findall(reply)
        if not _MANDATED_LATIN_RE.match(w)
        and not (w.casefold() in caller_words and not _TAMIL_RE.search(w))
    ]
    tamil_words = [w for w in words if _TAMIL_RE.search(w)]
    return TurnScore(
        reply=reply,
        tamil_ratio=len(tamil_words) / len(words) if words else 0.0,
        has_english=bool(_LATIN_RE.search(reply)),
        wrong_script=bool(_WRONG_SCRIPT_RE.search(reply)),
        symbols=bool(_SPOKEN_SYMBOL_RE.search(reply)),
        word_count=len(_WORD_RE.findall(reply)),
        scored_words=len(words),
        question_count=reply.count("?"),
        fabricated=sorted(set(ungrounded) | set(_FABRICATION_RE.findall(reply))),
        unbacked_claims=tuple(claims),
    )


async def main() -> None:
    llm_settings = LlmSettings()
    print(f"LLM_BASE_URL={llm_settings.base_url}  LLM_MODEL={llm_settings.model}\n")

    llm = LlmClient(llm_settings)
    llm.load()
    manager = ConversationManager(ConversationSettings())
    manager.load()

    total = clean = 0
    call_warnings: list[str] = []
    for index, scenario in enumerate(SCENARIOS):
        call_words: list[str] = []
        connection_id = f"register-{index}"
        print("=" * 72)
        print(f"{scenario.name}")
        if scenario.note:
            print(f"  note: {scenario.note}")
        print("=" * 72)
        print(f"agent: {manager.start_call(connection_id, **SAMPLE_METADATA)}")

        for turn in scenario.turns:
            print(f"\ncaller: {turn}")
            try:
                # stream_utterance rather than handle_utterance: the final
                # AgentTurn carries the grounding verdict for this turn, which
                # is the check that distinguishes a looked-up ID from a
                # parroted one.
                reply: str = ""
                ungrounded: tuple[str, ...] = ()
                claims: tuple[str, ...] = ()
                async for event in manager.stream_utterance(connection_id, llm, turn):
                    if isinstance(event, AgentTurn):
                        reply, ungrounded = event.text, event.ungrounded
                        claims = event.unbacked_claims
            except Exception as error:
                print(f"  [FAILED: {error!r}]")
                total += 1
                continue

            score = score_reply(reply, ungrounded, claims, caller_said=" ".join(scenario.turns))
            call_words += _WORD_RE.findall(reply)
            print(f"agent : {reply}")
            print(
                f"        tamil={score.tamil_ratio:.0%} english={'y' if score.has_english else 'n'} "
                f"words={score.word_count} questions={score.question_count}"
            )
            total += 1
            if score.problems:
                for problem in score.problems:
                    print(f"        !! {problem}")
            else:
                clean += 1
                print("        ok")
        manager.end_call(connection_id)

        # Sec2's 65/35 Tamil-English balance, checked where it belongs: across
        # the whole call. A single all-Tamil sentence is normal; an entire call
        # without one English hospital word is the register drifting into
        # literary Tamil, and an entire call without Tamil is the English-drift
        # failure this eval was built for.
        if call_words:
            ratio = len([w for w in call_words if _TAMIL_RE.search(w)]) / len(call_words)
            if ratio > 0.95:
                call_warnings.append(f"{scenario.name}: no English code-mix across the whole call ({ratio:.0%} Tamil)")
            elif ratio < 0.35:
                call_warnings.append(f"{scenario.name}: call drifted to English ({ratio:.0%} Tamil)")
            print(f"\n  [call-level Tamil ratio: {ratio:.0%}]")

    print("\n" + "=" * 72)
    print(f"{clean}/{total} turns mechanically clean")
    if call_warnings:
        print(f"\n{len(call_warnings)}/{len(SCENARIOS)} calls broke the call-level register balance:")
        for warning in call_warnings:
            print(f"  !! {warning}")
    else:
        print(f"\nall {len(SCENARIOS)} calls held the call-level Tamil/English balance")
    print(
        "\nFloor check only - clean means nothing is mechanically wrong, NOT that\n"
        "the Tamil reads naturally. Read the transcript above before trusting it."
    )


if __name__ == "__main__":
    asyncio.run(main())
