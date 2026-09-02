"""The five-flow contract, in the words callers actually use.

AICA answers five flows and declines the rest (prompt_builder.SUPPORTED_INTENTS).
That makes the router load-bearing in a way it was not when it served twenty:
a miss no longer means a slightly-wrong playbook, it means the scope line said
to a caller the desk does serve. So the tables here are paraphrases, not the
trigger phrases - every line is a way of asking for one of the five that the
pattern does NOT contain literally.

The out-of-scope half is the same bargain from the other side: a turn belonging
to one of the fifteen dropped flows must reach the caller as a decline and must
not reach the model at all.

Nothing here needs Ollama. Register and Tamil quality are measured live by
backend/scripts/flow_sweep.py and register_eval.py; this is the floor beneath
them - if the routing is wrong, the register does not matter.
"""

from __future__ import annotations

import pytest

from .conversation import _OUT_OF_SCOPE, ConversationManager
from .llm import LlmReply
from .prompt_builder import SUPPORTED_INTENTS, detect_intent, names_a_department
from .settings import ConversationSettings, LlmSettings


def _route(text: str) -> str | None:
    """What conversation.py will pick, department fallback included."""
    detected = detect_intent(text)
    if detected is None and names_a_department(text):
        return "appointment.book"
    return detected


# --- the five, said every way a caller says them -------------------------


@pytest.mark.parametrize(
    "caller_turn",
    [
        "எனக்கு ஒரு appointment book பண்ணனும்",
        "Doctor-a பாக்கணும், time வேணும்",
        "நாளைக்கு consult ஒண்ணு வேணுமே",
        "Cardiology-ல slot இருக்கா?",
        "என் அம்மாவை doctor-கிட்ட காட்டணும்",
        "புது appointment வேணும்",
        "Dr. Kumar-a பாக்க முடியுமா?",
        "OP-க்கு வர token வேணும்",
        "next week doctor appointment வேணும்",
        "ஒரு டாக்டர் அப்பாயின்ட்மென்ட் போட்டு தாங்க",
        "Neuro specialist-a எப்போ பாக்கலாம்?",
        "என் அப்பாவுக்கு ஒரு checkup fix பண்ணுங்க",
    ],
)
def test_booking_is_recognised_however_it_is_asked(caller_turn: str) -> None:
    assert _route(caller_turn) == "appointment.book"


@pytest.mark.parametrize(
    "caller_turn",
    [
        "நாளைக்கு appointment இருக்கு, வேற date-க்கு மாத்தணும்",
        "appointment-ஐ postpone பண்ணணும்",
        "என் appointment-ஐ prepone பண்ண முடியுமா?",
        "அன்னைக்கு வர முடியாது, வேற நாள் இருக்கா?",
        "Time-ஐ கொஞ்சம் மாத்தி தர முடியுமா?",
        "reschedule பண்ணனும்",
        "Friday-க்கு பதிலா Monday-க்கு மாத்துங்க",
        "appointment-ஐ அடுத்த வாரத்துக்கு தள்ளி போடுங்க",
        "morning-ல வர முடியாது, evening-க்கு மாத்துங்க",
        "booking date change பண்ணனும்",
    ],
)
def test_rescheduling_is_recognised_however_it_is_asked(caller_turn: str) -> None:
    assert _route(caller_turn) == "appointment.reschedule"


@pytest.mark.parametrize(
    "caller_turn",
    [
        "இந்த வெள்ளிக்கிழமை appointment இருக்கு, cancel பண்ணணும்",
        "appointment வேணாம், cancel பண்ணிடுங்க",
        "என் booking-ஐ ரத்து பண்ணுங்க",
        "நான் வர மாட்டேன், appointment-ஐ எடுத்துடுங்க",
        "கேன்சல் பண்ணிடுங்க Sir",
        "appointment cancel பண்ண தான் call பண்ணேன்",
        "இனிமே அந்த appointment வேணாம்",
        "Booking-ஐ cancel பண்ணிட முடியுமா?",
    ],
)
def test_cancellation_is_recognised_however_it_is_asked(caller_turn: str) -> None:
    assert _route(caller_turn) == "appointment.cancel"


@pytest.mark.parametrize(
    "caller_turn",
    [
        "Visiting hours என்ன, parking இருக்கா?",
        "OP timing எத்தனை மணி?",
        "ICU-ல எப்போ பாக்கலாம்?",
        "Hospital எங்க இருக்கு?",
        "Wheelchair கிடைக்குமா?",
        "Canteen எத்தனை மணி வரைக்கும் open?",
        "Sunday hospital வேலை பாக்குமா?",
        "Attender ஒருத்தர் தங்கலாமா?",
        "எப்படி வர்றது, address சொல்லுங்க",
    ],
)
def test_general_information_is_recognised_however_it_is_asked(caller_turn: str) -> None:
    assert _route(caller_turn) == "info.general"


@pytest.mark.parametrize(
    "caller_turn",
    [
        "என் அம்மாவுக்கு திடீர்னு நெஞ்சு வலி, மூச்சு வாங்குது!",
        "ambulance வேணும் சீக்கிரம்",
        "அவரு மயக்கம் போட்டு விழுந்துட்டாரு",
        "ரத்தம் நிக்க மாட்டேங்குது",
        "வலிப்பு வந்துடுச்சு, என்ன பண்றது",
        "அவரால பேச முடியல, ஒரு பக்கம் செயலிழந்துடுச்சு",
        "எமர்ஜென்சி, உடனே வாங்க",
        "அம்புலான்ஸ் அனுப்புங்க",
    ],
)
def test_an_emergency_is_recognised_however_it_is_said(caller_turn: str) -> None:
    assert _route(caller_turn) == "emergency.escalate"


def test_an_emergency_still_outranks_an_in_scope_booking_in_the_same_breath() -> None:
    # The one row where a false positive is cheaper than a miss keeps that
    # property now that four of the five flows it competes with are served.
    assert _route("appointment book பண்ணனும், ஆனா அப்பாவுக்கு நெஞ்சு வலி") == "emergency.escalate"


# --- and the fifteen it declines -----------------------------------------


OUT_OF_SCOPE_TURNS = [
    "பில்-ல ஒரு charge தப்பா இருக்கு",
    "நேத்து blood test பண்ணேன், report வந்துடுச்சா?",
    "என் அப்பாவுக்கு tablets தீர்ந்துடுச்சு, refill வேணும்",
    "Gall bladder surgery insurance-ல cover ஆகுமா?",
    "என் அப்பாவோட discharge summary copy வேணும்",
    "வேற hospital-க்கு referral letter status என்ன?",
    "நான் புதுசா register பண்ணணும், முதல் தடவை வர்றேன்",
    "ரெண்டு மணி நேரம் காக்க வெச்சீங்க, staff மோசமா பேசுனாங்க",
    "இது மூணாவது தடவை call பண்றது, என் பணம் இன்னும் வரல்",
    "மருந்து சாப்பிட்ட பிறகு தூக்கம் வர்றது normal-ஆ?",
    "B positive blood நாளைக்கு கிடைக்குமா?",
    "என் mobile number update பண்ணணும்",
    "நாளைக்கு மழை பெய்யுமா",
    "cricket score என்ன சொல்லுங்க",
]


@pytest.mark.parametrize("caller_turn", OUT_OF_SCOPE_TURNS)
def test_out_of_scope_work_never_resolves_to_one_of_the_five(caller_turn: str) -> None:
    assert _route(caller_turn) not in SUPPORTED_INTENTS


class _NeverCalledLlm:
    """A scripted reply here would mean the model was asked to improvise."""

    settings = LlmSettings()

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        self.calls.append(messages)
        raise AssertionError("an out-of-scope turn reached the model")


def _manager() -> ConversationManager:
    manager = ConversationManager(ConversationSettings())
    manager._deterministic_flows = False
    manager.prompts._core = "core"
    manager.prompts._playbooks = {}
    return manager


@pytest.mark.parametrize("caller_turn", OUT_OF_SCOPE_TURNS)
async def test_out_of_scope_work_is_declined_without_an_llm_call(caller_turn: str) -> None:
    manager = _manager()
    manager.start_call("conn-scope", agent_name="Gayathri")
    llm = _NeverCalledLlm()

    events = [event async for event in manager.stream_utterance("conn-scope", llm, caller_turn)]

    assert events[-1].text == _OUT_OF_SCOPE
    assert llm.calls == []


class _OneReply:
    settings = LlmSettings()

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[dict]] = []

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        from .llm import ReplyComplete, TextDelta

        self.calls.append(messages)
        yield TextDelta(self._content)
        yield ReplyComplete(LlmReply(content=self._content))


async def test_a_bill_question_mid_booking_is_declined_without_losing_the_booking() -> None:
    """The scope line is not a hang-up. A caller who wanders off the five for
    one turn hears what the desk does and carries on where they were, so the
    decline must not clear the flow the call is already running."""
    manager = _manager()
    manager.start_call("conn-mixed", agent_name="Gayathri")
    llm = _OneReply("கண்டிப்பா Sir. Patient பேரு சொல்லுங்க?")

    booking = [
        e async for e in manager.stream_utterance("conn-mixed", llm, "Ortho-ல appointment வேணும்")
    ][-1]
    bill = [
        e async for e in manager.stream_utterance("conn-mixed", llm, "பில் எவ்வளவு ஆகும்?")
    ][-1]

    assert booking.text != _OUT_OF_SCOPE
    assert bill.text == _OUT_OF_SCOPE
    assert manager._sessions["conn-mixed"].intent == "appointment.book"
    assert len(llm.calls) == 1, "the bill turn was sent to the model anyway"


async def test_a_department_answered_mid_cancellation_stays_a_cancellation() -> None:
    """names_a_department is why this test exists: "Cardiology" said on its own
    is a new booking on turn one and an ANSWER on turn three."""
    manager = _manager()
    manager.start_call("conn-dept", agent_name="Gayathri")
    llm = _OneReply("சரி Sir. எந்த department?")

    [e async for e in manager.stream_utterance("conn-dept", llm, "என் appointment cancel பண்ணணும்")]
    [e async for e in manager.stream_utterance("conn-dept", llm, "Cardiology")]

    assert manager._sessions["conn-dept"].intent == "appointment.cancel"


def test_the_scope_line_names_every_flow_the_desk_serves() -> None:
    """A decline that does not say what IS answered leaves the caller nowhere.
    Checked against SUPPORTED_INTENTS so adding a sixth flow without telling
    callers about it fails here."""
    spoken = _OUT_OF_SCOPE.lower()
    for word in ("appointment", "cancel", "timing", "emergency"):
        assert word in spoken, f"the scope line never mentions {word}"
    assert "மாத்த" in _OUT_OF_SCOPE, "the scope line never mentions rescheduling"
    assert len(SUPPORTED_INTENTS) == 5, "the scope line lists five things; update both together"


def test_no_spoken_fixed_line_guesses_the_callers_gender() -> None:
    """_normalize_spoken_register drops the address from the model's speech
    whenever the caller has not revealed their gender. A canned line that says
    "Sir" anyway is then the one sentence in the call that guessed, and these
    are the lines a caller hears most - the scope line, the stuck ladder and
    the clinical refusals are all fixed strings.

    Enumerated rather than reflected over every uppercase name in the module,
    because _LANGUAGE_REMINDER legitimately names both address forms: it is an
    instruction TO the model, not a sentence anybody says.
    """
    from . import conversation as c

    spoken: list[str] = [
        c._OUT_OF_SCOPE,
        c._GO_AHEAD,
        c._CANNOT_RECALL,
        c._ECHO_RECOVERY,
        c._ACTION_RECOVERY,
        c._EMERGENCY_ACTION_RECOVERY,
        c._EMERGENCY_OPENING,
        c._DISTRESS_CLARIFICATION,
        c._DIAGNOSIS_REFUSAL,
        c._EMERGENCY_MEDICINE_REFUSAL,
        c._LAB_VALUE_REFUSAL,
        c.OPENING_LINE,
        *c._STUCK_REPLIES,
        *c._EMERGENCY_STUCK_REPLIES,
    ]

    offenders = [line for line in spoken if c._ADDRESS_RE.search(line)]

    assert not offenders, "fixed lines that assume a gender: " + "; ".join(offenders)
