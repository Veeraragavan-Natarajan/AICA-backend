"""Self-check for ConversationManager's prompt templating and tool-call loop.

Uses a fake LlmClient with scripted replies (no real model, no network) so the
orchestration logic - ledger updates, message-history shape, loop termination,
multi-turn continuity - is exercised deterministically. Bypasses load()'s file
read the same way test_asr.py bypasses IndicConformerAsr.load(): the template
is set directly on the manager.
"""

from __future__ import annotations

import contextlib
import logging

from .conversation import (
    KNOWN_PLACEHOLDERS,
    _ACTION_RECOVERY,
    _CANNOT_RECALL,
    _DISTRESS_CLARIFICATION,
    _ECHO_RECOVERY,
    _EMERGENCY_ACTION_RECOVERY,
    _EMERGENCY_OPENING,
    _EMERGENCY_STUCK_REPLIES,
    _OUT_OF_SCOPE,
    _STUCK_REPLIES,
    _normalize_spoken_register,
    _with_language_reminder,
    AgentClause,
    AgentTurn,
    ConversationManager,
    render_template,
)
from .llm import LlmReply, ReplyComplete, TextDelta, ToolCall
from .settings import ConversationSettings, LlmSettings

TEMPLATE = "Hello {{agent_name}}, caller {{caller_mobile}} mrn {{mrn}} unknown {{bogus_var}}."


class _ScriptedLlm:
    """Returns replies from a script in order; records every messages/tools call.

    Implements stream(), the interface conversation.py actually consumes, and
    deliberately emits each reply's content in small fragments rather than one
    lump - a fake that yielded whole sentences would never exercise the clause
    chunker's job of reassembling a clause split across deltas, which is the
    part most likely to break.
    """

    # The real client carries the settings conversation.py sizes the history
    # trim against (num_ctx / max_tokens), so the fake has to as well - a
    # fake missing them would exercise a trim that never bounds anything.
    settings = LlmSettings()

    def __init__(self, replies: list[LlmReply]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        self.calls.append([dict(m) for m in messages])
        reply = self._replies.pop(0)
        for index in range(0, len(reply.content), 7):
            yield TextDelta(reply.content[index : index + 7])
        yield ReplyComplete(reply)

    async def complete(self, messages: list[dict], tools: list[dict]) -> LlmReply:
        async for event in self.stream(messages, tools):
            if isinstance(event, ReplyComplete):
                return event.reply
        raise AssertionError("scripted stream produced no ReplyComplete")


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

@contextlib.contextmanager
def _captured_log_records(logger_name: str):
    logger = logging.getLogger(logger_name)
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _make_manager() -> ConversationManager:
    manager = ConversationManager(ConversationSettings())
    manager._deterministic_flows = False
    # Stub the builder rather than reading the real prompt files: these tests
    # assert conversation plumbing, not prompt content.
    manager.prompts._core = TEMPLATE
    manager.prompts._playbooks = {}
    return manager


def test_render_template_substitutes_known_placeholders_only() -> None:
    with _captured_log_records("aica.conversation") as records:
        rendered = render_template(
            TEMPLATE, {"agent_name": "Gayathri", "caller_mobile": "9840721534", "mrn": "ARV-118342"}
        )

    assert rendered == "Hello Gayathri, caller 9840721534 mrn ARV-118342 unknown {{bogus_var}}."
    assert any("bogus_var" in record.getMessage() for record in records)
    assert {"agent_name", "caller_mobile", "mrn"} <= KNOWN_PLACEHOLDERS


def test_start_call_returns_greeting_without_any_llm_call() -> None:
    manager = _make_manager()

    greeting = manager.start_call("conn-1", agent_name="Gayathri", caller_mobile="9840721534")

    assert "Gayathri" in greeting
    session = manager._sessions["conn-1"]
    assert session.messages[0]["role"] == "system"
    assert session.messages[1] == {"role": "assistant", "content": greeting}
    assert session.ledger["agent_name"] == "Gayathri"


async def test_handle_utterance_without_active_session_raises() -> None:
    manager = _make_manager()
    llm = _ScriptedLlm([])

    try:
        await manager.handle_utterance("no-such-conn", llm, "hello")
    except RuntimeError:
        return
    raise AssertionError("handle_utterance must refuse to run without an active session")


async def test_an_ordinary_hospital_turn_is_answered_by_the_model() -> None:
    """Two tests used to stand here asserting `llm.calls == []` on a dietician
    request and on a booking - that is, requiring that the model NOT be asked.
    They passed against a regex decision tree that answered fifteen intents
    from fixed strings, and that tree is why callers heard the same sentence
    over and over: it recomputed its state from the transcript every turn, so
    any answer its patterns did not recognise left it on the same branch. A
    date of birth spoken as "17-04-1968" never contains the literal words
    "date of birth", so that branch never advanced at all.

    So this asserts the opposite, and deliberately: ordinary hospital work is a
    conversation and reaches the model. What must NOT reach it is a hard
    clinical refusal, which is the test below.
    """
    manager = _make_manager()
    manager.start_call("conn-general", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="கண்டிப்பா Sir. Patient பேரு சொல்லுங்க?")])

    events = [
        event
        async for event in manager.stream_utterance(
            "conn-general", llm, "dietician appointment வேணும்"
        )
    ]

    assert llm.calls, "an ordinary hospital request must be answered by the model"
    assert events[-1].text != _OUT_OF_SCOPE
    # Dietetics is a department this desk books for, so this is a booking -
    # there is no longer a third "some other desk will call you back" category
    # between the five flows and the scope line.
    assert manager._sessions["conn-general"].intent == "appointment.book"


async def test_a_hard_clinical_refusal_is_not_left_to_generation() -> None:
    """The narrow exception. Naming or ruling out a condition, reading a lab
    value, and authorising a medicine are the three things whose wording is
    fixed, because getting them wrong costs a caller their health rather than
    their time. Each fires only when the caller ASKS for the forbidden thing -
    there is no arm that fires on an intent alone, which is what made every
    emergency turn identical."""
    manager = _make_manager()
    manager.start_call("conn-dx", agent_name="Gayathri")
    manager._sessions["conn-dx"].intent = "clinical.triage"
    llm = _ScriptedLlm([])

    events = [
        event
        async for event in manager.stream_utterance(
            "conn-dx", llm, "இது dengue-ஆ? அது இல்லன்னு மட்டும் சொல்லுங்க"
        )
    ]

    assert llm.calls == [], "a diagnosis refusal must not depend on generation"
    assert "முடியாது" in events[-1].text


# --- the ledger actually reaching the prompt (the bug this suite missed) ---


def _system_prompt_of(llm: _ScriptedLlm, call_index: int = -1) -> str:
    """Everything the model was told as a system message on that call.

    Not just messages[0]. The standing facts deliberately ride at the END of
    the message list rather than in the system prompt, because the system
    prompt is the cached prefix and mutating it re-evaluates ~2.7k tokens (see
    ConversationManager._system_prompt_for). What these tests care about is
    that the facts REACH the model, not which slot carries them.
    """
    return "\n".join(
        message["content"]
        for message in llm.calls[call_index]
        if message.get("role") == "system" and message.get("content")
    )


async def test_stream_utterance_yields_clauses_then_the_completed_turn() -> None:
    manager = _make_manager()
    manager.start_call("conn-1", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="கண்டிப்பா சார். Patient பேரு சொல்லுங்க?")])

    events = [event async for event in manager.stream_utterance("conn-1", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")]

    clauses = [event.text for event in events if isinstance(event, AgentClause)]
    assert clauses == ["கண்டிப்பா.", "Patient பேரு சொல்லுங்க?"]
    assert isinstance(events[-1], AgentTurn)
    assert events[-1].text == "கண்டிப்பா. Patient பேரு சொல்லுங்க?"


async def test_generated_honorific_is_normalized_without_rewriting_caller_history() -> None:
    """Spelling AND presence. LANGUAGE says the male address is Latin "Sir" so
    the Tamil TTS says the English word rather than "saar" - and, in the same
    breath, that an unestablished gender means no address word at all. This
    caller has given a name and nothing else, so there is nothing to guess
    from."""
    manager = _make_manager()
    manager.start_call("conn-sir", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="சரி சார். சொல்லுங்க sir?")])

    events = [
        event
        async for event in manager.stream_utterance(
            "conn-sir", llm, "சார், appointment ஒண்ணு வேணும்"
        )
    ]

    assert events[-1].text == "சரி. சொல்லுங்க?"
    assert llm.calls[-1][-2]["content"] == "சார், appointment ஒண்ணு வேணும்"


async def test_the_address_follows_the_caller_not_the_models_guess() -> None:
    """The model always guesses male - it said "Kavitha Sir" to a woman who had
    just given her name. The caller saying "மேடம்" is addressing the AGENT and
    reveals nothing; only their own identity does."""
    manager = _make_manager()
    manager.start_call("conn-her", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="சரி Sir. Patient பேரு சொல்லுங்க?")])

    events = [
        e
        async for e in manager.stream_utterance(
            "conn-her", llm, "மேடம், நான் அவரோட மனைவி பேசுறேன். Appointment வேணும்."
        )
    ]

    assert "மேடம்" in events[-1].text, events[-1].text
    assert "Sir" not in events[-1].text, "the caller was called Sir"


async def test_metadata_gender_beats_anything_said_on_the_call() -> None:
    """A telephony leg or CRM that knows is better evidence than a guess."""
    manager = _make_manager()
    manager.start_call("conn-meta", agent_name="Gayathri", caller_gender="male")
    llm = _ScriptedLlm([LlmReply(content="சரி மேடம். Patient பேரு சொல்லுங்க?")])

    events = [e async for e in manager.stream_utterance("conn-meta", llm, "appointment வேணும்")]

    assert "Sir" in events[-1].text
    assert "மேடம்" not in events[-1].text


async def test_unbacked_completed_action_is_replaced_before_speech() -> None:
    manager = _make_manager()
    manager.start_call("conn-action", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="Appointment book பண்ணிட்டேன் Sir.")])

    events = [
        event
        async for event in manager.stream_utterance(
            "conn-action", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்"
        )
    ]

    assert events[-1].text == _ACTION_RECOVERY
    assert all("book பண்ணிட்டேன்" not in event.text for event in events)


async def test_first_emergency_turn_is_warm_and_dispatches_before_asking_address() -> None:
    manager = _make_manager()
    manager.start_call("conn-emergency-open", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    events = [
        event
        async for event in manager.stream_utterance(
            "conn-emergency-open", llm, "என் அப்பாவுக்கு நெஞ்சு வலி. மூச்சு வாங்குது!"
        )
    ]

    assert events[-1].text == _EMERGENCY_OPENING
    assert "ambulance அனுப்புறேன்" in events[-1].text
    assert "address சொல்லுங்க" in events[-1].text
    assert llm.calls == [], "the fixed first emergency response should not wait for the model"


async def test_emergency_address_is_read_back_verbatim_without_exemplar_facts() -> None:
    manager = _make_manager()
    manager.start_call("conn-emergency-address", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    [e async for e in manager.stream_utterance("conn-emergency-address", llm, "எனக்கு நெஞ்சு வலிக்குது")]
    address = "number 3 காந்திநகர் கொளத்தூர் Chennai"
    response = [
        e async for e in manager.stream_utterance("conn-emergency-address", llm, address)
    ][-1]

    assert response.text.startswith(address + ", சரி.")
    assert "Kanchan" not in response.text and "Cross" not in response.text
    assert "அப்பா" not in response.text
    assert manager._sessions["conn-emergency-address"].ledger["emergency_address"] == address
    assert llm.calls == []


async def test_emergency_accepts_a_response_update_before_the_onset_answer() -> None:
    manager = _make_manager()
    manager.start_call("conn-emergency-order", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    [e async for e in manager.stream_utterance("conn-emergency-order", llm, "அம்மாவுக்கு நெஞ்சு வலி")]
    [e async for e in manager.stream_utterance("conn-emergency-order", llm, "Velachery number 9")]
    response = [
        e
        async for e in manager.stream_utterance(
            "conn-emergency-order", llm, "ஆமா, அவங்க பேசுறாங்க. ரொம்ப வியர்க்குது"
        )
    ][-1]

    assert response.text == "சரி, நான் line-ல இருக்கேன். Patient மூச்சு சீரா இருக்கா?"
    assert "கண் திறந்து" not in response.text
    assert llm.calls == []


async def test_interrupted_distress_fragments_route_together_without_inventing_a_disease() -> None:
    manager = _make_manager()
    manager.start_call("conn-distress", agent_name="Gayathri")
    llm = _ScriptedLlm([])
    session = manager._sessions["conn-distress"]

    # Live barge-in recorded the first transcript but no agent speech.
    from .conversation import _append_caller_turn

    _append_caller_turn(session, "ஐயோ அம்மா")
    clarified = [e async for e in manager.stream_utterance("conn-distress", llm, "முடியல")][-1]

    assert session.intent == "emergency.escalate"
    assert clarified.text == _DISTRESS_CLARIFICATION
    assert "நோய்" not in clarified.text
    assert llm.calls == []

    explicit = [
        e async for e in manager.stream_utterance("conn-distress", llm, "மூச்சு விட முடியல")
    ][-1]
    assert explicit.text == _EMERGENCY_OPENING


async def test_a_corrected_appointment_date_replaces_the_old_date_without_a_fake_hearing_error() -> None:
    manager = _make_manager()
    manager.start_call("conn-date-correction", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [
            LlmReply(content="சரி. Patient பேரு சொல்லுங்க?"),
            LlmReply(content="Murugan. எந்த நாள் convenient?"),
            LlmReply(content="வர Friday. Mobile number சொல்லுங்க?"),
        ]
    )

    [e async for e in manager.stream_utterance("conn-date-correction", llm, "Cardiology-ல appointment வேணும்")]
    [e async for e in manager.stream_utterance("conn-date-correction", llm, "Murugan")]
    [e async for e in manager.stream_utterance("conn-date-correction", llm, "வர Friday")]
    corrected = [
        e
        async for e in manager.stream_utterance(
            "conn-date-correction", llm, "இல்ல, எனக்கு 6th September வேணும்"
        )
    ][-1]

    assert "6th September, மாத்தி குறிச்சுக்கிட்டேன்" in corrected.text
    assert "வர Friday" not in corrected.text
    assert "கேட்கல" not in corrected.text
    assert "காலையா மாலையா" in corrected.text
    assert len(llm.calls) == 3, "a clear date correction should not be delegated to the model"


def _controlled_manager() -> ConversationManager:
    manager = _make_manager()
    manager._deterministic_flows = True
    return manager


async def test_production_booking_controller_advances_on_arbitrary_values() -> None:
    manager = _controlled_manager()
    manager.start_call("conn-controlled-book", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    replies = []
    for caller in (
        "Oncology-ல appointment வேணும்",
        "Zoya Rahman",
        "7th October evening",
        "9123456789",
    ):
        replies.append([e async for e in manager.stream_utterance("conn-controlled-book", llm, caller)][-1].text)

    assert "Patient பேரு" in replies[0]
    assert "எந்த நாள்" in replies[1]
    assert "mobile number" in replies[2].lower()
    assert "உறுதி" in replies[3] and "SMS" in replies[3]
    assert llm.calls == []


async def test_production_reschedule_controller_preserves_old_booking_until_callback() -> None:
    manager = _controlled_manager()
    manager.start_call("conn-controlled-move", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    replies = []
    for caller in (
        "நாளைக்கு appointment இருக்கு, வேற date-க்கு மாத்தணும்",
        "Lakshmi Devi",
        "அடுத்த Friday morning",
        "9876543210",
    ):
        replies.append([e async for e in manager.stream_utterance("conn-controlled-move", llm, caller)][-1].text)

    assert "Patient பேரு" in replies[0]
    assert "புதுசா எந்த நாள்" in replies[1]
    assert "mobile number" in replies[2].lower()
    assert "பழைய appointment அப்படியே இருக்கும்" in replies[3]
    assert llm.calls == []


async def test_production_cancel_controller_never_claims_it_already_cancelled() -> None:
    manager = _controlled_manager()
    manager.start_call("conn-controlled-cancel", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    replies = []
    for caller in (
        "இந்த Friday appointment cancel பண்ணணும்",
        "Ravi Kumar",
        "வேணாம், reschedule வேணாம்",
        "9840721534",
    ):
        replies.append([e async for e in manager.stream_utterance("conn-controlled-cancel", llm, caller)][-1].text)

    assert "Patient பேரு" in replies[0]
    assert "மாத்திக்கலாமா" in replies[1]
    assert "Appointment ID இல்ல mobile number" in replies[2]
    assert "Cancellation request குறிச்சுக்கிட்டேன்" in replies[3]
    assert "cancel பண்ணிட்டேன்" not in " ".join(replies)
    assert llm.calls == []


async def test_production_information_controller_answers_every_subject_without_exemplar_followup() -> None:
    manager = _controlled_manager()
    manager.start_call("conn-controlled-info", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    response = [
        e
        async for e in manager.stream_utterance(
            "conn-controlled-info", llm, "Visiting hours என்ன, parking இருக்கா?"
        )
    ][-1].text

    assert "General ward" in response and "ICU" in response and "Parking" in response
    assert "எத்தனை மணி காலை வரும்" not in response
    assert llm.calls == []


async def test_an_invented_identifier_is_never_spoken_to_the_caller() -> None:
    """The parroted-exemplar failure, end to end through the manager.

    Observed live over the socket: asked for a mobile number and given an age
    instead, the agent said "90045 33218 என்ன சொல்லுங்க?" - the phone number
    out of its own few-shot exemplar. grounding.py detected it, and the caller
    had already heard it, because detection ran after the clause was streamed.
    speakable() is a pre-speech choke point, so it never leaves the server now.
    """
    manager = _make_manager()
    manager.start_call("conn-1", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="ஆமாம், MRN ARV-604417-னு இருக்கு. சரியா?")])

    events = [e async for e in manager.stream_utterance("conn-1", llm, "Cardiology-ல ஒரு appointment வேணும்")]
    turn = events[-1]

    assert "ARV-604417" not in turn.text, f"the caller was told an invented MRN: {turn.text}"
    for event in events[:-1]:
        assert "ARV-604417" not in event.text, f"invented MRN reached a clause: {event.text}"
    # ...and the turn still says something coherent rather than trailing off
    # mid-sentence, which is what grounding.py's docstring warned against.
    assert turn.text.endswith(_CANNOT_RECALL)
    # Nothing ungrounded survives into the reported turn, because nothing
    # ungrounded was spoken.
    assert turn.ungrounded == ()


def test_record_interrupted_turn_keeps_history_honest_after_barge_in() -> None:
    """Barge-in cancels the turn mid-yield, so the assistant message is never
    appended and the model's next turn sees its own line answered by nothing."""
    manager = _make_manager()
    manager.start_call("conn-1", agent_name="Gayathri")
    # Simulate a caller turn having been taken, so the last message is not the
    # greeting's own assistant line.
    manager._sessions["conn-1"].messages.append({"role": "user", "content": "slots என்ன?"})

    manager.record_interrupted_turn("conn-1", "Dr. Ramanathan-oda slots")

    assert manager._sessions["conn-1"].messages[-1] == {
        "role": "assistant",
        "content": "Dr. Ramanathan-oda slots",
    }


def test_record_interrupted_turn_ignores_blank_duplicate_and_unknown_calls() -> None:
    manager = _make_manager()
    manager.start_call("conn-1", agent_name="Gayathri")
    before = len(manager._sessions["conn-1"].messages)

    manager.record_interrupted_turn("conn-1", "   ")
    # The greeting already left an assistant message last; appending another
    # would read as the agent taking two turns in a row.
    manager.record_interrupted_turn("conn-1", "something")
    manager.record_interrupted_turn("no-such-call", "something")

    assert len(manager._sessions["conn-1"].messages) == before


def test_language_reminder_forbids_claiming_a_system_action() -> None:
    """The inverse of the guard this replaces.

    _LANGUAGE_REMINDER is appended after the caller's turn, immediately before
    generation - the last thing the model reads before deciding what to say.
    While there WAS a tool layer this message had to open by naming "call a
    tool", because a speech-only version read as "produce speech now" and
    suppressed tool calling entirely across a four-turn booking.

    There are no tools now, so the failure mode flips: the risk is the model
    saying it looked something up, booked something or knows an MRN, none of
    which it can do. That claim is the one thing this message must keep
    forbidding, and no other test would catch its removal - they all script
    the LLM's output rather than generating it.
    """
    from .conversation import _LANGUAGE_REMINDER

    lowered = _LANGUAGE_REMINDER.lower()
    assert "mrn" in lowered
    # It must forbid the invention...
    assert "never claim you already booked" in lowered
    # ...without inviting the refusal that invention-avoidance produced live:
    # the agent answered a booking request with "book பண்ண முடியாது".
    assert "never refuse the request itself" in lowered
    # And it must not resurrect the tool vocabulary it used to require.
    assert "call a tool" not in lowered



def test_no_facts_block_is_sent_when_the_server_knows_nothing() -> None:
    """A browser call opens knowing only the agent's own name, so the block
    used to be five labels with blanks after them plus a paragraph explaining
    what a blank meant - ~70 tokens of empty scaffolding on every turn, and it
    put "mrn:" in front of a model that is told never to say an MRN.

    What the caller said is not lost: the transcript sits directly above in the
    message list, which is where a conversational agent's memory lives.
    """
    manager = _make_manager()
    manager.start_call("conn-blank", agent_name="Gayathri")
    session = manager._sessions["conn-blank"]

    assert manager._turn_facts_message(session) == ""

    messages = _with_language_reminder(session.messages, manager._turn_facts_message(session))
    assert not any("KNOWN FACTS" in str(m.get("content") or "") for m in messages)


def test_a_fact_the_server_does_know_is_still_carried() -> None:
    """The inverse: a telephony leg knows the caller's number before the call
    is answered, and that must not be re-asked."""
    manager = _make_manager()
    manager.start_call("conn-known", agent_name="Gayathri", caller_mobile="9840721534")
    session = manager._sessions["conn-known"]

    facts = manager._turn_facts_message(session)
    assert "caller_mobile: 9840721534" in facts
    # ...and the labels that are still unknown stay out of the prompt entirely.
    assert "mrn:" not in facts
    assert "patient_name:" not in facts


def test_a_long_call_never_pushes_the_system_prompt_out_of_the_context_window() -> None:
    """The assembled prompt is ~3.8k tokens against num_ctx 6144, so a call has
    ~2k tokens of room for history and nothing used to bound it. Overflow
    makes Ollama truncate from the FRONT, taking the language rules with it -
    the agent switches to English and invents identifiers, silently. That is
    the exact failure backend/prompt_builder.py exists to prevent.

    Driven through stream_utterance rather than by calling the trimmer
    directly: an earlier version of this test exercised the helper alone and
    still passed with the call site deleted, which is a test that cannot fail.
    """
    import asyncio

    from .conversation import MAX_HISTORY_MESSAGES

    turns = 60
    manager = _make_manager()
    llm = _ScriptedLlm([LlmReply(content=f"பதில் {i}.") for i in range(turns)])
    manager.start_call("conn-long", agent_name="Gayathri")
    session = manager._sessions["conn-long"]
    system_prompt = session.messages[0]

    async def run() -> None:
        for i in range(turns):
            async for _event in manager.stream_utterance("conn-long", llm, f"கேள்வி {i}."):
                pass

    asyncio.run(run())

    assert len(session.messages) <= MAX_HISTORY_MESSAGES + 1, (
        f"history grew to {len(session.messages)} messages - it will truncate the system prompt"
    )
    # The one message that must never be dropped.
    assert session.messages[0] is system_prompt
    # ...and the most recent exchange survives, because that is the context the
    # next turn actually depends on.
    assert session.messages[-1]["content"] == f"பதில் {turns - 1}."
    assert session.messages[-2]["content"] == f"கேள்வி {turns - 1}."

    # The prompt the model was last handed must still be the system prompt,
    # intact and in position 0 - that is what overflow destroys.
    last_messages = llm.calls[-1]
    assert last_messages[0]["role"] == "system"
    assert last_messages[0]["content"] == session.messages[0]["content"]
    assert len(last_messages) <= MAX_HISTORY_MESSAGES + 3  # + facts/reminder tail


# The longest turns this server has actually produced, out of the 794 real turn
# texts in call_events.db: 177 characters for an agent turn (median 60) and 67
# for a caller turn (median 20). Quoted rather than read from the database so
# this test stays offline, and used for EVERY turn of the long call below,
# because "24 messages" is only a safe bound at median length.
LONGEST_REAL_AGENT_TURN = (
    "கண்டிப்பா மேடம். General ward-க்கு காலை 11 to 12, மாலை 5 to 7. "
    "ICU-க்கு மாலை 5 to 5:30 மட்டும், அதுவும் ஒரு நேரத்துல ஒருத்தர் தான். "
    "எந்த ward-ல பார்க்க வேண்டும் என்று சொல்லுங்க?"
)
LONGEST_REAL_CALLER_TURN = (
    "பொறுமையாக பிடிச்சா நாலு மாசத்துக்கு ஒரு மாசத்துக்குள்ள பிரிச்சுடும்"
)


def _modelfile_num_ctx() -> int:
    import re
    from pathlib import Path

    modelfile = (Path(__file__).resolve().parent.parent / "Modelfile").read_text(encoding="utf-8")
    match = re.search(r"^PARAMETER\s+num_ctx\s+(\d+)", modelfile, re.MULTILINE)
    assert match, "Modelfile has no num_ctx PARAMETER"
    return int(match.group(1))


def test_llm_num_ctx_matches_the_modelfile() -> None:
    """Two statements of one number, so they are checked rather than trusted.

    Ollama's window is set by the Modelfile and nothing at runtime can read it
    back, so conversation.py has to be told. A setting LARGER than the
    Modelfile's makes the history trim size itself against a window that does
    not exist, and the overflow it exists to prevent comes back silently.
    """
    assert LlmSettings().num_ctx == _modelfile_num_ctx()


def test_the_script_that_builds_the_model_agrees_with_the_modelfile() -> None:
    """The third statement of that number, and the one that had drifted.

    setup_model.py is what actually runs `ollama create`, and it carried its
    own NUM_CTX = 8192 while the Modelfile and LLM_NUM_CTX both said 6144. The
    two that were checked against each other agreed, so the guard above passed
    while the model Ollama really served had a window nothing in the repo
    claimed - visible only by reading `ollama ps` by hand.

    num_gpu and the anti-loop parameters are checked in the same breath and
    for the same reason: run.sh uses the generated file in production, so any
    parameter omitted there is silently absent from the model callers use.
    """
    from pathlib import Path
    import re

    from .scripts.setup_model import build_modelfile

    built = build_modelfile("some-base:tag")
    modelfile = (Path(__file__).resolve().parent.parent / "Modelfile").read_text(encoding="utf-8")

    def parameter(text: str, name: str) -> str | None:
        match = re.search(rf"^PARAMETER\s+{name}\s+(\S+)", text, re.MULTILINE)
        return match.group(1) if match else None

    for name in ("num_ctx", "num_gpu", "temperature", "repeat_penalty", "repeat_last_n"):
        assert parameter(built, name) == parameter(modelfile, name), (
            f"{name}: setup_model.py builds {parameter(built, name)!r} but the "
            f"Modelfile documents {parameter(modelfile, name)!r}. The build "
            "script wins at runtime, so the Modelfile is the one that lies."
        )


def test_a_call_of_long_turns_never_overflows_num_ctx() -> None:
    """The bound that message-counting cannot provide, driven end to end.

    Measured with Ollama's own prompt_eval_count while auditing this: the
    widest playbook plus 24 messages built from the LONGEST turns above came to
    7202 tokens against num_ctx 6144 - 1058 OVER, in the shipped
    configuration. Sec10.2 had sized MAX_HISTORY_MESSAGES against num_ctx 8192
    and the window was later lowered to 6144 for VRAM headroom without the
    budget being re-derived. Overflow makes Ollama truncate from the FRONT,
    taking the language and clinical-safety rules with it, and it is silent.

    Uses the REAL prompt files and the REAL emergency.escalate playbook - the
    widest, and the one a call can switch to at any moment - because the
    failure is a property of the assembled prompt, not of the plumbing.

    Asserts on what the model was ACTUALLY HANDED (llm.calls[-1]) rather than
    on the trimmer in isolation: an earlier version of the neighbouring test
    exercised the helper alone and still passed with its call site deleted.
    """
    import asyncio

    from .conversation import (
        _HISTORY_TOKENS_PER_CHAR,
        _LANGUAGE_REMINDER,
        _PROMPT_TOKENS_PER_CHAR,
        _TOKENS_PER_MESSAGE,
    )
    from .prompt_builder import PromptBuilder

    settings = ConversationSettings()
    manager = ConversationManager(settings)
    manager.prompts = PromptBuilder(
        settings.runtime_core_path, settings.prompt_path, settings.exemplars_path
    )
    manager.prompts.load()

    turns = 40
    llm = _ScriptedLlm([LlmReply(content=LONGEST_REAL_AGENT_TURN) for _ in range(turns)])
    manager.start_call("conn-ctx", agent_name="Gayathri")
    # The widest current runtime prompt, pinned. The caller fixture carries no
    # intent trigger, so it stays sticky for the whole sizing run.
    manager._sessions["conn-ctx"].intent = "records.request"

    async def run() -> None:
        for _ in range(turns):
            async for _event in manager.stream_utterance("conn-ctx", llm, LONGEST_REAL_CALLER_TURN):
                pass

    asyncio.run(run())

    num_ctx = _modelfile_num_ctx()
    max_tokens = LlmSettings().max_tokens
    for index, messages in enumerate(llm.calls):
        system_chars = sum(
            len(m["content"]) for m in messages if m.get("role") == "system"
        )
        history_chars = sum(
            len(m["content"] or "") for m in messages if m.get("role") != "system"
        )
        estimated = round(
            system_chars * _PROMPT_TOKENS_PER_CHAR
            + history_chars * _HISTORY_TOKENS_PER_CHAR
            + _TOKENS_PER_MESSAGE * len(messages)
        )
        assert estimated + max_tokens <= num_ctx, (
            f"turn {index} handed the model ~{estimated} prompt tokens; with "
            f"LLM_MAX_TOKENS {max_tokens} that is {estimated + max_tokens - num_ctx} "
            f"over num_ctx {num_ctx}. Ollama truncates from the front and drops "
            f"the language rules. ({system_chars} chars of prompt, "
            f"{history_chars} of history over {len(messages)} messages.)"
        )
        assert _LANGUAGE_REMINDER in messages[-1]["content"]

    # The trim has to have actually bitten, or this asserts nothing: 40 turns
    # of these lengths are far past the budget.
    assert len(llm.calls[-1]) < turns, "history was never trimmed - the test proves nothing"


def test_the_english_caller_detector_counts_words_not_letters() -> None:
    """Switching the register on this was built and measured TWICE, and made
    things worse both times - prose alone only half-moved it and introduced
    parroting; an English worked example alongside the twenty Tamil ones
    produced ungrammatical output mixing both. So it is not wired in: a
    coherent Tamil answer beats a broken half-English one.

    The detector is kept because the measurement is the correct one and any
    future attempt needs it. This guards the part that was genuinely hard: a
    code-mixed TAMIL line must not read as English. "Cardiology-ல ஒரு
    appointment book பண்ணணும்" is 64% Latin BY CHARACTER, which is why the
    count is by word.
    """
    from .conversation import caller_is_speaking_english

    assert caller_is_speaking_english("Hello, I need to book an appointment.")
    assert caller_is_speaking_english("Sometime this weekend would be good.")

    assert not caller_is_speaking_english("Cardiology-ல ஒரு appointment book பண்ணணும்.")
    assert not caller_is_speaking_english("Report வந்துடுச்சா?")
    # A phone number is evidence of neither language.
    assert not caller_is_speaking_english("98407 21534")


def test_the_reminder_keeps_one_register_instruction() -> None:
    """After the mirroring revert, exactly one register rule reaches the model
    and no {{register}} placeholder survives unfilled."""
    from .conversation import _LANGUAGE_REMINDER, _with_language_reminder

    assert "{{register}}" not in _LANGUAGE_REMINDER
    assert "never pure English" in _LANGUAGE_REMINDER
    assert "HOW THIS SOUNDS IN ENGLISH" not in "".join(
        str(m["content"]) for m in _with_language_reminder([], "")
    )


# --- turn discipline: ONE question per turn (LLM_STACK.md Sec9 item 1) ---
#
# runtime_core.txt states this rule three ways in one line and the model breaks
# it anyway. These drive the real stream_utterance path rather than a helper,
# because the two guards written before this one initially PASSED with the
# code deleted.


async def test_a_second_question_is_never_spoken() -> None:
    manager = _make_manager()
    manager.start_call("conn-q", agent_name="Gayathri")
    # The exact shape recorded in call_events.db: two questions, two clauses,
    # with a non-question closing line behind them that must survive.
    llm = _ScriptedLlm(
        [
            LlmReply(
                content=(
                    "சரி சார். உங்க mobile number சொல்லுங்களா? "
                    "எந்த நாள் convenient? Desk-ல இருந்து call பண்ணுவாங்க."
                )
            )
        ]
    )

    events = [event async for event in manager.stream_utterance("conn-q", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")]
    clauses = [event.text for event in events if isinstance(event, AgentClause)]

    assert "எந்த நாள் convenient?" not in clauses, "the second question reached TTS"
    assert sum(clause.count("?") for clause in clauses) == 1
    # The closing line is not a question and must NOT be collateral damage -
    # a guard that truncated the tail would drop the whole handoff promise.
    assert "Desk-ல இருந்து call பண்ணுவாங்க." in clauses
    assert events[-1].text == " ".join(clauses)


async def test_history_records_what_was_spoken_not_what_was_generated() -> None:
    """Otherwise the model believes it asked a question the caller never heard."""
    manager = _make_manager()
    manager.start_call("conn-q2", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="Patient பேரு சொல்லுங்க? வயசு என்ன?")])

    async for _event in manager.stream_utterance("conn-q2", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்"):
        pass

    said = manager._sessions["conn-q2"].messages[-1]
    assert said["role"] == "assistant"
    assert "வயசு என்ன?" not in said["content"]


async def test_one_question_per_turn_is_left_alone() -> None:
    """The guard must not fire on a well-formed turn."""
    manager = _make_manager()
    manager.start_call("conn-q3", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="கண்டிப்பா சார். Patient பேரு சொல்லுங்க?")])

    events = [event async for event in manager.stream_utterance("conn-q3", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")]

    assert events[-1].text == "கண்டிப்பா. Patient பேரு சொல்லுங்க?"


async def test_caller_turns_merge_when_the_agent_never_got_a_word_out() -> None:
    """A barge-in before the first clause leaves the history with no reply in it.

    record_interrupted_turn() has nothing to append when zero clauses were
    spoken, so without merging the next caller turn lands directly behind the
    previous one. On the real call in call_events.db (97dd5ac7) that put twelve
    consecutive user messages into a 21-message history and the model stopped
    answering, reproducing its own previous turn verbatim instead.
    """
    manager = _make_manager()
    manager.start_call("conn-merge", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="சரி சார்.") for _ in range(3)])

    # Exactly what main.py does on a barge-in that lands before the first
    # clause: abandon the generator without consuming a clause, then report
    # that nothing was spoken. Driving the real path, not _append_caller_turn.
    for cut_off in ("ஆஸ்டோ department க்கு வேணும்", "ஆற்று"):
        turn = manager.stream_utterance("conn-merge", llm, cut_off)
        await turn.asend(None)
        await turn.aclose()
        manager.record_interrupted_turn("conn-merge", "")

    async for _event in manager.stream_utterance("conn-merge", llm, "என் பேரு நானே"):
        pass

    messages = manager._sessions["conn-merge"].messages
    runs = [a for a, b in zip(messages, messages[1:]) if a["role"] == b["role"] == "user"]
    assert not runs, f"history still has consecutive user messages: {messages}"
    # Merged, not dropped: the department is the CONTENT of that stretch and
    # dropping the older message would lose it behind the noise that followed.
    caller_said = " ".join(m["content"] for m in messages if m["role"] == "user")
    assert "ஆஸ்டோ department க்கு வேணும்" in caller_said
    assert "ஆற்று" in caller_said


async def test_the_agent_never_opens_two_turns_with_the_same_clause() -> None:
    """runtime_core.txt has forbidden this in prose since before the tool removal
    and the model does it anyway - five turns running on the real call. The
    breaker is enforced in code for the same reason speakable() is."""
    manager = _make_manager()
    manager.start_call("conn-rep", agent_name="Gayathri")
    stuck = "உங்க registered mobile number சொல்லுங்க?"
    llm = _ScriptedLlm([LlmReply(content=stuck) for _ in range(4)])

    said = []
    for caller in ("98407", "வெண்ணூர் கிழவனை", "எல்லாம்", "தேடியா"):
        events = [e async for e in manager.stream_utterance("conn-rep", llm, caller)]
        said.append(events[-1].text)

    assert said[0] == stuck, "the FIRST time is not a repeat and must be spoken"
    assert stuck not in said[1:], f"the agent said its own last turn again: {said}"
    # Escalating, so a caller on a line that is not working reaches the handoff
    # instead of the same sentence until they hang up.
    assert len(set(said[1:])) == 3, f"the recovery lines repeated each other: {said}"
    assert "Desk-ல இருந்து" in said[-1], f"never offered the callback: {said}"


async def test_the_repeat_breaker_leaves_a_short_acknowledgement_alone() -> None:
    """Two turns may legitimately both open with "சரி சார்." - only a longer
    opening repeated verbatim is the model stuck rather than agreeing."""
    manager = _make_manager()
    manager.start_call("conn-ack", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [
            LlmReply(content="சரி சார். உங்க பேரு சொல்லுங்க?"),
            LlmReply(content="சரி சார். உங்க வயசு சொல்லுங்க?"),
        ]
    )

    first = [e async for e in manager.stream_utterance("conn-ack", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")][-1]
    second = [e async for e in manager.stream_utterance("conn-ack", llm, "நானே")][-1]

    assert first.text == "சரி. உங்க பேரு சொல்லுங்க?"
    assert second.text == "சரி. உங்க வயசு சொல்லுங்க?", "the breaker fired on an acknowledgement"


async def test_the_repeat_breaker_never_suppresses_a_repeated_refusal() -> None:
    """runtime_core.txt's CLINICAL SAFETY section requires a refused request to
    be refused AGAIN in the same words when the caller pushes - to a frightened
    caller a changed subject reads as being ignored. The repeat breaker forbids
    repeating an opening clause, so without an exemption it would replace the
    second refusal with "clear-ஆ கேட்கல", which is a safety regression.
    """
    manager = _make_manager()
    manager.start_call("conn-refuse", agent_name="Gayathri")
    refusal = "Phone-ல அதை நான் சொல்ல முடியாது சார்."
    llm = _ScriptedLlm([LlmReply(content=refusal) for _ in range(3)])

    said = []
    for caller in ("appointment cancel பண்ணணும்", "ஒரு தடவை சொல்லுங்க", "please சொல்லுங்க மேடம்"):
        events = [e async for e in manager.stream_utterance("conn-refuse", llm, caller)]
        said.append(events[-1].text)

    normalized_refusal = _normalize_spoken_register(refusal)
    assert said == [normalized_refusal] * 3, f"the refusal was suppressed or altered: {said}"


async def test_a_looping_decoder_is_never_spoken_to_the_caller() -> None:
    """Observed live, mid-emergency: "உங்க முழு முழு முழு ..." for twenty-two
    words, and the caller heard all of it. Neither existing guard could see it -
    the repeat breaker compares clauses ACROSS turns and this is one turn, and
    the clause chunker cuts the opening at 32 characters and then waits for
    punctuation a looping decoder never emits."""
    manager = _make_manager()
    manager.start_call("conn-loop", agent_name="Gayathri")
    loop = "உங்க முழு " + "முழு " * 20
    llm = _ScriptedLlm([LlmReply(content=loop)])

    # A non-emergency turn deliberately: which recovery ladder a stuck turn
    # draws from is the neighbouring test's business, not this one's.
    events = [e async for e in manager.stream_utterance("conn-loop", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")]

    assert "முழு முழு முழு" not in events[-1].text, f"the loop was spoken: {events[-1].text}"
    assert events[-1].text in _STUCK_REPLIES, f"no recovery was offered: {events[-1].text}"


async def test_the_agent_never_hands_the_callers_own_sentence_back() -> None:
    """_LANGUAGE_REMINDER has forbidden this for as long as it has existed and
    the model does it anyway. Observed live on both a question and, far worse,
    on an emergency: the caller said they were fighting for their life and the
    agent said it back to them."""
    manager = _make_manager()
    manager.start_call("conn-echo", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="appointment cancel பண்ணணும் சார்?")])

    events = [e async for e in manager.stream_utterance("conn-echo", llm, "appointment cancel பண்ணணும்?")]

    assert events[-1].text == _ECHO_RECOVERY, f"the parrot was spoken: {events[-1].text}"


async def test_the_echo_guard_leaves_a_read_back_alone() -> None:
    """The LEDGER section requires reading facts back to confirm them, and the
    EMERGENCY playbook requires it of the address specifically. A guard that
    could not tell a read-back from a parrot would be a safety regression, so
    the read-backs are what this actually protects."""
    manager = _make_manager()
    manager.start_call("conn-readback", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [
            LlmReply(content="98407 21534, குறிச்சுக்கிட்டேன். எந்த நாள் convenient சார்?"),
            LlmReply(content="Anna Nagar 2nd street, சரியா சார்?"),
        ]
    )

    first = [e async for e in manager.stream_utterance("conn-readback", llm, "98407 21534")][-1]
    second = [
        e async for e in manager.stream_utterance("conn-readback", llm, "Anna Nagar 2nd street")
    ][-1]

    assert "98407 21534" in first.text, f"the number read-back was withheld: {first.text}"
    assert "Anna Nagar" in second.text, f"the address read-back was withheld: {second.text}"


async def test_an_answer_that_uses_the_callers_words_is_not_a_parrot() -> None:
    """A real answer necessarily adds words the caller did not say, which is
    what both bars in _echoes_caller are measuring."""
    manager = _make_manager()
    manager.start_call("conn-answer", agent_name="Gayathri")
    reply = "Cardiology appointment-க்கு Friday morning slot இருக்கு சார்."
    llm = _ScriptedLlm([LlmReply(content=reply)])

    events = [e async for e in manager.stream_utterance("conn-answer", llm, "Cardiology appointment வேணும்?")]

    assert events[-1].text == _normalize_spoken_register(reply), f"a real answer was withheld: {events[-1].text}"


async def test_an_off_topic_opener_is_told_what_the_desk_answers() -> None:
    """detect_intent returning None used to fall through to the info.general
    playbook and let the model improvise, which is how a question the hospital
    desk does not handle gets a confident invented answer. Costs no LLM call."""
    manager = _make_manager()
    manager.start_call("conn-scope", agent_name="Gayathri")
    llm = _ScriptedLlm([])  # a scripted reply here would mean the LLM was called

    events = [
        e async for e in manager.stream_utterance("conn-scope", llm, "நாளைக்கு மழை பெய்யுமா")
    ]

    assert events[-1].text == _OUT_OF_SCOPE
    assert llm.calls == [], "an off-topic turn should not reach the model at all"


async def test_the_scope_line_never_displaces_a_turn_inside_a_live_flow() -> None:
    """Once a flow is running, a turn that matches no trigger is the caller
    ANSWERING a question - a day, a name, a number - and belongs to the model.
    This is the failure mode that makes an eager scope check worse than none."""
    manager = _make_manager()
    manager.start_call("conn-inflow", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [
            LlmReply(content="சரி சார். உங்க பேரு சொல்லுங்க?"),
            LlmReply(content="நன்றி முருகேசன் சார். Mobile number சொல்லுங்க?"),
        ]
    )

    [e async for e in manager.stream_utterance("conn-inflow", llm, "Cardiology-ல ஒரு appointment book பண்ணணும்")]
    second = [
        e async for e in manager.stream_utterance("conn-inflow", llm, "என் பேரு முருகேசன் சார்")
    ][-1]

    assert "முருகேசன்" in second.text, f"the scope line displaced a real turn: {second.text}"


async def test_the_scope_line_is_said_once_and_then_the_model_takes_over() -> None:
    """A caller who hears the list and still says nothing that routes is better
    served by the model than by the same list again."""
    manager = _make_manager()
    manager.start_call("conn-scope2", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="சொல்லுங்க சார், என்ன வேணும்?")])

    first = [e async for e in manager.stream_utterance("conn-scope2", llm, "நாளைக்கு மழை பெய்யுமா")][-1]
    second = [e async for e in manager.stream_utterance("conn-scope2", llm, "cricket score என்ன")][-1]

    assert first.text == _OUT_OF_SCOPE
    assert second.text != _OUT_OF_SCOPE, "the desk read out the same list twice"


async def test_the_repeat_breaker_never_hangs_up_on_an_emergency() -> None:
    """The ordinary escalation ends the call - "Desk-ல இருந்து call பண்ண
    சொல்றேன். நன்றி சார்." - and main_prompt.txt flow 18 forbids exactly that:
    "you do NOT end this call", "Never hang up". An emergency gets a ladder
    that keeps the line open and repeats the two things that matter."""
    manager = _make_manager()
    manager.start_call("conn-er", agent_name="Gayathri")
    repeated = "உங்க address சொல்லுங்க சார்?"
    llm = _ScriptedLlm([LlmReply(content=repeated) for _ in range(5)])

    said = []
    for caller in ("ambulance வேணும்", "சீக்கிரம்", "ஐயோ", "சார்", "என்ன பண்றது"):
        events = [e async for e in manager.stream_utterance("conn-er", llm, caller)]
        said.append(events[-1].text)

    assert manager._sessions["conn-er"].intent == "emergency.escalate"
    # There used to be an `assert llm.calls == []` here, requiring that an
    # emergency never reach the model at all. It was satisfied by answering
    # every emergency turn from a fixed string, which meant a frightened
    # caller who said anything at all - "சீக்கிரம்", "ஐயோ" - got the same
    # sentence back, forever, because nothing they said could change it. The
    # guarantee that actually matters is the one asserted below and it does
    # not need the model excluded: whatever is generated, the call does not
    # close and it never invents an ER-team alert.
    for turn in said:
        assert "நன்றி" not in turn, f"closed an emergency call: {turn}"
        assert "ER team" not in turn, f"claimed an alert this process cannot make: {turn}"


async def test_a_reply_cut_off_at_max_tokens_never_speaks_the_fragment() -> None:
    """Observed live on an info.general turn that ran long: the reply ended on
    a bare "எந்த" ("which"), because the chunker's flush() hands back whatever
    was in the buffer and at max_tokens that is a word cut in half. A caller on
    a phone has no way to tell a truncated word from a strange one."""
    manager = _make_manager()
    manager.start_call("conn-cut", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [LlmReply(content="Visiting hours 11 to 12 Sir. Parking basement-ல இருக்கு. எந்த", finish_reason="length")]
    )

    events = [e async for e in manager.stream_utterance("conn-cut", llm, "Visiting hours என்ன?")]
    spoken = [e.text for e in events if isinstance(e, AgentClause)]

    assert spoken, "the whole reply was dropped, not just the cut-off tail"
    assert not any(clause.strip() == "எந்த" for clause in spoken), spoken
    assert "Parking basement-ல இருக்கு." in spoken


async def test_a_reply_that_simply_ends_without_punctuation_is_still_spoken() -> None:
    """The mirror image, and why the guard keys off finish_reason rather than
    off punctuation: a model that stops on its own mid-sentence has still said
    something the caller needs."""
    manager = _make_manager()
    manager.start_call("conn-uncut", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="சரி Sir. Patient பேரு சொல்லுங்க", finish_reason="stop")])

    events = [e async for e in manager.stream_utterance("conn-uncut", llm, "appointment வேணும்")]

    assert "Patient பேரு சொல்லுங்க" in events[-1].text


async def test_a_diagnosis_question_is_refused_even_outside_a_clinical_flow() -> None:
    """clinical.triage is not one of the five flows any more, and the diagnosis
    refusal used to be gated on it - which silently switched the refusal off.
    safety_eval caught the cost: the symptom turn got the scope line, and the
    NEXT turn reached the model, which answered "dengue இல்லன்னு மட்டும்".
    A caller can ask this in any call, so it is checked in any call."""
    manager = _make_manager()
    manager.start_call("conn-dx2", agent_name="Gayathri")
    llm = _ScriptedLlm([])  # a scripted reply would mean the model was asked

    events = [
        e async for e in manager.stream_utterance("conn-dx2", llm, "இது dengue-ஆ இருக்குமா மேடம்?")
    ]

    assert llm.calls == []
    assert "முடியாது" in events[-1].text


async def test_the_refusal_holds_when_the_caller_drops_the_condition_name() -> None:
    """"இல்லன்னு மட்டும் சொல்லுங்க" names nothing at all, and CLINICAL SAFETY
    says a repeated ask is refused in the same words - so the clinical subject
    is looked for across the whole call, not just the turn."""
    manager = _make_manager()
    manager.start_call("conn-dx3", agent_name="Gayathri")
    llm = _ScriptedLlm([])

    first = [
        e async for e in manager.stream_utterance("conn-dx3", llm, "இது dengue-ஆ இருக்குமா?")
    ][-1]
    second = [
        e
        async for e in manager.stream_utterance(
            "conn-dx3", llm, "சரி, dengue இல்லன்னு மட்டும் சொல்லுங்க"
        )
    ][-1]

    assert first.text == second.text, "the second ask was answered differently"
    assert llm.calls == []


async def test_an_ordinary_booking_turn_is_not_mistaken_for_a_diagnosis_question() -> None:
    """The reason the refusal needs two keys rather than none: the diagnosis
    pattern's "(?:இது|அது) ... -ஆ" arm matches perfectly ordinary booking Tamil,
    and a booking caller told "I cannot diagnose that" has been answered a
    question they never asked."""
    manager = _make_manager()
    manager.start_call("conn-ok", agent_name="Gayathri")
    llm = _ScriptedLlm(
        [
            LlmReply(content="சரி Sir. எந்த நாள் convenient?"),
            LlmReply(content="ஆமாம் Sir. காலை 10 மணி."),
        ]
    )

    [e async for e in manager.stream_utterance("conn-ok", llm, "Cardiology-ல appointment வேணும்")]
    events = [
        e async for e in manager.stream_utterance("conn-ok", llm, "அது நாளைக்கு காலைல-ஆ Sir?")
    ]

    assert len(llm.calls) == 2, "a booking question was answered with a clinical refusal"
    assert "காலை 10 மணி" in events[-1].text


async def test_a_closing_read_back_is_not_treated_as_a_stuck_model() -> None:
    """THE LEDGER requires confirming by reading back, so a closing turn
    legitimately repeats a fact it already confirmed mid-call. The repeat
    breaker used to kill that turn and append "மன்னிச்சுடுங்க Sir, clear-ஆ
    கேட்கல" - telling a caller who had just been heard perfectly that they had
    not been heard."""
    manager = _make_manager()
    manager.start_call("conn-conf", agent_name="Gayathri")
    confirm = "அடுத்த திங்கள் காலைல், சரி."
    llm = _ScriptedLlm(
        [
            LlmReply(content=f"Kavitha Sir. {confirm} Mobile number சொல்லுங்க?"),
            LlmReply(content=f"98407 21534, குறிச்சுக்கிட்டேன் Sir. {confirm} Desk call பண்ணுவாங்க."),
        ]
    )

    [e async for e in manager.stream_utterance("conn-conf", llm, "அடுத்த திங்கள் காலைல")]
    second = [e async for e in manager.stream_utterance("conn-conf", llm, "98407 21534")][-1]

    assert confirm in second.text, f"the read-back was suppressed: {second.text}"
    assert second.text not in _STUCK_REPLIES
    assert "கேட்கல" not in second.text, "the caller was told they had not been heard"


async def test_a_repeated_question_is_still_a_stuck_model() -> None:
    """The exemption is for statements only. A stuck model loops by ASKING -
    the five-turns-running clause this breaker exists for was a question - and
    that must still be caught."""
    manager = _make_manager()
    manager.start_call("conn-loop2", agent_name="Gayathri")
    # No identifier in it: an invented number would be caught by the grounding
    # guard first and this would pass without exercising the breaker at all.
    asked = "உங்க பேரு என்ன Sir?"
    llm = _ScriptedLlm([LlmReply(content=asked) for _ in range(2)])

    [e async for e in manager.stream_utterance("conn-loop2", llm, "appointment வேணும்")]
    second = [e async for e in manager.stream_utterance("conn-loop2", llm, "ஆமாம்")][-1]

    assert second.text in _STUCK_REPLIES, f"the loop was spoken again: {second.text}"


async def test_asking_something_the_caller_did_not_ask_is_not_a_parrot() -> None:
    """The echo guard fires on shared wording, and a follow-up question
    necessarily reuses the caller's words. Observed: the caller said "அடுத்த
    திங்கள் காலைல" and the agent's "அடுத்த திங்கள் காலைல எப்போது சரி?" was
    dropped as a parrot, leaving the caller a two-word turn with no question."""
    manager = _make_manager()
    manager.start_call("conn-followup", agent_name="Gayathri")
    question = "அடுத்த திங்கள் காலைல எப்போது சரி?"
    llm = _ScriptedLlm(
        [
            LlmReply(content="சரி Sir. Patient பேரு சொல்லுங்க?"),
            LlmReply(content=f"Kavitha Sir. {question}"),
        ]
    )

    [e async for e in manager.stream_utterance("conn-followup", llm, "Dermatology-ல appointment வேணும்")]
    events = [
        e async for e in manager.stream_utterance("conn-followup", llm, "அடுத்த திங்கள் காலைல")
    ]

    assert question in events[-1].text, f"the follow-up was dropped: {events[-1].text}"


async def test_the_parrot_the_echo_guard_exists_for_is_still_caught() -> None:
    """Both observed parrots reuse the caller's own interrogative, or have
    none at all - which is what the exemption above keys off."""
    manager = _make_manager()
    manager.start_call("conn-parrot", agent_name="Gayathri")
    caller = "ICU visiting hours என்ன?"
    llm = _ScriptedLlm([LlmReply(content="ICU visiting hours என்ன Sir?")])

    events = [e async for e in manager.stream_utterance("conn-parrot", llm, caller)]

    assert events[-1].text == _ECHO_RECOVERY, f"the parrot was spoken: {events[-1].text}"


async def test_a_turn_gutted_by_a_guard_gets_a_real_turn_not_the_leftovers() -> None:
    """Every guard used to ask "is the whole turn gone?" before offering a
    recovery, and a turn does not have to be emptied to be ruined. The caller
    heard "Kavitha Sir." - not silence, so no recovery ran, and the call
    stalled for a turn."""
    manager = _make_manager()
    manager.start_call("conn-stub", agent_name="Gayathri")
    # Second clause is a pure parrot with no new interrogative, so the echo
    # guard drops it and only the two-word acknowledgement survives.
    llm = _ScriptedLlm([LlmReply(content="Kavitha Sir. ICU visiting hours என்ன Sir?")])

    events = [e async for e in manager.stream_utterance("conn-stub", llm, "ICU visiting hours என்ன?")]
    spoken = " ".join(e.text for e in events if isinstance(e, AgentClause))

    assert _ECHO_RECOVERY in spoken, f"the caller was left with a stub: {spoken}"


def test_carries_a_turn_accepts_a_real_closing_and_rejects_an_address_form() -> None:
    from .conversation import _carries_a_turn

    assert not _carries_a_turn([])
    assert not _carries_a_turn(["Kavitha Sir."])
    assert _carries_a_turn(["Mobile number சொல்லுங்க?"])
    assert _carries_a_turn(["எல்லாம் குறிச்சுக்கிட்டேன் — desk call பண்ணுவாங்க."])
