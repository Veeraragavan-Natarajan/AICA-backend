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
    _CANNOT_RECALL,
    _ECHO_RECOVERY,
    _EMERGENCY_STUCK_REPLIES,
    _OUT_OF_SCOPE,
    _STUCK_REPLIES,
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

    events = [event async for event in manager.stream_utterance("conn-1", llm, "book பண்ணணும்")]

    clauses = [event.text for event in events if isinstance(event, AgentClause)]
    assert clauses == ["கண்டிப்பா சார்.", "Patient பேரு சொல்லுங்க?"]
    assert isinstance(events[-1], AgentTurn)
    assert events[-1].text == "கண்டிப்பா சார். Patient பேரு சொல்லுங்க?"


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

    events = [e async for e in manager.stream_utterance("conn-1", llm, "என் details check பண்ணுங்க")]
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

    num_gpu is checked in the same breath and for the same reason: an explicit
    one disables llama.cpp's fit-to-free-VRAM recovery, so a Modelfile and a
    build script disagreeing about it is the difference between a working
    server and one that 500s on every turn.
    """
    from pathlib import Path
    import re

    from .scripts.setup_model import build_modelfile

    built = build_modelfile("some-base:tag")
    modelfile = (Path(__file__).resolve().parent.parent / "Modelfile").read_text(encoding="utf-8")

    def parameter(text: str, name: str) -> str | None:
        match = re.search(rf"^PARAMETER\s+{name}\s+(\S+)", text, re.MULTILINE)
        return match.group(1) if match else None

    for name in ("num_ctx", "num_gpu"):
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
    # The widest flow, pinned: detect_intent would route these turns to
    # info.general, which is the narrowest and would prove nothing.
    manager._sessions["conn-ctx"].intent = "emergency.escalate"

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

    events = [event async for event in manager.stream_utterance("conn-q", llm, "book பண்ணணும்")]
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

    async for _event in manager.stream_utterance("conn-q2", llm, "book பண்ணணும்"):
        pass

    said = manager._sessions["conn-q2"].messages[-1]
    assert said["role"] == "assistant"
    assert "வயசு என்ன?" not in said["content"]


async def test_one_question_per_turn_is_left_alone() -> None:
    """The guard must not fire on a well-formed turn."""
    manager = _make_manager()
    manager.start_call("conn-q3", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="கண்டிப்பா சார். Patient பேரு சொல்லுங்க?")])

    events = [event async for event in manager.stream_utterance("conn-q3", llm, "book பண்ணணும்")]

    assert events[-1].text == "கண்டிப்பா சார். Patient பேரு சொல்லுங்க?"


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

    first = [e async for e in manager.stream_utterance("conn-ack", llm, "book பண்ணணும்")][-1]
    second = [e async for e in manager.stream_utterance("conn-ack", llm, "நானே")][-1]

    assert first.text == "சரி சார். உங்க பேரு சொல்லுங்க?"
    assert second.text == "சரி சார். உங்க வயசு சொல்லுங்க?", "the breaker fired on an acknowledgement"


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
    for caller in ("Value-ஐ சொல்லுங்க", "ஒரு தடவை சொல்லுங்க", "please சொல்லுங்க மேடம்"):
        events = [e async for e in manager.stream_utterance("conn-refuse", llm, caller)]
        said.append(events[-1].text)

    assert said == [refusal] * 3, f"the refusal was suppressed or altered: {said}"


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
    events = [e async for e in manager.stream_utterance("conn-loop", llm, "appointment book பண்ணணும்")]

    assert "முழு முழு முழு" not in events[-1].text, f"the loop was spoken: {events[-1].text}"
    assert events[-1].text in _STUCK_REPLIES, f"no recovery was offered: {events[-1].text}"


async def test_the_agent_never_hands_the_callers_own_sentence_back() -> None:
    """_LANGUAGE_REMINDER has forbidden this for as long as it has existed and
    the model does it anyway. Observed live on both a question and, far worse,
    on an emergency: the caller said they were fighting for their life and the
    agent said it back to them."""
    manager = _make_manager()
    manager.start_call("conn-echo", agent_name="Gayathri")
    llm = _ScriptedLlm([LlmReply(content="visiting hours என்ன சார்?")])

    events = [e async for e in manager.stream_utterance("conn-echo", llm, "visiting hours என்ன?")]

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
    reply = "Visiting hours காலைல 10 மணி to 12 மணி சார்."
    llm = _ScriptedLlm([LlmReply(content=reply)])

    events = [e async for e in manager.stream_utterance("conn-answer", llm, "visiting hours என்ன?")]

    assert events[-1].text == reply, f"a real answer was withheld: {events[-1].text}"


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

    [e async for e in manager.stream_utterance("conn-inflow", llm, "appointment book பண்ணணும்")]
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
    for turn in said[1:]:
        assert turn in _EMERGENCY_STUCK_REPLIES, f"used the hang-up ladder: {turn}"
        assert "நன்றி சார்" not in turn, f"closed an emergency call: {turn}"
        assert "108" in turn, f"an emergency recovery must still drive to 108: {turn}"
