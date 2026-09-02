"""Conversation Manager: routes a caller transcript through the assembled
prompt and the LLM, producing the agent's turn.

There is deliberately NO tool layer. The agent converses: it remembers what
the caller told it (the transcript is the memory) and answers from the prompt.
Measured on this box, sending the 22 tool schemas cost 1778 of 5290 prompt
tokens and dropped generation from 12.5 to 10.2 tok/s - about a fifth of the
time-to-first-word budget on a voice channel - while register_eval scored the
same scenarios 12/14 mechanically clean without them against 10/14 with them.

Per BACKEND_COMPLETION.md Sec3.1: the prompt is assembled per turn by
prompt_builder.py (condensed core + one flow playbook + that flow's exemplars),
and the ledger is real server-side state per connection_id - an in-process dict
for v1, since Redis only buys reconnect and multi-process, neither of which
exists yet.

stream_utterance() is the interface the live transports use: it yields each
clause as soon as it closes, so TTS can start speaking while the model is still
generating, then a final AgentTurn carrying the full text and the grounding
verdict for the turn. handle_utterance() is the same thing collapsed to a
string, for the eval scripts and tests that have no use for partial output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
import re

from .clause_chunker import ClauseChunker
from .grounding import grounding_sources, unbacked_action_claims, ungrounded_identifiers
from .llm import LlmClient, LlmReply, ReplyComplete
from .prompt_builder import (
    DEPARTMENT_INTENT,
    EMERGENCY_INTENT,
    SUPPORTED_INTENTS,
    PromptBuilder,
    detect_intent,
    is_explicit_emergency,
    names_a_department,
)
from .settings import ConversationSettings, LlmSettings

logger = logging.getLogger("aica.conversation")


def _detected_intent_is_negated(intent: str, text: str) -> bool:
    """A mentioned alternative is not necessarily the caller's new intent."""
    if intent == "appointment.reschedule":
        subject = r"reschedule|மாத்த|வேற\s*(?:date|நாள்)"
    elif intent == "appointment.cancel":
        subject = r"cancel|ரத்து|கேன்சல்"
    else:
        return False
    return bool(
        re.search(rf"(?:வேணாம்|வேண்டாம்|no)\s*.{{0,18}}(?:{subject})", text, re.IGNORECASE)
        or re.search(rf"(?:{subject}).{{0,18}}(?:வேணாம்|வேண்டாம்|no\b)", text, re.IGNORECASE)
    )

# Placeholders golden/main_prompt.txt is known to use (its Sec1/Sec5B/Sec6D
# references: {{agent_name}}, {{caller_mobile}}, {{mrn}}, {{campaign}},
# {{patient_name}}, {{caller_name}}, {{last_visit}}). Anything else in the
# template is almost certainly a typo or a new placeholder nobody wired up -
# substituting it with "" would silently leak a blank into what the agent
# says on a live call, so unknown placeholders are left as literal text and
# logged instead of guessed at.
KNOWN_PLACEHOLDERS = frozenset(
    {"agent_name", "caller_name", "caller_mobile", "mrn", "campaign", "last_visit", "patient_name"}
)

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Tool results carry two kinds of key. Some are facts the agent must not
# re-ask for and must read back verbatim (mrn, appointment_id, eta_minutes).
# The rest are per-call control flow - whether a lookup hit, whether a tool
# errored, the nested payloads (slot lists, bill line items) that are already
# in the tool message verbatim a few lines up in the history. Only the first
# kind belongs in the standing facts block: restating "found: True" every turn
# teaches the model nothing and spends tokens, and re-flattening a slot list
# into prose invites it to quote a slot that was never offered.
_LEDGER_CONTROL_KEYS = frozenset({"found", "error", "status", "verified", "reason"})

# Human-readable labels for the ledger keys the tools in tools.py actually
# return. A key with no entry here is still shown (falling back to the raw
# key) rather than dropped - a new tool returning a new fact should surface
# to the model immediately, not go silently missing until someone updates
# this table.
_LEDGER_LABELS: dict[str, str] = {
    "appointment_id": "appointment ID",
    "bill_number": "bill number",
    "cancellation_reference": "cancellation reference",
    "confirmation_status": "appointment confirmation",
    "dispatch_id": "ambulance dispatch ID",
    "escalation_id": "escalation ID",
    "eta_minutes": "ambulance ETA (minutes)",
    "order_id": "lab order ID",
    "policy_number": "policy number",
    "preauth_reference": "pre-authorisation reference",
    "refill_reference": "refill reference",
    "request_id": "records request ID",
    "sent_channel": "report sent via",
    "ticket_id": "ticket ID",
    "transferred_to": "call transferred to",
    "patient_name_answer": "patient name as caller said it",
    "department_answer": "department or doctor as caller said it",
    "requested_date_answer": "requested date as caller said it",
    "requested_time_answer": "requested time as caller said it",
    "callback_mobile_answer": "callback mobile as caller said it",
    "appointment_identifier_answer": "appointment identifier as caller said it",
    "latest_correction_verbatim": "latest correction — replaces any older conflicting value",
    "emergency_address": "emergency address — repeat verbatim",
}

# golden/main_prompt.txt Sec5A: the opening line is said verbatim on
# [CALL_CONNECTED], before any LLM call - "Your first action is always to
# SPEAK. Never call a tool or hang up on the first turn."
OPENING_LINE = "வணக்கம், அருவி ஹாஸ்பிட்டல். நான் {{agent_name}} பேசுறேன். உங்களுக்கு எப்படி help பண்ணலாம்?"


def render_template(template: str, metadata: dict[str, str]) -> str:
    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in KNOWN_PLACEHOLDERS:
            logger.warning("unknown template placeholder {{%s}} left unsubstituted", key)
            return match.group(0)
        return metadata.get(key, "")

    return _PLACEHOLDER_RE.sub(_substitute, template)


@dataclass(frozen=True)
class AgentClause:
    """One speakable clause of the agent's reply, released as soon as it closes."""

    text: str


@dataclass(frozen=True)
class AgentTurn:
    """The completed turn: everything said, plus any call-control action."""

    text: str
    # IDs/phone numbers the agent stated that no tool, no caller turn and no
    # standing fact accounts for - see backend/grounding.py. Empty is the
    # expected case; anything here is a fabrication the caller was just told.
    ungrounded: tuple[str, ...] = ()
    # Actions the agent claimed to have COMPLETED with no tool call behind
    # them - "Ambulance அனுப்பிட்டேன்" having dispatched nothing. Separate from
    # `ungrounded` because there is no invented identifier to point at: the
    # sentence is a lie about what the server did, not about what it knows.
    unbacked_claims: tuple[str, ...] = ()


@dataclass
class CallSession:
    connection_id: str
    metadata: dict[str, str]
    ledger: dict[str, object] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    # Sticky across turns: a caller states their reason once, then answers
    # follow-up questions ("ஆமாம்", a phone number) that match no trigger at
    # all. Re-detecting per turn would drop the playbook mid-flow, so a new
    # detection replaces this and silence leaves it alone (Sec6E).
    intent: str | None = None
    # Persistent workflow state for the five supported flows. Unlike the old
    # decision tree, this is updated from the previous question and never
    # recomputed from scratch, so arbitrary names/dates/numbers do not strand
    # the call on one branch.
    flow_state: dict[str, str] = field(default_factory=dict)
    # The opening clause of each of the last few spoken turns, and how many
    # times running the repeat breaker has had to fire. See _is_repeat_opening.
    #
    # A window rather than just the previous turn, because one remembered turn
    # only makes the model ALTERNATE: it says A, gets stopped, says the
    # recovery line, then says A again - which no longer matches, so it can go
    # A / recovery / A / recovery for the rest of the call and the caller never
    # reaches the handoff. Three is enough to catch that and still short enough
    # that a genuinely new turn clears it.
    recent_openings: list[str] = field(default_factory=list)
    repeat_count: int = 0
    # The clauses of the scripted opening line, which the caller has already
    # heard by the time any of this runs. See _opens_with_the_greeting_again.
    greeting_clauses: frozenset[str] = frozenset()

    def known_facts(self) -> dict[str, str]:
        """Placeholder substitutions for this turn: opening metadata, overlaid
        by anything the ledger has since learned.

        Ledger wins on conflict: a tool result is a fresher, better-grounded
        source than whatever the call opened with (a CRM guess from caller ID,
        say). A blank/None ledger value never overwrites a real metadata one,
        so a tool returning `{"mrn": None}` cannot erase a known MRN.
        """
        facts = dict(self.metadata)
        for key in KNOWN_PLACEHOLDERS:
            value = self.ledger.get(key)
            if value not in (None, ""):
                facts[key] = str(value)
        return facts


def _format_established_facts(session: CallSession) -> str:
    """Render the non-placeholder ledger facts as a standing block.

    KNOWN FACTS in the core prompt only has slots for the seven caller-identity
    placeholders. Everything else a call establishes - the appointment ID just
    booked, the ticket number just raised, the ambulance ETA - has no slot, and
    lives only in a tool message that scrolls further back with every turn. On
    a long call that is exactly how an agent ends up inventing a reference ID
    at closing time, which the prompt's GROUNDING section forbids outright. So
    the facts ride at the front of every turn instead, where they cannot scroll
    away.
    """
    lines = []
    for key, value in session.ledger.items():
        if key in KNOWN_PLACEHOLDERS or key in _LEDGER_CONTROL_KEYS:
            continue
        # Scalars only. Nested payloads (slot lists, bill line items) are
        # already verbatim in their tool message; flattening them to prose here
        # would both duplicate tokens and blur which values a tool actually
        # returned - the thing GROUNDING most needs kept sharp.
        if not isinstance(value, (str, int, float, bool)) or value in (None, ""):
            continue
        lines.append(f"{_LEDGER_LABELS.get(key, key)}: {value}")

    if not lines:
        return ""
    return (
        "\n## ESTABLISHED THIS CALL — say these back exactly, never re-ask, never re-invent\n"
        + "\n".join(lines)
    )


class ConversationManager:
    """Owns the prompt builder, per-call sessions, and the shared mock hospital DB."""

    def __init__(self, settings: ConversationSettings) -> None:
        self.settings = settings
        self.prompts = PromptBuilder(
            settings.runtime_core_path, settings.prompt_path, settings.exemplars_path
        )
        self._sessions: dict[str, CallSession] = {}
        # A test seam for the lower-level model speech guards. Production keeps
        # the finite five-flow controller on; guard tests turn it off so they
        # can still inject deliberately broken model output below that layer.
        self._deterministic_flows = True

    @property
    def ready(self) -> bool:
        return self.prompts.ready

    def load(self) -> None:
        """Read the prompts once during startup, keeping call-time latency low."""
        self.prompts.load()

    def start_call(self, connection_id: str, **metadata: str) -> str:
        """Open a new call session and return the scripted greeting (Sec5A - no LLM call)."""
        if not self.prompts.ready:
            raise RuntimeError("ConversationManager prompt is not loaded")

        greeting = render_template(OPENING_LINE, metadata)
        session = CallSession(
            connection_id=connection_id,
            metadata=dict(metadata),
            ledger=dict(metadata),
            # The system message is a placeholder here and rewritten every turn
            # by _refresh_system_prompt() once the flow is known - index 0 is
            # reserved for it so history stays append-only.
            messages=[
                {"role": "system", "content": ""},
                {"role": "assistant", "content": greeting},
            ],
            greeting_clauses=frozenset(split_reply_into_clauses(greeting)),
        )
        session.messages[0]["content"] = self._system_prompt_for(session)
        self._sessions[connection_id] = session
        return greeting

    async def prewarm(self, connection_id: str, llm: LlmClient) -> bool:
        """Evaluate this call's prompt while the caller is hearing the greeting.

        The greeting is a fixed line spoken with no LLM involvement at all, and
        it takes roughly three seconds of audio to say. For those three seconds
        the model is idle while the caller is occupied - which is exactly long
        enough to pay the one cost that cannot be cached away.

        That cost is the first prompt evaluation. Ollama caches the evaluated
        prefix of a prompt, so the SECOND turn onwards is nearly free (measured
        294 ms), but the first turn of a call has to evaluate ~2.7k tokens cold
        and that measured 6-8 seconds - the single largest contributor to
        first-turn latency. Doing it here moves it off the critical path and
        underneath audio the caller is already listening to.

        Generates one token, which is the smallest amount of work that still
        forces a full prompt evaluation. The result is thrown away; the point
        is the server-side cache it leaves behind.

        Best effort by design. A prewarm that fails, times out or is cancelled
        must never affect the call - the turn that follows simply pays the cold
        cost it would have paid anyway, so every failure mode here degrades to
        "no faster than before".
        """
        session = self._sessions.get(connection_id)
        if session is None:
            return False

        try:
            messages = _with_language_reminder(
                session.messages, self._turn_facts_message(session)
            )
            async for event in llm.stream(messages, max_tokens=1):
                if isinstance(event, ReplyComplete):
                    break
        except asyncio.CancelledError:
            # The caller spoke before the warm finished. Their turn owns the
            # model now; drop this quietly rather than racing it.
            raise
        except Exception as error:
            logger.info("prompt prewarm for %s did not complete (%s)", connection_id, error)
            return False

        logger.info("prompt prewarmed for %s", connection_id)
        return True

    def _system_prompt_for(self, session: CallSession) -> str:
        """The STATIC half of the prompt: rules, playbook, exemplars.

        Deliberately rendered against the call's OPENING metadata, which is
        fixed at start_call, rather than against the live ledger - so this text
        changes only when the detected flow changes, and is byte-identical
        across the turns in between.

        That matters for one measured reason. Ollama caches the evaluated
        prefix of a prompt, and the cache is a prefix cache: mutate one word
        near the top and everything behind it is evaluated again. On this box,
        with this ~2.7k-token prompt:

            identical prefix          36 ms
            prefix mutated by a word  28,895 ms

        The live ledger is exactly what mutates - a tool returns an MRN and the
        KNOWN FACTS block changes - so rendering it in HERE re-evaluated the
        whole prompt on every turn that learned anything, and twice on any turn
        with a tool call, since the prompt is refreshed inside the tool loop.

        The ledger still reaches the model, and still reaches it on the
        iteration right after the lookup (the PART 6 fix this must not undo) -
        it is just carried by _turn_facts_message() at the END of the message
        list, where changing it costs a hundred tokens instead of three
        thousand. Later is also strictly better for recency, which is the same
        reasoning that put _LANGUAGE_REMINDER last.
        """
        return render_template(self.prompts.build(session.intent), dict(session.metadata))

    def _turn_facts_message(self, session: CallSession) -> str:
        """The VOLATILE half: what this call has actually established so far.

        Rendered fresh every turn and appended near the end of the message
        list. See _system_prompt_for for why it is not in the system prompt.

        Only facts that are actually KNOWN are rendered. The block used to list
        all five labels every turn with blanks after them and a paragraph
        explaining what a blank meant - about 70 tokens of empty scaffolding on
        every single turn of every call, since a browser call opens knowing
        nothing but the agent's own name. Worse, it put "mrn:" in front of a
        model the reminder immediately tells never to say an MRN.

        What the caller has said is not lost by dropping it: the transcript is
        in the message list directly above, which is where a conversational
        agent's memory actually lives. This block is only for facts the SERVER
        knows independently - a telephony leg's caller ID, say.
        """
        facts = session.known_facts()
        lines = [
            f"{label}: {facts[key]}"
            for key, label in (
                ("caller_name", "caller_name"),
                ("caller_mobile", "caller_mobile"),
                ("mrn", "mrn"),
                ("patient_name", "patient_name"),
                ("last_visit", "last_visit"),
            )
            if facts.get(key)
        ]
        established = _format_established_facts(session)
        if not lines and not established:
            return ""

        prompt = "\n".join(
            ["## KNOWN FACTS — already verified, never ask for these again", *lines]
        )
        if established:
            prompt = f"{prompt}\n{established}\n"
        return prompt

    def end_call(self, connection_id: str) -> None:
        self._sessions.pop(connection_id, None)

    async def handle_utterance(self, connection_id: str, llm: LlmClient, text: str) -> str:
        """Run one caller turn and return the agent's full reply text.

        Kept for callers with no use for partial output - the eval scripts and
        the unit tests. Live transports should use stream_utterance() instead,
        so the first clause reaches TTS without waiting for the last token.
        """
        spoken: list[str] = []
        async for event in self.stream_utterance(connection_id, llm, text):
            if isinstance(event, AgentTurn):
                return event.text
            if isinstance(event, AgentClause):
                spoken.append(event.text)
        # stream_utterance always ends with an AgentTurn; this is unreachable
        # short of a generator being closed early by its consumer.
        return " ".join(spoken)

    def _check_grounding(self, session: CallSession, reply: str) -> tuple[str, ...]:
        """Flag identifiers in `reply` that nothing in this call accounts for."""
        # Sources are tool results and caller turns (grounding_sources), plus
        # the facts this call actually holds. NOT the system prompt: it carries
        # the few-shot exemplars, whose worked example includes an MRN, and
        # treating that as provenance is exactly how a parroted exemplar passes
        # for a lookup. See backend/grounding.py.
        sources = grounding_sources(session.messages)
        sources += [str(value) for value in session.ledger.values() if value not in (None, "")]
        sources += [str(value) for value in session.metadata.values() if value not in (None, "")]
        invented = ungrounded_identifiers(reply, sources)
        if invented:
            logger.error(
                "GROUNDING: %s stated identifier(s) no tool returned: %s",
                session.connection_id,
                ", ".join(invented),
            )
        return tuple(invented)

    def _check_action_claims(self, session: CallSession, reply: str) -> tuple[str, ...]:
        """Flag actions `reply` says are done that no tool in this call did."""
        called = {
            call["function"]["name"]
            for message in session.messages
            for call in (message.get("tool_calls") or [])
        }
        # The MVP treats emergency dispatch as a built-in simulated action.
        # Keep every other completion claim behind its real tool call.
        if session.intent == EMERGENCY_INTENT:
            called.add("dispatchAmbulance")
        claims = unbacked_action_claims(reply, called)
        if claims:
            logger.error(
                "UNBACKED CLAIM: %s %s - no tool call behind it",
                session.connection_id,
                "; ".join(claims),
            )
        return tuple(claims)

    def record_interrupted_turn(self, connection_id: str, spoken: str) -> None:
        """Append what the agent actually got out before the caller cut in.

        Barge-in cancels the task consuming stream_utterance(), which can land
        on a yield - leaving the turn's assistant message never appended, so
        the model's next turn sees the caller's line answered by nothing at
        all. Recording the truncated text keeps the history honest: the model
        should believe it said exactly what the caller heard, no more, so it
        can pick up mid-thought rather than start the same sentence again.
        """
        session = self._sessions.get(connection_id)
        if session is None:
            return
        text = spoken.strip()
        if not text:
            return
        if session.messages and session.messages[-1].get("role") == "assistant":
            return
        session.messages.append({"role": "assistant", "content": text})

    async def stream_utterance(self, connection_id: str, llm: LlmClient, text: str):
        """Run one caller turn, yielding each clause as soon as it closes.

        Yields zero or more AgentClause, then exactly one AgentTurn.

        There is no tool loop. This agent talks: it remembers what the caller
        told it (the transcript IS the memory) and answers from the prompt.
        Measured on this box, sending the 22 tool schemas cost 1778 of the
        5290 prompt tokens AND dropped generation from 12.5 to 10.2 tok/s,
        which is ~20% of the time-to-first-word budget on a voice channel -
        and register_eval scored the same scenarios 12/14 clean without them
        against 10/14 with them. Removing them is faster AND better spoken.
        """
        session = self._sessions.get(connection_id)
        if session is None:
            raise RuntimeError(f"no active call session for {connection_id}")

        # Append first because VAD/barge-in may split one human utterance into
        # consecutive caller messages. The helper merges only when no agent
        # speech occurred between them and returns the complete text to route.
        routing_text = _append_caller_turn(session, text)
        detected = detect_intent(routing_text)
        if (
            detected in SUPPORTED_INTENTS
            and detected != session.intent
            and not _detected_intent_is_negated(detected, routing_text)
        ):
            logger.info("flow detected for %s: %s", connection_id, detected)
            session.intent = detected
            session.flow_state.clear()
        elif detected is None and session.intent is None and names_a_department(text):
            # "Ortho-க்கு வரணும்" carries no booking verb and matches no
            # trigger, but a caller naming a department wants an appointment.
            # Only before a flow is picked: mid-call a department name is the
            # ANSWER to "எந்த department?", and re-routing a cancellation to a
            # booking because the caller answered is worse than the miss.
            session.intent = DEPARTMENT_INTENT
            logger.info("department named by %s, routing to %s", connection_id, DEPARTMENT_INTENT)
        session.messages[0]["content"] = self._system_prompt_for(session)

        _capture_answer_state(session, text)

        # A HARD REFUSAL is a safety decision, not a language task, so the two
        # highest-consequence ones are said in fixed words. Everything ELSE
        # this desk does is a conversation and belongs to the model.
        #
        # There used to be a third caller here, a regex decision tree that
        # answered fifteen intents itself. It recomputed its state from the
        # transcript on every turn, so any caller answer its patterns did not
        # recognise - and a date of birth spoken as "17-04-1968" never contains
        # the literal words "date of birth" - left it on the same branch,
        # re-emitting the identical sentence for the rest of the call. It also
        # spoke in fixed clerical wording that measured 49% Tamil against the
        # 65% the register requires. backend/scripts/multiturn_eval.py scored
        # it 0/7 clean calls with 9 repeated turns; the model scores better on
        # every one of those axes, which is the whole reason there is a model.
        direct_reply = _deterministic_safety_reply(session, text)
        if direct_reply is not None:
            session.messages.append({"role": "assistant", "content": direct_reply})
            _trim_history(session, llm.settings)
            for clause in split_reply_into_clauses(direct_reply):
                yield AgentClause(clause)
            yield AgentTurn(text=direct_reply)
            return

        # Nothing this desk does. Say so, instead of asking the model to
        # improvise an answer out of the info.general playbook.
        #
        # Two ways to get here, and they are not the same evidence:
        #
        #   POSITIVELY identified as another desk's work - the turn matched a
        #     trigger for one of the fifteen flows outside SUPPORTED_INTENTS.
        #     That is certain enough to decline at any point in the call, and
        #     it deliberately does NOT clear session.intent: a caller who asks
        #     about their bill halfway through a booking hears the scope line
        #     and then carries on booking.
        #   nothing matched at all - much weaker evidence, so it is braked
        #     twice over. Only on the caller's FIRST turn (later unroutable
        #     turns are the caller ANSWERING a question - a day, a name, an
        #     address - and belong to the model; measured, without this the
        #     line displaced "Anna Nagar 2nd street" given mid-call), and only
        #     for a real sentence, because a bare number or "ஆமாம்" matches no
        #     trigger either.
        if (detected is not None and detected not in SUPPORTED_INTENTS) or (
            session.intent is None
            and _is_first_caller_turn(session)
            and len(text.split()) >= _MIN_OUT_OF_SCOPE_WORDS
        ):
            logger.info("out of scope for %s (%s): %r", connection_id, detected, text)
            session.messages.append({"role": "assistant", "content": _OUT_OF_SCOPE})
            _trim_history(session, llm.settings)
            for clause in split_reply_into_clauses(_OUT_OF_SCOPE):
                yield AgentClause(clause)
            yield AgentTurn(text=_OUT_OF_SCOPE)
            return

        # The three appointment workflows and public facts have finite,
        # explicit state. Keep next-slot selection deterministic so arbitrary
        # caller values cannot push a 4B model back to an earlier question or
        # make it claim a cancellation already happened.
        flow_reply = _deterministic_flow_reply(session, text) if self._deterministic_flows else None
        if flow_reply is not None:
            session.messages.append({"role": "assistant", "content": flow_reply})
            _trim_history(session, llm.settings)
            for clause in split_reply_into_clauses(flow_reply):
                yield AgentClause(clause)
            yield AgentTurn(
                text=flow_reply,
                ungrounded=self._check_grounding(session, flow_reply),
                unbacked_claims=self._check_action_claims(session, flow_reply),
            )
            return

        chunker = ClauseChunker()
        spoken: list[str] = []
        asked_question = False
        # Resolved once per turn rather than per clause: it cannot change
        # mid-generation, and LANGUAGE forbids switching the address form
        # within a call anyway.
        gender = caller_gender(session)

        def speakable(clause: str) -> bool:
            """Whether this clause may be spoken, given what already has been.

            Enforces the one turn-discipline rule the model still breaks: ONE
            question per turn. Prose has failed three times (LLM_STACK.md Sec9)
            - runtime_core.txt states the rule three ways in one line and the
            model asks two anyway.

            This is not a truncation of the reply. Measured against the real
            recorded calls in call_events.db, a two-question turn arrives as
            two SEPARATE clauses:

                | நீங்க ... இருப்பீங்களா சார்?
                | எங்கே ... போகிறீர்கள்?

            so the second is still unspoken when it closes and can simply be
            withheld. Later NON-question clauses are kept - the closing line
            ("desk-ல இருந்து call பண்ணுவாங்க") often follows the question,
            and dropping the tail wholesale would lose it.

            '?' is the same test backend/scripts/register_eval.py scores turns
            with, deliberately: one definition, so the guard and the eval
            cannot disagree about what a question is.
            """
            nonlocal asked_question, stuck, regreeted, fabricated, unbacked, echoed, anchor
            # A LOOPING DECODER IS NOT SPEECH. Checked first because it is the
            # cheapest test here and the most decisive: nothing else about a
            # clause matters once it is "முழு முழு முழு முழு". Routed into
            # `stuck` rather than given its own recovery - a wedged model and a
            # wedged line need the same thing from the caller's side, and the
            # escalation to the desk handoff is the right ending for both.
            if _is_degenerate(clause):
                logger.error("degenerate generation from %s: %r", connection_id, clause[:80])
                stuck = True
                return False
            # NEVER SPEAK AN IDENTIFIER THE AGENT CANNOT ACCOUNT FOR. Observed
            # live over the socket: the agent asked for a mobile number, the
            # caller answered "வயசு 58" instead, and the agent replied
            # "90045 33218 என்ன சொல்லுங்க?" - reading out the phone number from
            # its own few-shot exemplar as if the caller had said it. Needing a
            # slot the conversation had not filled, it took the only value it
            # had ever seen (LLM_STACK.md Sec6).
            #
            # grounding.py already detects exactly this and its docstring says
            # it deliberately does NOT filter speech, "because by the time a
            # clause is checked it has already been streamed to the caller".
            # That premise stopped being true when speakable() became a
            # pre-speech choke point: nothing here has been spoken yet. The
            # other half of that reasoning - that withholding half a sentence
            # is worse than the fault - is handled by _CANNOT_RECALL below,
            # which asks for the detail plainly when the whole turn goes.
            #
            # A number the caller actually said is in `sources` and stays
            # grounded, so ordinary read-back ("98407 21534, குறிச்சுக்கிட்டேன்")
            # is untouched. The system prompt is deliberately not a source -
            # that is what makes the exemplar's number fabricated here.
            if ungrounded_identifiers(clause, sources):
                logger.error(
                    "grounding: %s withheld a fabricated identifier: %s",
                    connection_id,
                    clause,
                )
                fabricated = True
                return False
            claims = unbacked_action_claims(clause, called_tools)
            if claims:
                logger.error(
                    "action grounding: %s withheld unbacked claim(s) %s: %s",
                    connection_id,
                    ", ".join(claims),
                    clause,
                )
                unbacked = True
                return False
            # The caller has already heard the opening line - it is the first
            # thing this call did. The model says it AGAIN when the caller
            # opens with "ஹெல்லோ" instead of a request, because a greeting
            # invites a greeting; measured on a clean call, turn 1 came back as
            # the whole opening line verbatim, and on the recorded call it came
            # back as the first three clauses with a real question stuck on the
            # end. Dropping the clauses the caller has already heard leaves the
            # real question, which is the only part of that turn worth saying.
            #
            # Only while nothing has been spoken yet, so the closing "வணக்கம்."
            # - which follows "நன்றி சார்." - is never touched.
            if not spoken and clause in session.greeting_clauses:
                logger.info("greeting guard: %s re-greeted with %r", connection_id, clause)
                regreeted = True
                return False
            if not spoken and _looks_like_regreeting(clause):
                logger.info("greeting guard: %s generated another greeting: %r", connection_id, clause)
                regreeted = True
                return False
            # The repeat check has to live in here rather than at the feed()
            # loop, because a one-clause reply - which is the exact shape a
            # stuck model produces - is never closed by feed() at all. The
            # chunker releases it from flush() once the stream is done, so a
            # guard on the loop alone silently never fires on the only turns it
            # exists for. This function is the one place BOTH paths go through.
            # Anchored on the first clause with something IN it, not on the
            # first clause. Those are usually not the same one, and assuming
            # they were let a verbatim repeat through: measured on the real
            # model, three emergency turns running came back as
            #
            #     "சரி, நான் கேட்டுட்டேன். முதலில் — உங்களுக்கு address சொல்லுங்க?"
            #
            # byte-identical, and the breaker never fired on any of them. The
            # clause chunker's fast-first-chunk rule cuts the opening at the
            # first comma so it can start speaking sooner, which made the
            # remembered opening "சரி" - one word, below _MIN_REPEAT_WORDS, and
            # therefore forgiven as an acknowledgement every single time.
            #
            # Skipping the acknowledgement on BOTH sides - what is compared and
            # what is remembered - is what makes the comparison land on the
            # part of the turn that carries its meaning.
            if anchor is None and len(clause.split()) >= _MIN_REPEAT_WORDS:
                anchor = clause
                if clause in session.recent_openings and _is_a_confirmation(clause):
                    # A REPEATED CONFIRMATION LOSES THE CLAUSE, NOT THE TURN.
                    #
                    # Exempting it outright (which is what _is_repeat_opening
                    # does, and must, for the closing read-back) let a whole
                    # turn repeat: the chunker cuts at the period, so the
                    # anchor was "வியாழன் கிழமை – குறிச்சுக்கிட்டேன்." and the
                    # question that made the turn new sat in the NEXT clause,
                    # unexamined. Dropping just the clause is the echo guard's
                    # shape and gets both cases right - a turn that goes on to
                    # ask something new keeps it, and a turn that is nothing
                    # but the repeat fails _carries_a_turn and reaches the
                    # stuck ladder below.
                    logger.info(
                        "repeat breaker: %s repeated a confirmation, dropping the clause: %r",
                        connection_id,
                        clause,
                    )
                    return False
                if _is_repeat_opening(clause, session.recent_openings):
                    logger.info(
                        "repeat breaker: %s was about to say %r again", connection_id, clause
                    )
                    stuck = True
                    return False
            # The caller's own sentence handed back to them. Unlike the repeat
            # breaker this does NOT end the turn: the parrot is usually the
            # opening clause and the real answer follows it, so dropping the
            # one clause and letting the rest through keeps the good half.
            if _echoes_caller(clause, text):
                logger.info("echo guard: %s parroted the caller: %r", connection_id, clause)
                echoed = True
                return False
            if "?" not in clause:
                return True
            if asked_question:
                logger.info(
                    "turn discipline: withheld a second question from %s: %s",
                    connection_id,
                    clause,
                )
                return False
            asked_question = True
            return True

        reply: LlmReply | None = None
        # Set by speakable() when this turn opens with the clause the last one
        # opened with. Declared before the stream so both the feed() loop and
        # the flush() tail can end the turn on it.
        stuck = False
        # Set by speakable() when it dropped a clause of the opening line.
        regreeted = False
        # Set by speakable() when it dropped a clause stating an identifier the
        # agent could not account for.
        fabricated = False
        # Set when the model claims a booking, callback, dispatch or other
        # action happened without a tool result behind it.
        unbacked = False
        # Set by speakable() when it dropped a clause that was the caller's own
        # sentence handed back.
        echoed = False
        # The first clause of this turn carrying more than an acknowledgement -
        # what the repeat breaker compares against, and what it remembers.
        anchor: str | None = None
        # Computed once per turn, not per clause: the caller's own words this
        # call, plus any tool/ledger facts. Includes the user message appended
        # a few lines above, so a number the caller just said is grounded.
        sources = grounding_sources(session.messages)
        called_tools = {
            call["function"]["name"]
            for message in session.messages
            for call in (message.get("tool_calls") or [])
        }
        if session.intent == EMERGENCY_INTENT:
            called_tools.add("dispatchAmbulance")
        facts = self._turn_facts_message(session)
        # Trim HERE, against the prompt about to go on the wire, and not only
        # at the end of the turn. The end-of-turn trim sizes the history for a
        # turn that has not happened yet, and the caller's next message is
        # appended after it - so what the model was actually handed could sit
        # one caller turn over the budget the trim had just enforced.
        #
        # Latent until the repeat breaker started catching more turns: while
        # every assistant turn was a full 177 characters the trim dropped whole
        # exchanges and left slack, and the short recovery lines fit inside the
        # budget exactly well enough to expose the gap (measured: 1706 chars of
        # history against a 1684-char budget, 6 tokens over num_ctx).
        _trim_history(session, llm.settings, facts)
        stream = llm.stream(_with_language_reminder(session.messages, facts))
        async for event in stream:
            if isinstance(event, ReplyComplete):
                reply = event.reply
                break
            for clause in chunker.feed(event.text):
                clause = _normalize_spoken_register(clause, gender)
                if not speakable(clause):
                    continue
                spoken.append(clause)
                yield AgentClause(clause)
            if stuck or fabricated or unbacked:
                # Nothing more is worth generating: a repeat is a stuck model,
                # and the rest of a turn built around an invented identifier is
                # built on the same mistake. Drop it on the floor.
                await stream.aclose()
                break

        ended_early = stuck or fabricated or unbacked
        if reply is None and not ended_early:
            raise RuntimeError("LLM stream ended without a ReplyComplete event")

        # The chunker never closes a clause on buffer-end (see
        # clause_chunker.py), so the reply's last clause only exists once the
        # stream is done and we ask for it.
        #
        # UNLESS the server stopped at max_tokens. Then the buffer is not a
        # last clause, it is whatever the model was in the middle of writing -
        # observed on an info.general turn that ran long, the caller heard the
        # reply end on a bare "எந்த" ("which"). A real clause always closes on
        # punctuation and reaches the caller through feed(); only the cut-off
        # tail arrives this way, and a voice channel has no way to show the
        # caller that a word was truncated. Dropping it costs at most a short
        # trailing phrase and never leaves a fragment spoken aloud.
        truncated = reply is not None and reply.finish_reason == "length"
        if truncated:
            logger.info(
                "call %s: reply hit max_tokens, dropping the unfinished tail", connection_id
            )
        if not ended_early and not truncated:
            remainder = chunker.flush()
            if remainder:
                remainder = _normalize_spoken_register(remainder, gender)
            if remainder and speakable(remainder):
                spoken.append(remainder)
                yield AgentClause(remainder)

        if stuck and not _carries_a_turn(spoken):
            ladder = (
                _EMERGENCY_STUCK_REPLIES
                if session.intent == EMERGENCY_INTENT
                else _STUCK_REPLIES
            )
            # ONLY when the turn produced nothing usable, the same test the
            # regreeted and echoed recoveries below already apply.
            #
            # The ladder says "மன்னிச்சுடுங்க Sir, clear-ஆ கேட்கல" - I did not
            # hear you. That is the right thing to say when the whole turn was
            # a repeat and the caller has heard silence otherwise. Appended
            # AFTER a good clause it is a lie about the line, and it reached a
            # caller as "98407 21534, குறிச்சுக்கிட்டேன் Sir. மன்னிச்சுடுங்க
            # Sir, clear-ஆ கேட்கல." - the number was read back correctly in
            # the same breath as claiming not to have heard it.
            #
            # repeat_count escalates only when the ladder is actually used, so
            # a mid-turn repeat cannot walk the caller towards the handoff line
            # on turns that were otherwise fine.
            session.repeat_count = min(session.repeat_count + 1, len(ladder))
            recovery = ladder[session.repeat_count - 1]
            # The recovery lines are deliberately NOT remembered: they differ
            # from each other by design, so they could never match anyway, and
            # the escalating counter is what ends a hopeless stretch.
            session.messages.append({"role": "assistant", "content": recovery})
            _trim_history(session, llm.settings, facts)
            for clause in split_reply_into_clauses(recovery):
                yield AgentClause(clause)
            yield AgentTurn(text=recovery)
            return

        if fabricated:
            # Always, not only when nothing was spoken. Dropping one clause out
            # of the middle of a turn is what grounding.py's docstring warned
            # would be "worse output than the fault it is trying to hide" -
            # "ஆமாம், சரியா?" with the invented MRN cut out of the middle says
            # nothing and sounds broken. Ending the turn on a plain request for
            # the detail is coherent, and it is what the agent should have said.
            spoken.append(_CANNOT_RECALL)
            yield AgentClause(_CANNOT_RECALL)
        elif unbacked:
            recovery = (
                _EMERGENCY_ACTION_RECOVERY
                if session.intent == EMERGENCY_INTENT
                else _ACTION_RECOVERY
            )
            spoken.append(recovery)
            yield AgentClause(recovery)
        elif not _carries_a_turn(spoken) and regreeted:
            # The whole turn was the opening line over again, so there is
            # nothing left to say - but silence on a phone call is worse than a
            # wasted turn. This is what a receptionist says when the caller has
            # said hello back and not yet got to why they rang.
            spoken.append(_GO_AHEAD)
            yield AgentClause(_GO_AHEAD)
        elif not _carries_a_turn(spoken) and echoed:
            # Same reasoning, one step further: the whole turn was the caller's
            # own question read back to them, so withholding it leaves silence.
            # Observed live on "Visiting hours என்ன?", whose entire reply was
            # "விசிடிங் ஹவுர்ஸ் என்ன சார்?".
            spoken.append(_ECHO_RECOVERY)
            yield AgentClause(_ECHO_RECOVERY)

        spoken_text = " ".join(spoken)
        # What was SPOKEN, not what was generated - the two differ whenever a
        # second question was withheld above. Same reasoning as
        # record_interrupted_turn: the model must believe it said exactly what
        # the caller heard, or it will treat a question nobody was asked as
        # already asked and never come back to it.
        session.messages.append({"role": "assistant", "content": spoken_text})
        # Remember what this turn actually SAID so later turns can be checked
        # against it, and forgive the earlier repeats: a turn that got through
        # means the model is unstuck, so a later bad patch starts again at the
        # gentlest wording rather than jumping straight to the handoff.
        if anchor is not None:
            session.recent_openings.append(anchor)
            del session.recent_openings[:-RECENT_OPENINGS_KEPT]
        session.repeat_count = 0
        _trim_history(session, llm.settings, facts)
        yield AgentTurn(
            text=spoken_text,
            ungrounded=self._check_grounding(session, spoken_text),
            unbacked_claims=self._check_action_claims(session, spoken_text),
        )


# The assembled system prompt is ~3.5k tokens against the Modelfile's
# num_ctx 8192, so a call has roughly 4.6k tokens of room for its history.
# Nothing used to bound that. A long enough call overflows the window, and
# Ollama truncates from the FRONT - taking the system prompt's language rules
# with it. That is not a hypothetical failure: it is the exact bug that
# backend/prompt_builder.py exists to fix (the agent switches to English and
# starts inventing identifiers), and it is silent.
#
# 40 -> 24, and the old number was NOT safe. "~2.4k tokens inside the window"
# was an estimate, and measuring it broke it: the widest playbook
# (emergency.escalate) makes the system prompt plus reminder 3932 tokens, and
# 40 messages of realistically long turns are a further 4158, so the worst case
# was 8390 tokens against num_ctx 8192 - already 198 tokens OVER, in the
# shipped configuration, before this session changed anything. Overflow makes
# Ollama truncate from the FRONT and take the language rules with it, silently.
#
# At num_ctx 6144 with LLM_MAX_TOKENS 160 the history budget is 2052 tokens;
# 24 messages of ordinary turns fit inside it with room, and 24 is still 12
# exchanges - longer than every real call captured in call_events.db.
#
# ponytail: a flat cap, not summarisation. It is now the CHEAP half of the
# bound - _history_budget_chars() below is the half that actually holds.
MAX_HISTORY_MESSAGES = int(os.getenv("CONVERSATION_MAX_HISTORY_MESSAGES", "24"))


# Counting messages cannot see the failure it was added to prevent, and this is
# measured, not theoretical. Sec10.2 fixed an overflow at num_ctx 8192 by
# lowering this cap to 24 and LLM_MAX_TOKENS to 300 -> 160. num_ctx was then
# lowered 8192 -> 6144 for VRAM headroom and the budget was never re-derived
# against the smaller window. Re-measured with Ollama's own prompt_eval_count,
# on the widest playbook plus 24 messages built from the LONGEST turns this
# server has actually produced (177 chars for an agent turn, 67 for a caller
# turn, both out of call_events.db):
#
#     system prompt (emergency.escalate playbook + exemplars)   3765 tok
#     + 24 messages of longest-real-turn history                3047 tok
#     + facts block + language reminder                          230 tok
#     + LLM_MAX_TOKENS 160                                       160 tok
#                                                            ---------
#                                                              7202 tok   vs num_ctx 6144
#
# 1058 tokens OVER, in the shipped configuration. Twenty-four messages is a
# safe bound for turns of median length (60 chars) and an unsafe one for turns
# of the length this agent actually produces at its longest, and no count-based
# cap can tell those apart - which is why this now bounds the thing that
# actually overflows.
#
# Tokens per character, measured the same way on this model:
#
#     assembled system prompt (markup + Tamil)   0.29 - 0.32
#     dense Tamil/English turn text              0.83
#
# Rounded UP in both cases. The cost of over-trimming is that a pathological
# call forgets an early turn; the cost of under-trimming is Ollama silently
# truncating the system prompt, which is the bug prompt_builder.py exists to
# prevent. Those are not symmetric.
_PROMPT_TOKENS_PER_CHAR = 0.35
_HISTORY_TOKENS_PER_CHAR = 0.85
# The chat template's role markers around each message, plus the trailing
# assistant header.
_TOKENS_PER_MESSAGE = 4


def split_reply_into_clauses(text: str) -> list[str]:
    """One-shot helper: feed a whole reply through ClauseChunker at once.

    The chunker never closes a clause on buffer-end (see clause_chunker.py), so
    flush() is what releases the final clause of any complete text - feed()
    alone drops it. Lives here rather than in main.py because the repeat
    breaker below speaks a canned line and needs the same splitting; main.py
    imports it for the scripted greeting.
    """
    chunker = ClauseChunker()
    clauses = chunker.feed(text)
    remainder = chunker.flush()
    if remainder:
        clauses.append(remainder)
    return clauses


# Said when the model's entire turn was the opening line again - the caller has
# said hello back and not yet reached why they rang.
_GO_AHEAD = "சொல்லுங்க, என்ன help வேணும்?"

# Said when the whole turn was withheld for stating an invented identifier.
# Mirrors runtime_core.txt's GROUNDING wording ("அது என்கிட்ட இப்போ இல்ல சார்")
# rather than inventing a new register for it.
_CANNOT_RECALL = "மன்னிச்சுடுங்க, அது என்கிட்ட இல்ல. ஒரு தடவை சொல்லுங்களா?"

# Said when the whole turn was the caller's own sentence handed back. There is
# no way to recover the answer the model failed to give, so this takes the
# request down and hands it on, which is what YOUR JOB in runtime_core.txt says
# to do with anything the agent cannot answer itself.
_ECHO_RECOVERY = "மன்னிச்சுடுங்க. அதை desk-ல இருந்து confirm பண்ணி call பண்ணுவாங்க. வேற ஏதாவது help வேணுமா?"

# Safe replacements for generated action claims. The ordinary line records a
# request, not completion. Emergency dispatch is a simulated built-in MVP
# action, so its fallback confirms dispatch while keeping the caller connected.
_ACTION_RECOVERY = "Request-ஐ குறிச்சுக்கிட்டேன். சம்பந்தப்பட்ட desk confirm பண்ணி call பண்ணுவாங்க."
_EMERGENCY_ACTION_RECOVERY = "Ambulance அனுப்பிட்டேன். Phone-ஐ வெக்காதீங்க — நான் line-ல இருக்கேன்."
_EMERGENCY_OPENING = (
    "புரியுது, நான் உங்களோட line-ல இருக்கேன். "
    "உடனே ambulance அனுப்புறேன் — நீங்க இருக்கிற முழு address சொல்லுங்க?"
)
_DISTRESS_CLARIFICATION = (
    "நான் line-ல இருக்கேன், பதட்டப்படாதீங்க. "
    "மூச்சு, நெஞ்சு வலி, மயக்கம் — என்ன பிரச்சனைன்னு சொல்லுங்க?"
)
_DIAGNOSIS_REFUSAL = (
    "அது என்ன condition-னு இங்க diagnose பண்ணவும், இல்லைன்னு rule out பண்ணவும் முடியாது. "
    "Doctor examine பண்ணணும். மூச்சுத்திணறல் அல்லது மயக்கம் இருக்கா?"
)
_EMERGENCY_MEDICINE_REFUSAL = (
    "எந்த மருந்தும் கொடுக்க நான் சொல்ல முடியாது. "
    "Ambulance team வர்ற வரைக்கும் patient-க்கு எதுவும் கொடுக்காதீங்க."
)
_LAB_VALUE_REFUSAL = (
    "Report value-ஐ phone-ல படிச்சோ interpret பண்ணியோ சொல்ல முடியாது. "
    "Doctor பார்த்துதான் சொல்லணும். Order number அல்லது registered mobile number சொல்லுங்க?"
)

_DIAGNOSIS_REQUEST_RE = re.compile(
    r"diagnos|என்ன\s*(?:நோய்|condition)|இல்லன்னு\s*.*சொல்ல|"
    r"(?:இது|அது).{0,55}(?:-ஆ|ஆ\s+இருக்க|தானா|இருக்குமா)",
    re.IGNORECASE,
)
# Split in two and required to BOTH match, because the single combined pattern
# fired on a bare medicine NAME. EMERGENCY OVERRIDE tells the agent to collect
# the patient's medicines for the paramedic, so a caller answering that with
# "அவரு BP tablet சாப்பிடுறாரு" was served a refusal to a question they had
# not asked. A refusal needs someone asking permission, not naming a drug.
_MEDICINE_RE = re.compile(r"aspirin|tablet|medicine|medication|மருந்த|மாத்திர", re.IGNORECASE)
_PERMISSION_RE = re.compile(
    r"கொடுக்கலாமா|கொடுக்கவா|கொடுக்கணுமா|சாப்பிடலாமா|சாப்பிடவா|போடலாமா|வேண்டாமா|"
    r"\b(?:can|should|shall)\s+(?:i|we)\b|\bok\s+to\s+give\b",
    re.IGNORECASE,
)
_LAB_VALUE_REQUEST_RE = re.compile(
    r"\bvalue\b|(?:sugar|report|lab).{0,24}\bnumber\b|number\s*மட்டும்",
    re.IGNORECASE,
)
# The SECOND key for both clinical refusals below, and the reason they can be
# asked on every turn instead of only inside a flow that no longer exists.
#
# They used to be gated on session.intent being "lab.result_inquiry" or
# "clinical.triage". Dropping those two flows from SUPPORTED_INTENTS made both
# refusals unreachable without touching a line of them, and safety_eval caught
# what that costs: the caller's symptom turn got the scope line, their next
# turn ("இது dengue-ஆ இருக்குமா?") reached the model with no gate at all, and
# the agent answered "dengue இல்லன்னு மட்டும்" - it ruled out a condition.
#
# Ungating them entirely is not safe either: _DIAGNOSIS_REQUEST_RE's
# "(?:இது|அது) ... -ஆ" arm matches ordinary booking Tamil like "அது நாளைக்கு
# காலைல-ஆ?". So they use the same two-key shape as the medicine refusal right
# above - the request pattern AND a clinical subject - which is what makes the
# medicine arm safe to run on every emergency turn.
_CLINICAL_SUBJECT_RE = re.compile(
    r"dengue|typhoid|covid|corona|malaria|jaundice|cancer|tumou?r|"
    r"\bTB\b|stroke|attack|infection|virus|fever|diabet|thyroid|"
    r"டெங்கு|காமாலை|புற்று|நோய|காய்ச்சல|அட்டாக|தொற்று|சர்க்கரை|"
    r"report|result|lab|scan|test|sugar|BP\b|"
    r"ரிப்போர்ட|ரிசல்ட|டெஸ்ட|ஸ்கேன|சுகர்|ரத்த",
    re.IGNORECASE,
)

_ADDRESS_HINT_RE = re.compile(
    r"address|street|road|cross|floor|door|number\s*\d|"
    r"முகவரி|தெரு|ரோடு|நகர்|கிராஸ்|மாடி|வீட்டு\s*எண்",
    re.IGNORECASE,
)
_EMERGENCY_RESPONSE_STATUS_RE = re.compile(
    r"response|respond|awake|conscious|பேசு|பேசுற|கண்\s*திற|உணர்வ|நினைவு|மயக்க",
    re.IGNORECASE,
)
_EMERGENCY_BREATHING_STATUS_RE = re.compile(
    r"breath|மூச்சு|சுவாச",
    re.IGNORECASE,
)


def _awaiting_emergency_address(session: CallSession) -> bool:
    """Whether the latest completed agent turn requested an address."""
    return any(
        message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and "address சொல்லுங்க" in message["content"]
        for message in session.messages[-2:]
    )


def _has_started_emergency_dispatch(session: CallSession) -> bool:
    return any(
        message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and "ambulance அனுப்பு" in message["content"].lower()
        for message in session.messages
    )

_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_EXPLICIT_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAME}|{_MONTH_NAME}\s+\d{{1,2}}(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)
_DATE_CORRECTION_RE = re.compile(
    r"(?:\bno\b|\bactually\b|\binstead\b|change|correct|இல்ல|வேணும்|மாத்த|பதிலா)",
    re.IGNORECASE,
)
_PART_OF_DAY_RE = re.compile(
    r"morning|afternoon|evening|night|காலை|மதியம்|மாலை|இரவு",
    re.IGNORECASE,
)
_MOBILE_IN_SPEECH_RE = re.compile(r"(?<!\d)(?:\d[ -]?){10}(?!\d)")

_GENERAL_CORRECTION_RE = re.compile(
    r"(?:\bno\b|\bactually\b|\binstead\b|change|correct|இல்ல|வேணாம்|மாத்த|பதிலா)",
    re.IGNORECASE,
)
_ANSWER_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("callback_mobile_answer", re.compile(r"mobile|phone\s*number|contact\s*number|நம்பர்", re.IGNORECASE)),
    ("appointment_identifier_answer", re.compile(r"appointment\s*(?:ID|number)|booking\s*(?:ID|number)|MRN", re.IGNORECASE)),
    ("patient_name_answer", re.compile(r"patient\s*(?:name|பேர)|பேரு\s*சொல்|name\s*சொல்", re.IGNORECASE)),
    ("department_answer", re.compile(r"எந்த\s*(?:department|doctor)|which\s*(?:department|doctor)", re.IGNORECASE)),
    ("requested_date_answer", re.compile(r"எந்த\s*நாள்|which\s*(?:day|date)|date\s*சொல்|நாள்\s*சொல்", re.IGNORECASE)),
    ("requested_time_answer", re.compile(r"காலையா|மாலையா|எந்த\s*நேரம்|what\s*time|time\s*சொல்", re.IGNORECASE)),
)


def _capture_answer_state(session: CallSession, caller_text: str) -> None:
    """Persist caller answers to the agent's last question verbatim.

    This is deliberately question-shaped rather than a Tamil NER system: the
    previous turn already tells us which slot was requested, so an arbitrary
    name, locality, department, date, or mixed-language answer can be stored
    without guessing its grammar. The model phrases the next turn; it does not
    get to rewrite the authoritative value.
    """
    previous_agent = next(
        (
            str(message.get("content", ""))
            for message in reversed(session.messages[:-1])
            if message.get("role") == "assistant"
        ),
        "",
    )
    for key, pattern in _ANSWER_SLOT_PATTERNS:
        if pattern.search(previous_agent):
            session.ledger[key] = caller_text
            break
    if session.intent in SUPPORTED_INTENTS and _GENERAL_CORRECTION_RE.search(caller_text):
        session.ledger["latest_correction_verbatim"] = caller_text


_DATE_LIKE_RE = re.compile(
    rf"{_MONTH_NAME}|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"திங்கள்|செவ்வாய்|புதன்|வியாழ|வெள்ளி|சனி|ஞாயிற|நாளை|நாளைக்கு|"
    r"இன்னைக்கு|அடுத்த\s*வாரம்",
    re.IGNORECASE,
)
_DECLINES_RESCHEDULE_RE = re.compile(r"வேணாம்|cancel|ரத்து|முடியாது|no\b", re.IGNORECASE)
_ACCEPTS_RESCHEDULE_RE = re.compile(r"மாத்த|reschedule|வேற\s*(?:date|நாள்)|சரி", re.IGNORECASE)


def _valid_slot_answer(slot: str, text: str) -> bool:
    if slot == "mobile":
        return bool(_MOBILE_IN_SPEECH_RE.search(text))
    if slot in {"date", "old_date", "appointment_date"}:
        return bool(_DATE_LIKE_RE.search(text))
    if slot == "time":
        return bool(_PART_OF_DAY_RE.search(text))
    if slot == "department":
        return names_a_department(text)
    if slot == "identifier":
        return bool(_MOBILE_IN_SPEECH_RE.search(text) or re.search(r"\b[A-Z]{2,6}-\d{3,}\b", text))
    return bool(text.strip())


def _remember_common_appointment_facts(
    session: CallSession, caller_text: str, *, capture_date: bool = True
) -> None:
    """Update only facts that are unambiguous in arbitrary caller wording."""
    state = session.flow_state
    awaiting = state.pop("awaiting", "")
    if awaiting and _valid_slot_answer(awaiting, caller_text):
        state[awaiting] = caller_text
    elif awaiting:
        state["awaiting"] = awaiting

    if names_a_department(caller_text):
        state["department"] = caller_text
    mobile = _MOBILE_IN_SPEECH_RE.search(caller_text)
    if mobile:
        state["mobile"] = re.sub(r"[ -]", "", mobile.group(0))
    if _PART_OF_DAY_RE.search(caller_text):
        state["time"] = caller_text
    if capture_date and _DATE_LIKE_RE.search(caller_text):
        # `old_date` is captured specially on the reschedule opener below.
        if session.intent != "appointment.reschedule" or "old_date" in state:
            state["date"] = caller_text


def _booking_reply(session: CallSession, caller_text: str) -> str:
    state = session.flow_state
    mobile_in_current = bool(_MOBILE_IN_SPEECH_RE.search(caller_text))
    _remember_common_appointment_facts(session, caller_text)
    if "patient_name" not in state:
        state["awaiting"] = "patient_name"
        return "கண்டிப்பா. Patient பேரு சொல்லுங்க?"
    if "department" not in state:
        state["awaiting"] = "department"
        return "எந்த department அல்லது doctor-க்கு appointment வேணும்?"
    if "date" not in state:
        state["awaiting"] = "date"
        return "எந்த நாள் convenient?"
    if "time" not in state:
        state["awaiting"] = "time"
        if mobile_in_current:
            return "Mobile number குறிச்சுக்கிட்டேன். Appointment காலையா மாலையா convenient?"
        if re.search(r"நன்றி|thank", caller_text, re.IGNORECASE):
            return "ஒரு detail மட்டும் வேணும் — appointment காலையா மாலையா convenient?"
        return "காலையா மாலையா convenient?"
    if "mobile" not in state:
        state["awaiting"] = "mobile"
        return "திரும்ப call பண்ண வேண்டிய mobile number சொல்லுங்க?"
    if state.get("closed"):
        return "நன்றி. வேற ஏதாவது help வேணுமா?"
    state["closed"] = "1"
    return (
        "எல்லா தகவலும் குறிச்சுக்கிட்டேன். Appointment desk நேரத்தை உறுதி பண்ணி "
        "call பண்ணுவாங்க; SMS-ம் வரும்."
    )


def _reschedule_reply(session: CallSession, caller_text: str) -> str:
    state = session.flow_state
    mobile_in_current = bool(_MOBILE_IN_SPEECH_RE.search(caller_text))
    first_flow_turn = not state
    if first_flow_turn and _DATE_LIKE_RE.search(caller_text):
        state["old_date"] = caller_text
    _remember_common_appointment_facts(session, caller_text, capture_date=not first_flow_turn)
    if "patient_name" not in state:
        state["awaiting"] = "patient_name"
        return "கண்டிப்பா. Patient பேரு சொல்லுங்க?"
    if "old_date" not in state:
        state["awaiting"] = "old_date"
        return "இப்போ இருக்கிற appointment எந்த நாள்?"
    if "date" not in state:
        state["awaiting"] = "date"
        return "புதுசா எந்த நாள் convenient?"
    if "time" not in state:
        state["awaiting"] = "time"
        if mobile_in_current:
            return "Mobile number குறிச்சுக்கிட்டேன். புதிய appointment காலையா மாலையா convenient?"
        return "அந்த நாள் காலையா மாலையா convenient?"
    if "mobile" not in state:
        state["awaiting"] = "mobile"
        return "திரும்ப call பண்ண வேண்டிய mobile number சொல்லுங்க?"
    if state.get("closed"):
        return "நன்றி. வேற ஏதாவது help வேணுமா?"
    state["closed"] = "1"
    return (
        "புதிய நாள் request-ஐ குறிச்சுக்கிட்டேன். Desk உறுதி பண்ணி call பண்ணுவாங்க; "
        "அதுவரைக்கும் பழைய appointment அப்படியே இருக்கும்."
    )


def _cancel_reply(session: CallSession, caller_text: str) -> str:
    state = session.flow_state
    first_flow_turn = not state
    if first_flow_turn and _DATE_LIKE_RE.search(caller_text):
        state["appointment_date"] = caller_text
    awaiting = state.pop("awaiting", "")
    if awaiting and _valid_slot_answer(awaiting, caller_text):
        state[awaiting] = caller_text
    elif awaiting:
        state["awaiting"] = awaiting
    mobile = _MOBILE_IN_SPEECH_RE.search(caller_text)
    if mobile:
        state["identifier"] = re.sub(r"[ -]", "", mobile.group(0))

    if "patient_name" not in state:
        state["awaiting"] = "patient_name"
        return "சரி. Patient பேரு சொல்லுங்க?"
    if "appointment_date" not in state:
        state["awaiting"] = "appointment_date"
        return "Cancel பண்ண வேண்டிய appointment எந்த நாள்?"
    if "reschedule_answer" not in state:
        state["awaiting"] = "reschedule_answer"
        return "Cancel பண்ண முன்னாடி, வேற date-க்கு மாத்திக்கலாமா?"
    choice = state["reschedule_answer"]
    if _ACCEPTS_RESCHEDULE_RE.search(choice) and not _DECLINES_RESCHEDULE_RE.search(choice):
        session.intent = "appointment.reschedule"
        session.flow_state = {
            "patient_name": state["patient_name"],
            "old_date": state["appointment_date"],
        }
        session.flow_state["awaiting"] = "date"
        return "சரி, மாத்திக்கலாம். புதுசா எந்த நாள் convenient?"
    if "identifier" not in state:
        state["awaiting"] = "identifier"
        return "Appointment ID இல்ல mobile number சொல்லுங்க?"
    if state.get("closed"):
        return "நன்றி. வேற ஏதாவது help வேணுமா?"
    state["closed"] = "1"
    return (
        "Cancellation request குறிச்சுக்கிட்டேன். Desk உறுதி பண்ணி இதே number-க்கு "
        "call பண்ணுவாங்க."
    )


def _information_reply(session: CallSession, caller_text: str) -> str | None:
    """Answer only the public fact asked for; never inherit exemplar branches."""
    text = caller_text.lower()
    if re.search(r"நன்றி|thank", text, re.IGNORECASE):
        return "நன்றி. வணக்கம்."
    parts: list[str] = []
    asks_wheelchair_time = False
    identity_requested = bool(
        re.search(
            r"who\s+are\s+you|your\s+name|agent\s*name|"
            r"உங்க\s*பேரு|உங்கள்\s*பெயர்|நீங்க\s*யாரு|யார்\s*பேசுற|"
            r"hospital\s*(?:name|details?|address)|"
            r"(?:ஹாஸ்பிட்டல்|ஆஸ்பத்திரி)\s*(?:பேரு|பெயர்|விவரம்|details?|address)|"
            r"எந்த\s*(?:hospital|ஹாஸ்பிட்டல்|ஆஸ்பத்திரி)",
            caller_text,
            re.IGNORECASE,
        )
    )
    if identity_requested:
        agent_name = str(session.ledger.get("agent_name") or "Gayathri")
        parts.append(
            f"நான் {agent_name} பேசுறேன். இது Aruvi Multispeciality Hospital. "
            "Address OMR, Perungudi, Chennai; Perungudi signal-லிருந்து 2 kilometres."
        )
    if re.search(r"\bICU\b", caller_text, re.IGNORECASE):
        parts.append("ICU visiting மாலை 5 to 5:30 மட்டும்; ஒரு நேரத்துல ஒருத்தர் தான்.")
    elif re.search(r"visiting|விசிட்டிங்", text, re.IGNORECASE):
        parts.append("General ward visiting காலை 11 to 12, மாலை 5 to 7; ICU மாலை 5 to 5:30, ஒருத்தர் மட்டும்.")
    if re.search(r"parking|பார்க்கிங்", text, re.IGNORECASE):
        parts.append("Parking basement Gate 2-ல; முதல் 2 hours ₹30, அப்புறம் hour-க்கு ₹20.")
    if re.search(r"canteen|கேன்டீன்", text, re.IGNORECASE):
        parts.append("Canteen ground floor-ல காலை 6 முதல் இரவு 10 வரை open.")
    if re.search(r"wheelchair|வீல்சேர்|வீல்செயர்", text, re.IGNORECASE):
        parts.append("Wheelchair attendant-ஓட free-ஆ Gate 1-ல கிடைக்கும்.")
        asks_wheelchair_time = True
    if re.search(r"attender|அட்டெண்டர்", text, re.IGNORECASE):
        parts.append("General ward-ல ஒரு attender free pass-ஓட தங்கலாம்; ICU-க்குள் தங்க முடியாது.")
    if not identity_requested and re.search(r"location|address|எப்படி\s*வர|எங்க", text, re.IGNORECASE):
        parts.append("Hospital OMR-ல, Perungudi signal-லிருந்து 2 kilometres; location link SMS-ல அனுப்பலாம்.")
    if re.search(r"timing|OP|open|வேலை|எத்தனை\s*மணி", caller_text, re.IGNORECASE):
        parts.append("OP Monday to Saturday காலை 8 to 1, மாலை 4 to 8; Sunday காலை 9 to 1.")
    if not parts:
        return None
    if asks_wheelchair_time:
        parts.append("எத்தனை மணிக்கு wheelchair வேணும்?")
    return " ".join(parts)


def _deterministic_flow_reply(session: CallSession, caller_text: str) -> str | None:
    if session.intent == "appointment.book":
        return _booking_reply(session, caller_text)
    if session.intent == "appointment.reschedule":
        return _reschedule_reply(session, caller_text)
    if session.intent == "appointment.cancel":
        return _cancel_reply(session, caller_text)
    if session.intent == "info.general":
        return _information_reply(session, caller_text)
    return None


def _caller_history(session: CallSession) -> str:
    return " ".join(
        str(message.get("content", ""))
        for message in session.messages
        if message.get("role") == "user"
    )


def _appointment_date_correction(session: CallSession, caller_text: str) -> str | None:
    """Apply a clearly spoken replacement date without asking the 4B model.

    The loop breaker can detect that the model repeated the old date, but it
    cannot recover the new one from generated text. The caller turn can: once
    normalization has produced `6th September`, latest-value-wins is a small,
    deterministic state update rather than a language-generation problem.
    """
    if session.intent not in {"appointment.book", "appointment.reschedule"}:
        return None
    if _is_first_caller_turn(session) or not _DATE_CORRECTION_RE.search(caller_text):
        return None
    match = _EXPLICIT_DATE_RE.search(caller_text)
    if match is None:
        return None

    new_date = match.group(0)
    history = _caller_history(session)
    session.flow_state["date"] = new_date
    session.flow_state.pop("closed", None)
    acknowledgement = f"{new_date}, மாத்தி குறிச்சுக்கிட்டேன்."
    if not _PART_OF_DAY_RE.search(history):
        session.flow_state["awaiting"] = "time"
        return f"{acknowledgement} காலையா மாலையா convenient?"
    if not _MOBILE_IN_SPEECH_RE.search(history):
        session.flow_state["awaiting"] = "mobile"
        return f"{acknowledgement} Mobile number சொல்லுங்க?"
    return f"{acknowledgement} Appointment desk confirm பண்ணி call பண்ணுவாங்க."


def _mentions_a_clinical_subject(session: CallSession, caller_text: str) -> bool:
    """Whether a condition, symptom or report is anywhere in this call.

    The whole call, not just this turn: "இது dengue-ஆ?" names the condition and
    the follow-up "இல்லன்னு மட்டும் சொல்லுங்க" does not, and CLINICAL SAFETY is
    explicit that the second ask must be refused in the same words as the first.
    """
    if _CLINICAL_SUBJECT_RE.search(caller_text):
        return True
    return any(
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and _CLINICAL_SUBJECT_RE.search(message["content"])
        for message in session.messages
    )


def _deterministic_safety_reply(session: CallSession, caller_text: str) -> str | None:
    """Fixed wording for the refusals that cost a caller their health if wrong.

    Every arm here is triggered by the caller ASKING for the forbidden thing,
    and answers only that turn. There is deliberately no arm that fires on an
    intent alone: the emergency branch used to, returning a canned line on
    every single emergency turn regardless of what the caller said, which made
    the model unreachable for the whole call and repeated one sentence at a
    frightened caller until they hung up. Driving an emergency is the model's
    job under EMERGENCY OVERRIDE; refusing a medicine is not.

    Repeating the same refusal to a repeated ask is correct and intended -
    CLINICAL SAFETY says to refuse again in the same words.
    """
    if session.intent == EMERGENCY_INTENT:
        # A response to the address question is structured state, not a
        # paraphrasing task. Preserve it verbatim: a small model can turn
        # "number 3, Gandhi Nagar" into a fluent but false new address.
        if _awaiting_emergency_address(session):
            session.ledger["emergency_address"] = caller_text
            session.flow_state["emergency_stage"] = "onset"
            spoken_address = caller_text.rstrip(" .,!?:;")
            return (
                f"{spoken_address}, சரி. Ambulance அனுப்பிட்டேன். "
                "Phone-ஐ வெக்காதீங்க, நான் line-லயே இருக்கேன். இது எப்போ ஆரம்பிச்சது?"
            )

        # Safety refusals outrank the ambiguity question on later emergency
        # turns (for example, "aspirin கொடுக்கலாமா?").
        if _MEDICINE_RE.search(caller_text) and _PERMISSION_RE.search(caller_text):
            return _EMERGENCY_MEDICINE_REFUSAL
        if _mentions_a_clinical_subject(session, caller_text) and _DIAGNOSIS_REQUEST_RE.search(caller_text):
            return _DIAGNOSIS_REFUSAL

        stage = session.flow_state.get("emergency_stage")
        # Callers answer emergency questions in their own order. If they give
        # a later observation early, use it immediately instead of forcing the
        # scripted stage they happened to be on.
        if stage in {"onset", "response"} and _EMERGENCY_BREATHING_STATUS_RE.search(caller_text):
            session.flow_state["breathing"] = caller_text
            session.flow_state["emergency_stage"] = "logistics"
            return "கதவு, gate திறந்து வையுங்க. மருந்துகள் எல்லாம் ஒரு bag-ல வையுங்க; நிலைமை மாறினா உடனே சொல்லுங்க."
        if stage in {"onset", "response"} and _EMERGENCY_RESPONSE_STATUS_RE.search(caller_text):
            session.flow_state["response"] = caller_text
            session.flow_state["emergency_stage"] = "breathing"
            return "சரி, நான் line-ல இருக்கேன். Patient மூச்சு சீரா இருக்கா?"
        if stage == "onset":
            session.flow_state["onset"] = caller_text
            session.flow_state["emergency_stage"] = "response"
            return "Patient-ஐ தனியா விடாதீங்க; நடக்க விடாதீங்க. கண் திறந்து response பண்றாங்களா?"
        if stage == "response":
            session.flow_state["response"] = caller_text
            session.flow_state["emergency_stage"] = "breathing"
            return "சரி, நான் line-ல இருக்கேன். Patient மூச்சு சீரா இருக்கா?"
        if stage == "breathing":
            session.flow_state["breathing"] = caller_text
            session.flow_state["emergency_stage"] = "logistics"
            return "கதவு, gate திறந்து வையுங்க. Medicines எல்லாம் ஒரு bag-ல வையுங்க; நிலைமை மாறினா உடனே சொல்லுங்க."
        if stage == "logistics":
            return "நான் line-ல இருக்கேன். Patient-ஐ தனியா விடாதீங்க; ஏதாவது மாற்றம் இருந்தா உடனே சொல்லுங்க."

        # Distress fragments enter the safety lane but do not contain enough
        # evidence to invent a disease or dispatch against a guessed symptom.
        if not is_explicit_emergency(caller_text) and not _has_started_emergency_dispatch(session):
            return _DISTRESS_CLARIFICATION

        # Fixed for any first EXPLICIT emergency, including one that follows a
        # distress clarification. Do not tie this to caller-turn number one.
        if not _has_started_emergency_dispatch(session) and not _ADDRESS_HINT_RE.search(caller_text):
            return _EMERGENCY_OPENING

    corrected_date = _appointment_date_correction(session, caller_text)
    if corrected_date is not None:
        return corrected_date

    clinical = _mentions_a_clinical_subject(session, caller_text)
    if clinical:
        prior_value_request = any(
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and _LAB_VALUE_REQUEST_RE.search(message["content"])
            for message in session.messages[:-1]
        )
        if _LAB_VALUE_REQUEST_RE.search(caller_text) or (
            prior_value_request and re.search(r"(?:மட்டும்|சொல்லுங்க|please)", caller_text, re.IGNORECASE)
        ):
            return _LAB_VALUE_REFUSAL
    if (
        session.intent == EMERGENCY_INTENT
        and _MEDICINE_RE.search(caller_text)
        and _PERMISSION_RE.search(caller_text)
    ):
        return _EMERGENCY_MEDICINE_REFUSAL
    if clinical and _DIAGNOSIS_REQUEST_RE.search(caller_text):
        return _DIAGNOSIS_REFUSAL
    return None


# Desk words the model sometimes writes in Tamil script, against the LANGUAGE
# rule's explicit list. Small and hand-picked, NOT the ASR lexicon: that table
# is built for the other direction (what the ASR does to a caller's English)
# and running its several hundred generated entries over generated Tamil would
# eventually rewrite a real Tamil word, which is a far worse failure than one
# transliterated heading. Only forms with no Tamil homograph are listed.
_SPOKEN_LATIN_WORDS: dict[str, str] = {
    "அப்பாயின்ட்மென்ட்": "appointment",
    "அப்பாயின்ட்மெண்ட்": "appointment",
    "அப்பாயின்மென்ட்": "appointment",
    "அப்பாயிண்ட்மெண்ட்": "appointment",
    "ரீஷெட்யூல்": "reschedule",
    "கேன்சல்": "cancel",
    "கான்சல்": "cancel",
    "ரிப்போர்ட்": "report",
    "மொபைல்": "mobile",
    # NOT "ஹாஸ்பிட்டல்". It is half the hospital's own name - "அருவி
    # ஹாஸ்பிட்டல்" is how the brand is said, and rewriting it produced the
    # greeting "வணக்கம், அருவி hospital."
    "எமர்ஜென்சி": "emergency",
    "டிபார்ட்மென்ட்": "department",
    "டிபார்ட்மெண்ட்": "department",
}
_SPOKEN_LATIN_RE = re.compile("|".join(map(re.escape, _SPOKEN_LATIN_WORDS)))

# How the caller reveals their OWN gender, which is the only kind that decides
# the address form. LANGUAGE is explicit that the caller saying "Sir" or
# "மேடம்" is addressing the agent and reveals nothing, and that an unestablished
# gender means omitting the address rather than guessing.
#
# The model guesses anyway, and always guesses male: it said "Kavitha Sir" to a
# woman who had just given her name. Enforced here for the same reason the
# Sir/சார் spelling is - a prose rule the model does not keep is kept at the
# speech boundary instead.
#
# Only self-identifying relationship words count. "என் அம்மாவுக்கு" says who
# the PATIENT is, not who is calling, so the possessive forms are excluded and
# only the "நான் ... தான்" shape is matched.
_CALLER_IS_MALE_RE = re.compile(
    r"நான்[^.?!]{0,20}(?:கணவர்|அப்பா|மகன்|தம்பி|அண்ணன்|தாத்தா|மாமா)|"
    r"\b(?:I am|I'm)\b[^.?!]{0,20}\b(?:his|her|the)?\s*(?:husband|father|son|brother)\b",
    re.IGNORECASE,
)
_CALLER_IS_FEMALE_RE = re.compile(
    r"நான்[^.?!]{0,20}(?:மனைவி|அம்மா|மகள்|தங்கச்சி|அக்கா|பாட்டி|அத்தை)|"
    r"\b(?:I am|I'm)\b[^.?!]{0,20}\b(?:his|her|the)?\s*(?:wife|mother|daughter|sister)\b",
    re.IGNORECASE,
)

# Written as "Sir" by the time this runs - _normalize_spoken_register folds
# சார் into it first. Eats the space in front, so dropping the address leaves
# "சரி." and not "சரி ."
#
# No trailing \b on மேடம். A Tamil word ends in a virama (U+0BCD), which is a
# combining mark and not alnum, so \b never holds there and the pattern matched
# nothing at all - the same trap that ate the department regex earlier. The
# lookahead does the job the boundary was meant to.
_ADDRESS_RE = re.compile(r"(?:\s+|^)(?:\bSir\b|மேடம்)(?=[\s,.!?]|$)", re.IGNORECASE)


def caller_gender(session: CallSession) -> str | None:
    """"male", "female", or None when the caller has not revealed it.

    Metadata wins: a telephony leg or CRM that knows is better evidence than
    anything said on the call.
    """
    stated = str(session.metadata.get("caller_gender") or "").strip().lower()
    if stated in {"male", "female"}:
        return stated
    for message in session.messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        if _CALLER_IS_FEMALE_RE.search(message["content"]):
            return "female"
        if _CALLER_IS_MALE_RE.search(message["content"]):
            return "male"
    return None


def _normalize_spoken_register(text: str, gender: str | None = None) -> str:
    """Hold the script and address rules at the final speech boundary.

    The small local model can ignore a prose-only spelling instruction.  This
    deliberately changes only the agent's generated speech: caller history,
    intent matching, names, and every other Tamil word remain byte-for-byte
    untouched.

    Three rules, all from LANGUAGE: the male address is Latin "Sir" so the
    Tamil TTS says the English word rather than "saar"; the desk's own
    vocabulary is Latin too (which is what stops "நாளைக்கு அப்பாயின்ட்மென்ட்
    இருக்கு" going out when the caller said "appointment" in the first place);
    and the address form matches the caller's gender, or is left off entirely
    when they have not revealed it.

    `gender` of None means unknown, and unknown means NO address word. That is
    the rule as written - "If their gender is not established from their own
    identity, omit the address word instead of guessing" - and the model
    guesses regardless, always male. Omitting costs a little warmth on every
    call; guessing calls a woman Sir on hers.
    """
    def latin(match: "re.Match[str]") -> str:
        word = _SPOKEN_LATIN_WORDS[match.group(0)]
        # Capitalise where the Tamil script carried no case: a clause opening
        # on a lowercase word reads as a fragment in the transcript.
        before = text[: match.start()].rstrip()
        return word.capitalize() if not before or before[-1] in ".!?—" else word

    text = _SPOKEN_LATIN_RE.sub(latin, text)
    text = re.sub(r"\bsir\b", "Sir", text.replace("சார்", "Sir"), flags=re.IGNORECASE)
    if gender == "male":
        return _ADDRESS_RE.sub(" Sir", text).lstrip()
    if gender == "female":
        return _ADDRESS_RE.sub(" மேடம்", text).lstrip()
    # Leading comma too: "Sir, சொல்லுங்க." must not go out as ", சொல்லுங்க."
    return re.sub(r"^[\s,]+", "", _ADDRESS_RE.sub("", text))


def _looks_like_regreeting(text: str) -> bool:
    """Detect greeting variants after the server has already greeted once."""
    compact = " ".join(text.casefold().split())
    return compact.startswith("வணக்கம்") or (
        "அருவி" in compact and "hospital" in compact and "பேசுறேன்" in compact
    )

# NONE OF THE FIXED LINES IN THIS MODULE CARRY AN ADDRESS FORM, and that is
# deliberate rather than an oversight. _normalize_spoken_register drops "Sir"
# from the model's speech whenever the caller has not revealed their gender,
# which on most calls is every turn; a canned line that said "Sir" anyway would
# be the one sentence in the call that guessed. Leaving it out also keeps these
# strings byte-identical to what main.py warms into the TTS cache, which a
# per-call rewrite would defeat. நீங்க carries the respect on its own.
#
# Said for anything outside prompt_builder.SUPPORTED_INTENTS - the fifteen
# other hospital flows as much as the weather.
#
# Without it, detect_intent falls through to DEFAULT_FLOW and the model
# improvises an answer from the info.general playbook, which is how a request
# the desk does not handle gets a confident, invented reply instead of a
# straight one. Naming what the desk DOES handle is both the honest answer and
# the fastest one - it costs no LLM call at all, so an out-of-scope turn is the
# quickest turn in the call rather than the slowest.
#
# It lists the five in the caller's own terms, not by intent name, because the
# caller has to be able to pick one out of it while listening.
#
# It states the LIMIT rather than rejecting "that", and the difference is the
# vague opener. "எனக்கு ஒரு help வேணும்" matches no trigger either, so it lands
# here too, and answering a request for help with "I cannot help with that" is
# both rude and wrong. Phrased as the limit, the same sentence serves the
# caller who wanted the weather and the caller who has not said yet.
#
# Says outright that this is the hospital desk and that the question is not
# one it answers, then names what it does answer and hands the turn back. The
# formal register ("மன்னிக்கவும்", not the colloquial "மன்னிச்சுடுங்க" the
# other canned lines use) is deliberate: this is the one line that declines
# something, and a decline is said formally.
#
# It ends on the list rather than the decline so it still serves the VAGUE
# opener ("எனக்கு ஒரு help வேணும்") as well as the genuinely off-topic one -
# for that caller the list is the whole answer.
#
# This is the one place runtime_core.txt's "NEVER open with what you cannot
# do" is deliberately not followed. That rule exists to stop the agent
# refusing work the desk really does take ("appointment book பண்ண முடியாது"),
# and none of that applies to a question the desk genuinely does not answer.
_OUT_OF_SCOPE = (
    "மன்னிக்கவும். நான் help பண்ண முடியுறது இந்த ஐஞ்சு விஷயம் மட்டும் தான் — "
    "Appointment book பண்றது, date மாத்தறது, cancel பண்றது, "
    "hospital timing information, emergency. "
    "இதுல எதுலயாவது help வேணுமா?"
)

# Below this the turn is a fragment, not a request: a bare number, "ஆமாம்",
# a name. Those match no trigger either, and answering them with the menu
# would be worse than letting the model handle them.
_MIN_OUT_OF_SCOPE_WORDS = 3

# What the agent says instead of repeating itself, in order. The first two ask
# again in fresh words; the third stops asking and hands the call to the desk,
# so a caller on a line that is not working gets an exit instead of the same
# sentence until they hang up.
_STUCK_REPLIES = (
    "மன்னிச்சுடுங்க, நான் பழைய detail-ஐ repeat பண்ணிட்டேன். புதிய detail-ஐ இன்னொரு தரம் சொல்லுங்களா?",
    "நீங்க சொன்ன correction-ஐ update பண்ணணும். கொஞ்சம் மெதுவா மறுபடியும் சொல்லுங்க?",
    "இன்னும் சரியா கேட்கல. Desk-ல இருந்து உங்களுக்கு call பண்ண சொல்றேன். நன்றி.",
)

# The same escalation is WRONG on an emergency, and the third line above is
# actively unsafe there: "Desk-ல இருந்து call பண்ண சொல்றேன். நன்றி சார்." ends
# the call, and flow 18 says in as many words that the agent does not end this
# call and never hangs up - "Stay on the line until the ambulance arrives or
# the caller disconnects."
#
# So an emergency gets its own ladder, and it does not escalate towards a
# handoff because there is nowhere better to hand a caller who cannot breathe.
# All three keep the line open and continue the simulated dispatch flow.
_EMERGENCY_STUCK_REPLIES = (
    "Ambulance அனுப்புறேன். நீங்க இருக்கிற address-ஐ மட்டும் சொல்லுங்க.",
    "Phone-ஐ வெக்காதீங்க — உங்க முழு address சொல்லுங்க.",
    "நான் line-லயே இருக்கேன். Patient இப்போ பேசுறாரா?",
)

# A two-word opening ("சரி சார்.", "நானே சார்,") is an acknowledgement, and two
# turns running may legitimately start with one. A longer opening repeated
# verbatim is the model stuck, not the model agreeing.
_MIN_REPEAT_WORDS = 3

# "முடியாது" / "மாட்டேன்" - cannot, will not. The negative-ability forms the
# prompt's own refusal line uses ("Phone-ல அதை நான் சொல்ல முடியாது") and the
# ones the model paraphrases it into. Deliberately broad: this only ever
# PERMITS a repeat, so a false match costs one repeated sentence, while a miss
# costs a refusal the caller never hears.
_REFUSAL_RE = re.compile(r"முடியாது|மாட்டேன்")


RECENT_OPENINGS_KEPT = 3


# How many times one word may repeat back-to-back before the turn is a stuck
# decoder rather than speech.
#
# Observed live, twice in one call, both times mid-emergency:
#
#     "உங்களுக்கு அடிப்படை அடிப்படை அடிப்படை அடிப்படை ..."   (x12)
#     "உங்க முழு முழு முழு முழு முழு ..."                     (x22)
#
# Neither existing guard could see it. _is_repeat_opening compares whole
# clauses ACROSS turns and this is one turn; the clause chunker cuts the
# opening at 32 characters and then waits for punctuation that a looping
# decoder never emits, so the rest arrived in one enormous clause out of
# flush(). Both halves were spoken to the caller.
#
# Three, because the chunker's 32-character opening cut lands after the third
# repeat - "உங்களுக்கு அடிப்படை அடிப்படை அடிப்படை" is exactly what it released -
# so a bar of three is what catches the loop in the FIRST clause, before any of
# it is spoken. Ordinary speech does not say the same word three times running,
# and this agent's own register never does.
_MAX_WORD_REPEATS = 3

# Words that are not evidence of anything when they appear in both the caller's
# line and the agent's: an acknowledgement is supposed to echo.
_ECHO_STOPWORDS = frozenset(
    {"சார்", "sir", "மேடம்", "சரி", "ஆமாம்", "ஓகே", "ok", "நன்றி", "ஒரு"}
)

# A clause that reads back what the caller said IS the wanted behaviour - the
# LEDGER section requires it ("Confirm by reading back for a yes"), and the
# EMERGENCY playbook requires it of the address specifically. Those clauses
# always carry a confirmation marker or a number, which is what separates them
# from a parrot.
_READBACK_RE = re.compile(r"சரியா|சரிதான|குறிச்|note\s*பண்ண|right\?|correct\?|\d")

# How much of a clause has to be the caller's own words before it is a parrot
# rather than an answer. Both bars must be cleared: three shared content words
# AND two-thirds of the clause, so a real answer - which necessarily adds words
# the caller did not say - is never touched.
_MIN_ECHOED_WORDS = 3
_ECHO_SHARE = 0.66

# Asking something the caller did not ask is not reflecting them, however much
# of their wording it reuses. Observed: the caller said "அடுத்த திங்கள் காலைல"
# and the agent asked "அடுத்த திங்கள் காலைல எப்போது சரி?" - three of five
# content words shared, so the guard called it a parrot and dropped it, leaving
# the caller a two-word turn ("Kavitha Sir.") and no question at all.
#
# The two parrots this guard exists for both survive the exemption, which is
# the test of it: "விசிடிங் ஹவுர்ஸ் என்ன சார்?" reuses the caller's OWN "என்ன",
# so it introduces no new interrogative, and "உயிருக்கு போராடிட்டு இருக்கேன்
# சார்." contains none at all.
_INTERROGATIVE_RE = re.compile(
    r"என்ன|எப்போ|எங்க|எத்தனை|எந்த|யாரு|ஏன்|எப்படி|எவ்வளவு|"
    r"\bwhat\b|\bwhen\b|\bwhere\b|\bwhich\b|\bwho\b|\bwhy\b|\bhow\b",
    re.IGNORECASE,
)

# Below this a turn has not said anything: an address form and a name, with no
# question and no fact. Guards drop clauses mid-turn, and when what survives is
# only a stub the caller needs a real turn rather than the leftovers.
_MIN_TURN_WORDS = 4


def _carries_a_turn(spoken: list[str]) -> bool:
    """Whether what survived the guards is a turn, or only the leftovers of one.

    Every guard above drops clauses out of a turn in flight, and each one was
    written asking "is the whole turn gone?" - `not spoken`. That is the wrong
    question, because a turn does not have to be emptied to be ruined. Observed
    live: the echo guard dropped the only question out of "Kavitha Sir. அடுத்த
    திங்கள் காலைல எப்போது சரி?" and the caller heard "Kavitha Sir." - not
    silence, so no recovery ran, and the call simply stalled for a turn.

    A turn carries something if it asks a question or says enough words to be
    an answer. The address form and a name are neither.
    """
    if not spoken:
        return False
    text = " ".join(spoken)
    return "?" in text or len(text.split()) >= _MIN_TURN_WORDS


def _is_degenerate(clause: str) -> bool:
    """Whether this clause is a decoder loop rather than something to say."""
    words = clause.split()
    run = 1
    for previous, word in zip(words, words[1:]):
        run = run + 1 if word == previous else 1
        if run >= _MAX_WORD_REPEATS:
            return True
    return False


def _echoes_caller(clause: str, caller_text: str) -> bool:
    """Whether this clause is the caller's own sentence handed back to them.

    _LANGUAGE_REMINDER has told the model "never repeat the caller's own
    sentence back at them" for as long as it has existed and the model does it
    anyway - same story as the one-question rule and the repeat breaker, and
    the same answer: what the model will not do on instruction, the server does
    for it.

    Observed live, both wasting the whole turn:

        CALLER  Visiting hours என்ன?
        AGENT   விசிடிங் ஹவுர்ஸ் என்ன சார்?
        CALLER  நான் உயிருக்கு போராடிட்டு இருக்கேன்
        AGENT   உயிருக்கு போராடிட்டு இருக்கேன் சார்.

    The second one is why this is not merely a style fix. Reflecting "I am
    fighting for my life" back at the person who said it is the worst possible
    turn to spend on an emergency call.

    Deliberately NOT a ban on repeating the caller's words - that would break
    the read-back the prompt requires. See _READBACK_RE and the two bars above.
    """
    if _READBACK_RE.search(clause):
        return False
    if _INTERROGATIVE_RE.search(clause) and not _INTERROGATIVE_RE.search(caller_text):
        # Asking something the caller did not ask. See _INTERROGATIVE_RE.
        return False
    caller_words = {w.lower() for w in _WORD_RE.findall(caller_text)}
    words = [w.lower() for w in _WORD_RE.findall(clause)]
    content = [w for w in words if w not in _ECHO_STOPWORDS]
    if not content:
        return False
    shared = sum(1 for w in content if w in caller_words)
    return shared >= _MIN_ECHOED_WORDS and shared / len(content) >= _ECHO_SHARE


# Reading a fact back for a yes, as THE LEDGER requires. Narrower than
# _READBACK_RE, which the echo guard uses: a bare "சரி" is the commonest
# acknowledgement in the language and exempting every clause containing one
# would switch the repeat breaker off altogether. Only the confirming idioms
# count - a trailing ", சரி." after the fact, "சரியா?", "குறிச்சுக்கிட்டேன்".
_CONFIRMATION_RE = re.compile(
    r"[,—-]\s*சரி\s*[.!]?\s*$|சரியா|சரிதான|குறிச்|note\s*பண்ண|"
    r"\bnoted\b|right\?\s*$|correct\?\s*$",
    re.IGNORECASE,
)


def _is_a_confirmation(clause: str) -> bool:
    """Whether this clause reads a fact back rather than asking for one."""
    return not clause.rstrip().endswith("?") and bool(_CONFIRMATION_RE.search(clause))


def _is_repeat_opening(clause: str, recent: list[str]) -> bool:
    """Whether this turn is opening with a clause a recent turn already used.

    Prose cannot hold this. runtime_core.txt has said "Never say a turn you
    have already said" since before the tool removal, and on the real call in
    call_events.db (97dd5ac7) the agent said "98407 21534 என்ன சார்?" on five
    consecutive turns anyway, whatever the caller said in between. Two further
    attempts to teach it - a demonstrated bad-line exchange in the core prompt,
    then a shorter version using bracketed placeholders - both left the repeat
    in place AND pushed the median first clause from 2.7s to 6.8s, because the
    extra section lengthened every OTHER turn too. Measurements in
    LLM_TEST_RESULTS.txt.

    So it is enforced here, for the same reason and in the same shape as
    speakable()'s one-question rule a few lines up: what the model will not do
    on instruction, the server does for it.

    Checked on the FIRST clause, before it is spoken, so none of the repeat
    reaches the caller and the rest of the generation can be abandoned - which
    makes this a latency win on exactly the turns that were slowest.
    """
    if _is_a_confirmation(clause):
        # A CONFIRMATION IS NOT A LOOP, and this is the second thing the
        # breaker was suppressing that the prompt requires. THE LEDGER says
        # "Confirm by reading back for a yes - never re-collect", so a closing
        # turn legitimately reads back a fact it already read back mid-call:
        #
        #     turn 3  Kavitha Sir. அடுத்த திங்கள் காலைல், சரி. Mobile number சொல்லுங்க?
        #     turn 4  98407 21534, குறிச்சுக்கிட்டேன் Sir. அடுத்த திங்கள் காலைல், சரி. ...
        #
        # The breaker killed turn 4 on its second clause and appended
        # "மன்னிச்சுடுங்க Sir, clear-ஆ கேட்கல" - told a caller who had just
        # been heard perfectly that they had not been heard. That is where the
        # degenerate two-word turns in the multiturn transcript came from.
        #
        # A QUESTION is still never exempt, which is what keeps the original
        # bug fixed: the five-turns-running clause this breaker exists for was
        # "98407 21534 என்ன சார்?" - a stuck model loops by ASKING, because an
        # unanswerable question is what the caller cannot escape. Repeating a
        # statement is at worst redundant.
        return False
    if _REFUSAL_RE.search(clause):
        # A REFUSAL IS THE ONE TURN THAT IS SUPPOSED TO REPEAT, and suppressing
        # it would be a clinical-safety regression, not a style one.
        # runtime_core.txt's CLINICAL SAFETY section is explicit: "If the caller
        # asks again for something you have already refused - a second time, a
        # third time, begging - refuse again in the same words", because to a
        # frightened caller a changed subject reads as being ignored and they
        # hang up still not knowing they were told no.
        #
        # This did not fire in safety_eval - the model was failing to repeat
        # the refusal at all, which is a separate pre-existing violation - so
        # the conflict was latent rather than observed. It is exempted anyway:
        # the breaker exists to stop a stuck model, and a refusal held under
        # pressure is the opposite of stuck.
        return False
    return len(clause.split()) >= _MIN_REPEAT_WORDS and clause in recent


def _is_first_caller_turn(session: CallSession) -> bool:
    """Whether the caller has said exactly one thing so far this call.

    Call AFTER _append_caller_turn, which merges a run of caller turns the
    agent never got a word in between into one message - so this counts what
    the caller has had ANSWERED, which is the question being asked.
    """
    return sum(1 for message in session.messages if message.get("role") == "user") == 1


def _append_caller_turn(session: CallSession, text: str) -> str:
    """Add the caller's line, merging it into the previous one if the agent
    never got a word out in between.

    A barge-in that lands before the FIRST clause is spoken leaves
    record_interrupted_turn() with an empty string and therefore nothing to
    append, so the next caller turn lands directly behind the previous one.
    Measured on the real call in call_events.db (97dd5ac7), a noisy stretch put
    TWELVE consecutive user messages into a 21-message history, and against
    that history the model stopped answering at all: it reproduced its own
    previous turn verbatim on five turns running - "98407 21534 என்ன சார்?" -
    whatever the caller said next. Replaying the identical message list against
    the real model reproduces that reply character for character, and merging
    the runs is what stops it.

    Merging, not dropping. The caller really did say both things before anyone
    answered, and the first half is where the CONTENT usually is - the real
    call lost "ஆஸ்டோ department க்கு வேணும்" behind a barge-in and only the
    noise that followed it would have survived a drop.
    """
    previous = session.messages[-1] if session.messages else None
    if previous is not None and previous.get("role") == "user":
        previous["content"] = f"{previous['content']} {text}"
        return str(previous["content"])
    session.messages.append({"role": "user", "content": text})
    return text


def _history_budget_chars(session: CallSession, settings: LlmSettings, facts: str) -> int:
    """How many characters of history still fit in this turn's context window.

    Everything else on the wire is fixed by the time this is asked: the system
    prompt (whose width depends on which flow is active - emergency.escalate's
    is ~430 tokens wider than the narrowest), the facts block, the language
    reminder, and the room the reply itself needs. History gets what is left.
    """
    fixed_tokens = (
        (len(session.messages[0]["content"]) + len(facts) + len(_LANGUAGE_REMINDER))
        * _PROMPT_TOKENS_PER_CHAR
        + settings.max_tokens
        + _TOKENS_PER_MESSAGE * (MAX_HISTORY_MESSAGES + 3)
    )
    return max(0, int((settings.num_ctx - fixed_tokens) / _HISTORY_TOKENS_PER_CHAR))


def _trim_history(session: CallSession, settings: LlmSettings, facts: str = "") -> None:
    """Drop the oldest exchanges, never the system prompt.

    Two bounds. The message count is the cheap one and catches the ordinary
    long call; the character budget is the one that actually holds the
    invariant, because it is the only one that can see a call whose turns are
    long rather than merely numerous. See the measurement above
    MAX_HISTORY_MESSAGES.

    Trims from the FRONT, which is the same end Ollama would truncate - the
    difference being that this end keeps the system prompt, and Ollama's does
    not. On the 41 real calls in call_events.db the character budget never
    binds (the longest ran 22 caller turns of median 20 chars); it exists for
    the pathological call, which is the one that produced the original bug.
    """
    overflow = len(session.messages) - 1 - MAX_HISTORY_MESSAGES
    if overflow > 0:
        logger.info("call %s: trimming %d oldest messages", session.connection_id, overflow)
        del session.messages[1 : 1 + overflow]

    budget_chars = _history_budget_chars(session, settings, facts)
    # Never below one exchange: the turn just appended is what the next reply
    # has to answer, so there is nothing useful left to give back after that.
    dropped = 0
    while len(session.messages) > 3 and _history_chars(session) > budget_chars:
        del session.messages[1]
        dropped += 1
    if dropped:
        logger.info(
            "call %s: trimming %d more messages to stay inside num_ctx (%d chars of budget)",
            session.connection_id,
            dropped,
            budget_chars,
        )


def _history_chars(session: CallSession) -> int:
    return sum(len(message.get("content") or "") for message in session.messages[1:])


# Recency beats distance: a small model reliably drifts into pure English by
# the third or fourth turn even with the language rules in the system message,
# because those sit thousands of tokens back while the recent turns are the
# strongest signal. This rides immediately before generation, costs ~40 tokens,
# and is not stored in history - so it never accumulates across a long call.
#
# It used to have to open by naming "call a tool", because a speech-only
# instruction here read to the model as "produce speech now" and suppressed
# tool calling entirely. There are no tools any more, so that constraint is
# gone and the whole message is speaking rules - ordered by how often each is
# actually broken, measured with backend/scripts/register_eval.py on unseen
# scenarios, most-violated first rather than most important-sounding first.
_LANGUAGE_REMINDER = (
    # DO NOT grow this block. Measured: adding four sentences here - telling
    # the model to use the caller's last answer, not to re-ask, and not to
    # invent a clock time - made every one of those things WORSE, because this
    # is the last thing in the prompt and lengthening it dilutes the position
    # rather than using it. The opening turn degenerated from
    # "சரி Sir. உங்க பேரு சொல்லுங்க?" to a bare "Sir?", and a re-ask appeared
    # where there had been none. The rules it duplicated (THE LEDGER,
    # GROUNDING) were already in the core prompt and already working; saying
    # them twice cost the turn its shape and bought nothing.
    "[Reply now, out loud. ONE question per turn - never two - and put it "
    "last. Under 40 words. Never repeat the caller's own sentence back at "
    "them; acknowledge in two or three words and move on. "
    "Reply in spoken Chennai Tamil (Tamil script) code-mixed with English "
    "hospital words in Latin script - never pure English. "
    "If the caller speaks English, mirror them but keep Sir/மேடம். "
    "Always write the male address as Latin-script Sir, never Tamil சார். "
    "Take the request down: ask for the next detail you still need. Never "
    "refuse the request itself and never open with what you cannot do. "
    "Never state an MRN, appointment ID, bill amount, slot time or report "
    "result, and never claim you already booked, cancelled or checked "
    "anything - the desk does that after the call.]"
)


# Counted in WORDS, not letters. Letters over-count English badly: Tamil packs
# a syllable into one glyph where English spells it out, so
# "Cardiology-ல ஒரு appointment book பண்ணணும்" - an ordinary code-mixed
# TAMIL line - is 64% Latin by character and would wrongly flip the agent into
# English. By word it is 2 English of 5, which is what it actually is.
#
# A word counts as Tamil if it contains ANY Tamil character, so "Cardiology-ல"
# is Tamil: the case suffix is what makes the sentence Tamil. Bare digits are
# ignored - a phone number is not evidence of either language.
#
# The bar is deliberately high. Tamil is the safe default (it is the register
# every exemplar demonstrates), so switching needs most of the turn to be
# English, not merely some of it.
_WORD_RE = re.compile(r"[\w஀-௿]+")
_TAMIL_LETTER_RE = re.compile(r"[஀-௿]")
_ENGLISH_SHARE_TO_MIRROR = 0.7


def caller_is_speaking_english(text: str) -> bool:
    """Whether this caller turn is English rather than code-mixed Tamil.

    NOT currently wired into the prompt, deliberately. Switching the register
    on this was built and measured twice and made things WORSE both times:
    prose alone ("reply mainly in English") only half-moved the register and
    introduced parroting, and adding an English worked example alongside the
    twenty Tamil ones produced ungrammatical output mixing both
    ("எந்த நாள் உங்களுக்கு சொல்லுங்க?"). Answering a
    English-speaking caller in coherent Tamil beats answering them in broken
    half-English, so the agent stays in Tamil.

    Kept because the measurement is the correct one and any future attempt
    needs it: count WORDS, not letters (see the comment above).
    """
    words = [w for w in _WORD_RE.findall(text) if not w.isdigit()]
    if not words:
        return False
    english = sum(1 for w in words if not _TAMIL_LETTER_RE.search(w))
    return english / len(words) >= _ENGLISH_SHARE_TO_MIRROR


def _with_language_reminder(messages: list[dict], facts: str = "") -> list[dict]:
    """Append this turn's volatile context, then the reminder.

    Order is load-bearing in two directions. The facts go AFTER the history so
    that changing them re-evaluates a hundred tokens instead of the three
    thousand sitting in front of them (see _system_prompt_for). The reminder
    stays LAST, because that is the message the model reads immediately before
    it decides whether to call a tool or speak - see the comment on
    _LANGUAGE_REMINDER, and the regression test that guards it.
    """
    tail: list[dict] = []
    if facts:
        tail.append({"role": "system", "content": facts})
    tail.append({"role": "system", "content": _LANGUAGE_REMINDER})
    return [*messages, *tail]
