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
    EMERGENCY_INTENT,
    PromptBuilder,
    detect_intent,
    looks_like_hospital_business,
)
from .settings import ConversationSettings, LlmSettings

logger = logging.getLogger("aica.conversation")

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

        detected = detect_intent(text)
        if detected is not None and detected != session.intent:
            logger.info("flow detected for %s: %s", connection_id, detected)
            session.intent = detected
        session.messages[0]["content"] = self._system_prompt_for(session)

        _append_caller_turn(session, text)

        # Nothing this desk does. Say so, instead of asking the model to
        # improvise an answer out of the info.general playbook.
        #
        # Three conditions, and every one of them is a brake rather than a
        # trigger, because deflecting a real caller is far worse than letting
        # an off-topic one through to the model:
        #
        #   no flow, and no hospital words either - detect_intent finding
        #     nothing is not evidence of anything on its own, the trigger table
        #     has gaps between its twenty flows (see looks_like_hospital_business)
        #   the caller's FIRST turn - this is the "why are you ringing" moment.
        #     Later turns that match no trigger are the caller ANSWERING a
        #     question - a day, a name, an address - and belong to the model.
        #     Measured: without this the line displaced "Anna Nagar 2nd street"
        #     given as an address mid-call.
        #   a real sentence, not a fragment - a bare number or "ஆமாம்" matches
        #     no trigger either
        if (
            session.intent is None
            and _is_first_caller_turn(session)
            and not looks_like_hospital_business(text)
            and len(text.split()) >= _MIN_OUT_OF_SCOPE_WORDS
        ):
            logger.info("off-topic opener from %s: %r - naming what the desk does", connection_id, text)
            session.messages.append({"role": "assistant", "content": _OUT_OF_SCOPE})
            _trim_history(session, llm.settings)
            for clause in split_reply_into_clauses(_OUT_OF_SCOPE):
                yield AgentClause(clause)
            yield AgentTurn(text=_OUT_OF_SCOPE)
            return

        chunker = ClauseChunker()
        spoken: list[str] = []
        asked_question = False

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
            nonlocal asked_question, stuck, regreeted, fabricated, echoed, anchor
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
                if not speakable(clause):
                    continue
                spoken.append(clause)
                yield AgentClause(clause)
            if stuck or fabricated:
                # Nothing more is worth generating: a repeat is a stuck model,
                # and the rest of a turn built around an invented identifier is
                # built on the same mistake. Drop it on the floor.
                await stream.aclose()
                break

        ended_early = stuck or fabricated
        if reply is None and not ended_early:
            raise RuntimeError("LLM stream ended without a ReplyComplete event")

        # The chunker never closes a clause on buffer-end (see
        # clause_chunker.py), so the reply's last clause only exists once the
        # stream is done and we ask for it.
        if not ended_early:
            remainder = chunker.flush()
            if remainder and speakable(remainder):
                spoken.append(remainder)
                yield AgentClause(remainder)

        if stuck:
            ladder = (
                _EMERGENCY_STUCK_REPLIES
                if session.intent == EMERGENCY_INTENT
                else _STUCK_REPLIES
            )
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
        elif not spoken and regreeted:
            # The whole turn was the opening line over again, so there is
            # nothing left to say - but silence on a phone call is worse than a
            # wasted turn. This is what a receptionist says when the caller has
            # said hello back and not yet got to why they rang.
            spoken.append(_GO_AHEAD)
            yield AgentClause(_GO_AHEAD)
        elif not spoken and echoed:
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
_GO_AHEAD = "சொல்லுங்க சார், என்ன help வேணும்?"

# Said when the whole turn was withheld for stating an invented identifier.
# Mirrors runtime_core.txt's GROUNDING wording ("அது என்கிட்ட இப்போ இல்ல சார்")
# rather than inventing a new register for it.
_CANNOT_RECALL = "மன்னிச்சுடுங்க சார், அது என்கிட்ட இல்ல. ஒரு தடவை சொல்லுங்களா?"

# Said when the whole turn was the caller's own sentence handed back. There is
# no way to recover the answer the model failed to give, so this takes the
# request down and hands it on, which is what YOUR JOB in runtime_core.txt says
# to do with anything the agent cannot answer itself.
_ECHO_RECOVERY = "மன்னிச்சுடுங்க சார். அதை desk-ல இருந்து confirm பண்ணி call பண்ணுவாங்க. வேற ஏதாவது help வேணுமா?"

# Said when the caller opens with something no flow covers at all.
#
# This is the branch the twenty-flow trigger table never had: detect_intent
# returning None fell through to DEFAULT_FLOW and the model improvised an
# answer from the info.general playbook, which is how a request the hospital
# desk does not handle gets a confident, invented reply instead of a straight
# one. Naming what the desk DOES handle is both the honest answer and the
# fastest one - it costs no LLM call at all, so an off-topic turn is the
# quickest turn in the call rather than the slowest.
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
    "மன்னிக்கவும் சார். இது அருவி ஹாஸ்பிட்டல் desk number — "
    "அதுக்கு நாங்க பதில் சொல்ல முடியாது. "
    "Appointment, report, bill, insurance, records, ambulance — "
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
    "மன்னிச்சுடுங்க சார், clear-ஆ கேட்கல. இன்னொரு தரம் சொல்லுங்களா?",
    "Line-ல கொஞ்சம் disturbance சார். கொஞ்சம் மெதுவா சொல்லுங்க?",
    "இன்னும் சரியா கேட்கல சார். Desk-ல இருந்து உங்களுக்கு call பண்ண சொல்றேன். நன்றி சார்.",
)

# The same escalation is WRONG on an emergency, and the third line above is
# actively unsafe there: "Desk-ல இருந்து call பண்ண சொல்றேன். நன்றி சார்." ends
# the call, and flow 18 says in as many words that the agent does not end this
# call and never hangs up - "Stay on the line until the ambulance arrives or
# the caller disconnects."
#
# So an emergency gets its own ladder, and it does not escalate towards a
# handoff because there is nowhere better to hand a caller who cannot breathe.
# All three keep the line open, and all three carry the two things that
# actually matter while the model is not producing a usable turn: ring 108 now,
# and give me the address. A caller who hears only these three has still been
# told the thing that saves them.
_EMERGENCY_STUCK_REPLIES = (
    "சார், இப்பவே 108-க்கு call பண்ணுங்க. நீங்க இருக்கிற address-ஐ மட்டும் சொல்லுங்க.",
    "108-க்கு call பண்ணிட்டீங்களா சார்? உங்க address சொல்லுங்க, ER team-கிட்ட சொல்லிடறேன்.",
    "நான் line-லயே இருக்கேன் சார், phone-ஐ வெக்காதீங்க. 108-க்கு call பண்ணுங்க — address சொன்னா ER team ready-யா இருப்பாங்க.",
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
_ECHO_STOPWORDS = frozenset({"சார்", "மேடம்", "சரி", "ஆமாம்", "ஓகே", "ok", "நன்றி", "ஒரு"})

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
    caller_words = {w.lower() for w in _WORD_RE.findall(caller_text)}
    words = [w.lower() for w in _WORD_RE.findall(clause)]
    content = [w for w in words if w not in _ECHO_STOPWORDS]
    if not content:
        return False
    shared = sum(1 for w in content if w in caller_words)
    return shared >= _MIN_ECHOED_WORDS and shared / len(content) >= _ECHO_SHARE


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


def _append_caller_turn(session: CallSession, text: str) -> None:
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
        return
    session.messages.append({"role": "user", "content": text})


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
    "[Reply now, out loud. ONE question per turn - never two - and put it "
    "last. Under 40 words. Never repeat the caller's own sentence back at "
    "them; acknowledge in two or three words and move on. "
    "Reply in spoken Chennai Tamil (Tamil script) code-mixed with English "
    "hospital words in Latin script - never pure English. "
    "If the caller speaks English, mirror them but keep சார்/மேடம். "
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
