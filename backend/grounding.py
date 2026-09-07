"""Catches identifiers the agent states that no tool ever returned.

golden/main_prompt.txt's GROUNDING section is blunt: "NEVER invent or guess an
ID, price, slot, date, doctor name, room, phone number, email or timeline -
including the caller's own mobile number or MRN." Until now that was a rule
addressed only to the model, with nothing checking whether it held. It does not
hold: a small model reading this repo's few-shot exemplars will happily say
"ஒரு நிமிஷம் சார், system-ல check பண்றேன்..." and then read back an MRN and a
pair of appointment slots lifted straight out of the exemplar, having called no
tool at all. That is the single highest-consequence failure this pipeline can
have - a caller told a confident, wrong reference number - and it is invisible
in the transcript, because a fabricated ID looks exactly like a real one.

So this checks the only class of fact where "invented" is decidable without a
model: structured identifiers and phone numbers. Anything the agent says that
matches those shapes must already appear somewhere it could legitimately have
come from - a tool result, the ledger, or the caller's own words. Everything
else (prose, reassurance, clinical wording) is out of scope here and stays a
matter for the prompt and the evals.

This USED to say it was deliberately not a filter on speech, "because by the
time a clause is checked it has already been streamed to the caller". That
premise stopped being true: conversation.speakable() is now a pre-speech choke
point that every clause passes through before it is spoken, and it calls
ungrounded_identifiers() there. Observed live over the socket, the agent asked
for a mobile number, was given an age, and read out the phone number from its
own few-shot exemplar - this had detected it and logged an ERROR, and the
caller had already heard it.

The other half of that reasoning was real and still holds: withholding half a
sentence is worse than the fault, because dropping the middle clause of
"ஆமாம், MRN ARV-604417-னு இருக்கு. சரியா?" leaves "ஆமாம், சரியா?", which says
nothing. So a fabrication ENDS the turn on a plain request for the detail
rather than punching a hole in it. See conversation.speakable().

unbacked_action_claims() below WAS report-only, on the reasoning that the
sentence it catches has no identifier to withhold. That is no longer true
either: conversation.speakable() withholds the clause outright and ends the
turn, exactly as it does for a fabricated identifier, because "I have booked
your appointment" is not improved by dropping a word out of the middle of it.
The one deliberate exception is emergency dispatch, which this MVP treats as a
built-in simulated action - see ConversationManager._check_action_claims().
"""

from __future__ import annotations

import json
import re

# ARV-118342, APT-77219, BILL-55210, LAB-33012, REF-90210, POL-4521, TCK-100001
# - every reference ID tools.py hands out has this shape, and so does every one
# the golden flows quote.
_STRUCTURED_ID_RE = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")

# Indian mobile numbers, in the two ways a caller or agent says them: as ten
# digits, and split 5+5 the way the golden flows write them ("98407 21534").
_MOBILE_RE = re.compile(r"\b[6-9]\d{9}\b|\b[6-9]\d{4}\s\d{5}\b")


def extract_identifiers(text: str) -> set[str]:
    """Return every structured ID and phone number appearing in `text`.

    Phone numbers are normalized to bare digits so "98407 21534" and
    "9840721534" compare equal - the agent routinely reads back, in spaced
    form, a number a tool returned unspaced.
    """
    identifiers = set(_STRUCTURED_ID_RE.findall(text))
    for match in _MOBILE_RE.findall(text):
        identifiers.add(re.sub(r"\s", "", match))
    return identifiers


def grounded_identifiers(sources: list[str]) -> set[str]:
    """Every identifier the agent could legitimately repeat, from all sources."""
    grounded: set[str] = set()
    for source in sources:
        grounded |= extract_identifiers(source)
    return grounded


def ungrounded_identifiers(reply: str, sources: list[str]) -> list[str]:
    """Identifiers stated in `reply` that appear in none of `sources`.

    Sorted so the result is stable for logging, assertions and event payloads.
    """
    return sorted(extract_identifiers(reply) - grounded_identifiers(sources))


def grounding_sources(messages: list[dict]) -> list[str]:
    """Everything in a call's history an identifier may legitimately come from.

    That is exactly two things: tool results (what the hospital systems
    actually returned) and the caller's own turns (they may state their MRN
    before any lookup runs). Callers holding facts from elsewhere - the ledger,
    the prompt's standing facts - pass them in alongside this.

    Two roles are excluded, both deliberately:

    - The agent's own previous turns. An ID it invented on turn two must not
      become self-justifying on turn three.
    - The SYSTEM PROMPT. This looks wrong and is the whole point: the prompt
      carries the few-shot exemplars, and those contain a full worked example
      with an MRN in it. Counting the prompt as a source is what let the
      observed failure through - the model read back the exemplar's MRN having
      called no tool, and the check called it grounded because the exemplar was
      "in the prompt". The exemplars say in as many words never to reuse their
      identifiers, so an identifier whose only provenance is the prompt is
      precisely the fabrication worth catching.
    """
    sources: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in {"tool", "user"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            sources.append(content)
        elif content is not None:
            sources.append(json.dumps(content, ensure_ascii=False))
    return sources


# --------------------------------------------------------------------------
# Claims about actions, as opposed to claims about facts.
#
# The identifier check above answers "where did that number come from?". It
# cannot answer the question that turned out to matter more: the agent, given
# a chest-pain call and an address, says
#
#     "Ambulance அனுப்பிட்டேன், இப்பவே கிளம்பிடுச்சு."
#     (I have sent an ambulance, it has left right now.)
#
# and calls no tool. There is no invented identifier in that sentence - there
# is nothing for the check above to see - and it is worse than a wrong MRN:
# the caller stops looking for help because they have been told help is coming,
# and it is not. Three separate prompt fixes failed to make dispatchAmbulance
# fire (runtime_core.txt's EMERGENCY section, its GROUNDING section, and the
# pre-generation reminder in conversation.py), so this stops being a thing the
# prompt is trusted to get right and becomes a thing that is checked.
#
# Like everything else here it REPORTS: the sentence has already been spoken by
# the time it is checked. What it buys is that the failure is now visible in
# the console, in the call log and in the evals, instead of reading as a
# perfectly normal turn.
#
# Each entry is (what was claimed, the tools that would make it true, how it is
# said). The patterns deliberately match only COMPLETED forms - "அனுப்பிட்டேன்"
# (I have sent), never "அனுப்பணுமா?" (shall I send?) - because offering to do
# something is not claiming to have done it.
# Exported so a caller that wants to act on one specific claim can match it
# without duplicating this literal. (It used to name
# conversation.py's _dispatch_ambulance_fallback(), which went with the tool
# layer - there is nothing to fall back TO now, which is the whole reason this
# claim is worth reporting.)
AMBULANCE_CLAIM = "said an ambulance has been dispatched"

_ACTION_CLAIMS: tuple[tuple[str, frozenset[str], re.Pattern[str]], ...] = (
    (
        AMBULANCE_CLAIM,
        frozenset({"dispatchAmbulance"}),
        # FUTURE forms as well as completed ones, and this row is the one place
        # that exception is right. Everywhere else "offering to do something is
        # not claiming to have done it" holds, but there is nothing to offer
        # here: this process cannot dispatch, so "ambulance உடனே வரும்" is not
        # an offer, it is a promise of an arrival nobody has arranged. Observed
        # live at temperature 0 on a chest-pain call - "இப்பவே 108-க்கு call
        # பண்ணுங்க, ambulance உடனே வரும்" - with the completed-only pattern
        # letting it through to a caller who then stopped calling 108.
        #
        # The paramedic exclusion is not incidental. Flow 18 wants the agent to
        # say "Ambulance-ல paramedic வர்றாங்க, அவங்க பாத்துட்டு கொடுப்பாங்க"
        # when refusing aspirin - the people in the ambulance are coming, which
        # is a statement about who administers medicine and not a promise that
        # a vehicle is on its way. Only the AMBULANCE arriving is the claim.
        re.compile(
            r"ambulance(?:(?!paramedic|பாராமெடிக்)[^.!?]){0,40}"
            r"(?:அனுப்பிட்ட|அனுப்பிவிட்ட|அனுப்பினேன்|கிளம்பிட்ட|கிளம்பிடுச்|"
            r"வரும்|வந்துடும்|வந்துக்கிட்|வர்றது|கிளம்பு|"
            r"sent|dispatch|on\s*the\s*way|coming)",
            re.IGNORECASE,
        ),
    ),
    (
        # Same reasoning: flow 18 forbids "the ER team is pre-alerted" outright
        # unless a tool confirmed it, and there is no such tool. Observed in the
        # same turn as the ambulance promise above.
        "said the ER team has been alerted",
        frozenset({"dispatchAmbulance", "alertEmergencyTeam"}),
        re.compile(
            r"(?:ER|emergency)\s*(?:team|desk|department)?[^.!?]{0,30}"
            r"(?:சொல்லிட|சொல்லிவிட|தெரிவிச்|alert|inform|ready|தயாரா)",
            re.IGNORECASE,
        ),
    ),
    (
        "said the appointment is booked",
        frozenset({"bookAppointment", "confirmAppointment"}),
        # These rows used to match ONLY the exact English-verb-plus-"பண்ணிட்ட"
        # form the exemplars happen to use, which is a fraction of how the
        # claim is actually said. Measured against 17 false completions across
        # the five served flows, the table caught 6 - and appointment.reschedule,
        # one of the five, had NO row at all: "மாத்திட்டேன்" (I have changed it)
        # went straight through to the caller.
        #
        # Each alternative is a specific ACTION VERB in completed aspect, not a
        # subject noun near a generic completion marker. That distinction is
        # load-bearing in both directions:
        #
        #   "Appointment book பண்ணட்டுமா?"    an OFFER - must stay clean, or the
        #                                     agent can no longer offer anything
        #   "எல்லாம் குறிச்சுக்கிட்டேன் —      the CORRECT closing line, said on
        #    appointment desk-ல இருந்து ..."   29 of the 208 recorded calls. It
        #                                     carries both a completion marker
        #                                     ("குறிச்சுக்கிட்டேன்", I have noted
        #                                     it down) and the word "appointment",
        #                                     so any proximity rule withholds it
        #                                     and breaks every booking call.
        #
        # Noting something down is not doing it, which is why the verb, not the
        # marker, is what these match.
        re.compile(
            r"(?:book|confirm)\s*பண்ணிட்ட"
            r"|ஒதுக்கிட்ட"
            r"|(?:appointment|booking|slot)[^.!?]{0,20}(?:போட்டுட்ட|முடிஞ்சிடுச்|ஆயிடுச்)",
            re.IGNORECASE,
        ),
    ),
    (
        # appointment.reschedule is a SERVED flow and had no row until now.
        "said the appointment is rescheduled",
        frozenset({"rescheduleAppointment", "bookAppointment"}),
        re.compile(
            r"(?:reschedule|change)\s*பண்ணிட்ட"
            r"|மாத்திட்ட"
            r"|மாத்தி\s*வெச்சிட்ட",
            re.IGNORECASE,
        ),
    ),
    (
        "said the appointment is cancelled",
        frozenset({"cancelAppointment"}),
        re.compile(
            r"(?:cancel|ரத்து)\s*பண்ணிட்ட"
            r"|(?:appointment|booking)[^.!?]{0,20}நீக்கிட்ட",
            re.IGNORECASE,
        ),
    ),
    (
        "said a ticket has been raised",
        frozenset({"createTicket", "escalate"}),
        re.compile(r"ticket[^.!?]{0,40}(?:போட்டுட்ட|raise\s*பண்ணிட்ட)", re.IGNORECASE),
    ),
    (
        # A caller told an SMS is already sent stops waiting for the desk's
        # call and waits for a message that is never coming. The scripted
        # closing line says "SMS-ம் வரும்" - an SMS WILL come, from the desk -
        # which is a different sentence and stays clean.
        "said an SMS has already been sent",
        frozenset({"resendReport", "confirmAppointment"}),
        re.compile(r"(?:SMS|message)[^.!?]{0,20}அனுப்பிட்ட", re.IGNORECASE),
    ),
    (
        # First person completed ONLY. "Doctor-கிட்ட சொல்லிடுங்க" is the agent
        # telling the CALLER to mention it to the doctor, which is ordinary
        # advice; "சொல்லிட்டேன்" claims this process notified a clinician, and
        # it has no tool that can.
        "said a doctor or the staff has been informed",
        frozenset({"escalate", "createTicket"}),
        re.compile(
            r"(?:doctor|டாக்டர்|staff|desk)[^.!?]{0,25}(?:சொல்லிட்டே|சொல்லிட்டோ|தெரிவிச்சிட்டே)",
            re.IGNORECASE,
        ),
    ),
)


def unbacked_action_claims(reply: str, tools_called: set[str]) -> list[str]:
    """Actions `reply` claims to have completed that no tool in this call did.

    `tools_called` is every tool name invoked so far in the call, not just this
    turn: the agent may legitimately dispatch on one turn and mention it on the
    next.
    """
    return [
        claim
        for claim, tools, pattern in _ACTION_CLAIMS
        if pattern.search(reply) and not (tools & tools_called)
    ]
