"""Builds the per-turn system prompt: condensed core + ONE flow's playbook.

golden/main_prompt.txt is a ~15k-token specification document. Sending it whole
on every turn is what broke the agent in practice: Ollama's VRAM-derived default
num_ctx (2048-4096) silently truncated it *before* section 2's language rules,
so the model answered in English and invented a caller mobile number. Raising
num_ctx to hold it fixes correctness but allocates a KV cache far too large for
CPU-only inference - a trivial generation measured 222s on this machine.

So the prompt is assembled instead of shipped whole: golden/runtime_core.txt
carries the rules that apply to EVERY turn (language, turn discipline, ledger,
grounding, clinical safety, emergency override), and only the ACTIVE flow's
section-8 playbook is appended. main_prompt.txt stays the single source of truth
for those 20 playbooks - they are parsed out of it here, never duplicated.

Result: ~1.5k tokens per turn instead of ~15k, which is both fast enough for a
voice channel and small enough that the language rules are never truncated away.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger("aica.prompt_builder")

# Section 8 entries look like "--- FLOW 01 — appointment.book ---", but four of
# them carry a trailing qualifier the intent name must not absorb:
# "(OUTBOUND)", "(nurse-led)", "(OVERRIDE — speed beats everything)".
_FLOW_HEADER_RE = re.compile(r"^---\s*FLOW\s+(\d+)\s*—\s*([\w.]+).*?---\s*$", re.MULTILINE)

# Flow 14's standing facts are the only hard facts the agent may state without a
# tool result, so they ship with that playbook (see main_prompt.txt Sec8/Sec10).
DEFAULT_FLOW = "info.general"

# THE FIVE FLOWS THIS DESK ACTUALLY HANDLES.
#
# main_prompt.txt still specifies all twenty and parse_flow_playbooks() still
# reads all twenty out of it - the spec is not the deliverable. What ships is
# these five, and every other flow is answered with the scope line in
# conversation.py instead of being improvised at by a 4B model.
#
# Why: a flow the desk half-answers is worse than one it declines. The other
# fifteen need tools this runtime does not have (a bill amount, a lab value, a
# policy limit, a referral status) and their playbooks demonstrate stating
# facts the process cannot know. Declining them is the honest answer AND the
# fast one - it costs no LLM call at all.
SUPPORTED_INTENTS = frozenset(
    {
        "appointment.book",
        "appointment.reschedule",
        "appointment.cancel",
        "info.general",
        "emergency.escalate",
    }
)

# Which flow a bare department name belongs to. "Ortho-க்கு வரணும்" carries no
# booking verb at all and matches no trigger below, but there is only one thing
# a caller naming a department wants from this desk.
DEPARTMENT_INTENT = "appointment.book"

# Every department the hospital runs, in both scripts, because a caller says
# "cardiology" and the ASR may hand over either "Cardiology" (normalised by
# transcript_norm.py) or the raw Tamil transliteration.
#
# Only consulted when nothing else matched, and only on a turn that has not
# yet picked a flow - see names_a_department(). A department name is also a
# perfectly ordinary ANSWER mid-call ("எந்த department?" / "Cardiology"), and
# re-routing a cancellation to a booking because the caller answered the
# question would be worse than the miss this closes.
#
# Tamil stems are chosen for distinctiveness, not completeness: கண் (eye) and
# தோல் (skin) are left out because they are prefixes of கண்டிப்பா and தோல்வி,
# and a false department is a wrongly-routed call.
_DEPARTMENT_RE = re.compile(
    r"cardio|ortho|paediat|pediat|neuro|gyn|obstetric|derma|dental|dentist|"
    r"ophthal|urolog|gastro|pulmo|psychiat|nephro|oncolog|diabet|endocrin|"
    r"physio|dietici|dietitian|nutrition|vaccin|immunis|immuniz|"
    r"\bENT\b|ear\s*nose|\beye\b|\bskin\b|\bchest\s*(?:doctor|special)|"
    r"general\s*(?:medicine|surgery)|\bdepartment\b|specialist|"
    r"கார்டி|இதய|ஆர்த்தோ|எலும்பு|குழந்தை|நியூரோ|நரம்பு|மகப்பேறு|சிறுநீரக|"
    r"மனநல|பிசியோ|தடுப்பூசி|டெர்ம|டென்ட|கைனக|டிபார்ட்மெண|டிபார்ட்மென",
    re.IGNORECASE,
)


def names_a_department(text: str) -> bool:
    """Whether the caller named a hospital department or specialty."""
    return bool(_DEPARTMENT_RE.search(text))


# The master prompt documents a future TOOL-ENABLED agent. This runtime has no
# clinical-record tools, so three of the five playbooks demonstrate
# claims this process cannot make: reading an MRN back off a mobile number,
# offering slots out of searchSlots, and quoting a fee. Emergency dispatch is
# intentionally treated as a simulated built-in capability for this MVP.
#
# These replace the playbook body rather than being appended after it, keeping
# the runtime contract short and free of unavailable record-system operations.
#
# What must NOT go with them is the flow's exemplars: rules describe the
# register, examples ARE the register, and a previous attempt at this that also
# dropped the exemplars measured 49% Tamil against a 65% target. build() keeps
# them.
#
# info.general has no entry and keeps its playbook whole - its standing facts
# (OP timings, visiting hours, parking) are the one set of facts the agent may
# state without a tool, so there is nothing there to contradict.
_RUNTIME_PLAYBOOK_OVERRIDES: dict[str, str] = {
    "emergency.escalate": """Emergency outranks every other flow. Speed beats everything: no verification, no MRN, no insurance, no money.
React with urgency and compassion first. Say you are staying on the line and dispatching an ambulance immediately, then ask for the ADDRESS before any clinical question. Read the address back exactly once and say the ambulance has been dispatched. If they already gave the address, never ask again.
Copy the caller's address VERBATIM. Never translate, expand, autocorrect, infer a cross/street/floor, or replace a locality with a similar-sounding one. Never copy a patient relationship from an example: use only the relationship stated in this call, otherwise say Patient.
Then assess in SHORT questions, one per turn — is the patient speaking, are the eyes open, when did it start, is the breathing laboured.
Give safe positioning only: never let a chest-pain patient walk, sit them propped up rather than flat, loosen tight clothing, clear the crowd. Ask them to put all current medicines in one bag for the paramedic, and to open the door and the gate.
Tell them to say immediately if speech stops, if the patient collapses, or if breathing stops.
Never give an ETA and never say the ER was alerted. Never authorise any medicine, aspirin included. Never name a condition. Never end the call.""",
    "appointment.book": """You are TAKING DOWN a booking request, not completing one. Never state or read back an MRN, a slot, a doctor's availability, a fee, a block/floor/room, or an appointment ID.
Acknowledge what they just gave you in two or three words — by name if they gave a name — and then ask for ONE detail you do not have yet. You need: patient name, department or doctor, the day and part of the day, and a callback mobile number. Never ask for one the caller has already said, even in passing: asking again is the commonest way this call goes wrong.
The caller may correct any detail mid-call. The LATEST value replaces the earlier one immediately: acknowledge only the new value, never repeat the old value, and continue from the next missing detail. A clear correction is not a hearing problem.
Any department the caller names is fine — take it down in their words. Never say a department does not exist and never substitute a different one.
Close by saying the desk will confirm the slot and call back, and that an SMS follows.""",
    "appointment.reschedule": """You are TAKING DOWN a reschedule request. Never read back a booking you have not been told, never offer a slot, never confirm the old one is cancelled.
Acknowledge what they just gave you in two or three words, then ask for ONE detail you do not have yet. You need: patient name, the existing appointment's day, the new day and part of the day, and a callback mobile number. Never ask for one they have already said — "அடுத்த வெள்ளிக்கிழமை மாலை" is BOTH the day and the time, so do not then ask what time.
Say the desk will confirm the new slot and call back, that the old booking stays until then, and that there is no extra charge.""",
    "appointment.cancel": """You are TAKING DOWN a cancellation request. Never state a cancellation reference, a refund amount, a refund date, or that anything has been cancelled.
Offer a reschedule ONCE, gently. If they decline, cancel without a second attempt and never make them justify it.
Acknowledge what they just gave you in two or three words — by name if they gave a name — and then ask for ONE detail you do not have yet. You need: patient name, the appointment day, and the appointment ID or mobile number. Never ask for one they have already said. Ask the reason only to log it, and offer a neutral category.
If an advance was paid, say the billing desk will confirm the refund route and timeline in the callback. Never promise a refund date.""",
}

# Emergency outranks everything and is already in the core prompt; it is listed
# here too so an explicit emergency turn still pulls in flow 18's full playbook.
EMERGENCY_INTENT = "emergency.escalate"

# Strong emergencies may enter the dispatch sequence immediately. Distress
# fragments are routed to the emergency safety lane too, but conversation.py
# asks what is wrong before dispatching; "ஐயோ ... முடியல" is urgent evidence,
# not enough clinical detail to invent a symptom or diagnosis.
_STRONG_EMERGENCY_PATTERN = (
    r"நெஞ்சு\s*வலி|chest\s*pain|மயக்க|மூச்சு|வலிப்ப|seizure|"
    r"unconscious|ரத்தம்\s*(போ|வ|கொட்|நி)|ரத்த\s*போக்கு|bleeding|சுத்த\s*முடிய|108|"
    r"உயிர|தூக்கி|விழுந்துட்டா|பேச\s*முடிய|"
    r"ambulance|அம்புல|ஆம்புல|emergency|எமர்ஜென்|அவசர"
)
_AMBIGUOUS_DISTRESS_PATTERN = (
    r"(?:ஐயோ|அய்யோ|காப்பாத்த|help|பயமா|தாங்க)[^.?!]{0,40}(?:முடியல|தாங்க|கஷ்டம்|வலி)|"
    r"(?:முடியல|தாங்க\s*முடியல|கஷ்டமா)[^.?!]{0,30}(?:ஐயோ|அய்யோ|காப்பாத்த|help)"
)
_STRONG_EMERGENCY_RE = re.compile(_STRONG_EMERGENCY_PATTERN, re.IGNORECASE)


def is_explicit_emergency(text: str) -> bool:
    """Whether `text` names a concrete emergency rather than distress alone."""
    return bool(_STRONG_EMERGENCY_RE.search(text))

# Ordered most-specific first: the first intent whose pattern matches wins, so
# e.g. "report வந்துடுச்சா" routes to lab.result_inquiry rather than lab.book.
# Patterns cover the caller's own words in both scripts, per main_prompt.txt
# Sec6C's trigger table. This is a deterministic pre-router, not a classifier:
# it only picks which playbook to show the model, and the model still detects
# the real flow from the conversation - so a miss degrades to a slightly
# less-specific playbook, never to a wrong answer.
_INTENT_PATTERNS: list[tuple[str, str]] = [
    (
        # THIS ROW IS A SAFETY DEVICE, and it is the one row where a false
        # positive is cheaper than a miss. A wrongly-triggered emergency costs
        # one embarrassing turn; a missed one routes a caller who cannot
        # breathe to the info.general playbook.
        #
        # It missed a whole live call, every turn of it. The caller said
        # "எனக்கு அம்புலான்ஸ் வேணும்" four times and then "நான் உயிருக்கு
        # போராடிட்டு இருக்கேன்", and detect_intent returned None for all five,
        # for two separate reasons that both had to be fixed:
        #
        #   1. The trigger was the LATIN "ambulance". The ASR is a Tamil-only
        #      model and can only ever emit Tamil script, so the caller's word
        #      arrived as அம்புலான்ஸ் and matched nothing. Every English trigger
        #      in this table has that hole; it is closed for this row here and
        #      for the desk vocabulary generally by transcript_norm.py.
        #   2. "உயிர்" does not match "உயிருக்கு". The pulli is not a stable
        #      thing to anchor on - it disappears whenever a vowel suffix
        #      attaches (உயிர் + உக்கு -> உயிருக்கு), which is most of the time
        #      in a real sentence. Tamil stems here are written WITHOUT the
        #      trailing pulli so they survive inflection.
        #
        # "அவசர" (urgent) is included knowing it also matches "அவசரம் இல்ல"
        # (no hurry). That is the trade above, taken deliberately.
        EMERGENCY_INTENT,
        rf"{_STRONG_EMERGENCY_PATTERN}|{_AMBIGUOUS_DISTRESS_PATTERN}",
    ),
    (
        # Services this desk does not handle. They keep their own rows rather
        # than falling through to "no match" because a POSITIVE identification
        # is what lets conversation.py say "not us" instead of handing the turn
        # to the model with the info.general playbook and hoping.
        "other.desk",
        r"certificate|blood\s*(bank|avail|stock|donor)|"
        r"(?:a|b|ab|o)\s*(?:positive|negative|\+|-)\s*blood|"
        r"mobile\s*(number\s*)?(update|change)|"
        r"phone\s*(number\s*)?(update|change)|contact\s*(update|change)",
    ),
    (
        "complaint.escalation_angry",
        r"மூணாவது\s*தடவை|மூன்றாவது|consumer\s*court|social\s*media|"
        r"எத்தனை\s*தடவை|இன்னும்\s*வரல|காசு\s*வரல|refund\s*வரல|கத்த|"
        r"manager|supervisor|மேனேஜர்|மூணு\s*தடவை\s*(call|கூப்பிட்|போன்)|"
        r"respond\s*பண்ணல|யாரும்\s*(பதில்|respond|கேட்க)|ரொம்ப\s*அதிகம்",
    ),
    (
        "complaint.register",
        r"complaint|புகார்|மோசமா|காக்க\s*வெச்|காத்திருந்த|சரி\s*இல்ல|rude|"
        r"மரியாதை\s*இல்ல|கம்ப்ளைண்ட்|service.{0,12}(மோசம்|சரி\s*இல்ல)",
    ),
    (
        "postprocedure.checkin",
        r"surgery\s*ஆகி|operation\s*ஆகி|operation.{0,6}அப்புறம்|surgery.{0,6}அப்புறம்|"
        r"stitch|தையல்|dressing|drops\s*போட|discharge\s*ஆன|சர்ஜரி\s*ஆகி|"
        r"ஆப்பரேஷன்.{0,8}அப்புறம்|post.?op|ஆபரேஷன்\s*ஆகி",
    ),
    (
        "clinical.triage",
        r"காய்ச்சல|fever|வாந்தி|vomit|rash|தடிப்ப|வலிக்குது|வலி\s|\sவலி|"
        r"என்ன\s*பண்றதுன்னு\s*தெரியல|உடம்பு\s*சரி\s*இல்ல|ஃபீவர்|வாமிட்|"
        r"மயக்கம்|சளி|இருமல்|என்ன\s*பண்ணனும்",
    ),
    (
        "prescription.refill",
        r"refill|தீர்ந்து|மாத்திரை\s*வேண|tablets?\s*வேண|மருந்து\s*வேண|stock\s*இல்ல|ரீஃபில்|ரீபில்|டேப்லெட்|டாப்லெட்|மருந்து\s*தீர",
    ),
    (
        "medication.query",
        r"side\s*effect|தூக்கம்\s*வர|சாப்பிட்ட\s*பிறகு|எப்போ\s*சாப்பிட|dose|"
        r"tablet.*பிரச்ச|மருந்து.*பிரச்ச|சைட்\s*எஃபெக்ட்|டோஸ்|"
        r"எப்படி\s*சாப்பிட|எத்தனை\s*தடவை\s*சாப்பிட|மாத்திரை.{0,12}எப்படி|"
        r"மருந்து.{0,12}எப்படி|சாப்பிடணும்|சாப்பிடலாமா",
    ),
    (
        "lab.result_inquiry",
        r"report\s*வந்த|result|report\s*கிடைக்|SMS\s*வரல|report\s*எப்போ|value|ரிப்போர்ட்|ரிசல்ட்",
    ),
    (
        "lab.book",
        r"test\s*எழுதி|scan\s*book|blood\s*test|sample\s*எடுக்|ultrasound|scan\s*பண்ண|"
        r"lab.*book|test.*book|டெஸ்ட்|ஸ்கேன்|ப்ளட்\s*டெஸ்ட்|சாம்பிள்|"
        r"scan.{0,10}(appointment|வேண|book)|x.?ray|எக்ஸ்.?ரே|MRI|CT\s*scan",
    ),
    (
        "insurance.query",
        r"insurance|policy|cover\s*ஆகும|cashless|pre.?auth|TPA|room\s*rent|co.?pay|claim|இன்சூரன்ஸ்|பாலிசி|கிளைம்",
    ),
    (
        "billing.query",
        r"bill|(?<!dis)charge|extra\s*போட்|itemised|itemized|EMI|தவணை|கட்டணம்|"
        r"amount.*தப்ப|பில்|சார்ஜ்|எவ்வளவு\s*ஆகும்|எவ்வளவு\s*ஆச்|receipt|ரசீது|"
        r"payment.{0,10}வரல|கட்டணம்",
    ),
    (
        "records.request",
        r"case\s*sheet|discharge\s*summary|records?\s*வேண|medical\s*records|"
        r"copy\s*வேண|file\s*வேண|ரெக்கார்ட்|கேஸ்\s*ஷீட்|டிஸ்சார்ஜ்\s*சம்மரி",
    ),
    (
        "referral.status",
        r"referral|வேற\s*hospital|letter.*எழுத|refer\s*பண்ண|ரெஃபரல்|ரிபரல்",
    ),
    (
        "patient.register",
        r"register\s*பண்ண|புதுசா|new\s*patient|MRN\s*இல்ல|முதல்\s*தடவை|first\s*time.*register|ரெஜிஸ்டர்|புது\s*பேஷண்ட்",
    ),
    (
        "appointment.followup",
        r"follow.?up|review.{0,12}வர|course\s*முடிஞ்|திரும்ப\s*வர\s*சொன்|ஃபாலோ\s*அப்|"
        r"ஃபாலோஅப்|ரிவ்யூ|சொன்ன\s*மாதிரி\s*வர",
    ),
    (
        # The five supported flows carry MORE synonyms than the fifteen, and
        # deliberately: a caller who says "மாத்தி தர முடியுமா" instead of
        # "reschedule" is one of the five calls this MVP exists to answer, and
        # the cost of missing them is no longer a slightly-wrong playbook - it
        # is the scope line, told to a caller the desk does serve.
        "appointment.reschedule",
        r"postpone|prepone|reschedule|date\s*மாத்த|நேரம்\s*மாத்த|வேற\s*date|"
        r"அன்னைக்கு\s*வர\s*முடியா|போஸ்ட்போன்|ரீஷெட்யூல்|தேதி\s*மாத்த|"
        r"நாள்.{0,8}மாத்த|நாளுக்கு\s*மாத்த|வேற\s*நாள்|வேற\s*நேரம்|"
        # The bare verb. "மாத்துங்க" / "மாத்தி தர முடியுமா" / "மாத்திக்கலாமா"
        # are how this is actually said; every form above needs the caller to
        # have named the thing being changed first, and they usually have not.
        r"மாத்துங்க|மாத்தி\s*(?:தர|கொடு|போடு|விடு)|மாத்திக்க|மாற்றி\s*தர|"
        r"தள்ளி\s*(?:போடு|வெ|வைக்க)|முன்னாடி\s*போடு|"
        r"(?:date|time|நாள்|நேரம்|booking|appointment|slot)\s*(?:-?ஐ\s*)?change",
    ),
    (
        "appointment.cancel",
        # Bound ரத்து as a word: without this, வாரத்துக்கு contains the same
        # four code points and a physio frequency answer became a cancellation.
        r"cancel|(?<!\w)ரத்து(?!\w)|வேணாம்.*appointment|appointment.*வேணாம|கேன்சல்|கான்சல்|"
        # After reschedule, so "வர மாட்டேன், வேற நாள் இருக்கா" is still a
        # reschedule and only a flat refusal to come lands here.
        r"வர\s*மாட்ட|appointment.{0,20}எடுத்து(?:டு|விடு)|வேணாம்னு\s*சொல்ல",
    ),
    (
        "appointment.confirm",
        r"confirm\s*(ஆயி|ஆச்|ஆகி|ஆன|பண்ணிட்)|appointment\s*இருக்கா|"
        r"booking.{0,8}confirm|கன்ஃபர்ம்|கன்பர்ம்|appointment.{0,10}check\s*பண்ண|"
        r"appointment.{0,10}உறுதி",
    ),
    (
        "appointment.book",
        r"appointment|book\s*பண்ண|doctor.*பாக்க|consult|slot|சந்திக்க|அப்பாயின்|அபாயின்|"
        r"புக்\s*பண்ண|டாக்டர.{0,4}\s*பாக்க|கன்சல்ட்|ஸ்லாட்|see\s+a\s+[\w\s]{0,16}doctor|"
        r"time\s*வேண|நேரம்\s*வேண|டாக்டர்.{0,8}(வேண|இருக்கா)|"
        # Showing someone TO a doctor is the commonest phrasing of all and
        # contains neither "appointment" nor "book".
        r"(?:doctor|டாக்டர்|dr\.?)[^.?!]{0,20}(?:காட்ட|பாக்க|meet)|"
        r"(?:doctor|டாக்டர்|dr\.?)\s*(?:appointment|அப்பாயின்)|"
        r"checkup|check.?up|செக்கப்|master\s*health|"
        r"token\s*வேண|டோக்கன்\s*வேண|OP-?க்கு\s*வர",
    ),
    (
        "info.general",
        r"timing|visiting\s*hours|parking|canteen|wheelchair|ICU|attender|"
        r"who\s+are\s+you|your\s+name|agent\s*name|"
        r"உங்க\s*பேரு|உங்கள்\s*பெயர்|நீங்க\s*யாரு|யார்\s*பேசுற|"
        r"hospital\s*(?:name|details?|address)|"
        r"(?:ஹாஸ்பிட்டல்|ஆஸ்பத்திரி)\s*(?:பேரு|பெயர்|விவரம்|details?|address)|"
        r"எந்த\s*(?:hospital|ஹாஸ்பிட்டல்|ஆஸ்பத்திரி)|"
        r"எப்படி\s*வர|எத்தனை\s*மணி|எங்க\s*இருக்கு|விசிட்டிங்|பார்க்கிங்|டைமிங்|"
        # Is the place open. Bound to the place, because a bare weekday or a
        # bare "open" is far more often part of a booking turn.
        r"(?:hospital|OP|clinic|ஹாஸ்பிட்டல்|ஆஸ்பத்திரி)[^.?!]{0,15}"
        r"(?:open|close|வேலை|இருக்கும|திறந்|சாத்)|"
        r"address\s*சொல்|எப்படி\s*வந்து\s*சேர|location",
    ),
]

_COMPILED_PATTERNS = [(intent, re.compile(pattern, re.IGNORECASE)) for intent, pattern in _INTENT_PATTERNS]


def detect_intent(text: str) -> str | None:
    """Return the first intent whose trigger pattern matches, or None.

    Deterministic and cheap on purpose - a voice turn cannot afford an extra
    LLM round-trip just to pick which playbook to show.

    Three passes, and the order IS the scope policy:

      1. Emergency. Absolute priority, as it is everywhere else.
      2. The five SUPPORTED_INTENTS, in table order. A supported flow always
         beats an unsupported one that matched the same sentence, because the
         two mistakes are not the same size: taking "scan-க்கு appointment
         வேணும்" down as a consult booking is a slightly wrong playbook, and
         declining it is a caller told to go away over a word.
      3. Everything else, only so the turn can be POSITIVELY identified as
         something this desk does not do. conversation.py answers those with
         the scope line; nothing here ever loads their playbook.
    """
    emergency_pattern = next(
        pattern for intent, pattern in _COMPILED_PATTERNS if intent == EMERGENCY_INTENT
    )
    if emergency_pattern.search(text):
        return EMERGENCY_INTENT
    for supported_only in (True, False):
        for intent, pattern in _COMPILED_PATTERNS:
            if intent == EMERGENCY_INTENT:
                continue
            if (intent in SUPPORTED_INTENTS) is not supported_only:
                continue
            if pattern.search(text):
                return intent
    return None


@dataclass(frozen=True)
class FlowPlaybook:
    flow_number: int
    intent: str
    body: str


def parse_flow_playbooks(master_prompt: str) -> dict[str, FlowPlaybook]:
    """Extract section 8's twenty per-flow playbooks, keyed by intent."""
    matches = list(_FLOW_HEADER_RE.finditer(master_prompt))
    if not matches:
        raise ValueError("no '--- FLOW NN — intent ---' headers found in the master prompt")

    playbooks: dict[str, FlowPlaybook] = {}
    for index, match in enumerate(matches):
        flow_number = int(match.group(1))
        intent = match.group(2)
        start = match.end()
        # A flow body runs to the next flow header, or to section 9's rule for
        # the last one (section 8 is the final flow-bearing section).
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            section_end = master_prompt.find("\n====", start)
            end = section_end if section_end != -1 else len(master_prompt)
        playbooks[intent] = FlowPlaybook(
            flow_number=flow_number, intent=intent, body=master_prompt[start:end].strip()
        )
    return playbooks


def _format_exemplars(exchanges: list[list[str]]) -> str:
    """Render one flow's worked example, including its tool steps.

    The third role, "tool", matters more than it looks. Without it an exemplar
    reads as: caller gives a mobile number, then the agent says "ஒரு நிமிஷம்
    சார், system-ல check பண்றேன்..." and immediately states an MRN and an
    address. That is a demonstration of INVENTING a lookup result - the agent
    narrates a database query and then produces the answer out of nowhere - and
    a small model copies exactly that, calling no tool at all. Showing the
    lookup as a step makes the example demonstrate the behaviour we want:
    narrate, call the tool, then read back what the tool returned.
    """
    lines = []
    for role, text in exchanges:
        if role == "tool":
            lines.append(f"[you call {text}]")
            continue
        # A "note" is a branch the example cannot show without being twice
        # as long: the emergency exemplar has the caller give the address on
        # the SECOND turn, so recited literally it makes the agent ask for an
        # address the caller has already said. Bracketed like the tool steps,
        # which the model has not been observed reading aloud.
        if role == "note":
            lines.append(f"[{text}]")
            continue
        # Agent examples teach spoken output, so normalize only those. Caller
        # lines are evidence and must stay verbatim even when the caller says
        # the Tamil honorific themselves.
        if role != "caller":
            text = text.replace("சார்", "Sir")
        speaker = "CALLER" if role == "caller" else "YOU"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


class PromptBuilder:
    """Assembles core + one flow's playbook and exemplars into the turn prompt."""

    def __init__(
        self, core_path: Path, master_prompt_path: Path, exemplars_path: Path | None = None
    ) -> None:
        self.core_path = core_path
        self.master_prompt_path = master_prompt_path
        self.exemplars_path = exemplars_path
        self._core: str | None = None
        self._playbooks: dict[str, FlowPlaybook] = {}
        self._exemplars: dict[str, str] = {}

    @property
    def ready(self) -> bool:
        return self._core is not None

    def load(self) -> None:
        self._core = self.core_path.read_text(encoding="utf-8")
        # main_prompt.txt still specifies all twenty flows and stays the spec.
        # Only the five this desk serves are ever loadable as a playbook - so a
        # bug that routes a billing turn past the scope check cannot answer it
        # out of the billing playbook, it falls back to info.general like any
        # other unknown intent.
        self._playbooks = {
            intent: playbook
            for intent, playbook in parse_flow_playbooks(
                self.master_prompt_path.read_text(encoding="utf-8")
            ).items()
            if intent in SUPPORTED_INTENTS
        }

        # is_file(), not exists(): once the register is fine-tuned in, exemplars
        # are switched off with a blank CONVERSATION_EXEMPLARS_PATH, and a blank
        # path resolves to the CWD - a directory, which exists() happily accepts.
        if self.exemplars_path is not None and self.exemplars_path.is_file():
            raw = json.loads(self.exemplars_path.read_text(encoding="utf-8"))
            self._exemplars = {
                intent: _format_exemplars(exchanges)
                for intent, exchanges in raw.items()
                if not intent.startswith("_") and intent in SUPPORTED_INTENTS
            }
            missing = set(self._playbooks) - set(self._exemplars)
            if missing:
                # Not fatal - a flow without exemplars still gets rules and a
                # playbook - but it is the flow most likely to drift, so say so.
                logger.warning("no few-shot exemplars for: %s", ", ".join(sorted(missing)))

        logger.info(
            "prompt builder loaded: core=%d chars, %d playbooks, %d exemplar sets",
            len(self._core),
            len(self._playbooks),
            len(self._exemplars),
        )

    def build(self, intent: str | None) -> str:
        """Return the system prompt for a turn whose detected intent is `intent`."""
        if self._core is None:
            raise RuntimeError("PromptBuilder is not loaded")

        resolved = intent if intent in self._playbooks else DEFAULT_FLOW
        playbook = self._playbooks.get(resolved)
        if playbook is None:
            return self._core

        # The override REPLACES the spec's body where one exists; the spec's
        # own body is used where it does not. See _RUNTIME_PLAYBOOK_OVERRIDES
        # for why sending both was worse than sending either.
        body = _RUNTIME_PLAYBOOK_OVERRIDES.get(resolved, playbook.body)
        parts = [
            self._core,
            "## THIS CALL'S PLAYBOOK — the flow you are handling right now",
            "Wording shown is a MODEL, not a script. Say it in your own words, one point per turn.",
            # Playbook examples describe agent wording. Keep the male address
            # in Latin script so Tamil TTS says English "Sir", not "saar".
            body.replace("சார்", "Sir"),
        ]

        exemplars = self._exemplars.get(resolved)
        if exemplars:
            # Rules describe the register; examples ARE the register. This is
            # what actually holds a small model in Tamil-English code-mix.
            parts += [
                "",
                "## HOW A REAL CALL SOUNDS — copy this register, not these facts",
                "Never reuse a name, number or ID from this example. Match only the "
                "language mix, the sentence length, and the one-question-per-turn pacing.",
                # Observed on the 4B model: it read a CALLER line out of this block
                # aloud as its own turn, and invented a child the caller had never
                # mentioned because the example happened to be about one. The block
                # has to say whose words are whose, not just "do not copy the facts".
                "The CALLER lines are a different person in a different call. Never say a "
                "CALLER line yourself, and never assume this caller has the same patient, "
                "relative, test or problem as the example.",
                exemplars,
            ]

        return "\n".join(parts) + "\n"
