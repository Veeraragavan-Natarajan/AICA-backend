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

# Emergency outranks everything and is already in the core prompt; it is listed
# here too so an explicit emergency turn still pulls in flow 18's full playbook.
EMERGENCY_INTENT = "emergency.escalate"

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
        r"நெஞ்சு\s*வலி|chest\s*pain|மயக்க|மூச்சு|வலிப்ப|seizure|"
        r"unconscious|ரத்தம்\s*(போ|வ|கொட்)|bleeding|சுத்த\s*முடிய|108|"
        r"உயிர|தூக்கி|விழுந்துட்டா|பேச\s*முடிய|"
        r"ambulance|அம்புல|ஆம்புல|emergency|எமர்ஜென்|அவசர",
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
        "appointment.reschedule",
        r"postpone|prepone|reschedule|date\s*மாத்த|நேரம்\s*மாத்த|வேற\s*date|"
        r"அன்னைக்கு\s*வர\s*முடியா|போஸ்ட்போன்|ரீஷெட்யூல்|தேதி\s*மாத்த|"
        r"நாள்.{0,8}மாத்த|நாளுக்கு\s*மாத்த|வேற\s*நாள்|வேற\s*நேரம்",
    ),
    (
        "appointment.cancel",
        r"cancel|ரத்து|வேணாம்.*appointment|appointment.*வேணாம|கேன்சல்|கான்சல்",
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
        r"time\s*வேண|நேரம்\s*வேண|டாக்டர்.{0,8}(வேண|இருக்கா)",
    ),
    (
        "info.general",
        r"timing|visiting\s*hours|parking|canteen|wheelchair|ICU|attender|"
        r"எப்படி\s*வர|எத்தனை\s*மணி|எங்க\s*இருக்கு|விசிட்டிங்|பார்க்கிங்|டைமிங்",
    ),
]

_COMPILED_PATTERNS = [(intent, re.compile(pattern, re.IGNORECASE)) for intent, pattern in _INTENT_PATTERNS]


# Is this turn hospital business AT ALL, whatever flow it belongs to?
#
# detect_intent() returning None is NOT evidence that a caller is off-topic,
# and treating it that way is a mistake that costs a real request. The table
# above needs a specific phrasing to fire, and the twenty flows have gaps
# between them: "என் details check பண்ணுங்க" is an ordinary thing to ring a
# hospital about and matches nothing in it.
#
# So being off-topic has to be established POSITIVELY - no flow matched AND
# nothing in the turn is hospital business - and this is the second half. It is
# deliberately generous and deliberately NOT flow-specific: departments, staff,
# body and symptom words, the desk's own verbs. The two errors are not
# symmetric. Missing an off-topic turn costs nothing new (the model improvises
# from info.general, exactly as it did before this existed); deflecting a real
# request tells a caller with a genuine problem to go away.
_HOSPITAL_CONTEXT_RE = re.compile(
    r"hospital|clinic|doctor|dr\.?\s|patient|nurse|ward|ICU|OP|emergency|ambulance|"
    r"appointment|department|medicine|medical|health|surgery|operation|scan|test|report|"
    r"bill|payment|insurance|policy|record|discharge|admit|treatment|consult|checkup|"
    r"check|details|help|token|counter|desk|reception|visit|timing|pharmacy|lab|"
    r"blood|tablet|prescription|MRN|refill|referral|complaint|"
    r"ஹாஸ்பிட்டல்|ஆஸ்பத்திரி|மருத்துவ|மருந்து|டாக்டர|நர்ஸ்|பேஷண்ட|நோயாளி|"
    r"உடம்பு|உடல்|வலி|காய்ச்சல|சுகர்|ரத்த|மாத்திரை|சிகிச்சை|அறுவை|ஆபரேஷன்|"
    r"வார்டு|பரிசோதனை|ரிப்போர்ட|பில்|கட்டணம்|காப்பீ|சிகிச|செக்கப|"
    r"அப்பாயின்|டெஸ்ட|ஸ்கேன|மெடிக|நேரம்|டைமிங|விசிட",
    re.IGNORECASE,
)


def looks_like_hospital_business(text: str) -> bool:
    """Whether this turn is about the hospital at all, whatever flow it is.

    Used only to decide whether an unroutable turn is genuinely off-topic. See
    the comment above _HOSPITAL_CONTEXT_RE for why the test is positive rather
    than "detect_intent found nothing".
    """
    return bool(_HOSPITAL_CONTEXT_RE.search(text))


def detect_intent(text: str) -> str | None:
    """Return the first intent whose trigger pattern matches, or None.

    Deterministic and cheap on purpose - a voice turn cannot afford an extra
    LLM round-trip just to pick which playbook to show.
    """
    for intent, pattern in _COMPILED_PATTERNS:
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
        self._playbooks = parse_flow_playbooks(self.master_prompt_path.read_text(encoding="utf-8"))

        # is_file(), not exists(): once the register is fine-tuned in, exemplars
        # are switched off with a blank CONVERSATION_EXEMPLARS_PATH, and a blank
        # path resolves to the CWD - a directory, which exists() happily accepts.
        if self.exemplars_path is not None and self.exemplars_path.is_file():
            raw = json.loads(self.exemplars_path.read_text(encoding="utf-8"))
            self._exemplars = {
                intent: _format_exemplars(exchanges)
                for intent, exchanges in raw.items()
                if not intent.startswith("_")
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

        parts = [
            self._core,
            "## THIS CALL'S PLAYBOOK — the flow you are handling right now",
            "Wording shown is a MODEL, not a script. Say it in your own words, one point per turn.",
            playbook.body,
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
