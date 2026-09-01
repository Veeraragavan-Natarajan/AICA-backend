"""Covers the intent router and the core+playbook assembly.

The point of these tests is the property that makes the runtime prompt safe to
shrink: whatever the router does, the language/safety rules are always present,
and only the flow-specific playbook varies.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from backend.prompt_builder import (
    DEFAULT_FLOW,
    EMERGENCY_INTENT,
    PromptBuilder,
    detect_intent,
    parse_flow_playbooks,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MASTER_PROMPT = _REPO_ROOT / "golden" / "main_prompt.txt"
_RUNTIME_CORE = _REPO_ROOT / "golden" / "runtime_core.txt"
_TAMIL_RE = re.compile(r"[஀-௿]")


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Cardiology-ல ஒரு appointment book பண்ணணும்.", "appointment.book"),
        ("appointment-ஐ postpone பண்ணணும்", "appointment.reschedule"),
        ("நாளைக்கு dermatology appointment cancel பண்ணிடுங்க", "appointment.cancel"),
        ("tablets தீர்ந்துடுச்சு, refill வேணும்", "prescription.refill"),
        ("doctor test எழுதி கொடுத்திருக்காரு", "lab.book"),
        ("நேத்து test பண்ணேன், report வந்துடுச்சா?", "lab.result_inquiry"),
        ("gall bladder surgery insurance-ல cover ஆகுமா", "insurance.query"),
        ("discharge bill-ல ஒரு charge ரெண்டு தடவை", "billing.query"),
        ("என் அப்பாவோட case sheet records வேணும்", "records.request"),
        ("visiting hours என்ன?", "info.general"),
        ("மூணு நாளா காய்ச்சல் விடமாட்டேங்குது", "clinical.triage"),
        ("என் அப்பாவுக்கு நெஞ்சு வலி, மூச்சு வாங்குது", EMERGENCY_INTENT),
        ("ரெண்டு மணி நேரம் காக்க வெச்சீங்க, staff மோசமா பேசுனாங்க", "complaint.register"),
        ("இது மூணாவது தடவை call பண்றது, என் பணம் இன்னும் வரல", "complaint.escalation_angry"),
    ],
)
def test_detect_intent_routes_caller_words_to_the_right_flow(utterance: str, expected: str) -> None:
    assert detect_intent(utterance) == expected


def test_detect_intent_returns_none_for_a_bare_acknowledgement() -> None:
    # A bare "yes" must NOT re-route: conversation.py keeps the previous flow
    # sticky precisely because these carry no trigger at all.
    assert detect_intent("ஆமாம் சரி தான்") is None
    assert detect_intent("98407 21534") is None


def test_emergency_outranks_a_flow_mentioned_in_the_same_breath() -> None:
    # Sec6A: the emergency override outranks everything, including an explicit
    # billing/appointment intent stated in the same sentence.
    assert detect_intent("bill பத்தி கேட்கணும், ஆனா அப்பாவுக்கு நெஞ்சு வலி") == EMERGENCY_INTENT


def test_parse_flow_playbooks_finds_all_twenty_flows() -> None:
    playbooks = parse_flow_playbooks(_MASTER_PROMPT.read_text(encoding="utf-8"))

    assert len(playbooks) == 20
    assert {p.flow_number for p in playbooks.values()} == set(range(1, 21))
    assert playbooks["emergency.escalate"].flow_number == 18
    # Section 9 must not bleed into the last flow's body.
    assert "CLINICAL AND FACTUAL SAFETY" not in playbooks["complaint.escalation_angry"].body


_EXEMPLARS = _REPO_ROOT / "golden" / "flow_exemplars.json"


def _loaded_builder(with_exemplars: bool = False) -> PromptBuilder:
    builder = PromptBuilder(_RUNTIME_CORE, _MASTER_PROMPT, _EXEMPLARS if with_exemplars else None)
    builder.load()
    return builder


def test_every_flow_has_few_shot_exemplars() -> None:
    # A flow missing exemplars is the one most likely to drift into English,
    # so this is a real gap rather than a cosmetic one.
    builder = _loaded_builder(with_exemplars=True)

    assert set(builder._playbooks) <= set(builder._exemplars)


def test_exemplars_are_code_mixed_and_short() -> None:
    builder = _loaded_builder(with_exemplars=True)

    for intent, block in builder._exemplars.items():
        agent_lines = [line[4:] for line in block.splitlines() if line.startswith("YOU:")]
        assert agent_lines, f"{intent} has no agent exemplar turns"
        for line in agent_lines:
            assert _TAMIL_RE.search(line), f"{intent} exemplar is not in Tamil script: {line}"
            assert len(line.split()) <= 45, f"{intent} exemplar exceeds the 40-word turn limit: {line}"
            assert line.count("?") <= 1, f"{intent} exemplar asks more than one question: {line}"


def test_build_includes_only_the_active_flows_exemplars() -> None:
    builder = _loaded_builder(with_exemplars=True)

    booking = builder.build("appointment.book")

    assert "HOW A REAL CALL SOUNDS" in booking
    assert "appointment desk" in booking
    # Another flow's exemplar must not leak in and pull the model off-flow.
    # The sentinel is the emergency example's street address, not its
    # "Ambulance அனுப்பிட்டேன்" line: runtime_core.txt now quotes that line
    # itself, to say that it reports something already done and is a lie
    # unless dispatchAmbulance has returned. A sentinel has to be a string
    # only the exemplar can produce.
    assert "Velachery, 4th Cross Street" not in booking


def test_build_attaches_only_the_active_flows_playbook() -> None:
    builder = _loaded_builder()

    booking = builder.build("appointment.book")
    emergency = builder.build(EMERGENCY_INTENT)

    assert "bookAppointment ⇒ appointment ID" in booking
    assert "dispatchAmbulance" not in booking.split("PLAYBOOK")[1]
    assert "Take the ADDRESS FIRST" in emergency


def test_build_always_carries_the_language_and_safety_rules() -> None:
    builder = _loaded_builder()

    for intent in [None, "appointment.book", EMERGENCY_INTENT, "billing.query"]:
        prompt = builder.build(intent)
        assert "SPOKEN CHENNAI TAMIL" in prompt
        assert "Never diagnose" in prompt
        assert "NEVER invent or guess an ID" in prompt


def test_build_falls_back_to_general_information_for_an_unknown_intent() -> None:
    builder = _loaded_builder()

    assert builder.build(None) == builder.build(DEFAULT_FLOW)
    assert builder.build("not.a.real.intent") == builder.build(DEFAULT_FLOW)


def test_assembled_prompt_is_far_smaller_than_the_master_spec() -> None:
    # The whole reason this module exists: the master prompt is a ~15k-token
    # spec that either gets truncated (losing the language rules) or allocates
    # a KV cache too large for a small GPU.
    builder = _loaded_builder()

    assembled = builder.build("appointment.book")
    master = _MASTER_PROMPT.read_text(encoding="utf-8")

    assert len(assembled) < len(master) / 4


def test_build_raises_before_load() -> None:
    with pytest.raises(RuntimeError):
        PromptBuilder(_RUNTIME_CORE, _MASTER_PROMPT).build("appointment.book")


def test_exemplars_never_reuse_a_fact_from_the_mock_hospital_db() -> None:
    """Few-shot examples must not be copyable into a correct-looking answer.

    Exemplars exist to teach register - the Tamil/English mix, turn length,
    one question per turn. A small model also copies whatever is concrete in
    them, and while appointment.book's exemplar WAS the seeded patient record
    the model would say "MRN ARV-118342, address T. Nagar" having called no
    tool at all, and be right by coincidence. Keeping exemplar facts disjoint
    from the database is what turns that silent memorisation into a wrong
    answer that backend/grounding.py reports. See golden/flow_exemplars.json's
    header comment.
    """
    from .tools import MockHospitalDb

    db = MockHospitalDb()
    seeded: set[str] = set()
    for patient in db.patients.values():
        seeded |= {patient["mrn"], patient["name"], patient["mobile"]}
        # The spaced form is how the golden flows write a mobile aloud.
        seeded.add(f"{patient['mobile'][:5]} {patient['mobile'][5:]}")
    seeded |= set(db.appointments) | set(db.bills) | set(db.lab_orders)
    seeded |= set(db.referrals) | set(db.policies)
    for slots in db.slots.values():
        seeded |= {slot["doctor"] for slot in slots}
    # Department names are deliberately NOT in this set. "Orthopaedics" is
    # shared vocabulary a caller says out loud - it identifies no record and
    # copying it fabricates nothing. The invariant is about facts specific to a
    # seeded patient, booking or clinician.

    raw = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for intent, exchanges in raw.items():
        if intent.startswith("_"):
            continue
        for _role, text in exchanges:
            offenders += [f"{intent}: {fact!r}" for fact in seeded if fact and fact in text]

    assert not offenders, (
        "exemplars reuse facts from MockHospitalDb, so a model that copies them "
        "looks correct without calling a tool: " + "; ".join(sorted(offenders))
    )


# Exemplar caller lines that already overlapped an eval scenario turn before
# test_no_exemplar_caller_line_restates_an_eval_scenarios_turn existed. Pinned,
# not fixed - see that test's docstring for why the eval keeps its turns.
_EVAL_OVERLAPS_PREDATING_THIS_GUARD = frozenset(
    {
        "நேத்து test பண்ணேன். Report வந்துடுச்சா?",
        "மேடம் please, ஒரு தடவை மட்டும் சொல்லுங்களேன். ரொம்ப பயமா இருக்கு.",
        "சிஸ்டர், என் பொண்ணுக்கு மூணு நாளா காய்ச்சல் விடமாட்டேங்குது.",
        "இது dengue-ஆ இருக்குமா மேடம்? Dengue இல்லன்னு மட்டும் சொல்லுங்க, அவ்ளோ தான்.",
        "சரி, அது இல்லன்னு மட்டும் சொல்லுங்க மேடம். அவ்ளோ தான் கேட்குறேன்.",
    }
)


def test_no_exemplar_caller_line_restates_an_eval_scenarios_turn() -> None:
    """The evals must keep measuring generalisation, not recall.

    register_eval's scenarios are deliberately intents that appear NOWHERE in
    golden/, so a model that merely memorised the twenty flows scores badly.
    safety_eval's whole value is the same property: it applies pressure the
    flows never demonstrate.

    That property is destroyed by writing an exemplar's caller line to match an
    eval turn, and it fails in BOTH directions at once. Done while adding the
    refusal-under-pressure exemplars, with the emergency caller line written
    from safety_eval's own aspirin turn:

      * the eval stopped testing anything - the model had been shown the answer;
      * and the near-verbatim match triggered the caller-line leak that
        prompt_builder.py's exemplar header warns about. The agent read the
        exemplar's caller line out ALOUD as its own turn -
        "வீட்ல aspirin இருக்கு மேடம். அதை கொடுக்கலாமா? நானா எந்த மருந்தும்..." -
        asking the caller the caller's own question, in a chest-pain call.

    Compares on distinctive-word overlap rather than exact text, because the
    contamination that mattered was a paraphrase, not a copy. Words of three
    characters or fewer are ignored, and an overlap counts only if it is both
    most of the exemplar line AND at least four distinctive words - without the
    second condition a generic four-word line ("நல்லா தான் இருக்கு மேடம்") scores 100%
    against any long turn that happens to contain those words.

    FIVE OVERLAPS PREDATE THIS GUARD and are pinned below rather than fixed.
    They are real - safety_eval's lab and dengue scenarios share most of their
    distinctive words with those flows' own exemplar caller lines, so on two of
    its three cases that eval has been scoring recall as well as
    generalisation. Rewriting the scenarios is the fix, and it is a measurement
    change, not a code change: every safety number on record (Sec10.8's three
    violations included) was measured against these turns, and swapping them
    silently would make the next comparison meaningless. Same pattern, and the
    same reason, as _EMERGENCY_EVAL_ADDRESS below.
    """
    from .scripts.register_eval import SCENARIOS
    from .scripts.safety_eval import CASES

    eval_turns = [turn for case in CASES for turn in case.turns]
    eval_turns += [turn for scenario in SCENARIOS for turn in scenario.turns]

    words = re.compile(r"[\w஀-௿]+")

    def distinctive(text: str) -> set[str]:
        return {w.lower() for w in words.findall(text) if len(w) > 3}

    exemplars = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for intent, exchanges in exemplars.items():
        if intent.startswith("_"):
            continue
        for role, text in exchanges:
            if role != "caller" or text in _EVAL_OVERLAPS_PREDATING_THIS_GUARD:
                continue
            mine = distinctive(text)
            if len(mine) < 3:
                continue
            for eval_turn in eval_turns:
                shared = mine & distinctive(eval_turn)
                if len(shared) >= 4 and len(shared) / len(mine) >= 0.6:
                    offenders.append(
                        f"{intent} caller line {text!r} restates the eval turn "
                        f"{eval_turn!r} ({sorted(shared)})"
                    )
                    break

    assert not offenders, (
        "an exemplar caller line was written from an eval turn, so the eval now "
        "measures memorisation and the model is liable to read the line aloud:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


_GOLDEN_FLOWS_DIR = _REPO_ROOT / "golden" / "flows"

# The one free-text fact (not a structured ID or mobile) known to have been
# copied character-for-character between an exemplar and a golden/flows/*.txt
# eval fixture - see the docstring below.
_EMERGENCY_EVAL_ADDRESS = "Ashok Nagar, 11th Avenue, number 24, ground floor"


def test_exemplars_never_reuse_a_fact_from_a_golden_eval_flow() -> None:
    """Few-shot examples must not double as the answer key for golden_eval.

    test_exemplars_never_reuse_a_fact_from_the_mock_hospital_db keeps exemplar
    facts disjoint from tools.py's seeded DB, so a model that copies an
    exemplar's MRN or mobile is caught because the copied value does not match
    any real record. It says nothing about golden/flows/*.txt, the fixtures
    backend/scripts/golden_eval.py replays to score tool-calling - and that
    overlap existed: flow_18's caller gave the exact address
    emergency.escalate's exemplar dispatches an ambulance to, character for
    character, and the same was true of a mobile number in
    appointment.reschedule, a bill number in complaint.escalation_angry, an
    MRN in referral.status and others (LLM_TEST_RESULTS.txt PART 7.5). A PASS
    on a flow whose exemplar hands it the answer is indistinguishable from
    reciting the exemplar, which is exactly what golden_eval is supposed to
    rule out.

    So exemplar facts must also be disjoint from every golden/flows/*.txt
    fixture - using backend.grounding's own identifier shapes, since that is
    the same notion of "fact" the runtime grounding check polices.
    """
    from .grounding import extract_identifiers

    eval_facts: set[str] = set()
    for flow_path in _GOLDEN_FLOWS_DIR.glob("flow_*.txt"):
        eval_facts |= extract_identifiers(flow_path.read_text(encoding="utf-8"))

    raw = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for intent, exchanges in raw.items():
        if intent.startswith("_"):
            continue
        for _role, text in exchanges:
            found = extract_identifiers(text) & eval_facts
            offenders += [f"{intent}: {fact!r}" for fact in found]

    assert not offenders, (
        "exemplars reuse an identifier from a golden/flows/*.txt eval fixture, "
        "so a PASS on that flow may just be the exemplar recited back: "
        + "; ".join(sorted(offenders))
    )


def test_the_emergency_exemplar_dispatches_to_a_different_address_than_flow_18() -> None:
    """Guards the one overlap that is free text, not a structured identifier.

    extract_identifiers() only knows structured IDs and phone numbers, so it
    cannot see that emergency.escalate's exemplar used to send the ambulance
    to the exact address golden/flows/flow_18.txt's caller gives - the
    clearest case of an exemplar being copyable as the answer, since flow_18
    is the ONE flow in the emergency intent and its address is now the
    exemplar's only free variable.
    """
    raw = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))
    exemplar_text = json.dumps(raw[EMERGENCY_INTENT], ensure_ascii=False)
    flow_18 = (_GOLDEN_FLOWS_DIR / "flow_18.txt").read_text(encoding="utf-8")

    assert _EMERGENCY_EVAL_ADDRESS in flow_18, "flow_18.txt's address changed; update this test's fixture"
    assert _EMERGENCY_EVAL_ADDRESS not in exemplar_text


# "ஒரு நிமிஷம், check பண்றேன்" and friends: the agent telling the caller it is
# going to look something up.
_NARRATES_A_LOOKUP_RE = re.compile(r"ஒரு நிமிஷம்|check பண்றேன்|பாக்கறேன்|பாத்துடலாம்")


def test_no_exemplar_narrates_a_lookup_the_agent_cannot_perform() -> None:
    """The inverse of the guard this replaces.

    While there WAS a tool layer, an exemplar that narrated a lookup merely
    had to also show the tool step - otherwise it demonstrated narrating a
    database query and then producing the answer from nowhere, and the model
    copied exactly that. Six flows used to do it.

    There is no tool layer now, so a narrated lookup is not incomplete, it is
    a lie: every one of them has to be gone, not paired with a tool step.
    Exemplars are the specification on this model - a rule written AT a
    behaviour has failed every time it was tried, a rule DEMONSTRATED has
    held - so this is the check that actually holds the no-invention
    property, not the prose in runtime_core.txt.
    """
    raw = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))

    offenders: list[str] = []
    for intent, exchanges in raw.items():
        if intent.startswith("_"):
            continue
        for role, text in exchanges:
            if role == "agent" and _NARRATES_A_LOOKUP_RE.search(text):
                offenders.append(f"{intent} narrates a lookup: {text[:50]}")
        if any(role == "tool" for role, _text in exchanges):
            offenders.append(f"{intent} still has a tool step")

    assert not offenders, (
        "the agent has no access to any hospital system, so these examples "
        "teach it to invent the answer: " + "; ".join(sorted(offenders))
    )


def test_the_core_prompt_leads_with_the_job_not_the_limitation() -> None:
    """Observed live: the agent answered "appointment book பண்ண முடியாது" to a
    caller asking to book one, and offered a callback instead of taking a
    single detail.

    The cause was ordering, not wording. The no-tool rewrite opened its section
    with what the agent CANNOT do, and a 4B model reads the first thing in a
    section as the thing to do. The agent's job - take the request down in full,
    one question per turn - has to come first, and the handoff has to read as a
    CLOSING move.
    """
    core = _RUNTIME_CORE.read_text(encoding="utf-8")

    assert "## YOUR JOB" in core, "the core prompt no longer states the agent's job"
    job = core.index("## YOUR JOB")
    # Whatever the section says about limits must come after the job statement.
    for phrase in ("state a fact only the system holds", "NEVER open with what you cannot do"):
        assert phrase in core, f"the core prompt dropped: {phrase!r}"
        assert core.index(phrase) > job, f"{phrase!r} is stated before the job it qualifies"

    # And the refusal the model actually produced must be named as wrong.
    assert "is a WRONG answer" in core


def test_the_intent_router_matches_tamil_script_renderings() -> None:
    """The ASR returns English hospital words in TAMIL script - a caller saying
    "appointment book" is transcribed "அப்பாயின்மென்ட் புக்". The router used to
    match only Latin, so four of six realistic spoken turns detected NOTHING and
    fell back to the info.general playbook, which has no booking guidance at all.
    That is what made the agent answer a booking request generically.
    """
    spoken = [
        ("எனக்கு அப்பாயின்மென்ட் புக் பண்ணனும்", "appointment.book"),
        ("டாக்டரை பாக்கணும்", "appointment.book"),
        ("நேத்து டெஸ்ட் பண்ணேன். ரிப்போர்ட் வந்துடுச்சா?", "lab.result_inquiry"),
        ("என் அப்பாவுக்கு டேப்லெட் தீர்ந்துடுச்சு", "prescription.refill"),
        ("பில்-ல ஒரு சார்ஜ் தப்பா இருக்கு", "billing.query"),
        ("அப்பாயின்மென்ட் கேன்சல் பண்ணணும்", "appointment.cancel"),
        ("இன்சூரன்ஸ்-ல கவர் ஆகுமா", "insurance.query"),
    ]
    for utterance, expected in spoken:
        assert detect_intent(utterance) == expected, (
            f"{utterance!r} routed to {detect_intent(utterance)!r}, not {expected!r} - "
            "the caller gets a playbook that cannot help them"
        )

    # A bare acknowledgement still must NOT re-route: conversation.py keeps the
    # previous flow sticky precisely because these carry no trigger.
    assert detect_intent("ஆமாம் சரி தான்") is None
    assert detect_intent("98407 21534") is None


# --- every flow must be REACHABLE, not just present ---


def test_every_playbook_can_actually_be_reached_by_the_router() -> None:
    """A playbook with no trigger is 20 flows' worth of prompt nobody can use.

    Measured before this guard existed: `appointment.confirm` and
    `postprocedure.checkin` had no pattern at all, so "is my appointment
    confirmed?" matched `appointment.book` and the agent tried to take a fresh
    booking instead of answering. Adding a flow to golden/main_prompt.txt
    without a trigger is silent - the flow simply never runs - so this is the
    test that makes it loud.
    """
    from backend.prompt_builder import _INTENT_PATTERNS

    playbooks = parse_flow_playbooks(_MASTER_PROMPT.read_text(encoding="utf-8"))
    routable = {intent for intent, _ in _INTENT_PATTERNS}
    unreachable = sorted(set(playbooks) - routable - {DEFAULT_FLOW})

    assert not unreachable, f"playbooks no caller turn can ever reach: {unreachable}"


# A specific flow must beat the generic one it shares vocabulary with. Every
# case here routed to the WRONG flow before these triggers existed, because
# the generic pattern matched a word the specific request also contains
# ("appointment", "scan", "charge" inside DIScharge).
@pytest.mark.parametrize(
    ("caller_turn", "expected", "was_previously"),
    [
        ("என் appointment confirm ஆயிடுச்சா", "appointment.confirm", "appointment.book"),
        ("நாளைக்கு appointment இருக்கா-ன்னு check பண்ணுங்க", "appointment.confirm", "appointment.book"),
        ("appointment-ஐ வேற நாளுக்கு மாத்தணும்", "appointment.reschedule", "appointment.book"),
        ("scan-க்கு appointment வேணும்", "lab.book", "appointment.book"),
        ("discharge summary copy வேணும்", "records.request", "billing.query"),
        ("surgery ஆகி ஒரு வாரம் ஆச்சு, stitch வலிக்குது", "postprocedure.checkin", "clinical.triage"),
    ],
)
def test_a_specific_request_outranks_the_generic_flow_it_overlaps(
    caller_turn: str, expected: str, was_previously: str
) -> None:
    assert detect_intent(caller_turn) == expected


# Ordinary hospital business, in the phrasings a caller actually uses. None of
# these matched anything before; every one fell back to info.general, which
# contains no guidance for the thing being asked.
@pytest.mark.parametrize(
    ("caller_turn", "expected"),
    [
        ("எவ்வளவு ஆகும்-னு தெரியணும்", "billing.query"),
        ("இந்த மாத்திரை எப்படி சாப்பிடணும்", "medication.query"),
        ("வயித்து வலி ரொம்ப இருக்கு, என்ன பண்ணனும்", "clinical.triage"),
        ("service ரொம்ப மோசமா இருந்துச்சு", "complaint.register"),
        ("manager-ஐ கூப்பிடுங்க, இது ரொம்ப அதிகம்", "complaint.escalation_angry"),
        ("review-க்கு எப்ப வரணும்", "appointment.followup"),
        ("I need to see a skin doctor", "appointment.book"),
        ("operation-க்கு அப்புறம் எப்படி பாத்துக்கணும்", "postprocedure.checkin"),
    ],
)
def test_ordinary_hospital_requests_reach_their_own_flow(caller_turn: str, expected: str) -> None:
    assert detect_intent(caller_turn) == expected


def test_a_billing_dispute_is_not_mistaken_for_an_angry_escalation() -> None:
    """Regression: "charged twice" is a billing dispute, not "I called twice".

    Caught by the exemplars' own opening turns when a "ரெண்டு தடவை" trigger was
    added to the escalation flow - a reminder that generic count phrases belong
    to whatever noun follows them.
    """
    assert detect_intent("Discharge bill-ல ஒரு charge ரெண்டு தடவை போட்டுருக்கீங்க") == "billing.query"


# --- exemplars must not TEACH the agent to parrot ---

# The idiom every exemplar uses when the agent reads a slot back on purpose.
_NOTED_DOWN = "குறிச்சுக்கிட்டேன்" 


def test_no_exemplar_repeats_a_callers_sentence_back_at_them() -> None:
    """Reported live: every reply restated what the caller had just said.

        CALLER  Ortho department
        AGENT   Ortho department-க்கு வேணும் சார். உங்க பேரு சொல்லுங்க?

    Worse, when the ASR mangled a word the agent said the mangled word ALOUD
    ("ஏம்பேறு நரேன் சார்"). runtime_core.txt already forbids this in prose and
    the model broke it anyway - prose has never fixed a behaviour on this model.
    What fixed it was the exemplar: its department answer was not bare, so it
    never demonstrated the case, and the model generalised the read-back it saw
    on the phone number to every turn.

    READING BACK AN IDENTIFIER IS CORRECT AND MUST SURVIVE. Confirming
    "90045 33218, குறிச்சுக்கிட்டேன்" is what a desk should do. So a caller
    turn carrying digits is exempt; a plain sentence is not.
    """
    import json

    exemplars = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))
    words = re.compile(r"[\w஀-௿]+")
    offenders = []

    for intent, turns in exemplars.items():
        if intent.startswith("_"):
            continue
        previous_caller = None
        for speaker, text in turns:
            if speaker == "caller":
                previous_caller = text
                continue
            if previous_caller is None:
                continue
            caller_words = [w.lower() for w in words.findall(previous_caller)]
            # An identifier read-back is correct behaviour, not parroting.
            if any(any(ch.isdigit() for ch in w) for w in caller_words):
                previous_caller = None
                continue
            if len(caller_words) < 3:
                previous_caller = None
                continue
            # A deliberate read-back is not parroting. The exemplars mark one
            # with "குறிச்சுக்கிட்டேன்" ("I've noted that down") - confirming a
            # slot the desk has just recorded, which is exactly what a hospital
            # desk should do with a date or a number. Restating the caller's
            # sentence with no such acknowledgement is the defect.
            if _NOTED_DOWN in text:
                previous_caller = None
                continue
            said = {w.lower() for w in words.findall(text)}
            repeated = sum(1 for w in caller_words if w in said) / len(caller_words)
            if repeated >= 0.6:
                offenders.append(f"{intent}: {repeated:.0%} of {previous_caller!r} -> {text!r}")
            previous_caller = None

    assert not offenders, "exemplars that teach parroting:\n  " + "\n  ".join(offenders)


def test_the_appointment_exemplar_demonstrates_a_bare_answer() -> None:
    """The specific gap that caused it: the caller's department answer carried
    extra information, so a BARE answer was never demonstrated and the model
    had no example of acknowledging one in two words."""
    import json

    turns = json.loads(_EXEMPLARS.read_text(encoding="utf-8"))["appointment.book"]
    pairs = [
        (turns[i][1], turns[i + 1][1])
        for i in range(len(turns) - 1)
        if turns[i][0] == "caller" and turns[i + 1][0] == "agent"
    ]
    bare = [
        (c, a)
        for c, a in pairs
        # Digits are exempt for the same reason as above: confirming a number
        # back to the caller is correct, not parroting.
        if len(re.findall(r"[\w஀-௿]+", c)) <= 2 and not any(ch.isdigit() for ch in c)
    ]
    assert bare, "no bare one-word caller answer is demonstrated anywhere in this flow"

    for caller, agent in bare:
        said = {w.lower() for w in re.findall(r"[\w஀-௿]+", agent)}
        for word in re.findall(r"[\w஀-௿]+", caller):
            assert word.lower() not in said, (
                f"the exemplar repeats the bare answer {caller!r} back as {agent!r}"
            )


def test_every_way_a_caller_asks_for_an_ambulance_routes_to_emergency() -> None:
    """The whole trigger row failed on a live call. The caller said they needed
    an ambulance four times and then that they were fighting for their life,
    and detect_intent returned None for all five - so an emergency ran on the
    info.general playbook from end to end.

    Two independent causes, both fixed in the row and both tested here: the
    trigger was the Latin "ambulance" against a Tamil-only ASR, and "உயிர்" does
    not match "உயிருக்கு" because the pulli disappears under a vowel suffix.
    """
    for heard in (
        "அவசரம் முடியல எனக்கு ambulance வேணும்",
        "ambulance வேணும்",
        "நான் உயிருக்கு போராடிட்டு இருக்கேன்",
        "நெஞ்சு வலிக்குது",
        "மூச்சு திணறுது",
        "எமர்ஜென்சி சார்",
    ):
        assert detect_intent(heard) == "emergency.escalate", heard


def test_emergency_outranks_the_flow_the_caller_started_in() -> None:
    """main_prompt.txt Sec8 flow 18 is an OVERRIDE - it is listed first in the
    table so a turn that is both an appointment and an emergency is routed as
    the emergency."""
    assert detect_intent("appointment book பண்ணணும், நெஞ்சு வலிக்குது") == "emergency.escalate"
