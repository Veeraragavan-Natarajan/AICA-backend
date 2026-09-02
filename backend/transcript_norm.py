"""Normalise what the Tamil ASR hears back into how the agent writes.

IndicConformer is a Tamil model with a Tamil character vocabulary, so it can
only ever emit Tamil script. A caller saying an English hospital word therefore
comes back transliterated - "appointment" as "அப்பாயின்மென்ட்", "department" as
"டிபார்ட்மெண்ட்" - and a dictated phone number comes back as Tamil NUMBER WORDS
rather than digits.

Measured, real model, ten phrases synthesised and transcribed round-trip:

    SAID   Cardiology-ல ஒரு appointment book பண்ணணும்
    HEARD  ஒரு அப்பாயிண்ட்மெண்ட் புக் பண்ணணும்
    SAID   என் bill-ல ஒரு charge தப்பா இருக்கு
    HEARD  என் பில்ல ஒரு சார்ஜ் தப்பா இருக்கு
    SAID   என் mobile number 98407 21534
    HEARD  மொபைல் நம்பர் ஒன்பது எட்டு நான்கு பூஜ்ஜியம் ஏழு இரண்டு ஒன்று ஐந்து மூன்று நான்கு

Three things go wrong downstream if that is passed through untouched:

  1. The transcript shown to a human reads as mangled Tamil, not as the
     code-mix that was actually spoken.
  2. The model sees a register that does not match the one it is asked to
     produce - every exemplar writes English hospital words in Latin script.
  3. A phone number spelled out in words is not a phone number. Nothing
     downstream can read it back to the caller or write it down.

The vocabulary here is CLOSED - it is a hospital desk, not open dictation - so
a lookup table is the right tool and a fuzzy matcher is not: fuzzy matching
over Tamil script would eventually mangle a real Tamil word, which is a far
worse failure than leaving one English word transliterated.

Only exact, whole-word matches are rewritten. Anything not in the table is left
exactly as the ASR produced it.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re

logger = logging.getLogger("aica.transcript_norm")

# Tamil-script renderings -> the Latin spelling the agent itself uses.
#
# Several spellings per word on purpose: the ASR is not consistent about the
# pulli (்) or about ண/ன, and emitted "அப்பாயின்மென்ட்", "அப்பாயிண்ட்மெண்ட்" and
# "அப்பாயின்ட்மெண்ட்" for the same word across three runs. Every variant listed
# was either observed in a round-trip or is the same word with the one
# character the model flips.
_ENGLISH_WORDS: dict[str, str] = {
    # observed in round-trip transcription
    "அப்பாயின்மென்ட்": "appointment",
    "அப்பாயின்மெண்ட்": "appointment",
    "அப்பாயின்ட்மென்ட்": "appointment",
    "அப்பாயின்ட்மெண்ட்": "appointment",
    "அப்பாயிண்ட்மென்ட்": "appointment",
    "அப்பாயிண்ட்மெண்ட்": "appointment",
    "புக்": "book",
    "டிபார்ட்மென்ட்": "department",
    "டிபார்ட்மெண்ட்": "department",
    "டாக்டர்": "doctor",
    "பில்": "bill",
    "பில்ல": "bill-ல",
    "சார்ஜ்": "charge",
    "டெஸ்ட்": "test",
    "ப்ளூட்": "blood",
    "ப்ளட்": "blood",
    "மொபைல்": "mobile",
    "நம்பர்": "number",
    "இன்ஷூரன்ஸ்": "insurance",
    "இன்சூரன்ஸ்": "insurance",
    "கவர்": "cover",
    # the rest of the desk's vocabulary, same transliteration rules
    "ரிப்போர்ட்": "report",
    "ரிசல்ட்": "result",
    "ஸ்கேன்": "scan",
    "டேப்லெட்": "tablet",
    "டாப்லெட்": "tablet",
    "ரீஃபில்": "refill",
    "ரீபில்": "refill",
    "கேன்சல்": "cancel",
    "கான்சல்": "cancel",
    "ஸ்லாட்": "slot",
    "கன்சல்ட்": "consult",
    "ரெக்கார்ட்": "record",
    "பாலிசி": "policy",
    "கிளைம்": "claim",
    "ரெஜிஸ்டர்": "register",
    "ரெஃபரல்": "referral",
    "டிஸ்சார்ஜ்": "discharge",
    "ஃபாலோஅப்": "follow-up",
    "ரிவ்யூ": "review",
    "கம்ப்ளைண்ட்": "complaint",
    "பேஷண்ட்": "patient",
    "ஆப்பரேஷன்": "operation",
    "சர்ஜரி": "surgery",
    # EVERY department, not the five the exemplars happen to use.
    #
    # The router's department fallback (prompt_builder.names_a_department) and
    # the model both see Latin department names; the ASR can only emit Tamil
    # script. A department missing from here reaches the router as Tamil the
    # patterns do not know, and the caller is told the desk cannot help with
    # booking - the exact failure this MVP exists to not have.
    #
    # Two or three spellings each, for the same reason as "appointment" above:
    # this ASR is not consistent about the pulli, ண/ன, or ா/அ.
    "கார்டியாலஜி": "Cardiology",
    "கார்டியாலாஜி": "Cardiology",
    "காத்யாலோஜி": "Cardiology",
    "கார்டியாலஜிஸ்ட்": "Cardiologist",
    "கார்டியாலாஜிஸ்ட்": "Cardiologist",
    "டெர்மட்டாலஜி": "Dermatology",
    "டெர்மடாலஜி": "Dermatology",
    "டெர்மட்டாலோஜி": "Dermatology",
    "நியூராலஜி": "Neurology",
    "நியூரோலஜி": "Neurology",
    "நியூரோ": "Neuro",
    "ஆர்த்தோ": "Ortho",
    "ஆர்த்தோபெடிக்": "Orthopaedics",
    "ஆர்த்தோபீடிக்": "Orthopaedics",
    "ஆர்த்தோபெடிக்ஸ்": "Orthopaedics",
    "ஆர்த்தோபீடிக்ஸ்": "Orthopaedics",
    "பீடியாட்ரிக்": "Paediatrics",
    "பீடியாட்ரிக்ஸ்": "Paediatrics",
    "பீடியாட்ரிஷியன்": "Paediatrician",
    "கைனகாலஜி": "Gynaecology",
    "கைனாகாலஜி": "Gynaecology",
    "கைனக்": "Gynaec",
    # Human-microphone variants for "gynaecologist". The first form was
    # observed live; the others are the same conservative whole-word spelling
    # wobble this Tamil-only ASR produces for ன/னோ and the final pulli.
    "மினோகாலஜிஸ்ட்": "Gynaecologist",
    "கைனகாலஜிஸ்ட்": "Gynaecologist",
    "கைனாகாலஜிஸ்ட்": "Gynaecologist",
    "கைனோகாலஜிஸ்ட்": "Gynaecologist",
    "கினோகாலஜிஸ்ட்": "Gynaecologist",
    "ஈஎன்டி": "ENT",
    "இஎன்டி": "ENT",
    "டென்டல்": "Dental",
    "டெண்டல்": "Dental",
    "டென்டிஸ்ட்": "Dentist",
    "ஆப்தல்மாலஜி": "Ophthalmology",
    "ஆப்தால்மாலஜி": "Ophthalmology",
    "யூராலஜி": "Urology",
    "யூரோலஜி": "Urology",
    "காஸ்ட்ரோ": "Gastro",
    "கேஸ்ட்ரோ": "Gastro",
    "காஸ்ட்ரோஎன்ட்ராலஜி": "Gastroenterology",
    "பல்மனாலஜி": "Pulmonology",
    "பல்மோனாலஜி": "Pulmonology",
    "நெஃப்ராலஜி": "Nephrology",
    "நெஃப்ரோலோஜி": "Nephrology",
    "சைக்யாட்ரி": "Psychiatry",
    "சைகியாட்ரி": "Psychiatry",
    "ஆன்காலஜி": "Oncology",
    "ஆங்காலஜி": "Oncology",
    "என்டோக்ரினாலஜி": "Endocrinology",
    "டயாபடாலஜி": "Diabetology",
    "பிசியோதெரபி": "Physiotherapy",
    "பிசியோ": "Physio",
    "டயட்டீஷியன்": "Dietician",
    "டயட்டிஷியன்": "Dietician",
    "ஸ்பெஷலிஸ்ட்": "specialist",
    "ஸ்பெஷாலிட்டி": "speciality",
    # The rest of the five flows' own vocabulary, same rules. reschedule and
    # its synonyms matter as much as the departments: they are the ONLY trigger
    # appointment.reschedule has that a caller reliably says in English.
    "ரீஷெட்யூல்": "reschedule",
    "ரீஸ்கெஜூல்": "reschedule",
    "ரீஷெட்யூள்": "reschedule",
    "போஸ்ட்போன்": "postpone",
    "போஸ்ட்கோன்": "postpone",
    "ப்ரீபோன்": "prepone",
    "வீல்சேர்": "wheelchair",
    "வீல்செயர்": "wheelchair",
    "விசிட்டிங்": "visiting",
    "விசிடிங்": "visiting",
    "விஸிட்டிங்": "visiting",
    "அவர்ஸ்": "hours",
    "ஹவர்ஸ்": "hours",
    "அவுர்ஸ்": "hours",
    "டைமிங்": "timing",
    "பார்க்கிங்": "parking",
    "கேன்டீன்": "canteen",
    "எமர்ஜென்சி": "emergency",
    "அட்மிஷன்": "admission",
    # Observed live over the microphone, not through the TTS round-trip, and
    # the difference matters: the generated lexicon below learned "ambulance"
    # as "அண்டிலின்ஸ்" because that is what the ASR does to the TTS voice
    # saying it. A person saying it produces none of these forms, so an
    # emergency caller's word for "ambulance" reached prompt_builder.py's
    # EMERGENCY trigger table still in Tamil script and matched nothing at all.
    # Three spellings because three consecutive turns of one call produced
    # three (the ASR flips ல/லா and அ/ஆ here the way it flips ண/ன elsewhere).
    "அம்புலான்ஸ்": "ambulance",
    "அம்புலன்ஸ்": "ambulance",
    "ஆம்புலன்ஸ்": "ambulance",
    "ஹாஸ்பிட்டல்": "hospital",
    "ஹாஸ்பிடல்": "hospital",
    "ஹாஸ்படல்": "hospital",
    "ஆஸ்பிட்டல்": "hospital",
    "ஆஸ்பிடல்": "hospital",
    "ப்ரீ": "free",
    # Calendar words observed (or expected by the same Tamil-only ASR). These
    # stay in Latin script so the LLM sees the same register as the prompt and
    # can distinguish a date from an ordinary Tamil number word.
    "மண்டே": "Monday",
    "ட்யூஸ்டே": "Tuesday",
    "டியூஸ்டே": "Tuesday",
    "வென்ஸ்டே": "Wednesday",
    "வெட்னஸ்டே": "Wednesday",
    "தர்ஸ்டே": "Thursday",
    "தேர்ஸ்டே": "Thursday",
    "ஃப்ரைடே": "Friday",
    "ப்ரைடே": "Friday",
    "ப்ரிட்": "Friday",
    "சாட்டர்டே": "Saturday",
    "சாடர்டே": "Saturday",
    "சண்டே": "Sunday",
    "ஜனவரி": "January",
    "ஃபெப்ரவரி": "February",
    "பெப்ரவரி": "February",
    "மார்ச்": "March",
    "ஏப்ரல்": "April",
    "மே": "May",
    "ஜூன்": "June",
    "ஜூலை": "July",
    "ஆகஸ்ட்": "August",
    "செப்டம்பர்": "September",
    "அக்டோபர்": "October",
    "நவம்பர்": "November",
    "டிசம்பர்": "December",
    "ஃபர்ஸ்ட்": "1st",
    "பர்ஸ்ட்": "1st",
    "செகண்ட்": "2nd",
    "தேர்ட்": "3rd",
    "ஃபோர்த்": "4th",
    "பிஃப்த்": "5th",
    "ஃபிப்த்": "5th",
    "சிக்ஸ்த்": "6th",
    "ஸிக்ஸ்த்": "6th",
    "செவன்த்": "7th",
    "எய்த்": "8th",
    "நைந்த்": "9th",
    "டென்த்": "10th",
}

# Generated entries shorter than this are dropped. See the merge below.
_MIN_GENERATED_KEY_LEN = 6

# Generated coverage, merged UNDER the table above.
#
# The hand table is small, and "small" is the real complaint: a caller who says
# a word nobody typed into it hears it come back as mangled Tamil.
# backend/scripts/build_asr_lexicon.py grows it WITHOUT anyone maintaining a
# list - it takes every Latin word out of golden/ (the prompt, the exemplars,
# the flow transcripts), speaks each one with the agent's own Tamil voice,
# transcribes it with the caller's own ASR, and records what came back. Add a
# department to the prompt and it is covered on the next build.
#
# Two properties make this safe to merge blindly, and both matter:
#
#   1. It is still EXACT, whole-word matching. HANDOFF.md Sec6c measured every
#      fuzzy/transliterating alternative and all of them corrupted ordinary
#      Tamil - "சொல்லுங்க" (tell me) came back as "silence". Going forwards
#      (English -> expected Tamil form) instead of backwards means a form no
#      English source produced can never be matched at all.
#   2. The builder REJECTS any generated form that collides with real Tamil
#      appearing in golden/, which is what stops the romanised-Tamil words in
#      the prompt ("aamaam", "anga") from teaching this to rewrite ஆமாம்.
#
# The hand table wins every conflict: these entries can only add coverage, and
# a missing or malformed file simply means the hand table alone, which is
# exactly the behaviour before this existed.
#
# PROPERTY 2 IS NOT ENOUGH ON ITS OWN, and _MIN_GENERATED_KEY_LEN is what makes
# up the difference. golden/ is a hospital prompt, not a Tamil dictionary, so a
# generated form only has to avoid the few thousand words that happen to appear
# in it. Auditing the 771 shipped entries found ordinary Tamil among the short
# ones, mapped to English words it has nothing to do with:
#
#     கீழ்  -> keep      (கீழ் is "below")
#     சூழ்  -> phone     (சூழ் is "surround")
#     காபி  -> copy      (காபி is "coffee")
#     கேரளா -> care      (கேரளா is "Kerala")
#     சாய்  -> sign      (சாய் is "lean")
#     நம்ப  -> number    (நம்ப is "believe" - "நம்ப முடியல" became "number முடியல")
#
# Every one of those corrupts the transcript the model then has to answer,
# which is the failure this module's docstring calls "far worse than leaving
# one English word transliterated". They cluster at the short end because a
# real transliteration of an English word is LONG in Tamil script -
# "appointment" is அப்பாயின்ட்மெண்ட், fourteen characters. Below six there is
# not room for an English word and there is room for a Tamil one.
#
# Six drops 154 of 771 (19%) and costs almost nothing real: the short English
# words a caller actually says at a hospital desk - bill, book, test, scan,
# slot - are all in the hand-measured table above, which this threshold does
# not touch.
_LEXICON_PATH = pathlib.Path(__file__).resolve().parent.parent / "golden" / "asr_lexicon.json"


def _load_generated_lexicon() -> dict[str, str]:
    try:
        payload = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logger.warning("generated ASR lexicon at %s is unusable: %s", _LEXICON_PATH, error)
        return {}
    entries = payload.get("lexicon")
    if not isinstance(entries, dict):
        logger.warning("generated ASR lexicon at %s has no 'lexicon' object", _LEXICON_PATH)
        return {}
    return {k: v for k, v in entries.items() if isinstance(k, str) and isinstance(v, str) and k and v}


_GENERATED = {k: v for k, v in _load_generated_lexicon().items() if len(k) >= _MIN_GENERATED_KEY_LEN}
if _GENERATED:
    # Hand table last: a measured entry always beats a generated one.
    _ENGLISH_WORDS = {**_GENERATED, **_ENGLISH_WORDS}
    logger.info(
        "ASR normaliser: %d generated entries merged under %d hand-measured ones",
        len(_GENERATED),
        len(_ENGLISH_WORDS) - len(_GENERATED),
    )


# Number WORDS -> digits, in all three forms a caller actually produces.
#
# 1. Literary Tamil, which the model prefers when it writes out a number:
#    "ஒன்று, இரண்டு".
# 2. Spoken Tamil, which is what a caller says: "ஒண்ணு, ரெண்டு".
# 3. ENGLISH digit names, transliterated - because reading a phone number out
#    in English is the normal way to do it in Chennai, and a Tamil-only ASR
#    renders those in Tamil script too. Observed live: a caller reading
#    9840721534 was transcribed
#        "நீன் ஏஐட் போர் ஜெரோ செவன் டூ ஒன் பைவ் த்ரீ போர்"
#    which is nine-eight-four-zero-seven-two-one-five-three-four and matched
#    nothing at all in the Tamil-only table this replaces.
#
# Several of the English forms collide with ordinary Tamil words - "போர்" is
# "war", "ஒன்" is a fragment - which is exactly what _MIN_DIGIT_RUN protects
# against: they are only ever read as digits inside a run of four or more.
_DIGIT_WORDS: dict[str, str] = {
    # literary Tamil
    "பூஜ்ஜியம்": "0", "சுழியம்": "0",
    "ஒன்று": "1", "இரண்டு": "2", "மூன்று": "3", "நான்கு": "4", "ஐந்து": "5",
    "ஆறு": "6", "ஏழு": "7", "எட்டு": "8", "ஒன்பது": "9",
    # spoken Tamil
    "ஒண்ணு": "1", "ரெண்டு": "2", "மூணு": "3", "நாலு": "4", "அஞ்சு": "5",
    "ஒம்பது": "9",
    # English digit names as this ASR transliterates them
    "ஜீரோ": "0", "ஜெரோ": "0", "சீரோ": "0", "ஓ": "0",
    "ஒன்": "1", "வன்": "1",
    "டூ": "2", "டு": "2",
    "த்ரீ": "3", "திரீ": "3",
    "போர்": "4", "ஃபோர்": "4",
    "பைவ்": "5", "ஃபைவ்": "5",
    "சிக்ஸ்": "6", "ஸிக்ஸ்": "6",
    "செவன்": "7", "சேவன்": "7",
    "ஏஐட்": "8", "எயிட்": "8", "ஏட்": "8",
    "நீன்": "9", "நைன்": "9", "நயின்": "9",
}

# How many number-words in a row before they are read as a dictated number.
#
# Inside a sentence the bar is high, because "ரெண்டு தடவை" ("twice") and
# "மூணு நாள்" ("three days") are ordinary Tamil and turning those into digits
# would corrupt what the caller said.
#
# But a caller reading out a phone number PAUSES between groups, and the VAD
# endpoints on those pauses. Observed live, one number arrived as three
# separate transcripts:
#
#     ஒன்பது எட்டு        <- "nine eight"
#     செவன் சிக்ஸ்         <- "seven six"
#
# Two words each, so a four-in-a-row rule never fired and the caller watched
# their number come back as Tamil words. An utterance that is NOTHING BUT
# number-words is a dictated number whatever its length - there is no sentence
# around it to misread - so that case only needs two.
_MIN_DIGIT_RUN = 4
_MIN_DIGIT_RUN_WHEN_WHOLE_UTTERANCE = 2

_TOKEN_RE = re.compile(r"(\s+)")

_MONTHS = frozenset(
    {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }
)


def _ordinal(day: int) -> str:
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _rewrite_calendar_dates(words: list[str]) -> list[str]:
    """Recognise a spoken cardinal as a date only beside a month.

    `சிக்ஸ்` on its own remains untouched by the conservative digit-run rules,
    but `சிக்ஸ் செப்டம்பர்` is unambiguously a calendar date and becomes
    `6th September`. Month-first speech is handled too.
    """
    out = list(words)
    for index, word in enumerate(words):
        core = word.rstrip(".,?!")
        month = _ENGLISH_WORDS.get(core, core)
        if month not in _MONTHS:
            continue
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(words):
                continue
            candidate = words[neighbour]
            candidate_core = candidate.rstrip(".,?!")
            digit = _DIGIT_WORDS.get(candidate_core)
            if digit is None or not 1 <= int(digit) <= 9:
                continue
            trailing = candidate[len(candidate_core):]
            out[neighbour] = _ordinal(int(digit)) + trailing
    return out


def _rewrite_labeled_numbers(words: list[str]) -> list[str]:
    """Convert a single spoken digit when its label makes the meaning clear.

    Ordinary Tamil numbers stay protected by the digit-run threshold, while
    `number மூணு` and `வீட்டு எண் மூணு` are unambiguously identifiers inside
    an address and become `number 3` / `வீட்டு எண் 3`.
    """
    out = list(words)
    for index, word in enumerate(words):
        core = word.rstrip(".,?!").casefold()
        previous = words[index - 1].rstrip(".,?!").casefold() if index else ""
        labeled = previous in {"number", "no", "எண்", "நம்பர்"}
        if not labeled:
            continue
        # Leave a full dictated phone-number run intact for _join_digit_runs;
        # rewriting only its first digit would split `984...` into `9 84...`.
        next_core = words[index + 1].rstrip(".,?!") if index + 1 < len(words) else ""
        if next_core in _DIGIT_WORDS:
            continue
        digit = _DIGIT_WORDS.get(word.rstrip(".,?!"))
        if digit is None:
            continue
        trailing = word[len(word.rstrip(".,?!")):]
        out[index] = digit + trailing
    return out


def _rewrite_english(token: str) -> str:
    """Map one whole word, preserving any trailing punctuation."""
    core = token.rstrip(".,?!")
    trailing = token[len(core):]
    replacement = _ENGLISH_WORDS.get(core)
    return f"{replacement}{trailing}" if replacement else token


def _join_digit_runs(words: list[str], minimum: int) -> list[str]:
    """Collapse runs of >= `minimum` number-words into one digit string."""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= minimum:
            digits = "".join(_DIGIT_WORDS[w.rstrip(".,?!")] for w in run)
            # The clause chunker splits on '.', '?' and '!', so a punctuation
            # mark swallowed here changes how the turn is spoken.
            trailing = run[-1][len(run[-1].rstrip(".,?!")):]
            out.append(digits + trailing)
        else:
            out.extend(run)
        run.clear()

    for word in words:
        if word.rstrip(".,?!") in _DIGIT_WORDS:
            run.append(word)
        else:
            flush()
            out.append(word)
    flush()
    return out


def normalize_transcript(text: str) -> str:
    """Rewrite one ASR transcript into the register the agent writes in.

    Whole-word, exact matches only. Anything unrecognised is passed through
    untouched, so the worst case is the transcript the ASR already produced.
    """
    if not text:
        return text
    words = [w for w in _TOKEN_RE.split(text) if w and not w.isspace()]
    words = _rewrite_calendar_dates(words)
    words = _rewrite_labeled_numbers(words)
    whole_utterance_is_digits = all(w.rstrip(".,?!") in _DIGIT_WORDS for w in words)
    minimum = (
        _MIN_DIGIT_RUN_WHEN_WHOLE_UTTERANCE
        if whole_utterance_is_digits
        else _MIN_DIGIT_RUN
    )
    return " ".join(_rewrite_english(w) for w in _join_digit_runs(words, minimum))


if __name__ == "__main__":  # pragma: no cover - manual check
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for line in (
        "ஒரு அப்பாயிண்ட்மெண்ட் புக் பண்ணணும்",
        "மொபைல் நம்பர் ஒன்பது எட்டு நான்கு பூஜ்ஜியம் ஏழு இரண்டு ஒன்று ஐந்து மூன்று நான்கு",
        "என் பில்ல ஒரு சார்ஜ் தப்பா இருக்கு",
        "ரெண்டு தடவை கூப்பிட்டேன்",
    ):
        print(f"{line}\n  -> {normalize_transcript(line)}")
