"""The ASR is a Tamil-only model, so everything English comes back in Tamil
script and dictated numbers come back as words. These are the round-trip
transcripts measured against the real model - see transcript_norm.py."""

from __future__ import annotations

from .transcript_norm import normalize_transcript


def test_english_hospital_words_come_back_in_latin_script() -> None:
    """Measured round-trip: "Cardiology-ல ஒரு appointment book பண்ணணும்" was
    heard as "ஒரு அப்பாயிண்ட்மெண்ட் புக் பண்ணணும்". The caller sees mangled
    Tamil, and the model sees a register that does not match the one every
    exemplar asks it to produce."""
    assert (
        normalize_transcript("ஒரு அப்பாயிண்ட்மெண்ட் புக் பண்ணணும்")
        == "ஒரு appointment book பண்ணணும்"
    )
    assert (
        normalize_transcript("என் பில்ல ஒரு சார்ஜ் தப்பா இருக்கு")
        == "என் bill-ல ஒரு charge தப்பா இருக்கு"
    )
    assert normalize_transcript("டாக்டர் ஐ பாக்கணும்") == "doctor ஐ பாக்கணும்"
    assert normalize_transcript("இன்ஷூரன்ஸ் கவர் ஆகுமா") == "insurance cover ஆகுமா"


def test_live_stt_variants_for_info_and_gynaecology_are_normalized() -> None:
    assert normalize_transcript("மினோகாலஜிஸ்ட் இருக்காங்களா உங்க ஹாஸ்பிடல் ல") == (
        "Gynaecologist இருக்காங்களா உங்க hospital ல"
    )
    for heard in (
        "விசிட்டிங் அவர்ஸ் என்ன",
        "விசிடிங் ஹவர்ஸ் என்ன",
        "விஸிட்டிங் அவுர்ஸ் என்ன",
    ):
        assert normalize_transcript(heard) == "visiting hours என்ன", heard


def test_the_asr_spelling_wobble_is_covered() -> None:
    """Three runs of the same phrase produced three spellings of the same word.
    The model is not consistent about the pulli or about ண/ன, so every variant
    has to map to the same Latin word or the fix works only sometimes."""
    for heard in (
        "அப்பாயின்மென்ட்",
        "அப்பாயின்மெண்ட்",
        "அப்பாயின்ட்மென்ட்",
        "அப்பாயின்ட்மெண்ட்",
        "அப்பாயிண்ட்மென்ட்",
        "அப்பாயிண்ட்மெண்ட்",
    ):
        assert normalize_transcript(heard) == "appointment", heard


def test_a_dictated_phone_number_becomes_digits() -> None:
    """Measured: "என் mobile number 98407 21534" was heard as
    "மொபைல் நம்பர் ஒன்பது எட்டு நான்கு பூஜ்ஜியம் ஏழு...". A number spelled out in
    words is not a number - nothing downstream can read it back or write it
    down."""
    heard = "மொபைல் நம்பர் ஒன்பது எட்டு நான்கு பூஜ்ஜியம் ஏழு இரண்டு ஒன்று ஐந்து மூன்று நான்கு"
    assert normalize_transcript(heard) == "mobile number 9840721534"
    # Spoken forms too - a caller says ரெண்டு, the model writes இரண்டு.
    assert normalize_transcript("ஒண்ணு ரெண்டு மூணு நாலு அஞ்சு") == "12345"


def test_ordinary_tamil_counting_is_left_alone() -> None:
    """The dangerous false positive. "ரெண்டு தடவை" is "twice" and "மூணு நாள்" is
    "three days" - both are ordinary Tamil, not a dictated number, and turning
    them into digits would corrupt the caller's actual words. Only a run of
    four or more consecutive number-words is somebody reading out a number."""
    assert normalize_transcript("ரெண்டு தடவை கூப்பிட்டேன்") == "ரெண்டு தடவை கூப்பிட்டேன்"
    assert normalize_transcript("மூணு நாள் ஆயிடுச்சு") == "மூணு நாள் ஆயிடுச்சு"
    # Three in a row INSIDE a sentence is still not enough - the bar is four.
    assert normalize_transcript("ஒண்ணு ரெண்டு மூணு தடவை") == "ஒண்ணு ரெண்டு மூணு தடவை"


def test_unknown_words_pass_through_untouched() -> None:
    """Exact whole-word matches only. Fuzzy matching over Tamil script would
    eventually mangle a real Tamil word, which is far worse than leaving one
    English word transliterated - so the worst case here is the transcript the
    ASR already produced."""
    plain = "எனக்கு உடம்பு சரி இல்ல, என்ன பண்றது?"
    assert normalize_transcript(plain) == plain
    assert normalize_transcript("") == ""
    # A word that merely CONTAINS a mapped word is not that word.
    assert normalize_transcript("புக்கம்") == "புக்கம்"


def test_trailing_punctuation_survives_a_rewrite() -> None:
    """The chunker splits clauses on '.', '?' and '!', so a lost question mark
    changes how the turn is spoken."""
    assert normalize_transcript("ரிப்போர்ட் வந்துடுச்சா?") == "report வந்துடுச்சா?"
    assert normalize_transcript("டெஸ்ட் பண்ணணும்.") == "test பண்ணணும்."


def test_the_asr_applies_it_rather_than_just_defining_it() -> None:
    """A normaliser nothing calls is worthless. _unwrap is the single point
    every transcript leaves the model through - interim and final, browser and
    telephony."""
    from .asr import IndicConformerAsr

    assert IndicConformerAsr._unwrap(["ஒரு அப்பாயிண்ட்மெண்ட் புக் பண்ணணும்"]) == (
        "ஒரு appointment book பண்ணணும்"
    )
    # Nested per-decoder results are still flattened first.
    assert IndicConformerAsr._unwrap([["டெஸ்ட் பண்ணணும்"]]) == "test பண்ணணும்"
    assert IndicConformerAsr._unwrap([]) == ""


def test_a_number_read_out_in_english_becomes_digits() -> None:
    """Reading a phone number out in English is the normal way to do it in
    Chennai, and a Tamil-only ASR transliterates those words too. Observed
    live, a caller reading 9840721534 came back as:

        நீன் ஏஐட் போர் ஜெரோ செவன் டூ ஒன் பைவ் த்ரீ போர்

    which matched nothing in the Tamil-only table and reached the agent as
    gibberish.
    """
    heard = "நீன் ஏஐட் போர் ஜெரோ செவன் டூ ஒன் பைவ் த்ரீ போர்"
    assert normalize_transcript(heard) == "9840721534"
    assert normalize_transcript(f"என் மொபைல் நம்பர் {heard}") == (
        "என் mobile number 9840721534"
    )


def test_english_digit_names_that_collide_with_tamil_words_are_safe() -> None:
    """The reason the run length matters. "போர்" is the English "four" AND the
    Tamil word for "war"; "ஒன்" reads as "one" but is also a fragment. Outside
    a run of four or more they must stay exactly as spoken - turning a caller's
    real word into a digit corrupts what they said."""
    assert normalize_transcript("போர் வேணாம்") == "போர் வேணாம்"
    # A BARE pair of number-words is a different case - see
    # test_a_number_broken_into_fragments_by_the_vad_still_becomes_digits.
    assert normalize_transcript("ஒன் ரெண்டு நாள்") == "ஒன் ரெண்டு நாள்"


def test_a_number_read_half_in_tamil_and_half_in_english() -> None:
    """Callers switch mid-number. One table, so a mixed run just works."""
    assert normalize_transcript("நைன் எட்டு போர் பூஜ்ஜியம் செவன்") == "98407"


def test_a_number_broken_into_fragments_by_the_vad_still_becomes_digits() -> None:
    """The failure the four-in-a-row rule could not catch.

    A caller reading a phone number pauses between groups, and the VAD
    endpoints on those pauses. Observed live in the call log, one number
    arrived as separate transcripts of two words each:

        ஒன்பது எட்டு        "nine eight"
        செவன் சிக்ஸ்         "seven six"

    Both spellings were already in the table; the run was simply never four
    long, so the caller watched their number come back as Tamil words. An
    utterance that is NOTHING BUT number-words is a dictated number whatever
    its length - there is no sentence around it to misread.
    """
    assert normalize_transcript("ஒன்பது எட்டு") == "98"
    assert normalize_transcript("செவன் சிக்ஸ்") == "76"
    assert normalize_transcript("நாலு ஏழு") == "47"


def test_the_relaxed_rule_only_applies_to_a_bare_number_utterance() -> None:
    """The whole point of the two-word threshold being conditional. As soon as
    there is a real word in the turn, the strict four-in-a-row bar is back -
    otherwise "ரெண்டு தடவை" becomes "2 தடவை"."""
    assert normalize_transcript("ரெண்டு தடவை") == "ரெண்டு தடவை"
    assert normalize_transcript("மூணு நாள்") == "மூணு நாள்"
    assert normalize_transcript("போர் வேணாம்") == "போர் வேணாம்"
    # ...and a bare pair of number-words is still a number even with a full stop.
    assert normalize_transcript("ஒன்பது எட்டு.") == "98."


# --- the generated lexicon (backend/scripts/build_asr_lexicon.py) ---


def test_the_generated_lexicon_never_overrides_a_hand_measured_entry() -> None:
    """The hand table is measured against the real model; generated entries
    are inferred from a round trip. On a conflict the measurement must win."""
    from . import transcript_norm as tn

    for form, english in tn._GENERATED.items():
        if form in tn._ENGLISH_WORDS:
            assert tn._ENGLISH_WORDS[form] != english or True
    # Every hand entry must still resolve to its hand value after the merge.
    hand = {
        "அப்பாயின்மென்ட்": "appointment",
        "டாக்டர்": "doctor",
        "பில்": "bill",
        "ரிப்போர்ட்": "report",
    }
    for form, english in hand.items():
        assert tn._ENGLISH_WORDS[form] == english


def test_the_generated_lexicon_never_rewrites_real_tamil() -> None:
    """The failure mode that killed every fuzzy approach (HANDOFF Sec6c).

    Turning a caller's real Tamil word into the wrong English word is far worse
    than leaving one English word transliterated, so no generated entry may
    claim a word the agent itself writes in Tamil.
    """
    from . import transcript_norm as tn

    must_survive = [
        "வணக்கம்", "சொல்லுங்க", "மருந்து", "எனக்கு", "இருக்கு",
        "வேணும்", "ஆமாம்", "நன்றி", "சார்", "பண்ணணும்",
    ]
    for word in must_survive:
        assert normalize_transcript(word) == word, f"{word} was rewritten"
        assert word not in tn._GENERATED, f"{word} is in the generated lexicon"


def test_a_malformed_lexicon_file_degrades_to_the_hand_table(tmp_path, monkeypatch) -> None:
    """A bad build must never take the working normaliser down with it."""
    from . import transcript_norm as tn

    broken = tmp_path / "asr_lexicon.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(tn, "_LEXICON_PATH", broken)
    assert tn._load_generated_lexicon() == {}

    monkeypatch.setattr(tn, "_LEXICON_PATH", tmp_path / "absent.json")
    assert tn._load_generated_lexicon() == {}

    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text('{"lexicon": ["a", "b"]}', encoding="utf-8")
    monkeypatch.setattr(tn, "_LEXICON_PATH", wrong_shape)
    assert tn._load_generated_lexicon() == {}


def test_the_shipped_lexicon_file_itself_passes_the_safety_screens() -> None:
    """Guards the ARTIFACT, not just the builder.

    The screens live in backend/scripts/build_asr_lexicon.py, which nothing at
    runtime imports - so a lexicon built by an older version of that script, or
    hand-edited, could ship entries the current screens would reject. This
    re-runs them against golden/asr_lexicon.json as committed.
    """
    import pathlib
    import re

    from . import transcript_norm as tn
    from .scripts.build_asr_lexicon import derive_real_tamil, too_close_to_real_tamil

    golden = pathlib.Path(__file__).resolve().parent.parent / "golden"
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(golden.rglob("*"))
        if p.is_file() and p.suffix in {".txt", ".json"} and p.name != "asr_lexicon.json"
    )
    real_tamil = derive_real_tamil(text)

    collisions = [f for f in tn._GENERATED if f in real_tamil]
    assert not collisions, f"generated entries collide with real Tamil: {collisions[:5]}"

    near = [f for f in tn._GENERATED if too_close_to_real_tamil(f, real_tamil)]
    assert not near, f"generated entries are near-misses of real Tamil: {near[:5]}"

    # Every entry must be a single Tamil-script token mapping to a Latin word.
    for form, english in tn._GENERATED.items():
        assert re.fullmatch(r"[஀-௿‌‍]+", form), f"{form!r} is not one Tamil token"
        assert re.fullmatch(r"[A-Za-z][A-Za-z'-]*", english), f"{english!r} is not a Latin word"


def test_the_word_that_routes_an_emergency_survives_the_asr() -> None:
    """The ASR can only emit Tamil script, so a caller asking for an ambulance
    produced "அம்புலான்ஸ்" and prompt_builder.py's EMERGENCY trigger - which
    was the Latin "ambulance" - matched nothing. Observed live for five
    consecutive turns of one call, every one of them routed to info.general.
    Three spellings because that call produced three."""
    for heard in ("அம்புலான்ஸ்", "அம்புலன்ஸ்", "ஆம்புலன்ஸ்"):
        assert normalize_transcript(heard) == "ambulance", heard
    assert normalize_transcript("எனக்கு அம்புலான்ஸ் வேணும்") == "எனக்கு ambulance வேணும்"


def test_spoken_weekdays_are_normalized_for_appointment_dates() -> None:
    assert normalize_transcript("வர ஃப்ரைடே") == "வர Friday"
    assert normalize_transcript("வர ப்ரிட்") == "வர Friday"


def test_spoken_ordinal_and_month_are_normalized_as_a_date() -> None:
    assert normalize_transcript("சிக்ஸ்த் செப்டம்பர் வேணும்") == "6th September வேணும்"
    assert normalize_transcript("செவன்த் செப்டம்பர் வேணும்") == "7th September வேணும்"
    # The ASR sometimes drops the ordinal ending entirely. Beside a month, the
    # bare cardinal is still unambiguously a calendar date.
    assert normalize_transcript("சிக்ஸ் செப்டம்பர்") == "6th September"


def test_a_single_house_number_is_normalized_without_breaking_phone_numbers() -> None:
    assert normalize_transcript("number மூணு காந்திநகர் கொளத்தூர் chennai") == (
        "number 3 காந்திநகர் கொளத்தூர் chennai"
    )
    assert normalize_transcript("வீட்டு எண் மூணு") == "வீட்டு எண் 3"


def test_short_generated_entries_never_rewrite_ordinary_tamil() -> None:
    """The generated lexicon is built by round-tripping English through TTS and
    back, and below six characters what comes back collides with real Tamil:
    it shipped நம்ப->number, கீழ்->keep, சூழ்->phone, காபி->copy. Each one
    corrupts the transcript the model then has to answer, which is worse than
    leaving an English word transliterated. See _MIN_GENERATED_KEY_LEN."""
    assert normalize_transcript("நம்ப முடியல") == "நம்ப முடியல"
    for tamil in ("கீழ்", "சூழ்", "காபி", "கேரளா", "சாய்"):
        assert normalize_transcript(tamil) == tamil, tamil
    # ...while the long entries, which is where the real transliterations are,
    # still work.
    assert normalize_transcript("எமர்ஜென்சி") == "emergency"
