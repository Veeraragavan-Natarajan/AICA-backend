"""Self-check for TenVadSegmenter's turn-taking state machine.

Stubs the native ten_vad call so the pre-roll / endpoint-silence / max-duration
branches are exercised deterministically, without depending on real speech
audio or the native library's actual probabilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from .settings import AudioSettings
from .vad import TenVadSegmenter

HOP = 256

# A frame's AMPLITUDE now matters as well as its VAD flag: loudness gates turn
# ONSET (see AudioSettings.vad_onset_min_rms). Measured on real Tamil speech at
# full digital level, a voiced 16 ms frame runs ~931 RMS at p10 and ~2624 at
# p50, so SPOKEN is a realistic speaking level and ROOM is a quiet room.
SPOKEN = 2600
ROOM = 40


def _frame(value: int = SPOKEN) -> np.ndarray:
    return np.full(HOP, value, dtype=np.int16)


def _frames_for(flags: list[int], loud: int = SPOKEN) -> list[np.ndarray]:
    """One frame per flag: loud where the VAD says speech, quiet where it does not.

    Sending a loud frame for a NON-speech flag would teach the noise floor that
    the room is as loud as the caller, and onset would then never clear the SNR
    bar - which is a property of the test rig, not of the code under test.
    """
    return [_frame(loud if flag else ROOM) for flag in flags]


def _make_segmenter(flags: list[int], **overrides) -> TenVadSegmenter:
    settings = AudioSettings(**overrides)
    segmenter = TenVadSegmenter(settings)
    it = iter(flags)
    segmenter._vad.process = lambda frame: (1.0, next(it))
    return segmenter


def _run(segmenter: TenVadSegmenter, flags: list[int], loud: int = SPOKEN):
    update = None
    for frame in _frames_for(flags, loud):
        update = segmenter.process(frame)
        if update.speech_ended:
            break
    return update


def test_readme_hyperparameters_are_the_defaults() -> None:
    settings = AudioSettings()
    assert (settings.sample_rate, settings.vad_hop_size) == (16_000, 256)
    assert settings.vad_threshold == 0.35
    # 22 x 16 ms = 352 ms. Lowered from 30 (480 ms): this silence is spent
    # in front of every reply, so it is part of the latency budget.
    assert settings.endpoint_silence_frames == 22
    assert settings.pre_roll_frames == 8
    # Debounces. Flag-only, never energy - see AudioSettings.vad_start_frames.
    assert settings.vad_start_frames == 4
    assert settings.vad_resume_frames == 3
    assert (settings.language, settings.decoding) == ("ta", "rnnt")


def test_pre_roll_is_prepended_to_utterance() -> None:
    pre_roll = 3
    onset = AudioSettings().vad_start_frames
    silence = AudioSettings().endpoint_silence_frames
    flags = [0] * pre_roll + [1] * onset + [0] * silence
    segmenter = _make_segmenter(flags, pre_roll_frames=pre_roll)

    update = _run(segmenter, flags)

    assert update is not None and update.samples is not None
    # pre-roll only fills while silent, so the utterance is pre-roll + the
    # frames that CONFIRMED the onset + the silent tail that ended the turn.
    # The onset frames are kept, not discarded: confirming a turn must never
    # cost the caller their first syllable.
    assert len(update.samples) == (pre_roll + onset + silence) * HOP


def test_mid_sentence_pause_shorter_than_endpoint_does_not_split() -> None:
    # 10 silent frames (160 ms) is a breath, not a turn end - it must stay
    # comfortably under endpoint_silence_frames or the ASR gets half a
    # sentence, which is the failure that caps how low that can be tuned.
    settings = AudioSettings()
    endpoint = settings.endpoint_silence_frames
    onset, resume = settings.vad_start_frames, settings.vad_resume_frames
    flags = [1] * onset + [0] * 10 + [1] * resume + [0] * endpoint
    segmenter = _make_segmenter(flags)

    update = _run(segmenter, flags)

    assert update is not None and update.samples is not None
    assert len(update.samples) == (onset + 10 + resume + endpoint) * HOP


def test_a_blip_shorter_than_the_onset_debounce_never_opens_a_turn() -> None:
    """The 39%-empty-transcript bug: background noise opening real turns.

    Measured over the 87 captured turns in call_events.db, 34 transcribed to
    the EMPTY STRING - the VAD had opened a turn on noise, the ASR ran on it,
    and those frames were free to reach the barge-in gate and cancel the agent
    mid-sentence.
    """
    settings = AudioSettings()
    blip = settings.vad_start_frames - 1
    flags = [1] * blip + [0] * (settings.endpoint_silence_frames + 2)
    segmenter = _make_segmenter(flags)

    updates = [segmenter.process(f) for f in _frames_for(flags)]

    assert not any(u.speech_started for u in updates), "noise opened a turn"
    assert not any(u.speech_frame for u in updates), "a blip fed the barge-in gate"
    assert segmenter.in_speech is False
    assert segmenter.flush() is None


def test_a_confirmed_turn_still_reaches_asr_with_no_minimum_length_gate() -> None:
    """The debounce may DELAY a turn; it must never discard one."""
    settings = AudioSettings()
    flags = [1] * settings.vad_start_frames + [0] * settings.endpoint_silence_frames
    segmenter = _make_segmenter(flags)

    update = _run(segmenter, flags)

    assert update is not None and update.speech_ended and update.samples is not None


def test_an_isolated_blip_cannot_hold_the_microphone_open() -> None:
    """The stuck-mic bug: one noisy frame per silence window, turn never ends.

    Resetting the endpoint countdown on a SINGLE flagged hop meant background
    noise recurring anywhere inside the 352 ms window kept the turn open
    indefinitely - which is why a caller ends up toggling their microphone by
    hand after they have finished speaking.
    """
    settings = AudioSettings()
    tail: list[int] = []
    for _ in range(settings.endpoint_silence_frames * 3):
        tail += [0, 0, 0, 1]
    flags = [1] * settings.vad_start_frames + tail
    segmenter = _make_segmenter(flags)

    update = _run(segmenter, flags)

    assert update is not None and update.speech_ended, "noise held the turn open"
    assert update.end_reason == "silence"


def test_max_duration_forces_endpoint() -> None:
    flags = [1] * 10
    segmenter = _make_segmenter(flags, max_utterance_frames=5)

    update = _run(segmenter, flags)

    assert update is not None and update.end_reason == "max_duration"


def test_peek_utterance_is_none_before_speech_starts() -> None:
    segmenter = _make_segmenter([0, 0, 0])
    segmenter.process(_frame())
    assert segmenter.in_speech is False
    assert segmenter.peek_utterance() is None


def test_peek_utterance_returns_speech_so_far_without_consuming_it() -> None:
    onset = AudioSettings().vad_start_frames
    segmenter = _make_segmenter([1] * (onset + 2))
    for _ in range(onset):
        segmenter.process(_frame())

    assert segmenter.in_speech is True
    peeked = segmenter.peek_utterance()
    assert peeked is not None and len(peeked) == onset * HOP

    # A second peek and a further process() must see the same/growing buffer,
    # proving peek_utterance() never mutates or drains state.
    assert len(segmenter.peek_utterance()) == onset * HOP
    segmenter.process(_frame())
    assert len(segmenter.peek_utterance()) == (onset + 1) * HOP


def test_peek_utterance_is_none_again_once_the_turn_ends() -> None:
    settings = AudioSettings()
    flags = [1] * settings.vad_start_frames + [0] * settings.endpoint_silence_frames
    segmenter = _make_segmenter(flags)
    _run(segmenter, flags)

    assert segmenter.in_speech is False
    assert segmenter.peek_utterance() is None


def test_flush_emits_in_progress_utterance_and_resets() -> None:
    onset = AudioSettings().vad_start_frames
    segmenter = _make_segmenter([1] * onset)
    for _ in range(onset):
        segmenter.process(_frame())

    update = segmenter.flush()
    assert update is not None and update.end_reason == "call_ended" and update.samples is not None
    assert segmenter.flush() is None


if __name__ == "__main__":
    test_readme_hyperparameters_are_the_defaults()
    test_pre_roll_is_prepended_to_utterance()
    test_mid_sentence_pause_shorter_than_endpoint_does_not_split()
    test_a_blip_shorter_than_the_onset_debounce_never_opens_a_turn()
    test_a_confirmed_turn_still_reaches_asr_with_no_minimum_length_gate()
    test_an_isolated_blip_cannot_hold_the_microphone_open()
    test_quiet_background_speech_never_opens_a_turn()
    test_someone_actually_talking_to_the_microphone_still_opens_a_turn()
    test_a_quiet_syllable_can_never_end_a_turn_that_is_already_open()
    test_the_onset_bar_adapts_to_a_noisy_room()
    test_a_talking_caller_never_raises_the_bar_against_themselves()
    test_a_turn_always_ends_even_if_the_vad_never_stops_flagging_speech()
    test_a_caller_who_keeps_talking_never_trips_the_watchdog()
    test_max_duration_forces_endpoint()
    test_peek_utterance_is_none_before_speech_starts()
    test_peek_utterance_returns_speech_so_far_without_consuming_it()
    test_peek_utterance_is_none_again_once_the_turn_ends()
    test_flush_emits_in_progress_utterance_and_resets()
    print("ok")


# --- loudness at ONSET, and the rule that it may never end a turn ---


def test_quiet_background_speech_never_opens_a_turn() -> None:
    """A television, a fan, a conversation across the room.

    The VAD flags these as speech - they ARE speech, just not addressed to us -
    so the flag alone cannot reject them. Loudness can.
    """
    settings = AudioSettings()
    flags = [1] * (settings.vad_start_frames + 6) + [0] * 4
    segmenter = _make_segmenter(flags)

    # Flagged as speech the whole way, but far below the onset floor.
    quiet = int(settings.vad_onset_min_rms // 3)
    updates = [segmenter.process(_frame(quiet)) for _ in flags]

    assert not any(u.speech_started for u in updates), "quiet background opened a turn"
    assert not any(u.speech_frame for u in updates), "quiet background fed the barge-in gate"
    assert segmenter.in_speech is False


def test_someone_actually_talking_to_the_microphone_still_opens_a_turn() -> None:
    """The gate must reject the room, not the caller."""
    settings = AudioSettings()
    flags = [1] * settings.vad_start_frames + [0] * settings.endpoint_silence_frames
    segmenter = _make_segmenter(flags)

    update = _run(segmenter, flags)

    assert update is not None and update.speech_ended and update.samples is not None


def test_a_quiet_syllable_can_never_end_a_turn_that_is_already_open() -> None:
    """THE regression guard. This is the bug that reverted the last attempt.

    An energy gate applied to every frame scored a quiet trailing syllable as
    silence, so the endpoint countdown ran on through the middle of a word and
    turns came back as one-character transcripts ("ந", "ப", "க").

    What must hold is that a quiet syllable cannot END a turn and cannot lose
    its audio. It is NOT that a quiet frame reports speech_frame - that field
    feeds only the barge-in gate, and reporting it for quiet frames is what let
    a low voice cut the agent off mid-sentence. See
    test_a_quiet_flagged_run_cannot_cut_the_agent_off below.
    """
    settings = AudioSettings()
    onset, endpoint = settings.vad_start_frames, settings.endpoint_silence_frames
    whisper = int(settings.vad_onset_min_rms // 10)   # far below the onset floor

    # Flags: speech through the loud opening, the quiet trail and the
    # spoken-up recovery; then genuine non-speech for the closing silence.
    segmenter = _make_segmenter(
        [1] * (onset + endpoint + settings.vad_resume_frames) + [0] * (endpoint + 5)
    )
    # Open the turn at a normal speaking level.
    for _ in range(onset):
        assert not segmenter.process(_frame(SPOKEN)).speech_ended
    assert segmenter.in_speech

    # Now trail off. Every frame is still FLAGGED as speech, just very quiet -
    # exactly what the end of a Tamil word sounds like. A quiet passage as long
    # as the whole silence window must not end the turn: if it did, quiet and
    # silent would mean the same thing, which is the reverted behaviour.
    kept_before = len(segmenter._utterance)
    for _ in range(endpoint):
        update = segmenter.process(_frame(whisper))
        assert not update.speech_ended, "a quiet syllable ended the turn mid-word"
    assert len(segmenter._utterance) == kept_before + endpoint, (
        "a quiet syllable was dropped from the utterance instead of transcribed"
    )

    # Speaking up again clears the countdown completely, so the turn continues.
    for _ in range(settings.vad_resume_frames):
        update = segmenter.process(_frame(SPOKEN))
        assert not update.speech_ended

    # And it still ends normally on real silence.
    for _ in range(endpoint):
        update = segmenter.process(_frame(ROOM))
        if update.speech_ended:
            break
    assert update.speech_ended and update.end_reason == "silence"


def test_a_quiet_flagged_run_cannot_cut_the_agent_off() -> None:
    """The reported fault: the agent stops mid-sentence for a low voice.

    Measured over the 208 recorded calls in call_events.db, 45% of the turns
    the VAD opened transcribed to nothing at all (500 empty, 40 of 1-3 chars),
    and there were 332 agent_interrupted events. Onset probability does not
    separate those from real speech - real p50 0.796 vs empty p50 0.731 - so
    raising VAD_THRESHOLD costs real speech faster than it removes noise.
    Loudness does separate them, and it was already being computed here and
    thrown away: every flagged frame reported speech_frame regardless of level,
    so a quiet flagged run walked straight into the barge-in counter.

    speech_frame feeds the barge-in gate and nothing else (main.py and
    telephony.py pass it to ActiveSpeech.note_speech), so reporting
    loud-AND-flagged narrows exactly one behaviour and nothing else.
    """
    settings = AudioSettings()
    onset = settings.vad_start_frames
    whisper = int(settings.vad_onset_min_rms // 10)

    segmenter = _make_segmenter([1] * (onset + 40))
    for _ in range(onset):
        segmenter.process(_frame(SPOKEN))
    assert segmenter.in_speech

    quiet_updates = [segmenter.process(_frame(whisper)) for _ in range(20)]

    assert not any(u.speech_frame for u in quiet_updates), (
        "a quiet flagged run still feeds the barge-in gate and can cut the agent off"
    )
    assert not any(u.speech_ended for u in quiet_updates), "the quiet run ended the turn"
    # ...and a caller who actually speaks up is heard immediately.
    assert segmenter.process(_frame(SPOKEN)).speech_frame, "a real interruption was suppressed"


def test_the_onset_update_carries_the_level_the_loudness_gate_saw() -> None:
    """Recorded into the vad_start event so vad_onset_min_rms / vad_onset_snr
    can be tuned against real calls. Without it the only thing written down is
    the onset probability, which does not separate noise from speech - measured
    over the recorded calls, empty turns sat at p50 0.731 and real speech at
    0.796. A diagnostic that silently reports 0.0 forever is worse than none,
    so this asserts a real level reaches the field."""
    settings = AudioSettings()
    segmenter = _make_segmenter([1] * settings.vad_start_frames)

    for _ in range(settings.vad_start_frames - 1):
        segmenter.process(_frame(SPOKEN))
    onset = segmenter.process(_frame(SPOKEN))

    assert onset.speech_started
    assert onset.onset_rms == pytest.approx(SPOKEN, rel=0.01)
    assert onset.onset_rms >= settings.vad_onset_min_rms


def test_a_quiet_run_cannot_reach_the_barge_in_gate_end_to_end() -> None:
    """The same thing through the gate the VAD actually drives, because that is
    where it matters: ActiveSpeech resets its counter on a false speech_frame,
    so quiet frames must never accumulate towards an interrupt."""
    import asyncio

    from .barge_in import ActiveSpeech

    settings = AudioSettings()
    onset = settings.vad_start_frames
    whisper = int(settings.vad_onset_min_rms // 10)

    async def scenario() -> tuple[bool, bool]:
        agent_turn = asyncio.get_running_loop().create_future()  # stands in for speech
        task = asyncio.ensure_future(asyncio.sleep(30))
        active = ActiveSpeech(settings.barge_in_speech_frames)
        active.set(task)

        segmenter = _make_segmenter([1] * (onset + 200))
        for _ in range(onset):
            u = segmenter.process(_frame(SPOKEN))
            active.note_speech(u.speech_frame, u.speech_started)

        # Far more quiet flagged frames than the barge-in gate requires, but
        # still inside the quiet-endpoint watchdog so the turn stays open -
        # otherwise this would be testing the watchdog instead of the gate.
        quiet_run = settings.vad_quiet_endpoint_frames - 1
        assert quiet_run > settings.barge_in_speech_frames
        quiet_interrupted = False
        for _ in range(quiet_run):
            u = segmenter.process(_frame(whisper))
            quiet_interrupted |= active.note_speech(u.speech_frame, u.speech_started)
        assert segmenter.in_speech, "the turn closed before the gate could be tested"

        # Then someone genuinely talks over the agent.
        loud_interrupted = False
        for _ in range(settings.barge_in_speech_frames + 2):
            u = segmenter.process(_frame(SPOKEN))
            loud_interrupted |= active.note_speech(u.speech_frame, u.speech_started)

        task.cancel()
        agent_turn.cancel()
        return quiet_interrupted, loud_interrupted

    quiet_interrupted, loud_interrupted = asyncio.run(scenario())

    assert not quiet_interrupted, "quiet frames cut the agent off"
    assert loud_interrupted, "a real interruption no longer works"


def test_the_onset_bar_adapts_to_a_noisy_room() -> None:
    """A fixed number cannot be right for every microphone, so the bar is a
    multiple of the room level the segmenter has actually measured."""
    settings = AudioSettings()
    noisy = settings.vad_onset_min_rms * 2
    segmenter = _make_segmenter([0] * 60 + [1] * 40)

    # Sixty frames of a room noticeably louder than the absolute floor.
    for _ in range(60):
        segmenter.process(_frame(int(noisy)))
    assert segmenter.noise_floor > settings.vad_onset_min_rms

    # Speech only a little above that room is now rejected, where against a
    # quiet room the same level would have opened a turn.
    just_above = int(noisy * 1.2)
    assert just_above > settings.vad_onset_min_rms
    updates = [segmenter.process(_frame(just_above)) for _ in range(40)]
    assert not any(u.speech_started for u in updates)


def test_a_talking_caller_never_raises_the_bar_against_themselves() -> None:
    """The noise floor learns only from frames the VAD calls non-speech."""
    segmenter = _make_segmenter([1] * 50)
    for _ in range(50):
        segmenter.process(_frame(SPOKEN * 3))

    assert segmenter.noise_floor < SPOKEN, "loud speech was learned as room noise"


def test_a_turn_always_ends_even_if_the_vad_never_stops_flagging_speech() -> None:
    """THE hang. Reported live: the first turn endpointed, every turn after it
    listened indefinitely.

    Once the agent has spoken, residual echo and the browser's automatic gain
    control produce long runs of VAD-flagged frames out of an empty room. The
    endpoint countdown is restarted by any sustained run of them, so it never
    completed and the microphone stayed open to the 30 s hard cap.

    The watchdog is what makes termination unconditional: nothing LOUD for
    `vad_quiet_endpoint_frames` ends the turn, whatever the VAD is flagging.
    """
    settings = AudioSettings()
    onset = settings.vad_start_frames
    echo = int(settings.vad_onset_min_rms // 4)

    segmenter = _make_segmenter([1] * 4000)
    for _ in range(onset):
        segmenter.process(_frame(SPOKEN))
    assert segmenter.in_speech

    # Flagged as speech forever, but never loud: pure echo/room noise.
    ended = None
    for index in range(settings.max_utterance_frames):
        update = segmenter.process(_frame(echo))
        if update.speech_ended:
            ended = (index, update)
            break

    assert ended is not None, "the turn never ended - the microphone hung open"
    index, update = ended
    assert update.end_reason == "silence"
    assert index < settings.vad_quiet_endpoint_frames + 2, (
        f"took {index} frames to give up; the watchdog is {settings.vad_quiet_endpoint_frames}"
    )
    # Nothing was thrown away: the audio is still there for the ASR to judge.
    assert update.samples is not None and len(update.samples) > 0


def test_a_turn_ends_when_the_caller_stops_even_though_the_room_stays_loud() -> None:
    """The requirement in as many words: the end of MEANINGFUL SPEECH is
    detected even while background noise continues.

    The watchdog test above covers noise the VAD flags but that is QUIET
    (echo). This is the other half and the commoner one: a television or a
    conversation across the room going on at full speaking level after the
    caller has stopped. Those frames are loud, so if loudness were ever
    consulted when deciding that a turn has ENDED - the energy gate that was
    tried here and reverted - the countdown would never complete and the
    microphone would hang open to the 30 s cap.

    Endpointing is the VAD flag alone, which is exactly what makes this work:
    loudness may refuse to start a turn, never end one.

    The one case this cannot cover, named so nobody reports it as a bug: noise
    the VAD flags as speech AND that is loud (a television playing dialogue)
    is indistinguishable from a caller by every signal available here, and is
    bounded only by max_utterance_frames (30 s).
    """
    settings = AudioSettings()
    flags = [1] * settings.vad_start_frames + [0] * (settings.endpoint_silence_frames + 1)
    segmenter = _make_segmenter(flags)

    for _ in range(settings.vad_start_frames):
        segmenter.process(_frame(SPOKEN))
    assert segmenter.in_speech

    # The caller has stopped. The ROOM has not - every frame from here is as
    # loud as speech, and only the VAD flag says otherwise.
    ended = None
    for index in range(settings.endpoint_silence_frames + 1):
        update = segmenter.process(_frame(SPOKEN))
        if update.speech_ended:
            ended = (index, update)
            break

    assert ended is not None, (
        "the turn never ended while the room was loud - loudness has got into "
        "the endpoint decision, which is the reverted energy-gate bug"
    )
    index, update = ended
    assert update.end_reason == "silence"
    assert index < settings.endpoint_silence_frames + 1
    # And the caller's own audio is intact for the ASR.
    assert update.samples is not None and len(update.samples) > 0


def test_a_caller_who_keeps_talking_never_trips_the_watchdog() -> None:
    """The watchdog must be invisible to anyone actually speaking."""
    settings = AudioSettings()
    segmenter = _make_segmenter([1] * 4000)
    for _ in range(settings.vad_start_frames):
        segmenter.process(_frame(SPOKEN))

    # Ten seconds of real speech, far past the watchdog window.
    for _ in range(settings.vad_quiet_endpoint_frames * 6):
        update = segmenter.process(_frame(SPOKEN))
        assert not update.speech_ended, "the watchdog cut off a caller who was still talking"
