"""Splits agent reply text into clauses for pipelined TTS synthesis.

v1 (BACKEND_COMPLETION.md Sec3.2/Sec6 item 2) feeds this the whole LLM reply
in one feed() call rather than real token-level streaming, but the
feed()/flush() interface is written against a streaming source from the
start - swapping in token-level LLM streaming later shouldn't require
reshaping this class.
"""

from __future__ import annotations

import os
import re

# Guards against splitting mid-abbreviation ("Dr. Ramanathan") - a naive
# `.`/`?`/`!` splitter would cut it. Tamil/English code-mixed turns in this
# prompt commonly hit this (golden/main_prompt.txt).
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "st", "rs", "no", "vs", "e.g", "i.e"})

# A clause ends at '.'/'?'/'!' followed by whitespace - never at end-of-buffer.
# Requiring an actual trailing space (not just "no more text yet") is what
# keeps a decimal like "37." from matching before the "5" of "37.5" has even
# arrived: mid-stream, buffer-end isn't evidence of a real sentence end, only
# whitespace is. The one deliberate consequence: the last clause of any text
# never closes via feed() alone, even for a complete one-shot reply - callers
# always finish with flush() to release it (see _split_reply_into_clauses in
# main.py).
_BOUNDARY_RE = re.compile(r"[.!?]+\s+")

# The FIRST chunk of a turn is the one the caller is waiting on, and it is the
# only one whose latency is not hidden behind audio already playing. Measured:
# a turn whose opening sentence was 40 characters took 5.9 SECONDS to produce
# any audio at all, because nothing could be synthesized until the closing full
# stop arrived. Every later chunk is generated while the previous one is still
# being spoken, so they can wait for a proper sentence end and keep the better
# prosody.
#
# So for the first chunk only, a clause boundary is also a comma, a Tamil/Latin
# dash, or a semicolon - the places a speaker would naturally draw breath.
_FIRST_CHUNK_BOUNDARY_RE = re.compile(r"[.!?,;—]+\s+")

# ...and failing any punctuation at all, break at a word boundary once the
# opening has run this long. A model that opens with a long comma-less clause
# would otherwise still hold the audio.
#
# Measured (HANDOFF.md Sec6b): 32 -> 24 -> 18 changed NOTHING (2.38/2.43/2.43s
# to first clause) because the model's opening sentence ends at a period long
# before this cap applies. At 12 it finally moved (2.37 -> 1.89s) and did it by
# cutting mid-phrase, dropping the honorific. Lower it only with a stopwatch.
FIRST_CHUNK_MAX_CHARS = int(os.getenv("LLM_FIRST_CHUNK_MAX_CHARS", "32"))


class ClauseChunker:
    """Buffers streamed text and yields complete clauses as they close."""

    def __init__(self, fast_first_chunk: bool = True) -> None:
        self._buffer = ""
        # Whether this turn has released anything yet. Only the first release
        # gets the low-latency boundary rules - see _FIRST_CHUNK_BOUNDARY_RE.
        self._released_any = False
        self._fast_first_chunk = fast_first_chunk

    def feed(self, text: str) -> list[str]:
        """Add text to the buffer; return any clauses that closed as a result."""
        self._buffer += text
        clauses: list[str] = []

        while True:
            boundary = self._find_next_boundary(self._buffer)
            if boundary is None:
                break
            clause = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            if clause:
                clauses.append(clause)
                self._released_any = True
        return clauses

    def flush(self) -> str | None:
        """Return and clear any trailing text that never closed a clause."""
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder:
            self._released_any = True
        return remainder or None

    def _find_next_boundary(self, text: str) -> int | None:
        pattern = _BOUNDARY_RE
        if self._fast_first_chunk and not self._released_any:
            pattern = _FIRST_CHUNK_BOUNDARY_RE

        for match in pattern.finditer(text):
            if self._is_real_boundary(text, match.start()):
                return match.end()

        # No punctuation yet. For the opening chunk only, cut at a word
        # boundary rather than let the caller keep waiting in silence.
        if self._fast_first_chunk and not self._released_any:
            return self._word_boundary_after(text, FIRST_CHUNK_MAX_CHARS)
        return None

    @staticmethod
    def _word_boundary_after(text: str, minimum: int) -> int | None:
        """First whitespace at or after `minimum` characters, if any.

        Requires real trailing whitespace for the same reason the punctuation
        rule does: mid-stream, the end of the buffer is not evidence that a
        word has finished arriving.
        """
        if len(text) <= minimum:
            return None
        for index in range(minimum, len(text)):
            if text[index].isspace():
                return index + 1
        return None

    def _is_real_boundary(self, text: str, punct_start: int) -> bool:
        word_start = punct_start
        while word_start > 0 and (text[word_start - 1].isalpha() or text[word_start - 1] == "."):
            word_start -= 1
        word = text[word_start:punct_start].rstrip(".").lower()
        return word not in _ABBREVIATIONS
