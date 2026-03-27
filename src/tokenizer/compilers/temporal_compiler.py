"""
Temporal Compiler — Symbolic Expression Detector for Time
==========================================================
The first domain-specific plugin for the Domain-Typed Tokenizer.

Detects temporal constructs in natural language and compiles them
into Wolfram-style Head/Argument token sequences.

Coverage (v1):
  ✓ Explicit 24h times:  "14:20", "09:58"
  ✓ Explicit 12h times:  "2:20 PM", "9:58 am"
  ✓ Bare hour + period:  "3pm", "3 PM", "three pm"
  ✓ Dot-separated:       "2.15pm", "14.20"
  ✓ Fuzzy/spoken times:  "half past two", "quarter to ten", "noon", "midnight"
    (context-gated: only compiled when AM/PM context exists, or circadian default applies)
  ✓ Duration expressions: "5 minutes", "2 hours", "an hour and a half"

Context-Gated Fuzzy Compilation:
  Fuzzy times are inherently ambiguous (AM vs PM). We use circadian rhythm
  sensibility defaults: most human activity happens between 07:00-23:00.
  - Hours 7-11  → AM (waking, morning activities)
  - Hours 12    → PM (noon anchor, always unambiguous)
  - Hours 1-6   → PM (afternoon/evening, more common than 1am-6am)
  - Explicit context ("in the morning", "in the evening") overrides defaults
  - Confidence is reduced for ambiguous cases — 0.75 instead of 0.95

Known limitations (documented, not hidden):
  - Cultural variants ("half two" = 1:30 in German, 2:30 in British) not handled
  - "twenty-five past X" and arbitrary minute-word variants not in lookup table
  - Relative references ("4 weeks last Friday") NOT handled in v1 compiler

See: Issue #13, #7, #17
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.tokenizer.domain_tokenizer import SymbolicExpressionDetector, SymbolicSpan


# ---------------------------------------------------------------------------
# Circadian Rhythm Defaults
# ---------------------------------------------------------------------------
# Most human activity occurs between 07:00-23:00. When a fuzzy time like
# "half past six" is ambiguous, we assume the waking-hours interpretation.
#
# Hour word → (AM_hour, PM_hour, default_24h_hour)
# The default is the circadian-likely one.

HOUR_WORDS = {
    "one":    1,
    "two":    2,
    "three":  3,
    "four":   4,
    "five":   5,
    "six":    6,
    "seven":  7,
    "eight":  8,
    "nine":   9,
    "ten":    10,
    "eleven": 11,
    "twelve": 12,
}

def circadian_default(hour_12: int) -> int:
    """Apply circadian rhythm sensibility to a 12-hour time.

    Returns the 24-hour value that is more likely during waking hours.
    - 7, 8, 9, 10, 11 → morning (AM): 7, 8, 9, 10, 11
    - 12              → noon: 12
    - 1, 2, 3, 4, 5, 6 → afternoon/evening (PM): 13, 14, 15, 16, 17, 18
    """
    if 7 <= hour_12 <= 11:
        return hour_12        # AM — morning activities
    elif hour_12 == 12:
        return 12             # Noon
    else:
        return hour_12 + 12   # PM — afternoon/evening (1→13, 6→18)


# ---------------------------------------------------------------------------
# Context Cue Detection
# ---------------------------------------------------------------------------

AM_CONTEXT_CUES = re.compile(
    r'\b(?:in\s+the\s+morning|this\s+morning|a\.?m\.?|before\s+noon|'
    r'early\s+morning|at\s+dawn|sunrise|breakfast|wake|woke|alarm)\b',
    re.IGNORECASE
)
PM_CONTEXT_CUES = re.compile(
    r'\b(?:in\s+the\s+evening|in\s+the\s+afternoon|this\s+evening|'
    r'this\s+afternoon|p\.?m\.?|after\s+noon|tonight|dinner|supper|'
    r'sunset|dusk|late\s+afternoon|late\s+evening)\b',
    re.IGNORECASE
)
NIGHT_CONTEXT_CUES = re.compile(
    r'\b(?:at\s+night|late\s+night|overnight|after\s+midnight|'
    r'small\s+hours|wee\s+hours|insomnia|nightshift|night\s+shift)\b',
    re.IGNORECASE
)

def detect_period_context(text: str) -> Optional[str]:
    """Scan text for AM/PM/night context cues.

    Returns: 'AM', 'PM', 'NIGHT', or None if no context found.
    """
    if AM_CONTEXT_CUES.search(text):
        return 'AM'
    if PM_CONTEXT_CUES.search(text):
        return 'PM'
    if NIGHT_CONTEXT_CUES.search(text):
        return 'NIGHT'
    return None


def resolve_hour_with_context(hour_12: int, context: Optional[str]) -> Tuple[int, float]:
    """Resolve a 12-hour value to 24h using context and circadian defaults.

    Returns: (hour_24, confidence)
    """
    if context == 'AM':
        h24 = hour_12 if hour_12 != 12 else 0
        return (h24, 0.95)  # Explicit context → high confidence
    elif context == 'PM':
        h24 = hour_12 + 12 if hour_12 != 12 else 12
        return (h24, 0.95)
    elif context == 'NIGHT':
        # Night context: small hours (1-4 AM) or late evening (10-11 PM)
        if hour_12 <= 4:
            return (hour_12, 0.85)    # 1am-4am
        elif hour_12 >= 10:
            return (hour_12 + 12, 0.85) if hour_12 != 12 else (0, 0.85)
        else:
            return (circadian_default(hour_12), 0.70)  # Ambiguous even with night context
    else:
        # No context: use circadian defaults with reduced confidence
        h24 = circadian_default(hour_12)
        return (h24, 0.75)


# ---------------------------------------------------------------------------
# Fuzzy Time Lookup (with circadian-aware resolution)
# ---------------------------------------------------------------------------
# Instead of hardcoding PM hours, we store the 12h base and let
# circadian/context resolution determine AM/PM.

FUZZY_PATTERNS = {
    # (phrase_template, minute_offset_from_hour, is_past=True)
    # "half past X" → hour + 30min
    "half past":     (30, True),
    "quarter past":  (15, True),
    "quarter to":    (45, False),  # "quarter to X" → X-1 hour + 45min
    "twenty past":   (20, True),
    "twenty to":     (40, False),
    "ten past":      (10, True),
    "ten to":        (50, False),
    "five past":     (5,  True),
    "five to":       (55, False),
}

# Named anchors (unambiguous)
NAMED_TIMES = {
    "noon":     (12, 0, 0.95),
    "midday":   (12, 0, 0.95),
    "midnight": (0,  0, 0.95),
}


# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

RE_24H = re.compile(
    r'\b([01]?\d|2[0-3]):([0-5]\d)\b'
)

RE_12H = re.compile(
    r'\b(1[0-2]|0?[1-9]):([0-5]\d)\s*([AaPp]\.?[Mm]\.?)\b'
)

# Bare hour with AM/PM: "3pm", "3 PM", "3PM" (no minutes)
RE_BARE_12H = re.compile(
    r'\b(1[0-2]|0?[1-9])\s*([AaPp]\.?[Mm]\.?)\b'
)

RE_DOT_24H = re.compile(
    r'\b([01]?\d|2[0-3])\.([0-5]\d)\b(?!\s*[AaPp])'
)
RE_DOT_12H = re.compile(
    r'\b(1[0-2]|0?[1-9])\.([0-5]\d)\s*([AaPp]\.?[Mm]\.?)\b'
)

RE_DURATION_MINUTES = re.compile(
    r'\b(\d+)\s*(?:minutes?|mins?)\b', re.IGNORECASE
)
RE_DURATION_HOURS = re.compile(
    r'\b(\d+)\s*(?:hours?|hrs?)\b', re.IGNORECASE
)
RE_DURATION_HOUR_AND_HALF = re.compile(
    r'\b(?:an?\s+)?hour\s+and\s+a\s+half\b', re.IGNORECASE
)

# Fuzzy compound pattern: "{modifier} {past/to} {hour_word}"
RE_FUZZY_COMPOUND = re.compile(
    r'\b(half|quarter|twenty|ten|five)\s+(past|to)\s+'
    r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|midnight)\b',
    re.IGNORECASE
)

# Bare word hours with AM/PM: "three pm", "six am"
RE_WORD_HOUR_PERIOD = re.compile(
    r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+'
    r'([AaPp]\.?[Mm]\.?)\b',
    re.IGNORECASE
)

# O'clock: "three o'clock", "3 o'clock"
RE_OCLOCK_WORD = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+o'clock\b",
    re.IGNORECASE
)
RE_OCLOCK_DIGIT = re.compile(
    r"\b(1[0-2]|0?[1-9])\s+o'clock\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Temporal Compiler
# ---------------------------------------------------------------------------

class TemporalCompiler(SymbolicExpressionDetector):
    """Detects temporal expressions and compiles them to typed token sequences.

    Design principles:
      1. Only intercept when confident. Ambiguous → BPE fallback.
      2. Circadian sensibility: assume waking hours for ambiguous fuzzy times.
      3. Context cues ("in the morning") increase confidence and override defaults.
    """

    @property
    def domain_name(self) -> str:
        return "temporal"

    @property
    def special_tokens(self) -> List[str]:
        return []

    def compile_to_fullform(self, text: str) -> List[SymbolicSpan]:
        """Scan text for temporal constructs and return typed token spans.

        Note: Overlap resolution is NOT done here. It is done once in
        the parent DomainTypedTokenizer.compile() to resolve across all
        detector plugins globally. (Fix for Issue #6: duplicate resolution)
        """
        spans: List[SymbolicSpan] = []

        # Detect period context from the full text (used by fuzzy detectors)
        period_context = detect_period_context(text)

        # Priority 1: Named anchors (noon, midnight — always unambiguous)
        spans.extend(self._detect_named(text))

        # Priority 2: Fuzzy compound ("half past six", "quarter to ten")
        spans.extend(self._detect_fuzzy_compound(text, period_context))

        # Priority 3: Word hours with period ("three pm", "six am")
        spans.extend(self._detect_word_hour_period(text))

        # Priority 4: O'clock ("three o'clock", "3 o'clock")
        spans.extend(self._detect_oclock(text, period_context))

        # Priority 5: 12h format with AM/PM (before 24h)
        spans.extend(self._detect_12h(text))

        # Priority 6: Bare hour with AM/PM ("3pm", "3 PM")
        spans.extend(self._detect_bare_12h(text))

        # Priority 7: Dot-separated with AM/PM
        spans.extend(self._detect_dot_12h(text))

        # Priority 8: 24h format
        spans.extend(self._detect_24h(text))

        # Priority 9: Dot-separated 24h
        spans.extend(self._detect_dot_24h(text))

        # Priority 10: Duration expressions
        spans.extend(self._detect_durations(text))

        return spans

    # --- Detectors ---

    def _detect_named(self, text: str) -> List[SymbolicSpan]:
        """Detect named time anchors: noon, midnight, midday."""
        spans = []
        text_lower = text.lower()
        for name, (h, m, conf) in NAMED_TIMES.items():
            idx = 0
            while True:
                pos = text_lower.find(name, idx)
                if pos == -1:
                    break
                end = pos + len(name)
                if self._check_word_boundary(text_lower, pos, end):
                    spans.append(SymbolicSpan(
                        start=pos, end=end,
                        tokens=self._time_tokens(h, m),
                        confidence=conf,
                        source_text=text[pos:end],
                    ))
                idx = pos + 1
        return spans

    def _detect_fuzzy_compound(self, text: str, context: Optional[str]) -> List[SymbolicSpan]:
        """Detect fuzzy compound patterns: 'half past six', 'quarter to ten'."""
        spans = []
        for match in RE_FUZZY_COMPOUND.finditer(text):
            modifier = match.group(1).lower()
            direction = match.group(2).lower()
            hour_word = match.group(3).lower()

            if hour_word == "midnight":
                # "quarter to midnight" → 23:45
                if direction == "to":
                    pattern_data = FUZZY_PATTERNS.get(f"{modifier} to")
                    if pattern_data:
                        mins, _ = pattern_data
                        spans.append(SymbolicSpan(
                            start=match.start(), end=match.end(),
                            tokens=self._time_tokens(24 - 1, mins),
                            confidence=0.90,
                            source_text=match.group(),
                        ))
                continue

            hour_12 = HOUR_WORDS.get(hour_word)
            if hour_12 is None:
                continue

            pattern_key = f"{modifier} {direction}"
            pattern_data = FUZZY_PATTERNS.get(pattern_key)
            if pattern_data is None:
                continue

            mins, is_past = pattern_data
            if is_past:
                # "X past Y" → hour = Y, minute = mins
                resolved_h, conf = resolve_hour_with_context(hour_12, context)
                final_m = mins
            else:
                # "X to Y" → hour = Y-1, minute = mins
                prev_hour_12 = 12 if hour_12 == 1 else hour_12 - 1
                resolved_h, conf = resolve_hour_with_context(prev_hour_12, context)
                final_m = mins

            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(resolved_h, final_m),
                confidence=conf,
                source_text=match.group(),
            ))

        return spans

    def _detect_word_hour_period(self, text: str) -> List[SymbolicSpan]:
        """Detect word hours with AM/PM: 'three pm', 'six am'."""
        spans = []
        for match in RE_WORD_HOUR_PERIOD.finditer(text):
            hour_word = match.group(1).lower()
            period = match.group(2).upper().replace(".", "")
            hour_12 = HOUR_WORDS.get(hour_word)
            if hour_12 is None:
                continue
            if period == "PM" and hour_12 != 12:
                h24 = hour_12 + 12
            elif period == "AM" and hour_12 == 12:
                h24 = 0
            elif period == "AM":
                h24 = hour_12
            else:
                h24 = hour_12
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(h24, 0),
                confidence=0.95,
                source_text=match.group(),
            ))
        return spans

    def _detect_oclock(self, text: str, context: Optional[str]) -> List[SymbolicSpan]:
        """Detect o'clock patterns: 'three o'clock', '3 o'clock'."""
        spans = []
        for match in RE_OCLOCK_WORD.finditer(text):
            hour_word = match.group(1).lower()
            hour_12 = HOUR_WORDS.get(hour_word)
            if hour_12 is None:
                continue
            resolved_h, conf = resolve_hour_with_context(hour_12, context)
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(resolved_h, 0),
                confidence=conf,
                source_text=match.group(),
            ))
        for match in RE_OCLOCK_DIGIT.finditer(text):
            hour_12 = int(match.group(1))
            resolved_h, conf = resolve_hour_with_context(hour_12, context)
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(resolved_h, 0),
                confidence=conf,
                source_text=match.group(),
            ))
        return spans

    def _detect_24h(self, text: str) -> List[SymbolicSpan]:
        """Detect 24-hour format times: 09:58, 14:20, 0:05."""
        spans = []
        for match in RE_24H.finditer(text):
            h, m = int(match.group(1)), int(match.group(2))
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(h, m),
                confidence=0.90,
                source_text=match.group(),
            ))
        return spans

    def _detect_12h(self, text: str) -> List[SymbolicSpan]:
        """Detect 12-hour format times: 2:20 PM, 9:58am."""
        spans = []
        for match in RE_12H.finditer(text):
            h, m = int(match.group(1)), int(match.group(2))
            period = match.group(3).upper().replace(".", "")
            if period == "PM" and h != 12:
                h += 12
            elif period == "AM" and h == 12:
                h = 0
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(h, m),
                confidence=0.95,
                source_text=match.group(),
            ))
        return spans

    def _detect_bare_12h(self, text: str) -> List[SymbolicSpan]:
        """Detect bare hour with AM/PM: '3pm', '3 PM'."""
        spans = []
        for match in RE_BARE_12H.finditer(text):
            h = int(match.group(1))
            period = match.group(2).upper().replace(".", "")
            if period == "PM" and h != 12:
                h += 12
            elif period == "AM" and h == 12:
                h = 0
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(h, 0),
                confidence=0.95,
                source_text=match.group(),
            ))
        return spans

    def _detect_dot_12h(self, text: str) -> List[SymbolicSpan]:
        """Detect dot-separated 12h times: 2.15pm."""
        spans = []
        for match in RE_DOT_12H.finditer(text):
            h, m = int(match.group(1)), int(match.group(2))
            period = match.group(3).upper().replace(".", "")
            if period == "PM" and h != 12:
                h += 12
            elif period == "AM" and h == 12:
                h = 0
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=self._time_tokens(h, m),
                confidence=0.90,
                source_text=match.group(),
            ))
        return spans

    def _detect_dot_24h(self, text: str) -> List[SymbolicSpan]:
        """Detect dot-separated 24h times: 14.20."""
        spans = []
        for match in RE_DOT_24H.finditer(text):
            h, m = int(match.group(1)), int(match.group(2))
            if 0 <= h <= 23:
                spans.append(SymbolicSpan(
                    start=match.start(), end=match.end(),
                    tokens=self._time_tokens(h, m),
                    confidence=0.70,
                    source_text=match.group(),
                ))
        return spans

    def _detect_durations(self, text: str) -> List[SymbolicSpan]:
        """Detect duration expressions: 5 minutes, 2 hours, an hour and a half.

        Duration token ordering: always [HEAD_DURATION] [ARG_HOUR_XX] [ARG_MIN_XX]
        (hours first, then minutes — Fix for Issue #4: consistent ordering)
        """
        spans = []

        # "an hour and a half" → 1h30m
        for match in RE_DURATION_HOUR_AND_HALF.finditer(text):
            spans.append(SymbolicSpan(
                start=match.start(), end=match.end(),
                tokens=["[HEAD_DURATION]", "[ARG_HOUR_01]", "[ARG_MIN_30]"],
                confidence=0.95,
                source_text=match.group(),
            ))

        # "X minutes"
        for match in RE_DURATION_MINUTES.finditer(text):
            mins = int(match.group(1))
            if mins == 0:
                continue  # Zero-duration: don't intercept
            elif 1 <= mins <= 59:
                spans.append(SymbolicSpan(
                    start=match.start(), end=match.end(),
                    tokens=["[HEAD_DURATION]", f"[ARG_MIN_{mins:02d}]"],
                    confidence=0.90,
                    source_text=match.group(),
                ))
            else:
                # Multi-hour durations: decompose (cap at 23h59m)
                hours = min(mins // 60, 23)
                remaining_mins = mins % 60
                tokens = ["[HEAD_DURATION]"]
                if hours > 0:
                    tokens.append(f"[ARG_HOUR_{hours:02d}]")
                if remaining_mins > 0:
                    tokens.append(f"[ARG_MIN_{remaining_mins:02d}]")
                if len(tokens) > 1:
                    spans.append(SymbolicSpan(
                        start=match.start(), end=match.end(),
                        tokens=tokens,
                        confidence=0.85,
                        source_text=match.group(),
                    ))

        # "X hours"
        for match in RE_DURATION_HOURS.finditer(text):
            hours = int(match.group(1))
            if 0 < hours < 24:
                spans.append(SymbolicSpan(
                    start=match.start(), end=match.end(),
                    tokens=["[HEAD_DURATION]", f"[ARG_HOUR_{hours:02d}]"],
                    confidence=0.90,
                    source_text=match.group(),
                ))

        return spans

    # --- Helpers ---

    @staticmethod
    def _time_tokens(h: int, m: int) -> List[str]:
        return ["[HEAD_TIME]", f"[ARG_HOUR_{h:02d}]", f"[ARG_MIN_{m:02d}]"]

    @staticmethod
    def _check_word_boundary(text: str, start: int, end: int) -> bool:
        """Check that a match is at word boundaries (not mid-word)."""
        before_ok = (start == 0 or not text[start - 1].isalnum())
        after_ok = (end >= len(text) or not text[end].isalnum())
        return before_ok and after_ok
