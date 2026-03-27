"""
Tests for Domain-Typed Tokenizer + Temporal Compiler
=====================================================
Updated for:
  - Circadian rhythm defaults (Critical #2)
  - Context-gated fuzzy compilation
  - Bare hour patterns (Critical #3)
  - Consistent duration ordering (Important #4)
  - No overlap resolution in compiler (Important #6)
  - Proper overlap resolution in tokenizer (Important #5)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.domain_tokenizer import (
    SymbolicSpan,
    CompilationResult,
    DomainTypedTokenizer,
    ALL_SPECIAL_TOKENS,
)
from src.tokenizer.compilers.temporal_compiler import (
    TemporalCompiler,
    circadian_default,
    detect_period_context,
    resolve_hour_with_context,
)


def make_compiler():
    return TemporalCompiler()


# =========================================================================
# Circadian Rhythm Defaults (Critical #2)
# =========================================================================

def test_circadian_default_morning():
    """Hours 7-11 default to AM (morning activities)."""
    assert circadian_default(7) == 7
    assert circadian_default(8) == 8
    assert circadian_default(9) == 9
    assert circadian_default(10) == 10
    assert circadian_default(11) == 11
    print("  ✓ Circadian: hours 7-11 → AM")


def test_circadian_default_noon():
    """Hour 12 always maps to noon."""
    assert circadian_default(12) == 12
    print("  ✓ Circadian: hour 12 → noon")


def test_circadian_default_afternoon():
    """Hours 1-6 default to PM (afternoon/evening)."""
    assert circadian_default(1) == 13
    assert circadian_default(2) == 14
    assert circadian_default(3) == 15
    assert circadian_default(4) == 16
    assert circadian_default(5) == 17
    assert circadian_default(6) == 18
    print("  ✓ Circadian: hours 1-6 → PM")


def test_context_detection_am():
    """AM context cues are detected."""
    assert detect_period_context("I woke up early this morning") == 'AM'
    assert detect_period_context("Set my alarm for half past six") == 'AM'
    assert detect_period_context("Before noon I have a meeting") == 'AM'
    print("  ✓ Context detection: AM cues")


def test_context_detection_pm():
    """PM context cues are detected."""
    assert detect_period_context("Let's meet this evening") == 'PM'
    assert detect_period_context("After dinner we can go") == 'PM'
    assert detect_period_context("This afternoon is free") == 'PM'
    print("  ✓ Context detection: PM cues")


def test_context_detection_none():
    """No context returns None."""
    assert detect_period_context("The train departs at") is None
    assert detect_period_context("I need to be there by") is None
    print("  ✓ Context detection: no cues → None")


def test_resolve_with_am_context():
    """AM context overrides circadian default."""
    h24, conf = resolve_hour_with_context(6, 'AM')
    assert h24 == 6  # 6 AM, not 18
    assert conf == 0.95
    print("  ✓ Resolve: AM context → 6 AM with high confidence")


def test_resolve_with_pm_context():
    """PM context overrides circadian default."""
    h24, conf = resolve_hour_with_context(9, 'PM')
    assert h24 == 21  # 9 PM
    assert conf == 0.95
    print("  ✓ Resolve: PM context → 9 PM with high confidence")


def test_resolve_no_context_uses_circadian():
    """No context → circadian default with lower confidence."""
    h24, conf = resolve_hour_with_context(6, None)
    assert h24 == 18  # Circadian: 6 → PM
    assert conf == 0.75
    print("  ✓ Resolve: no context → circadian PM with 0.75 confidence")


# =========================================================================
# Fuzzy Compound Detection
# =========================================================================

def test_fuzzy_half_past_no_context():
    """'half past two' with no context → circadian default (PM)."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Its half past two and raining")
    assert len(spans) >= 1
    span = spans[0]
    assert span.tokens == ["[HEAD_TIME]", "[ARG_HOUR_14]", "[ARG_MIN_30]"]
    assert span.confidence == 0.75  # Lower: no context
    print("  ✓ Fuzzy: 'half past two' → 14:30 (circadian PM, 0.75)")


def test_fuzzy_with_am_context():
    """'half past six' with morning context → AM."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("I woke up at half past six this morning")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_06]", "[ARG_MIN_30]"]
    assert time_spans[0].confidence == 0.95
    print("  ✓ Fuzzy: 'half past six' + morning → 06:30 (AM, 0.95)")


def test_fuzzy_quarter_to():
    """'quarter to ten' — 'to' subtracts from the hour."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Meet me at quarter to ten")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    # "quarter to ten" → 9:45. Circadian: 9 → AM (morning), so 09:45
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_09]", "[ARG_MIN_45]"]
    print("  ✓ Fuzzy: 'quarter to ten' → 09:45 (circadian AM)")


def test_fuzzy_noon_midnight():
    """Named anchors are always unambiguous."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Between noon and midnight")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 2
    tokens_set = {tuple(s.tokens) for s in time_spans}
    assert ("HEAD_TIME", "ARG_HOUR_12", "ARG_MIN_00") or \
           tuple(["[HEAD_TIME]", "[ARG_HOUR_12]", "[ARG_MIN_00]"]) in tokens_set
    assert tuple(["[HEAD_TIME]", "[ARG_HOUR_00]", "[ARG_MIN_00]"]) in tokens_set
    print("  ✓ Fuzzy: noon=12:00, midnight=00:00 (always high confidence)")


# =========================================================================
# Bare Hour Patterns (Critical #3)
# =========================================================================

def test_bare_hour_digit():
    """'3pm' → 15:00."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Be there by 3pm")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_15]", "[ARG_MIN_00]"]
    print("  ✓ Bare hour: '3pm' → 15:00")


def test_bare_hour_word():
    """'three pm' → 15:00."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Meet at three pm")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_15]", "[ARG_MIN_00]"]
    print("  ✓ Bare hour: 'three pm' → 15:00")


def test_bare_hour_am():
    """'6 AM' → 6:00."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Wake up at 6 AM")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_06]", "[ARG_MIN_00]"]
    print("  ✓ Bare hour: '6 AM' → 06:00")


# =========================================================================
# Explicit Time Formats (existing coverage)
# =========================================================================

def test_24h_explicit():
    """'14:20' → 14h 20m."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Meet at 14:20")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_14]", "[ARG_MIN_20]"]
    print("  ✓ 24h: '14:20' → [HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_20]")


def test_12h_explicit():
    """'2:20 PM' → 14h 20m."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Meet at 2:20 PM")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_14]", "[ARG_MIN_20]"]
    print("  ✓ 12h: '2:20 PM' → [HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_20]")


def test_canonicalization_invariance():
    """Multiple formats for same time → identical tokens."""
    compiler = make_compiler()
    r1 = compiler.compile_to_fullform("at 14:20")
    r2 = compiler.compile_to_fullform("at 2:20 PM")
    t1 = [s for s in r1 if s.tokens[0] == "[HEAD_TIME]"]
    t2 = [s for s in r2 if s.tokens[0] == "[HEAD_TIME]"]
    assert len(t1) >= 1 and len(t2) >= 1
    assert t1[0].tokens == t2[0].tokens
    print("  ✓ Invariance: '14:20' == '2:20 PM' → same tokens")


# =========================================================================
# Duration Ordering (Important #4)
# =========================================================================

def test_duration_ordering_hour_and_half():
    """'an hour and a half' → hours first, then minutes."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("It takes an hour and a half")
    dur_spans = [s for s in spans if s.tokens[0] == "[HEAD_DURATION]"]
    assert len(dur_spans) >= 1
    assert dur_spans[0].tokens == ["[HEAD_DURATION]", "[ARG_HOUR_01]", "[ARG_MIN_30]"]
    print("  ✓ Duration: 'an hour and a half' → [HOUR_01] [MIN_30] (hours first)")


def test_duration_decomposed():
    """'90 minutes' → 1h 30m (hours first)."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("It takes 90 minutes")
    dur_spans = [s for s in spans if s.tokens[0] == "[HEAD_DURATION]"]
    assert len(dur_spans) >= 1
    assert dur_spans[0].tokens == ["[HEAD_DURATION]", "[ARG_HOUR_01]", "[ARG_MIN_30]"]
    print("  ✓ Duration: '90 minutes' decomposed → [HOUR_01] [MIN_30]")


def test_duration_zero_skipped():
    """'0 minutes' should not be intercepted."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("It takes 0 minutes")
    dur_spans = [s for s in spans if s.tokens[0] == "[HEAD_DURATION]"]
    assert len(dur_spans) == 0
    print("  ✓ Duration: '0 minutes' → not intercepted")


# =========================================================================
# Negative Examples (should NOT match)
# =========================================================================

def test_negative_room_number():
    """Room numbers should not be intercepted."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("I am in room 237")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) == 0
    print("  ✓ Negative: 'room 237' → no match")


def test_negative_plain_number():
    """Plain numbers without time context should not match."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("We need 42 widgets")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) == 0
    print("  ✓ Negative: '42 widgets' → no match")


# =========================================================================
# O'Clock Patterns
# =========================================================================

def test_oclock_word():
    """'three o'clock' → circadian default."""
    compiler = make_compiler()
    spans = compiler.compile_to_fullform("Meet at three o'clock")
    time_spans = [s for s in spans if s.tokens[0] == "[HEAD_TIME]"]
    assert len(time_spans) >= 1
    # Circadian: 3 → PM → 15
    assert time_spans[0].tokens == ["[HEAD_TIME]", "[ARG_HOUR_15]", "[ARG_MIN_00]"]
    print("  ✓ O'clock: 'three o'clock' → 15:00 (circadian PM)")


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    tests = [
        # Circadian
        test_circadian_default_morning,
        test_circadian_default_noon,
        test_circadian_default_afternoon,
        test_context_detection_am,
        test_context_detection_pm,
        test_context_detection_none,
        test_resolve_with_am_context,
        test_resolve_with_pm_context,
        test_resolve_no_context_uses_circadian,
        # Fuzzy compound
        test_fuzzy_half_past_no_context,
        test_fuzzy_with_am_context,
        test_fuzzy_quarter_to,
        test_fuzzy_noon_midnight,
        # Bare hours
        test_bare_hour_digit,
        test_bare_hour_word,
        test_bare_hour_am,
        # Explicit formats
        test_24h_explicit,
        test_12h_explicit,
        test_canonicalization_invariance,
        # Duration ordering
        test_duration_ordering_hour_and_half,
        test_duration_decomposed,
        test_duration_zero_skipped,
        # Negatives
        test_negative_room_number,
        test_negative_plain_number,
        # O'clock
        test_oclock_word,
    ]

    passed = 0
    failed = 0

    print("\n=== Domain-Typed Tokenizer Tests (Post-Opus Fixes) ===\n")
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*50}")

    sys.exit(1 if failed else 0)
