"""
Domain-Typed Tokenizer — Symbolic Expression Detection Layer
=============================================================
Option B Architecture: The LLM routes, Luxon computes.
This module implements Stage 1 (Pre-LLM): intercepting formal constructs
in natural language and canonicalizing them into typed symbolic tokens
before BPE encoding.

Wolfram-inspired: every formal token has a Head (type) and Arguments (operands).

See: Issues #13, #14, #15 in github.com/mgosal/tagzeit-gemma-2b
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
except ImportError:
    # Allow running compilation logic and tests without transformers installed
    PreTrainedTokenizerFast = None
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicSpan:
    """A detected formal construct within the input text.
    
    Attributes:
        start: Character offset where the formal construct begins.
        end: Character offset where the formal construct ends.
        tokens: The typed token sequence to replace this span with.
                e.g. ["[HEAD_TIME]", "[ARG_HOUR_14]", "[ARG_MIN_20]"]
        confidence: How confident the detector is (0.0 to 1.0).
                    Below the threshold, we fall back to BPE.
        source_text: The original text that was matched (for debugging).
    """
    start: int
    end: int
    tokens: List[str]
    confidence: float
    source_text: str = ""


@dataclass
class CompilationResult:
    """Result of compiling a full text string through all detectors.
    
    Attributes:
        compiled_text: The text with formal spans replaced by typed tokens.
        spans: The list of detected spans (for inspection / debugging).
        fallback_spans: Spans that were detected but fell below the
                        confidence threshold and were left for BPE.
    """
    compiled_text: str
    spans: List[SymbolicSpan] = field(default_factory=list)
    fallback_spans: List[SymbolicSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract Detector Protocol
# ---------------------------------------------------------------------------

class SymbolicExpressionDetector(ABC):
    """Base protocol for domain-specific symbolic expression detectors.
    
    Each detector is responsible for a single formal domain (temporal,
    arithmetic, spatial, etc.). It scans input text and identifies spans
    that contain formal computational content, compiling them into
    Wolfram-style Head/Argument token sequences.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Human-readable name for this detector's domain."""
        ...

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        """All special tokens this detector may emit.
        
        These are registered with the tokenizer at initialization time
        so that BPE never fragments them.
        """
        ...

    @abstractmethod
    def compile_to_fullform(self, text: str) -> List[SymbolicSpan]:
        """Scan text and return all detected formal construct spans.
        
        Each span contains the character offsets and the typed token
        sequence that should replace that region of text.
        
        Args:
            text: The raw natural language input.
            
        Returns:
            List of SymbolicSpan objects, sorted by start offset.
            Spans must NOT overlap.
        """
        ...


# ---------------------------------------------------------------------------
# Token Registry
# ---------------------------------------------------------------------------

# Routing tokens — emitted by the LLM to dispatch to external engines
ROUTING_TOKENS = [
    "[ROUTE_TIME_ADD]",
    "[ROUTE_TIME_SUB]",
    "[ROUTE_DURATION_BETWEEN]",
    "[ROUTE_CALENDAR_SHIFT]",
    "[ROUTE_TIMEZONE_CONVERT]",
]

# Head tokens — type identifiers for formal constructs
HEAD_TOKENS = [
    "[HEAD_TIME]",
    "[HEAD_DURATION]",
    "[HEAD_PLUS]",
    "[HEAD_MINUS]",
    "[HEAD_DATE]",
    "[HEAD_DATETIME]",
]

# Argument tokens — operands for temporal constructs
ARG_HOUR_TOKENS = [f"[ARG_HOUR_{h:02d}]" for h in range(24)]
ARG_MIN_TOKENS = [f"[ARG_MIN_{m:02d}]" for m in range(60)]

# Reference tokens — for multi-operand routing
REF_TOKENS = [f"[REF_{i}]" for i in range(1, 10)]

ALL_SPECIAL_TOKENS = (
    ROUTING_TOKENS + HEAD_TOKENS +
    ARG_HOUR_TOKENS + ARG_MIN_TOKENS +
    REF_TOKENS
)


# ---------------------------------------------------------------------------
# Domain-Typed Tokenizer Wrapper
# ---------------------------------------------------------------------------

class DomainTypedTokenizer:
    """Wraps a HuggingFace tokenizer with symbolic expression detection.
    
    This is the main entry point. It:
      1. Runs all registered SymbolicExpressionDetectors on the input text.
      2. Replaces detected spans (above confidence threshold) with typed tokens.
      3. Falls back to standard BPE for everything else.
    
    Usage:
        base = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        tokenizer = DomainTypedTokenizer(base, confidence_threshold=0.8)
        tokenizer.register_detector(TemporalCompiler())
        result = tokenizer.compile("its twenty past two and my train is at 3pm")
        ids = tokenizer.encode(result.compiled_text)
    """

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerFast,
        confidence_threshold: float = 0.8,
    ):
        self.base_tokenizer = base_tokenizer
        self.confidence_threshold = confidence_threshold
        self._detectors: List[SymbolicExpressionDetector] = []
        self._tokens_registered = False

    def register_detector(self, detector: SymbolicExpressionDetector) -> None:
        """Register a domain-specific symbolic expression detector."""
        self._detectors.append(detector)
        # Defer token registration until all detectors are added
        self._tokens_registered = False

    def _ensure_tokens_registered(self) -> None:
        """Register all special tokens from all detectors with the base tokenizer."""
        if self._tokens_registered:
            return

        all_tokens = list(ALL_SPECIAL_TOKENS)
        for detector in self._detectors:
            for token in detector.special_tokens:
                if token not in all_tokens:
                    all_tokens.append(token)

        num_added = self.base_tokenizer.add_special_tokens(
            {"additional_special_tokens": all_tokens}
        )
        if num_added > 0:
            print(f"[DomainTypedTokenizer] Registered {num_added} new special tokens.")

        self._tokens_registered = True

    def compile(self, text: str) -> CompilationResult:
        """Run all detectors on the input text and replace formal spans.
        
        Args:
            text: Raw natural language input.
            
        Returns:
            CompilationResult with the compiled text and span metadata.
        """
        self._ensure_tokens_registered()

        # Collect spans from all detectors
        all_spans: List[SymbolicSpan] = []
        for detector in self._detectors:
            spans = detector.compile_to_fullform(text)
            all_spans.extend(spans)

        # Sort by start offset, then by confidence descending
        all_spans.sort(key=lambda s: (s.start, -s.confidence))

        # Resolve overlapping spans properly:
        # - For full containment: keep the higher confidence span
        # - For partial overlap: keep the first (higher confidence due to sort)
        #   and drop the overlapping one entirely (don't fragment)
        resolved: List[SymbolicSpan] = []
        for span in all_spans:
            if not resolved:
                resolved.append(span)
                continue
            last = resolved[-1]
            if span.start >= last.end:
                # No overlap
                resolved.append(span)
            elif span.start >= last.start and span.end <= last.end:
                # Fully contained: keep whichever has higher confidence
                if span.confidence > last.confidence:
                    resolved[-1] = span
                # else: skip (contained with lower confidence)
            else:
                # Partial overlap: skip the new span (first-wins, already sorted by conf)
                pass

        # Split into accepted (above threshold) and fallback (below)
        accepted: List[SymbolicSpan] = []
        fallback: List[SymbolicSpan] = []
        for span in resolved:
            if span.confidence >= self.confidence_threshold:
                accepted.append(span)
            else:
                fallback.append(span)

        # Build the compiled text by replacing accepted spans
        parts: List[str] = []
        cursor = 0
        for span in accepted:
            if span.start > cursor:
                parts.append(text[cursor:span.start])
            parts.append(" ".join(span.tokens))
            cursor = span.end

        # Append any remaining text after the last span
        if cursor < len(text):
            parts.append(text[cursor:])

        compiled_text = "".join(parts)

        return CompilationResult(
            compiled_text=compiled_text,
            spans=accepted,
            fallback_spans=fallback,
        )

    def encode(self, text: str, **kwargs) -> List[int]:
        """Compile text through detectors, then encode with the base tokenizer."""
        result = self.compile(text)
        return self.base_tokenizer.encode(result.compiled_text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs back to text using the base tokenizer."""
        return self.base_tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size including all special tokens."""
        self._ensure_tokens_registered()
        return len(self.base_tokenizer)

    def get_token_id(self, token: str) -> int:
        """Get the ID for a specific special token.

        Raises:
            ValueError: If the token was not registered or was BPE-fragmented.
        """
        self._ensure_tokens_registered()
        ids = self.base_tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Token '{token}' was not registered as a single special token. "
                f"Got {len(ids)} sub-tokens: {ids}"
            )
        return ids[0]
