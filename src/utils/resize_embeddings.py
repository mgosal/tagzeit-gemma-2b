"""
Model Embedding Preparation — Geometric Initialization
========================================================
Resizes the model's embedding matrix to accommodate new Head/Argument tokens,
then initializes the new embeddings using sinusoidal geometric positioning.

This ensures:
  - [ARG_HOUR_14] is geometrically near [ARG_HOUR_13] and [ARG_HOUR_15]
  - [ARG_MIN_30] is geometrically near [ARG_MIN_29] and [ARG_MIN_31]
  - HEAD tokens cluster together in a separate region

Opus recommendation: geometric init must be v1 scope, not future scope.
Without it, every new token starts from random noise and the model has to
simultaneously learn what each token means AND how they compose.

See: Implementation Plan v2, Section 4
"""

from __future__ import annotations

import math
import hashlib
import argparse
from typing import List, Optional

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

from src.tokenizer.domain_tokenizer import (
    ALL_SPECIAL_TOKENS,
    HEAD_TOKENS,
    ARG_HOUR_TOKENS,
    ARG_MIN_TOKENS,
    ROUTING_TOKENS,
    REF_TOKENS,
)


def sinusoidal_encoding(position: int, d_model: int, max_period: int = 10000) -> np.ndarray:
    """Generate a sinusoidal positional encoding vector.
    
    Uses the same formula as the original Transformer paper
    (Vaswani et al. 2017), but applied to semantic position
    (hour index, minute index) instead of sequence position.
    """
    encoding = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = max_period ** (i / d_model)
        encoding[i] = math.sin(position / div_term)
        if i + 1 < d_model:
            encoding[i + 1] = math.cos(position / div_term)
    return encoding


def compute_geometric_init(
    token: str,
    d_model: int,
    existing_mean: np.ndarray,
    existing_std: float,
) -> np.ndarray:
    """Compute the geometric initialization for a single special token.

    Strategy — orthogonal subspaces (Issue #10):
      - ARG_HOUR_XX: Sinusoidal encoding on EVEN dimensions (0, 2, 4...).
      - ARG_MIN_XX:  Sinusoidal encoding on ODD dimensions (1, 3, 5...).
      This ensures hours and minutes are in orthogonal subspaces.
      - HEAD_*, ROUTE_*, REF_*: Small deterministic perturbation from mean.
    """
    def _deterministic_seed(s: str) -> int:
        """Reproducible seed from string (Issue #11: no Python hash())."""
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

    if token.startswith("[ARG_HOUR_"):
        hour = int(token.split("_")[-1].rstrip("]"))
        # Encode on even dimensions only
        base = np.zeros(d_model)
        for i in range(0, d_model, 2):
            dim_idx = i // 2
            div_term = 10000 ** (dim_idx / (d_model // 2))
            base[i] = math.sin(hour / div_term)
        return existing_mean + base * existing_std * 0.5

    elif token.startswith("[ARG_MIN_"):
        minute = int(token.split("_")[-1].rstrip("]"))
        # Encode on odd dimensions only
        base = np.zeros(d_model)
        for i in range(1, d_model, 2):
            dim_idx = i // 2
            div_term = 10000 ** (dim_idx / (d_model // 2))
            base[i] = math.sin(minute / div_term)
        return existing_mean + base * existing_std * 0.5

    elif token.startswith("[HEAD_"):
        rng = np.random.default_rng(_deterministic_seed(token))
        perturbation = rng.normal(0, existing_std * 0.1, d_model)
        return existing_mean + perturbation

    elif token.startswith("[ROUTE_"):
        rng = np.random.default_rng(_deterministic_seed(token))
        perturbation = rng.normal(0, existing_std * 0.1, d_model)
        offset = sinusoidal_encoding(50, d_model) * existing_std * 0.2
        return existing_mean + perturbation + offset

    else:
        rng = np.random.default_rng(_deterministic_seed(token))
        perturbation = rng.normal(0, existing_std * 0.1, d_model)
        return existing_mean + perturbation


def resize_and_initialize(
    model_id: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Resize a model's embeddings and initialize new tokens geometrically.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'google/gemma-2-2b')
        output_dir: Directory to save the modified model/tokenizer (optional)
        dry_run: If True, only report what would happen without loading model
        
    Returns:
        Dictionary with stats about the operation
    """
    tokens_to_add = list(ALL_SPECIAL_TOKENS)
    
    if dry_run:
        print(f"[DRY RUN] Would add {len(tokens_to_add)} new tokens to {model_id}")
        print(f"  HEAD tokens:    {len(HEAD_TOKENS)}")
        print(f"  ARG_HOUR tokens: {len(ARG_HOUR_TOKENS)}")
        print(f"  ARG_MIN tokens:  {len(ARG_MIN_TOKENS)}")
        print(f"  ROUTE tokens:    {len(ROUTING_TOKENS)}")
        print(f"  REF tokens:      {len(REF_TOKENS)}")
        print(f"  Total:           {len(tokens_to_add)}")
        return {
            "status": "dry_run",
            "tokens_to_add": len(tokens_to_add),
        }

    # Load model and tokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )

    original_vocab_size = len(tokenizer)
    original_embed_size = model.get_input_embeddings().weight.shape[0]
    d_model = model.get_input_embeddings().weight.shape[1]

    print(f"  Original vocab size: {original_vocab_size}")
    print(f"  Embedding dimension: {d_model}")

    # Add special tokens
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens_to_add}
    )
    print(f"  Added {num_added} new special tokens")

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    print(f"  New vocab size: {len(tokenizer)}")

    # Compute existing embedding statistics for scaling
    with torch.no_grad():
        existing_embeddings = model.get_input_embeddings().weight[:original_embed_size]
        existing_mean = existing_embeddings.mean(dim=0).numpy()
        existing_std = existing_embeddings.std().item()

    print(f"  Existing embedding mean norm: {np.linalg.norm(existing_mean):.4f}")
    print(f"  Existing embedding std: {existing_std:.4f}")

    # Geometric initialization
    print("  Applying geometric initialization...")
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        lm_head = model.get_output_embeddings()

        for token in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < original_embed_size:
                continue

            init_vector = compute_geometric_init(
                token, d_model, existing_mean, existing_std
            )
            init_tensor = torch.tensor(init_vector, dtype=torch.float32)

            # Set both input embedding and output (LM head) embedding
            embed_layer.weight[token_id] = init_tensor
            if lm_head is not None and lm_head.weight.shape[0] > token_id:
                lm_head.weight[token_id] = init_tensor

    # Verify geometric structure
    print("  Verifying geometric structure...")
    with torch.no_grad():
        h13_id = tokenizer.convert_tokens_to_ids("[ARG_HOUR_13]")
        h14_id = tokenizer.convert_tokens_to_ids("[ARG_HOUR_14]")
        h15_id = tokenizer.convert_tokens_to_ids("[ARG_HOUR_15]")
        h00_id = tokenizer.convert_tokens_to_ids("[ARG_HOUR_00]")

        e13 = embed_layer.weight[h13_id]
        e14 = embed_layer.weight[h14_id]
        e15 = embed_layer.weight[h15_id]
        e00 = embed_layer.weight[h00_id]

        dist_13_14 = torch.dist(e13, e14).item()
        dist_14_15 = torch.dist(e14, e15).item()
        dist_00_14 = torch.dist(e00, e14).item()

        print(f"    dist(HOUR_13, HOUR_14) = {dist_13_14:.4f}")
        print(f"    dist(HOUR_14, HOUR_15) = {dist_14_15:.4f}")
        print(f"    dist(HOUR_00, HOUR_14) = {dist_00_14:.4f}")
        print(f"    Adjacent hours closer than distant: {dist_13_14 < dist_00_14}")

    # Save if output dir specified
    if output_dir:
        print(f"  Saving to {output_dir}...")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        print("  Saved.")

    return {
        "status": "complete",
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(tokenizer),
        "tokens_added": num_added,
        "d_model": d_model,
        "adjacent_distance": dist_13_14,
        "distant_distance": dist_00_14,
        "geometric_verified": dist_13_14 < dist_00_14,
    }


def main():
    parser = argparse.ArgumentParser(description="Resize embeddings with geometric init")
    parser.add_argument("--model_id", type=str, default="google/gemma-2-2b")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report what would happen, don't load model")
    args = parser.parse_args()

    result = resize_and_initialize(args.model_id, args.output_dir, args.dry_run)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
