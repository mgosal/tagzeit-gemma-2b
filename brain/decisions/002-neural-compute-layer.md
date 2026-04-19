# ADR-002: Neural Compute Layers — Circuit-as-Module Architecture

**Status:** Accepted (Phase Zero validated)
**Date:** 2026-04-19
**Supercedes:** None (extends ADR-001)

## Context

ADR-001 (Route-to-Luxon) proved that Llama 3.2-1B can route temporal operations with 100% accuracy through an external Luxon engine, achieving 79.2% E2E accuracy. But the architecture requires a running Node.js process — the model cannot answer temporal questions standalone.

We needed to determine whether exact computation could be embedded directly inside the neural network as a differentiable layer.

## Decision

Implement exact digital circuits as PyTorch `nn.Module` instances with zero learnable parameters. The circuit topology (gate connections, carry chain wiring) IS the computation. Use algebraic identities over binary inputs (`AND = a*b`, `XOR = a+b-2ab`) rather than learned weight matrices.

The learned components (encoder, decoder, value extractor) operate AROUND the circuit, not through it:
- **Encoder** extracts operand values from hidden states (trained via auxiliary MSE loss)
- **Circuit** computes exact results (forward pass only, gradients detached)
- **Decoder** classifies circuit output into model predictions (trained via CE loss)

## Alternatives Considered

1. **Learn modular arithmetic end-to-end:** Train the model to discover mod/floor through gradient descent. Rejected — modular arithmetic is a known failure mode for gradient-based learning.

2. **External computation engine (ADR-001 status quo):** Keep Node.js Luxon. Rejected for this experiment — the goal is to eliminate the external dependency.

3. **Weight-matrix gates:** Encode AND/OR/XOR as specific weight values in `nn.Linear` layers with ReLU thresholds. Rejected — the algebraic form is cleaner, provably exact, and doesn't require careful threshold calibration.

4. **STE gradient flow through circuit to output heads:** Let gradients from the classification loss backpropagate through the entire circuit pipeline. Rejected — the accumulated STE approximation errors through 6+ discrete operations corrupted the loss landscape. Detaching the circuit output and training the heads independently was necessary.

## Consequences

- The circuit layer guarantees 100% arithmetic accuracy (verified exhaustively on all 256 4-bit inputs and 100 decimal digit inputs)
- Training is more complex (two gradient paths: aux loss for extractor, CE loss for heads) but converges reliably
- The architecture is extensible: the same pattern applies to base-60 and base-24 arithmetic without architectural changes — only the circuit topology changes
- No external runtime dependencies for computation
