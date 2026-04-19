"""
Compute Layer — Digital Circuits as Neural Network Layers
==========================================================
Implements exact digital logic (gates, adders, arithmetic) as
differentiable PyTorch modules with no learnable parameters.

The circuit topology IS the computation. Activations are signal values.
A forward pass IS circuit execution.

Hierarchy:
  gates.py        — AND, OR, XOR, NOT (algebraic form over {0,1})
  adders.py       — HalfAdder, FullAdder, RippleCarryAdder, DecimalDigitAdder
  circuit_layer.py — Drop-in nn.Module wrapper for use in neural networks
  proof_model.py  — Tiny model proving a network can learn to use a circuit layer

See: Experiment 011 (experiments/neural-compute/)
"""

from src.compute_layer.arithmetic import floor_ste, fmod_ste, round_ste
from src.compute_layer.gates import AND, OR, XOR, NOT
from src.compute_layer.adders import (
    HalfAdder,
    FullAdder,
    RippleCarryAdder,
    DecimalDigitAdder,
)
from src.compute_layer.circuit_layer import CircuitLayer
