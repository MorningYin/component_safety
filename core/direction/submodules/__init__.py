"""
Direction Submodules - Complete pipeline for direction extraction and selection.

Ported from refusal_direction/pipeline/submodules/

Contains:
- generate_directions: Mean-diff direction generation
- select_direction: Direction selection with KL/ablation/steering filtering
- evaluate_jailbreak: Jailbreak success evaluation
- evaluate_loss: Loss evaluation utilities
"""

from components_safety.core.direction.submodules.generate_directions import (
    generate_directions,
    get_mean_diff,
    get_mean_activations,
)
from components_safety.core.direction.submodules.select_direction import (
    select_direction,
    select_rdo_direction,
    get_refusal_scores,
    get_last_position_logits,
    refusal_score,
    filter_fn,
    plot_refusal_scores,
)

__all__ = [
    # Generate
    "generate_directions",
    "get_mean_diff",
    "get_mean_activations",
    # Select
    "select_direction",
    "select_rdo_direction",
    "get_refusal_scores",
    "get_last_position_logits",
    "refusal_score",
    "filter_fn",
    "plot_refusal_scores",
]
