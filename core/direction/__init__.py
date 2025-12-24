"""
Direction Generation and Selection.

This module provides the complete direction extraction and selection pipeline:

1. generate_directions - Compute mean-diff direction candidates
2. select_direction - Select best direction using ablation/steering/KL metrics
3. get_refusal_scores - Compute refusal probability scores
4. filter_fn - Filter candidates based on thresholds

Usage:
    from components_safety.core.direction import generate_directions, select_direction
"""

# Core functions from submodules
from components_safety.core.direction.submodules import (
    generate_directions,
    get_mean_diff,
    get_mean_activations,
    select_direction,
    select_rdo_direction,
    get_refusal_scores,
    get_last_position_logits,
    refusal_score,
    filter_fn,
    plot_refusal_scores,
)
from components_safety.core.direction.filter import (
    filter_instructions,
    filter_direction_data,
    filter_eval_datasets,
)

__all__ = [
    # Generation
    "generate_directions",
    "get_mean_diff",
    "get_mean_activations",
    # Selection
    "select_direction",
    "select_rdo_direction",
    "get_refusal_scores",
    "get_last_position_logits",
    "refusal_score",
    "filter_fn",
    "plot_refusal_scores",
    # Filtering
    "filter_instructions",
    "filter_direction_data",
    "filter_eval_datasets",
]
