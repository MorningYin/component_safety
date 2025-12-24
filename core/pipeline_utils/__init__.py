"""
Pipeline Utilities - Hook management and other utilities.

Ported from refusal_direction/pipeline/utils/
"""

from components_safety.core.pipeline_utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_all_direction_ablation_hooks,
)
from components_safety.core.pipeline_utils.utils import get_orthogonalized_matrix

__all__ = [
    "add_hooks",
    "get_activation_addition_input_pre_hook",
    "get_direction_ablation_input_pre_hook",
    "get_direction_ablation_output_hook",
    "get_all_direction_ablation_hooks",
    "get_orthogonalized_matrix",
]
