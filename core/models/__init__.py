"""
Model wrappers for various LLM families.

This module provides unified interfaces for different model families:
- Llama-3 (llama3.py)
- Llama-2 (llama2.py)
- Qwen (qwen.py) - supports both Qwen1 and Qwen2
- Gemma (gemma.py)
- Yi (yi.py)
- OLMo2 (olmo2.py)
- Mistral (mistral.py)

Use the factory to construct models:
    from components_safety.core.models import construct_model_base
    model = construct_model_base("llama-3-8b-it")
"""

from components_safety.core.models.base import ModelBase
from components_safety.core.models.factory import construct_model_base, MODEL_ALIASES
from components_safety.core.models.hooks import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_all_direction_ablation_hooks,
    get_activation_addition_input_pre_hook,
)
from components_safety.core.models.utils import get_orthogonalized_matrix

__all__ = [
    # Core
    "ModelBase",
    "construct_model_base",
    "MODEL_ALIASES",
    # Hooks
    "add_hooks",
    "get_direction_ablation_input_pre_hook",
    "get_direction_ablation_output_hook",
    "get_all_direction_ablation_hooks",
    "get_activation_addition_input_pre_hook",
    # Utils
    "get_orthogonalized_matrix",
]
