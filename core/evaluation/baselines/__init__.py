"""
Baseline classifiers for comparison experiments.

This module contains safety classifiers ported from safety-eval for baseline comparison:
- WildGuard: AllenAI's wildguard model (vLLM-based)
- LlamaGuard2: Meta's LlamaGuard 2 (transformers-based)
- LlamaGuardUserRequest: Prompt-only classification
- HarmbenchClassifier: HarmBench's Llama-2-13B classifier
- AegisLlamaGuard: AEGIS safety classifier (defensive/permissive modes)
- KeywordBasedRefusalClassifier: Simple keyword-based baseline
"""

from components_safety.core.evaluation.baselines.wildguard import WildGuard
from components_safety.core.evaluation.baselines.llama_guard import (
    LlamaGuard2,
    LlamaGuardUserRequest,
    LlamaGuardModelResponse,
)
from components_safety.core.evaluation.baselines.harmbench import (
    HarmbenchClassifier,
    HarmbenchValidationClassifier,
)
from components_safety.core.evaluation.baselines.aegis import (
    AegisLlamaGuardPermissive,
    AegisLlamaGuardDefensive,
)
from components_safety.core.evaluation.baselines.keyword import KeywordBasedRefusalClassifier

__all__ = [
    # WildGuard
    "WildGuard",
    # LlamaGuard family
    "LlamaGuard2",
    "LlamaGuardUserRequest",
    "LlamaGuardModelResponse",
    # HarmBench
    "HarmbenchClassifier",
    "HarmbenchValidationClassifier",
    # AEGIS
    "AegisLlamaGuardPermissive",
    "AegisLlamaGuardDefensive",
    # Keyword
    "KeywordBasedRefusalClassifier",
]
