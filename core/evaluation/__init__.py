"""
Evaluation Module - Safety classifiers and evaluation infrastructure.

This module provides:
- Base classes for safety classifiers (SafetyClassifierBase, SafetyClassifierOutput)
- Baseline classifiers for comparison (WildGuard, LlamaGuard2, HarmbenchClassifier, etc.)
- Utilities for vLLM-based inference

Usage:
    from components_safety.core.evaluation import SafetyClassifierBase
    from components_safety.core.evaluation.baselines import WildGuard, LlamaGuard2
"""

from components_safety.core.evaluation.base import (
    SafetyClassifierBase,
    SafetyClassifierOutput,
    LegacySafetyClassifierBase,
    PromptHarmfulness,
    ResponseRefusal,
    ResponseHarmfulness,
    HarmCategory,
    ConversationTurn,
    Role,
)

__all__ = [
    # Base classes
    "SafetyClassifierBase",
    "SafetyClassifierOutput",
    "LegacySafetyClassifierBase",
    # Enums
    "PromptHarmfulness",
    "ResponseRefusal",
    "ResponseHarmfulness",
    "HarmCategory",
    # Conversation
    "ConversationTurn",
    "Role",
]
