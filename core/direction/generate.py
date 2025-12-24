"""
Direction Generation - Compute mean-diff refusal direction vectors.

Ported from refusal_direction/pipeline/submodules/generate_directions.py
"""

import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from components_safety.core.models.hooks import add_hooks
from components_safety.core.models.base import ModelBase


def get_mean_activations_pre_hook(
    layer: int, 
    cache: Float[Tensor, "pos layer d_model"], 
    n_samples: int, 
    positions: List[int]
):
    """Create a pre-hook that accumulates mean activations."""
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn


def get_mean_activations(
    model, 
    tokenizer, 
    instructions: List[str], 
    tokenize_instructions_fn, 
    block_modules: List[torch.nn.Module], 
    batch_size: int = 32, 
    positions: List[int] = [-1]
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Compute mean activations across all instructions at specified positions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        instructions: List of instruction strings
        tokenize_instructions_fn: Function to tokenize instructions with chat template
        block_modules: List of transformer block modules
        batch_size: Batch size for processing
        positions: List of positions to collect activations from (e.g., [-1] for last token)
        
    Returns:
        Mean activations tensor of shape (n_positions, n_layers, d_model)
    """
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # Store mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), 
        dtype=torch.float64, 
        device=model.device
    )

    fwd_pre_hooks = [
        (block_modules[layer], get_mean_activations_pre_hook(
            layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions
        )) 
        for layer in range(n_layers)
    ]

    for i in tqdm(range(0, len(instructions), batch_size), desc="Computing activations"):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations


def get_mean_diff(
    model, 
    tokenizer, 
    harmful_instructions: List[str], 
    harmless_instructions: List[str], 
    tokenize_instructions_fn, 
    block_modules: List[torch.nn.Module], 
    batch_size: int = 32, 
    positions: List[int] = [-1]
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Compute mean activation difference between harmful and harmless instructions.
    
    This is the core of the "diff" method for extracting refusal direction.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        harmful_instructions: List of harmful instruction strings
        harmless_instructions: List of harmless instruction strings
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of transformer block modules
        batch_size: Batch size for processing
        positions: List of positions to collect activations from
        
    Returns:
        Mean difference tensor of shape (n_positions, n_layers, d_model)
    """
    mean_activations_harmful = get_mean_activations(
        model, tokenizer, harmful_instructions, tokenize_instructions_fn, 
        block_modules, batch_size=batch_size, positions=positions
    )
    mean_activations_harmless = get_mean_activations(
        model, tokenizer, harmless_instructions, tokenize_instructions_fn, 
        block_modules, batch_size=batch_size, positions=positions
    )

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = (
        mean_activations_harmful - mean_activations_harmless
    )

    return mean_diff


def generate_directions(
    model_base: ModelBase, 
    harmful_instructions: List[str], 
    harmless_instructions: List[str], 
    artifact_dir: str = None
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Generate candidate refusal directions using mean-diff method.
    
    Args:
        model_base: ModelBase instance
        harmful_instructions: List of harmful instruction strings
        harmless_instructions: List of harmless instruction strings
        artifact_dir: Optional directory to save intermediate results
        
    Returns:
        Mean diff tensor of shape (n_eoi_tokens, n_layers, d_model)
    """
    if artifact_dir and not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Use EOI token positions
    positions = list(range(-len(model_base.eoi_toks), 0))

    mean_diffs = get_mean_diff(
        model_base.model, 
        model_base.tokenizer, 
        harmful_instructions, 
        harmless_instructions, 
        model_base.tokenize_instructions_fn, 
        model_base.model_block_modules, 
        positions=positions
    )

    assert mean_diffs.shape == (
        len(model_base.eoi_toks), 
        model_base.model.config.num_hidden_layers, 
        model_base.model.config.hidden_size
    )
    assert not mean_diffs.isnan().any()

    if artifact_dir:
        torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs
