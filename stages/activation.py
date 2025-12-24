"""
Stage: Activation Collection for Component Safety.

Collects EOI projections for benign and harmful datasets.
Uses the defense_data module for loading evaluation data.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from components_safety.core.experiment import ExperimentRunner

from components_safety.core.functional.projections import collect_component_projections
from components_safety.core.models import construct_model_base
from components_safety.core.direction import filter_eval_datasets
from components_safety.data import load_defense_data, load_threshold_val_data


def collect_activations(runner: ExperimentRunner, direction: torch.Tensor):
    """
    Stage: Collect EOI projections for benign and harmful_safe datasets.
    Uses caching to avoid redundant model forwards.
    """
    config = runner.config
    
    # 1. Prepare Cache Key
    cache_key = {
        "model": config.model.alias,
        "benign_path": str(config.data.benign_path),
        "harmful_path": str(config.data.harmful_path),
        "direction_hash": hashlib.md5(direction.cpu().numpy().tobytes()).hexdigest()[:16],
    }
    
    cached_data = runner.cache.load("activations", cache_key)
    if cached_data and not getattr(runner.config, 'force', False):
        print("✓ Loaded activations from cache.")
        return cached_data

    # 2. Load model
    print("Collecting activations from model...")
    model_base = construct_model_base(config.model.alias)
    eoi_token_ids = model_base._get_eoi_toks()
    
    # 3. Load datasets
    # Train: simple harmful/benign
    train_datasets = load_defense_data(benign_path=config.data.benign_path, harmful_path=config.data.harmful_path, seed=config.seed)
    # Val: specific types for threshold search
    val_datasets = load_threshold_val_data(seed=config.seed)
    
    # 3.1 Filter datasets based on refusal scores
    print("\n>>> Filtering evaluation datasets based on refusal scores...")
    train_datasets = filter_eval_datasets(model_base=model_base, datasets=train_datasets, batch_size=config.data.batch_size)
    val_datasets = filter_eval_datasets(model_base=model_base, datasets=val_datasets, batch_size=config.data.batch_size)
    
    print(f"  - (After filtering) Train Benign: {len(train_datasets['benign'])}, Harmful: {len(train_datasets['harmful'])}")
    print(f"  - (After filtering) Val Benign: {len(val_datasets['benign'])}, Harmful: {len(val_datasets['harmful'])}")
    
    # 4. Collect projections
    print("Collecting projections for Train set...")
    train_proj = collect_component_projections(model_base=model_base, datasets=train_datasets, batch_size=config.data.batch_size, direction=direction, eoi_token_ids=eoi_token_ids)
    
    print("Collecting projections for Val set...")
    val_proj = collect_component_projections(model_base=model_base, datasets=val_datasets, batch_size=config.data.batch_size, direction=direction, eoi_token_ids=eoi_token_ids)
    
    proj_data = {
        "train": train_proj,
        "val": val_proj
    }
    
    # 5. Save to Cache
    runner.cache.save("activations", cache_key, proj_data)
    
    # Cleanup
    model_base.del_model()
    torch.cuda.empty_cache()
    
    return proj_data

