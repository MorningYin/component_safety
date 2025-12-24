"""
Robustness scoring via KDE overlap.

Computes the overlap coefficient between benign and harmful activation distributions
for each component. Lower overlap = better separation = more robust.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Union
from tqdm import tqdm

from components_safety.core.experiment import ExperimentRunner
from components_safety.core.functional.overlap import overlap_kde_fast

# ComponentKey: (module_type, layer, eoi_k) or (module_type, layer, eoi_k, head)
ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]


def compute_robustness(
    runner: ExperimentRunner, 
    proj_data: Dict
) -> Dict[ComponentKey, float]:
    """
    Compute robustness scores for each component using KDE overlap.
    
    Args:
        runner: ExperimentRunner with config and seed.
        proj_data: Dict with 'benign' and 'harmful' projection data.
                   Each is a Dict[ComponentKey, np.ndarray].
    
    Returns:
        score_q: Dict mapping ComponentKey -> overlap score.
                 Lower score means better separation (more robust).
    """
    # Set seed for reproducibility of random jitter in KDE
    np.random.seed(runner.config.seed)
    
    benign_proj = proj_data["train"]["benign"]
    harmful_proj = proj_data["train"]["harmful"]
    
    # Get all component keys (intersection of both datasets)
    all_keys = set(benign_proj.keys()) & set(harmful_proj.keys())
    
    score_q: Dict[ComponentKey, float] = {}
    
    print(f"Computing KDE overlap for {len(all_keys)} components...")
    
    for key in tqdm(all_keys, desc="KDE Overlap"):
        benign_scores = benign_proj[key].flatten()
        harmful_scores = harmful_proj[key].flatten()
        
        # Compute overlap coefficient
        overlap = overlap_kde_fast(benign_scores, harmful_scores, x_points=128)
        score_q[key] = overlap
    
    print(f"✓ Computed robustness scores for {len(score_q)} components")
    
    return score_q
