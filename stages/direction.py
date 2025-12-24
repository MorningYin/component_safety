"""
Stage: Direction Extraction using Complete Pipeline from refusal_direction.

This stage implements the COMPLETE direction extraction and selection pipeline:
1. Generate candidate directions using mean-diff method
2. Compute baseline refusal scores
3. Compute ablation scores (removing direction from harmful prompts)
4. Compute steering scores (adding direction to harmless prompts)
5. Compute KL divergence scores (model consistency check)
6. Filter and select the best direction
7. Generate visualization plots
"""
from __future__ import annotations
import torch
import os
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from components_safety.core.experiment import ExperimentRunner

from components_safety.core.models import construct_model_base
from components_safety.core.direction import (
    generate_directions, 
    select_direction,
    get_refusal_scores,
    filter_direction_data,
)
from components_safety.data import load_and_sample_direction_data


def extract_direction_diff(runner: ExperimentRunner) -> torch.Tensor:
    """
    Stage: Complete direction extraction pipeline.
    
    This is a 1:1 replication of refusal_direction's run_pipeline.py logic:
    1. Load harmful/harmless train and val data
    2. Generate candidate directions (mean-diff)
    3. Select best direction using ablation/steering/KL filtering
    4. Save direction and all artifacts
    """
    config = runner.config
    
    print(f"=== Direction Extraction for: {config.model.alias} ===")
    
    # 1. Check if direction already exists (for resuming)
    output_path = config.run_dir / "artifacts" / "direction.pt"
    if output_path.exists() and not getattr(config, 'force', False):
        print(f"✓ Loading existing direction from {output_path}")
        direction = torch.load(output_path, map_location='cpu')
        return direction
    
    # 2. Prepare artifact directory
    artifact_dir = str(config.run_dir / "artifacts" / "direction_selection")
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 3. Load model
    model_base = construct_model_base(config.model.alias)
    
    # 4. Load data using the same logic as refusal_direction
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_direction_data()
    
    # 4.1 Filter data based on refusal scores (consistent with refusal_direction/pipeline/run_pipeline.py)
    print("\n>>> Filtering directions data based on refusal scores...")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_direction_data(
        model_base=model_base,
        harmful_train=harmful_train,
        harmless_train=harmless_train,
        harmful_val=harmful_val,
        harmless_val=harmless_val,
        filter_train=config.data.filter_train,
        filter_val=config.data.filter_val,
        batch_size=config.data.batch_size
    )
    
    print(f"  - (After filtering) Harmful train: {len(harmful_train)}, Harmless train: {len(harmless_train)}")
    print(f"  - (After filtering) Harmful val: {len(harmful_val)}, Harmless val: {len(harmless_val)}")
    
    # 5. Generate candidate directions (mean-diff)
    print("\n>>> Generating candidate directions...")
    mean_diffs = generate_directions(
        model_base=model_base,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        artifact_dir=artifact_dir
    )
    # mean_diffs shape: (n_eoi_tokens, n_layers, d_model)
    
    # 6. Select best direction using complete evaluation
    print("\n>>> Selecting best direction...")
    pos, layer, best_direction = select_direction(
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        candidate_directions=mean_diffs,
        artifact_dir=artifact_dir,
        kl_threshold=getattr(config.direction, 'kl_threshold', 0.1),
        induce_refusal_threshold=getattr(config.direction, 'induce_refusal_threshold', 0.0),
        prune_layer_percentage=getattr(config.direction, 'prune_layer_percentage', 0.2),
        batch_size=config.data.batch_size
    )
    
    # 7. Normalize direction
    direction = best_direction.float()
    direction = direction / direction.norm()
    
    print(f"\n✓ Selected direction: position={pos}, layer={layer}")
    
    # 8. Save direction
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction, output_path)
    runner.manifest.log_output("refusal_direction", output_path)
    
    # Also save metadata
    metadata = {
        "position": int(pos),
        "layer": int(layer),
        "model_alias": config.model.alias,
        "n_harmful_train": len(harmful_train),
        "n_harmless_train": len(harmless_train),
    }
    import json
    with open(config.run_dir / "artifacts" / "direction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 9. Cleanup
    model_base.del_model()
    torch.cuda.empty_cache()
    
    print(f"✓ Direction saved to {output_path}")
    
    return direction