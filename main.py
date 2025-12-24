import os

os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import typer
import yaml
from pathlib import Path
import torch
import json

from components_safety.core.config_schema import ExperimentConfig
from components_safety.core.experiment import ExperimentRunner
from components_safety.stages.direction import extract_direction_diff
from components_safety.stages.activation import collect_activations
from components_safety.stages.robustness import compute_robustness
from components_safety.stages.search import run_grid_search
from components_safety.stages.evaluation import run_open_dataset_eval, run_autodan_eval

app = typer.Typer()

def run_pipeline(config_path: str):
    # 1. Load Config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config = ExperimentConfig(**config_data)
    config.ensure_dirs()
    
    runner = ExperimentRunner(config)
    
    # # 2. Direction Stage
    # direction = extract_direction_diff(runner)
    
    # # 3. Activation Stage
    # proj_data = collect_activations(runner, direction)
    
    # # 4. Robustness Stage
    # score_q = compute_robustness(runner, proj_data)
    
    # # 5. Search Stage
    classifier_config_path = config.run_dir / "artifacts" / "classifier_config.json"
    # if not classifier_config_path.exists() or getattr(config, 'force', False):
    #     best_params, all_grid_results = run_grid_search(runner, proj_data, score_q)
    # else:
    #     print("✓ Classifier config already exists. Skipping search.")

    # # 6. Evaluation Stage
    # run_open_dataset_eval(runner, classifier_config_path)
    run_autodan_eval(runner, classifier_config_path)

    print(f"\n✓ Pipeline complete! Results in: {config.run_dir}")

@app.command()
def run(config: str = typer.Option(..., help="Path to config yaml")):
    run_pipeline(config)

if __name__ == "__main__":
    app()
