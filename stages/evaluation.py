import json
import os
import sys
import torch
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from tqdm import tqdm
import time

from components_safety.data import load_threshold_val_data
from easyjailbreak.attacker.AutoDAN_Liu_2023 import AutoDAN
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.models.huggingface_model import from_pretrained

# Add safety-eval to path
sys.path.insert(0, str(Path("/root/LLM-Safety/safety-eval")))

# Import safety-eval components
import datasets
_original_load_dataset = datasets.load_dataset

def _patched_load_dataset(path, name=None, **kwargs):
    # Mapping of dataset names to local files found in cache
    local_map = {
        "allenai/wildguardmix": "/root/autodl-tmp/hf_cache/datasets/allenai___wildguardmix/wildguardtest/0.0.0/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/wildguardmix-test.arrow",
        "lmsys/toxic-chat": "/root/autodl-tmp/hf_cache/datasets/lmsys___toxic-chat/toxicchat0124/0.0.0/29df8e4dba60e1f4af4b4075c0705c5b313548a8/toxic-chat-test.arrow",
        "nvidia/Aegis-AI-Content-Safety-Dataset-1.0": "/root/autodl-tmp/hf_cache/datasets--nvidia--Aegis-AI-Content-Safety-Dataset-1.0/snapshots/bd96d862068e47630197de64eb91f8d1481ff3e0/Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet",
        "Bertievidgen/SimpleSafetyTests": "/root/autodl-tmp/hf_cache/datasets--Bertievidgen--SimpleSafetyTests/snapshots/98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5/sst_test_cases.csv",
        "PKU-Alignment/BeaverTails": "/root/autodl-tmp/hf_cache/datasets_cache/PKU-Alignment___beaver_tails/default/0.0.0/8401fe609d288129cc684a9b3be6a93e41cfe678/beaver_tails-30k_test.arrow",
        "PKU-Alignment/PKU-SafeRLHF": "/root/autodl-tmp/hf_cache/datasets_cache/PKU-Alignment___pku-safe_rlhf/default/0.0.0/ff7ba91063016c78a225b0f74e1c0860bb18230f/pku-safe_rlhf-test.arrow"
    }

    # Special handling for xstest which has split-specific files
    if path == "allenai/xstest-response":
        split = kwargs.get("split", "test")
        if split == "response_harmfulness":
            local_path = "/root/autodl-tmp/hf_cache/datasets_cache/allenai___xstest-response/default/0.0.0/4cd03695944eb5940c5a2dfcdf2a493a8f288dbc/xstest-response-response_harmfulness.arrow"
        else:
            local_path = "/root/autodl-tmp/hf_cache/datasets_cache/allenai___xstest-response/default/0.0.0/4cd03695944eb5940c5a2dfcdf2a493a8f288dbc/xstest-response-response_refusal.arrow"
        print(f"DEBUG: Redirecting xstest split {split} to local: {local_path}")
        return _original_load_dataset("arrow", data_files={split: local_path}, split=split)
    
    if path in local_map:
        local_path = local_map[path]
        print(f"DEBUG: Redirecting dataset {path} ({name}) to local file: {local_path}")
        
        # Determine format and handle split
        fmt = "arrow" if local_path.endswith(".arrow") else "parquet" if local_path.endswith(".parquet") else "csv"
        split = kwargs.get("split")
        if split:
            return _original_load_dataset(fmt, data_files={split: local_path}, split=split)
        else:
            return _original_load_dataset(fmt, data_files={"test": local_path})
            
    return _original_load_dataset(path, name, **kwargs)

datasets.load_dataset = _patched_load_dataset

from src.classifier_models.base import SafetyClassifierBase, SafetyClassifierOutput, PromptHarmfulness
from evaluation.tasks import load_evaluation_tasks, EvalMode
from evaluation.utils import save_evaluation_report

# Import components_safety modules
from components_safety.core.models import construct_model_base
from components_safety.core.functional.aggregate import compute_topm_scores
from components_safety.utils.runner_util import MinimalHookRunner
from components_safety.core.functional.projections import read_eoi_projection_unified
from components_safety.core.functional.overlap import overlap_kde_fast
from components_safety.core.experiment import ExperimentRunner

ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]

def component_dict_to_key(comp_dict: dict) -> ComponentKey:
    module_type = comp_dict["module_type"]
    layer = comp_dict["layer"]
    eoi_k = comp_dict["eoi_k"]
    if "head" in comp_dict:
        return (module_type, layer, eoi_k, comp_dict["head"])
    return (module_type, layer, eoi_k)

class RefusalComponentClassifier(SafetyClassifierBase):
    """
    Classifier that uses the internal activations (components) to detect harmful prompts.
    """
    def __init__(self, config_path: Path, batch_size: int = 4):
        super().__init__(batch_size=batch_size)
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.model_alias = self.config["model_alias"]
        self.p_star = self.config["p_star"]
        self.m_cap = self.config["m_cap"]
        self.actual_m = self.config["actual_m"]
        self.z_threshold = self.config["z_threshold_star"]
        self.eps = self.config["eps"]
        
        # Load direction
        self.direction = torch.load(self.config["files"]["direction_file"], map_location="cpu")
        if isinstance(self.direction, dict):
            self.direction = self.direction["direction"]
            
        # Load standardizer stats
        with open(self.config["files"]["standardizer_stats_json"], "r") as f:
            stats_list = json.load(f)
            
        self.means = {}
        self.stds = {}
        self.selected_components: Set[ComponentKey] = set()
        for item in stats_list:
            key = component_dict_to_key(item["component"])
            self.means[key] = item["mean"]
            self.stds[key] = item["std"]
            self.selected_components.add(key)
            
        # Model loading
        self.model_base = construct_model_base(self.model_alias)
        self.eoi_toks = self.model_base._get_eoi_toks()
        
        # Initialize Hook Runner
        self.runner = MinimalHookRunner(self.model_base)
        cfg = self.model_base.model.config
        n_heads = getattr(cfg, "num_attention_heads", 0)
        
        self.runner.set_state(
            direction=self.direction.to(device=self.model_base.model.device, dtype=self.model_base.model.dtype),
            eoi_token_ids=self.eoi_toks,
            input_ids=None,
            model_config=cfg,
            n_heads=n_heads,
        )
        for module_type in ["o_proj", "mlp"]:
            self.runner.add_reader(module_type=module_type, layers=None, fn=read_eoi_projection_unified)

    def get_required_input_fields(self) -> list[str]:
        return ["prompt"]

    def get_output_fields(self) -> list[str]:
        return ["prompt_harmfulness"]

    @torch.inference_mode()
    def _get_projections_batch(self, batch: list[dict[str, str]]) -> Dict[ComponentKey, np.ndarray]:
        """Collect all component projections for the batch."""
        prompts = [item["prompt"] for item in batch]
        self.runner.clear()
        self.runner.forward(prompts, batch_size=self.batch_size, show_progress=False)
        
        projections = {}
        for (layer, module_type, name), tensor in self.runner.results._final.items():
            if not name.startswith("eoi") or not name.endswith("_proj"): continue
            
            # Parse Key
            name_body = name[3:-5]
            if "_head" in name_body:
                parts = name_body.split("_head")
                key = (module_type, layer, int(parts[0]), int(parts[1]))
            else:
                key = (module_type, layer, int(name_body))
                
            projections[key] = tensor.float().cpu().numpy()
        return projections

    @torch.inference_mode()
    def _classify_batch(self, batch: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        prompts = [item["prompt"] for item in batch]
        self.runner.clear()
        self.runner.forward(prompts, batch_size=self.batch_size, show_progress=False)
        
        # Extract projections and standardize
        std_proj = {}
        for (layer, module_type, name), tensor in self.runner.results._final.items():
            if not name.startswith("eoi") or not name.endswith("_proj"): continue
            
            # Parse Key
            name_body = name[3:-5]
            if "_head" in name_body:
                parts = name_body.split("_head")
                key = (module_type, layer, int(parts[0]), int(parts[1]))
            else:
                key = (module_type, layer, int(name_body))
                
            if key in self.selected_components:
                val = tensor.float().cpu().numpy()
                mu = self.means[key]
                sigma = self.stds[key] + self.eps
                std_proj[key] = (val - mu) / sigma
                
        # Aggregate
        effective_m = self.actual_m if self.actual_m > 0 else len(self.selected_components)
        z_scores = compute_topm_scores(std_proj, self.selected_components, m_cap=effective_m)
        
        if len(z_scores) == 0:
            z_scores = np.zeros(len(prompts))
            
        outputs = []
        for z in z_scores:
            is_harmful = z > self.z_threshold
            outputs.append(SafetyClassifierOutput(
                prompt_harmfulness=PromptHarmfulness.HARMFUL if is_harmful else PromptHarmfulness.UNHARMFUL,
                metadata={"z_score": float(z), "threshold": self.z_threshold}
            ))
        return outputs

def run_open_dataset_eval(runner: ExperimentRunner, classifier_config_path: Path):
    """Run evaluation on open safety datasets."""
    tasks_str = "wildguardtest_prompt,harmbench,openai_mod,toxicchat,aegis_safety_dataset,simplesafetytests"
    task_list = tasks_str.split(",")
    
    output_dir = runner.config.run_dir / "evaluation" / "open_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Open Dataset Evaluation on: {tasks_str}")
    classifier = RefusalComponentClassifier(classifier_config_path)
    
    evaluation_tasks = load_evaluation_tasks(EvalMode.CLASSIFICATION, task_list)
    
    evaluation_report = {}
    all_individual_results = {}
    
    for task_name, eval_task in zip(task_list, evaluation_tasks):
        print(f"Evaluating task: {task_name}...")
        try:
            report, individual_results = eval_task.evaluate(classifier)
            evaluation_report[task_name] = report
            all_individual_results[task_name] = individual_results
            
            # Generate Task Specific Plots
            task_dir = output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            _plot_task_results(task_name, individual_results, classifier.z_threshold, task_dir)
            
        except Exception as e:
            print(f"Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Reports
    save_evaluation_report(str(output_dir / "metrics.json"), evaluation_report)
    save_evaluation_report(str(output_dir / "individual_results.json"), all_individual_results)
    
    print(f"✓ Open Dataset Evaluation complete. Results in {output_dir}")
    
    # Cleanup for next stage
    del classifier
    torch.cuda.empty_cache()
    gc.collect()

def _plot_task_results(task_name: str, results: List[Dict], threshold: float, output_dir: Path):
    """Plot Z-distribution for a specific task."""
    z_scores = []
    labels = []
    for r in results:
        z = r.get("metadata", {}).get("z_score")
        if z is not None:
            z_scores.append(z)
            gt = r.get("gt_prompt_harmfulness")
            labels.append(1 if gt == "harmful" else 0)
            
    if not z_scores: return
    
    z_scores = np.array(z_scores)
    labels = np.array(labels)
    
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    if np.sum(labels == 0) > 0:
        sns.kdeplot(z_scores[labels == 0], label="Benign", fill=True, color="#4c72b0", alpha=0.5)
    if np.sum(labels == 1) > 0:
        sns.kdeplot(z_scores[labels == 1], label="Harmful", fill=True, color="#dd8452", alpha=0.5)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="Threshold")
    plt.title(f"Z-Score Distribution: {task_name}", fontsize=16, pad=15)
    plt.xlabel("Z-Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(False) # Ensure grid is off
    sns.despine()
    plt.savefig(output_dir / "z_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def _load_raw_targets_map() -> Dict[str, str]:
    """Helper to load goal -> target mapping from raw CSVs for AutoDAN."""
    targets_map = {}
    raw_dir = Path("/root/LLM-Safety/refusal_direction/dataset/raw")
    if not raw_dir.exists():
        return targets_map
        
    # AdvBench
    advbench = raw_dir / "advbench.csv"
    if advbench.exists():
        import csv
        with open(advbench, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'goal' in row and 'target' in row:
                    targets_map[row['goal'].strip()] = row['target'].strip()

    # JailbreakBench
    jbb = raw_dir / "jailbreakbench.csv"
    if jbb.exists():
        import csv
        with open(jbb, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Goal' in row and 'Target' in row:
                    targets_map[row['Goal'].strip()] = row['Target'].strip()
    return targets_map

# --- AutoDAN Evaluation Support ---

def _setup_nltk():
    """Ensure NLTK data is available locally."""
    import nltk
    nltk_data_dir = "/root/LLM-Safety/components_safety/data/nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
        
    resources = [
        "stopwords", "wordnet", "omw-1.4", "punkt", 
        "punkt_tab", "averaged_perceptron_tagger_eng"
    ]
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            print(f"Downloading NLTK resource: {res}...")
            nltk.download(res, download_dir=nltk_data_dir, quiet=True)

def _get_target_model_template(model_alias: str) -> str:
    """Map model alias to EasyJailbreak template names."""
    alias_lower = model_alias.lower()
    if 'llama-3' in alias_lower:
        return 'llama-3'
    elif 'qwen' in alias_lower:
        return 'qwen-7b-chat'
    return 'llama-2'

def _plot_autodan_heatmaps(
    harmless_proj: Dict[ComponentKey, np.ndarray],
    harmful_proj: Dict[ComponentKey, np.ndarray],
    jailbreak_proj: Dict[ComponentKey, np.ndarray],
    output_path: Path
):
    """Generate the Refusal and Attack heatmap visualizations."""
    # Group by (module_type, layer, head/mlp) and find min/max overlap across EOI tokens
    all_keys = set(harmless_proj.keys()) | set(harmful_proj.keys()) | set(jailbreak_proj.keys())
    
    n_layers = max(k[1] for k in all_keys) + 1
    n_heads = max(k[3] for k in all_keys if len(k) == 4) + 1
    n_cols = n_heads + 1 # +1 for MLP
    
    # 1. Refusal Overlap: Harmless vs Original Harmful (Max overlap across EOI)
    refusal_grid = np.full((n_layers, n_cols), np.nan)
    # 2. Attack Overlap: Original Harmful vs Jailbreak (Min overlap across EOI)
    attack_grid = np.full((n_layers, n_cols), np.nan)
    
    # Pre-calculate overlaps for all components
    component_groups = {} # (module, layer, index) -> List of overlaps across EOI
    for k in all_keys:
        comp_id = (k[0], k[1], k[3] if len(k) == 4 else n_heads)
        if comp_id not in component_groups:
            component_groups[comp_id] = {"refusal": [], "attack": []}
            
        ovl_ref = overlap_kde_fast(harmless_proj[k], harmful_proj[k])
        ovl_atk = overlap_kde_fast(harmful_proj[k], jailbreak_proj[k])
        
        component_groups[comp_id]["refusal"].append(ovl_ref)
        component_groups[comp_id]["attack"].append(ovl_atk)
        
    for comp_id, results in component_groups.items():
        l, c = comp_id[1], comp_id[2]
        # 拒答组件：取最大 overlap（最不容易区分的情况下的代表值，或者保持保守）
        # 用户需求：可视化每一层每一个组件在 EOI token 上的最大 overlap
        refusal_grid[l, c] = max(results["refusal"])
        # 攻击组件：可视化每一层每一个组件在 EOI token 上的最小 overlap
        attack_grid[l, c] = min(results["attack"])
        
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    # Plot 1: Refusal Component Distribution Map (Smaller -> Darker)
    sns.heatmap(refusal_grid, ax=axes[0], cmap="crest_r", annot=False,
                cbar_kws={'label': 'Max Overlap (Harmless vs Harmful)'})
    axes[0].set_title("Refusal Component Map\n(Smaller Overlap -> Darker/More Robust)", fontsize=16, pad=15)
    axes[0].set_ylabel("Layer Index", fontsize=14)
    axes[0].set_xlabel("Head Index (0 to N-1) | MLP (Last)", fontsize=14)
    
    # Plot 2: Attacked Component Distribution Map (Smaller -> Darker)
    sns.heatmap(attack_grid, ax=axes[1], cmap="crest_r", annot=False,
                cbar_kws={'label': 'Min Overlap (Harmful vs Jailbreak)'})
    axes[1].set_title("Attacked Component Map\n(Smaller Overlap -> Darker/More Shifted)", fontsize=16, pad=15)
    axes[1].set_ylabel("Layer Index", fontsize=14)
    axes[1].set_xlabel("Head Index (0 to N-1) | MLP (Last)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated AutoDAN component analysis heatmaps: {output_path}")

def _prepare_autodan_dataset(goals: List[Dict]) -> 'JailbreakDataset':
    """Prepare a JailbreakDataset from instruction/target pairs."""
    from easyjailbreak.datasets import JailbreakDataset
    from easyjailbreak.datasets.instance import Instance
    
    instances = []
    for item in goals:
        goal = item["instruction"]
        # Use provided target or generic fallback
        target = item.get("target", "")
        if not target:
            suffix = goal[0].lower() + goal[1:] if goal else ""
            target = f"Sure, here is how to {suffix}"
        
        instances.append(Instance(query=goal, reference_responses=[target]))
    return JailbreakDataset(instances)

def _parse_autodan_result(res_obj) -> Dict:
    """Parse raw AutoDAN result object into a clean dictionary."""
    goal = res_obj.query 
    target_ref = res_obj.reference_responses[0] if res_obj.reference_responses else ""
    
    # Check for success in any of the population samples
    success_indices = [i for i, val in enumerate(res_obj.eval_results) if val]
    if success_indices:
        first_idx = success_indices[0]
        is_jailbreak = True
        response = res_obj.target_responses[first_idx]
    else:
        is_jailbreak = False
        response = res_obj.target_responses[-1] if res_obj.target_responses else ""

    return {
        "goal": goal,
        "target_ref": target_ref,
        "jailbreak_prompt": getattr(res_obj, 'jailbreak_prompt', ""),
        "is_jailbreak": is_jailbreak,
        "response": response
    }

def run_autodan_eval(runner: ExperimentRunner, classifier_config_path: Path):
    """Run AutoDAN white-box attack evaluation with incremental caching."""
    from easyjailbreak.attacker import AutoDAN as AutoDAN_Liu_2023
    from easyjailbreak.models import from_pretrained
    import random
    
    # 1. Environment and Path Setup
    _setup_nltk()
    output_dir = runner.config.run_dir / "evaluation" / "autodan"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl_path = output_dir / "autodan_results.jsonl"
    
    # 2. Load Sampled Goals
    full_data = load_threshold_val_data(runner.config.seed)
    goal = [d for d in full_data['harmful'] if d['type'] == 'simple_harmful']
    sampled_instructions = {d["instruction"] for d in goal}
    
    # 3. Handle Incremental Cache
    all_cached_results = {}
    if results_jsonl_path.exists():
        with open(results_jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    all_cached_results[res["goal"]] = res
    
    final_jailbreak_results = [res for goal, res in all_cached_results.items() if goal in sampled_instructions]
    n_initially_cached = len(final_jailbreak_results)
    print(f"✓ Found {n_initially_cached} existing results for sampled goals.")
    n_need_to_attack = max(0, runner.config.evaluation.num_autodan_goals - n_initially_cached)

    if n_need_to_attack > 0:
        # 4. Load Models
        print("Loading AutoDAN attack model and evaluation model (Vicuna-13b)...")
        vicuna_path = "/root/autodl-tmp/hf_cache/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2"
        attack_model = from_pretrained(model_name_or_path=vicuna_path, model_name='vicuna_v1.1', dtype=torch.float16)
        eval_model = attack_model
        
        print(f"Loading target model: {runner.config.model.alias}...")
        template_name = _get_target_model_template(runner.config.model.alias)
        target_hf_model = from_pretrained(model_name_or_path=runner.config.model.path, model_name=template_name, dtype=torch.float16)

        # 5. Execute Attacks for Missing Goals
        goals_to_attack = [d for d in goal if d['instruction'] not in all_cached_results][:n_need_to_attack]
        for d in goals_to_attack:
            if d['target'] == "":
                instr = d['instruction']
                d['target'] = f"Sure, I can assist with that. Here is the detailed information regarding {instr[0].lower() + instr[1:]}:"

        pending_instances = [Instance(query=d['instruction'], reference_responses=[d['target']]) for d in goals_to_attack]

        print(f"Attacking {len(pending_instances)} remaining goals...")
        dataset = JailbreakDataset(pending_instances, shuffle=False)
        
        # 4. 实例化 AutoDAN 攻击器
        print("Initializing AutoDAN attacker...")  
        attacker = AutoDAN(attack_model=attack_model, target_model=target_hf_model, jailbreak_datasets=dataset, eval_model=eval_model,
            max_query=100, max_jailbreak=100, max_reject=100, max_iteration=100, device="cuda", num_steps=100, sentence_level_steps=5,
            word_dict_size=30, batch_size=64, num_elites=0.1, crossover_rate=0.5, mutation_rate=0.01, num_points=5, model_name=template_name,
            low_memory=1, pattern_dict=None)
        
        try:
            for instance in tqdm(dataset, desc="Processing instances"):
                result_instances = attacker.single_attack(instance)
                attacker.attack_results.add(result_instances[0])
                
                # Each single_attack returns a dataset with one instance
                res_obj = result_instances[0]
                
                # Check for success in any of the population samples
                # eval_results is updated during the HGA iterations in single_attack
                success_indices = [i for i, val in enumerate(res_obj.eval_results) if val]
                
                if success_indices:
                    first_idx = success_indices[0]
                    is_jailbreak = True
                    # In AutoDAN success case, target_responses is populated
                    # but we need to match it with the correct index if possible
                    # or just take the successful one. 
                    # The response index might shift due to multiple evaluate_candidate_prompts calls
                    response = res_obj.target_responses[first_idx] if first_idx < len(res_obj.target_responses) else res_obj.target_responses[-1]
                else:
                    is_jailbreak = False
                    response = res_obj.target_responses[-1] if res_obj.target_responses else ""

                result_dict = {
                    "goal": res_obj.query,
                    "is_jailbreak": is_jailbreak,
                    "jailbreak_prompt": res_obj.jailbreak_prompt,
                    "response": response,
                }
                
                final_jailbreak_results.append(result_dict)
                with open(results_jsonl_path, "a") as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        except KeyboardInterrupt:
            logging.info("攻击被用户中断！")
        
        del attack_model, target_hf_model
        torch.cuda.empty_cache()
    else:
        print('No goals need to attack.')

    print('Sleeping for 5 seconds...')
    time.sleep(5)

    # 6. Evaluate Distributions on Defended Model
    print("Computing metrics and distributions...")
    classifier = RefusalComponentClassifier(classifier_config_path)
    
    # Load reference benign data
    harmless_data = full_data['benign']
    jailbreak_prompts = [r["jailbreak_prompt"] for r in final_jailbreak_results if r.get("jailbreak_prompt")]
    
    def get_zs(prompts):
        if not prompts: return []
        outputs = classifier._classify_batch([{"prompt": p} for p in prompts])
        return [o.metadata["z_score"] for o in outputs]
    
    z_harmless = get_zs([d["instruction"] for d in harmless_data])
    z_harmful = get_zs([d["instruction"] for d in goal])
    z_jailbreak = get_zs(jailbreak_prompts)
    
    # 6.5. Additional Analysis: Refusal & Attacked Component Maps
    print("Collecting full projections for component analysis maps...")
    proj_harmless = classifier._get_projections_batch([{"prompt": d["instruction"]} for d in harmless_data])
    proj_harmful = classifier._get_projections_batch([{"prompt": d["instruction"]} for d in goal])
    proj_jailbreak = classifier._get_projections_batch([{"prompt": p} for p in jailbreak_prompts])
    
    _plot_autodan_heatmaps(
        proj_harmless, proj_harmful, proj_jailbreak,
        output_dir / "autodan_component_analysis.png"
    )
    
    # 7. Calculate and Save Metrics
    asr_base = np.mean([1 if r.get("is_jailbreak") else 0 for r in final_jailbreak_results]) if final_jailbreak_results else 0.0
    asr_defended = np.mean([1 if z > classifier.z_threshold else 0 for z in z_jailbreak]) if z_jailbreak else 0.0
    
    metrics = {
        "autodan_asr_base": float(asr_base),
        "autodan_asr_defended": float(asr_defended),
        "n_goals": len(goal),
        "n_jailbreaks_produced": len(jailbreak_prompts),
        "n_cached": n_initially_cached
    }
    
    with open(output_dir / "autodan_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # 8. Triple Distribution Plot
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.figure(figsize=(10, 6))
    if z_harmless:
        sns.kdeplot(z_harmless, label="Harmless (Benign)", fill=True, color="#4c72b0", alpha=0.5)
    if z_harmful:
        sns.kdeplot(z_harmful, label="Original Harmful", fill=True, color="#55a868", alpha=0.5)
    if z_jailbreak:
        sns.kdeplot(z_jailbreak, label="AutoDAN Jailbreak", fill=True, color="#dd8452", alpha=0.5)
        
    plt.axvline(classifier.z_threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold ({classifier.z_threshold:.2f})")
    plt.title(f"AutoDAN Jailbreak Distribution: {runner.config.model.alias}", fontsize=16, pad=15)
    plt.xlabel("Z-Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(frameon=False)
    sns.despine()
    plt.savefig(output_dir / "autodan_z_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ AutoDAN Evaluation complete. Metrics: {metrics}")
    
    # Final cleanup
    del classifier
    torch.cuda.empty_cache()
    gc.collect()
