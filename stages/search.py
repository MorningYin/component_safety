import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Set, List
from tqdm import tqdm
import json

from components_safety.core.experiment import ExperimentRunner
from components_safety.core.functional.aggregate import compute_topm_scores
from components_safety.core.functional.standardize import Standardizer
from scipy.stats import gaussian_kde

def find_best_f1(z_benign: np.ndarray, z_harmful: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Find the threshold that maximizes F1 score.
    Returns: (best_f1, best_threshold, best_precision, best_recall)
    """
    if len(z_benign) == 0 or len(z_harmful) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Combine and sort all unique scores to use as candidate thresholds
    all_scores = np.sort(np.unique(np.concatenate([z_benign, z_harmful])))
    
    best_f1 = -1.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    n_harmful = len(z_harmful)
    
    for t in all_scores:
        tp = np.sum(z_harmful > t)
        fp = np.sum(z_benign > t)
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
            
        recall = tp / n_harmful
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall
            
    return float(best_f1), float(best_threshold), float(best_precision), float(best_recall)

def run_grid_search(runner: ExperimentRunner, proj_data: Dict, score_q: Dict):
    """
    Perform grid search for best p (alpha threshold) and m (top-m components).
    """
    config = runner.config.search
    results_dir = runner.config.run_dir / "artifacts"
    figures_dir = runner.config.run_dir / "figures"
    
    # 1. Standardize all projections based on train benign data
    print("Standardizing projections...")
    standardizer = Standardizer()
    standardizer.fit(proj_data["train"]["benign"])
    
    std_train_benign = standardizer.transform(proj_data["train"]["benign"])
    std_train_harmful = standardizer.transform(proj_data["train"]["harmful"])
    std_val_benign = standardizer.transform(proj_data["val"]["benign"])
    std_val_harmful = standardizer.transform(proj_data["val"]["harmful"])
    
    # 2. Grid Search
    p_values = np.linspace(config.p_min, config.p_max, config.n_p)
    m_values = np.linspace(config.m_min, config.m_max, config.n_m)
    
    all_scores = np.array(list(score_q.values()))
    
    best_f1 = -1.0
    best_params = None
    all_grid_results = []
    
    print(f"Starting Grid Search ( {len(p_values)} p x {len(m_values)} m )...")
    
    # Outer progress bar for p
    for p_threshold in tqdm(p_values, desc="Searching p (percentile)"):
        subset = {k for k, s in score_q.items() if s <= p_threshold}
        
        if not subset:
            continue
            
        for m in m_values:
            # m is a fraction of the subset size, e.g., 0.2 means 20%
            m_actual = max(1, int(m * len(subset)))
            
            # Compute Z-scores for Val set (to find best threshold)
            z_val_benign = compute_topm_scores(std_val_benign, subset, m_cap=m_actual)
            z_val_harmful = compute_topm_scores(std_val_harmful, subset, m_cap=m_actual)
            
            if len(z_val_benign) == 0 or len(z_val_harmful) == 0:
                continue
                
            # Find best F1 and threshold for this (p, m) on VAL set
            f1, threshold, precision, recall = find_best_f1(z_val_benign, z_val_harmful)
            
            all_grid_results.append({
                "p": float(p_threshold),
                "m": float(m),
                "m_actual": int(m_actual),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold)
            })
            
            # Update best params with tolerance
            if best_params is None or f1 > (best_f1 + config.f1_tolerance):
                best_f1 = f1
                best_params = {
                    "p": float(p_threshold), "m": float(m), "m_actual": int(m_actual),
                    "f1": float(f1), "threshold": float(threshold), "subset": list(subset),
                    "p_threshold": float(p_threshold)
                }
            elif abs(f1 - best_f1) <= config.f1_tolerance:
                # Tie-breaking: choose tougher config (higher p, higher m)
                if p_threshold >= best_params["p"] and m >= best_params["m"]:
                    best_params = {
                        "p": float(p_threshold), "m": float(m), "m_actual": int(m_actual),
                        "f1": float(f1), "threshold": float(threshold), "subset": list(subset),
                        "p_threshold": float(p_threshold)
                    }

    if best_params is None:
        raise ValueError("Search failed to find any valid parameters.")

    print(f"✓ Best Params: p={best_params['p']:.4f}, m={best_params['m']}, F1={best_params['f1']:.4f}")
    
    # 3. Save Results and Metadata
    standardizer_stats = []
    for k in best_params["subset"]:
        standardizer_stats.append({
            "component": {
                "module_type": k[0],
                "layer": k[1],
                "eoi_k": k[2],
                **({"head": k[3]} if len(k) == 4 else {})
            },
            "mean": float(standardizer.means[k]),
            "std": float(standardizer.stds[k]),
            "eps": float(standardizer.eps)
        })
    
    with open(results_dir / "standardizer_stats.json", "w") as f:
        json.dump(standardizer_stats, f, indent=2)

    classifier_config = {
        "model_alias": runner.config.model.alias,
        "p_star": best_params["p"],
        "m_cap": best_params["m"],
        "actual_m": m_actual,
        "z_threshold_star": best_params["threshold"],
        "eps": standardizer.eps,
        "files": {
            "direction_file": str(runner.config.run_dir / "artifacts" / "direction.pt"),
            "standardizer_stats_json": str(results_dir / "standardizer_stats.json")
        }
    }
    
    with open(results_dir / "classifier_config.json", "w") as f:
        json.dump(classifier_config, f, indent=2)

    with open(results_dir / "search_results.json", "w") as f:
        json.dump({"best": best_params, "all": all_grid_results}, f, indent=2)
        
    # 4. Generate Visualizations
    plot_search_results(runner, std_val_benign, std_val_harmful, score_q, best_params, all_grid_results, figures_dir)
    
    return best_params, all_grid_results

def plot_search_results(runner, std_benign, std_harmful, score_q, best_params, all_results, output_dir: Path):
    """
    Generate requested visualizations with scholarly styling.
    """
    # Use a clean, scholarly style
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    subset = set(tuple(k) if isinstance(k, list) else k for k in best_params["subset"])
    m_actual = min(best_params["m_actual"], len(subset))
    
    z_benign = compute_topm_scores(std_benign, subset, m_cap=m_actual)
    z_harmful = compute_topm_scores(std_harmful, subset, m_cap=m_actual)
    threshold = best_params["threshold"]

    # 1. Z-score Distribution
    plt.figure(figsize=(9, 6))
    sns.kdeplot(z_benign, label="Benign", fill=True, color="#4c72b0", alpha=0.5)
    sns.kdeplot(z_harmful, label="Harmful", fill=True, color="#dd8452", alpha=0.5)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"Best Threshold (F1={best_params['f1']:.3f})")
    plt.title(f"Z-Score Distribution: p={best_params['p']:.3f}, m={m_actual}", fontsize=16, pad=15)
    plt.xlabel("Z-Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(frameon=False)
    sns.despine()
    plt.savefig(output_dir / "z_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Overlap Distribution
    overlap_scores = list(score_q.values())
    plt.figure(figsize=(9, 6))
    plt.hist(overlap_scores, bins=50, density=True, color="#818fbd", alpha=0.8, edgecolor='white')
    plt.axvline(best_params["p_threshold"], color="#c44e52", linestyle="--", linewidth=2,
                label=f"p={best_params['p']:.2f} (threshold={best_params['p_threshold']:.4f})")
    plt.title("Distribution of Component Overlap Scores", fontsize=16, pad=15)
    plt.xlabel("Overlap Score (lower is better)", fontsize=14)
    plt.ylabel("Relative Density", fontsize=14)
    plt.legend(frameon=False)
    sns.despine()
    plt.savefig(output_dir / "overlap_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. F1 Heatmap
    p_vals = sorted(list(set(r["p"] for r in all_results)))
    m_vals = sorted(list(set(r["m"] for r in all_results)))
    
    f1_grid = np.zeros((len(m_vals), len(p_vals)))
    for r in all_results:
        i = m_vals.index(r["m"])
        j = p_vals.index(r["p"])
        f1_grid[i, j] = r["f1"]
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_grid, xticklabels=[f"{x:.2f}" for x in p_vals], 
                yticklabels=[f"{x:.2f}" for x in m_vals],
                cmap="YlGnBu", annot=False, cbar_kws={'label': 'F1 Score'})
    plt.xlabel("p (Percentile Threshold)")
    plt.ylabel("m (Top-m Fraction)")
    plt.title("Grid Search: F1 Score Heatmap")
    plt.savefig(output_dir / "f1_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


    # 3. Robustness & Selection Heatmaps (Combined)
    n_heads = 0
    layers = set()
    for key in score_q.keys():
        layers.add(key[1])
        if key[0] == "o_proj":
            n_heads = max(n_heads, key[3] + 1)
            
    n_layers = max(layers) + 1
    n_cols = n_heads + 1 
    
    robust_grid = np.full((n_layers, n_cols), np.nan)
    select_grid = np.zeros((n_layers, n_cols))
    
    for key, score in score_q.items():
        l = key[1]
        c = key[3] if key[0] == "o_proj" else n_heads
        if np.isnan(robust_grid[l, c]) or score < robust_grid[l, c]:
            robust_grid[l, c] = score
            
    for key in subset:
        l = key[1]
        c = key[3] if key[0] == "o_proj" else n_heads
        select_grid[l, c] += 1
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(robust_grid, ax=axes[0], cmap="crest_r", annot=False, 
                cbar_kws={'label': 'Lower Score is More Robust (q-statistic)'})
    axes[0].set_title("Component Robustness Map\n(Smaller q -> Darker)", fontsize=16, pad=15)
    axes[0].set_ylabel("Layer Index", fontsize=14)
    axes[0].set_xlabel("Head Index (0 to N-1) | MLP (Last)", fontsize=14)
    axes[0].grid(False)
    
    sns.heatmap(select_grid, ax=axes[1], cmap="flare", annot=False,
                cbar_kws={'label': 'Selection Count (Across EOI)'})
    axes[1].set_title("Component Selection Frequency\n(More Selected -> Darker)", fontsize=16, pad=15)
    axes[1].set_ylabel("Layer Index", fontsize=14)
    axes[1].set_xlabel("Head Index (0 to N-1) | MLP (Last)", fontsize=14)
    axes[1].grid(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "component_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated grid search and component analysis plots in: {output_dir}")
