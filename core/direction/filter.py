"""
Data filtering logic based on refusal scores.
Replicates filtering logic from refusal_direction project.
"""
from __future__ import annotations
import torch
from typing import List, Dict, Tuple, Callable, Any
from components_safety.core.direction.submodules import get_refusal_scores

def filter_instructions(
    model: Any, 
    instructions: List[str], 
    tokenize_fn: Callable, 
    refusal_toks: torch.Tensor,
    threshold: float = 0.0,
    keep_refused: bool = True,
    batch_size: int = 32
) -> List[str]:
    """
    Filter a list of instructions based on model refusal scores.
    
    Args:
        model: The language model.
        instructions: List of prompt strings.
        tokenize_fn: Function to tokenize instructions.
        refusal_toks: Tensor of refusal token IDs.
        threshold: Score threshold for filtering.
        keep_refused: If True, keep samples with score > threshold (refused).
                      If False, keep samples with score < threshold (not refused).
        batch_size: Inference batch size.
    """
    if not instructions:
        return []
        
    scores = get_refusal_scores(
        model=model,
        instructions=instructions,
        tokenize_instructions_fn=tokenize_fn,
        refusal_toks=refusal_toks,
        batch_size=batch_size,
        desc="过滤数据..."
    )
    
    scores_list = scores.tolist()
    if keep_refused:
        filtered = [inst for inst, score in zip(instructions, scores_list) if score > threshold]
    else:
        filtered = [inst for inst, score in zip(instructions, scores_list) if score < threshold]
        
    return filtered

def filter_direction_data(
    model_base: Any,
    harmful_train: List[str],
    harmless_train: List[str],
    harmful_val: List[str],
    harmless_val: List[str],
    filter_train: bool = True,
    filter_val: bool = True,
    batch_size: int = 32
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Filter training and validation datasets for direction extraction.
    """
    model = model_base.model
    tokenize_fn = model_base.tokenize_instructions_fn
    refusal_toks = model_base.refusal_toks
    
    if filter_train:
        print(f"  - Filtering training data (threshold=0)...")
        h_train_filtered = filter_instructions(model, harmful_train, tokenize_fn, refusal_toks, keep_refused=True, batch_size=batch_size)
        hl_train_filtered = filter_instructions(model, harmless_train[:5 * len(harmful_train)], tokenize_fn, refusal_toks, keep_refused=False, batch_size=batch_size)
        
        # Balance harmless to harmful count if necessary (consistent with refusal_direction)
        if len(h_train_filtered) > 0 and len(hl_train_filtered) > 0:
            harmful_train, harmless_train = h_train_filtered, hl_train_filtered
            print(f"    * Train filtered: harmful={len(harmful_train)}, harmless={len(harmless_train)}")
        else:
            print(f"    * Warning: Filtering resulted in empty set, skipping train filtering.")

    if filter_val:
        print(f"  - Filtering validation data (threshold=0)...")
        h_val_filtered = filter_instructions(model, harmful_val, tokenize_fn, refusal_toks, keep_refused=True, batch_size=batch_size)
        hl_val_filtered = filter_instructions(model, harmless_val[:5 * len(harmful_val)], tokenize_fn, refusal_toks, keep_refused=False, batch_size=batch_size)
        
        if len(h_val_filtered) > 0 and len(hl_val_filtered) > 0:
            harmful_val, harmless_val = h_val_filtered, hl_val_filtered
            print(f"    * Val filtered: harmful={len(harmful_val)}, harmless={len(harmless_val)}")
        else:
            print(f"    * Warning: Filtering resulted in empty set, skipping val filtering.")
            
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_eval_datasets(
    model_base: Any,
    datasets: Dict[str, List[Dict]],
    batch_size: int = 32
) -> Dict[str, List[Dict]]:
    """
    Filter datasets dictionary (usually 'benign' and 'harmful_safe').
    """
    model = model_base.model
    tokenize_fn = model_base.tokenize_instructions_fn
    refusal_toks = model_base.refusal_toks
    
    filtered_datasets = {}
    
    for name, data in datasets.items():
        if not data:
            filtered_datasets[name] = []
            continue
            
        instructions = [d['instruction'] for d in data]
        
        # Decide logic based on name
        # 'benign' or 'harmless' -> keep NOT refused
        # 'harmful' -> keep refused
        keep_refused = 'harmful' in name.lower()
        
        print(f"  - Filtering {name} (keep_refused={keep_refused})...")
        filtered_instrs = filter_instructions(
            model, instructions, tokenize_fn, refusal_toks, 
            keep_refused=keep_refused, batch_size=batch_size
        )
        
        # Reconstruct list of dicts
        instr_set = set(filtered_instrs)
        filtered_data = [d for d in data if d['instruction'] in instr_set]
        filtered_datasets[name] = filtered_data
        
        print(f"    * {name}: {len(data)} -> {len(filtered_data)}")
        
    return filtered_datasets
