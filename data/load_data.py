"""
方向提取的数据加载和准备。

该模块负责从原始来源加载有害和无害数据，并将它们分割为训练/验证/测试集。
它合并了之前的 defense_data.py 的功能，实现了统一的数据管理。
"""

import os

os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import json
import csv
import shutil
import random
from pathlib import Path
from typing import List, Dict, Union, Optional
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

# 路径配置
BASE_DIR = Path(__file__).parent
LOCAL_SPLITS_DIR = BASE_DIR / "splits"

RAW_DATA_DIR = BASE_DIR / "raw"

# 本地中间文件
LOCAL_HARMFUL_PATH = BASE_DIR / "harmful.json"
LOCAL_BENIGN_PATH = BASE_DIR / "benign.json"

# 常量定义
SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmful', 'harmless']


def get_data(force: bool = False):
    """
    聚合原始数据集并保存到本地有害/无害 JSON 文件。
    包含:
    - 有害: advbench, jailbreakbench, malicious_instruct, strongreject, tdc2023 (Raw)
    - 无害: alpaca (Processed) + OR-Bench-80k
    """
    if LOCAL_HARMFUL_PATH.exists() and LOCAL_BENIGN_PATH.exists() and not force:
        print("✓ 原始数据已同步到本地 (harmful.json, benign.json)。")
        return

    print("正在从原始源同步数据...")
    # 使用字典去重: goal -> target
    harmful_map = {}
    
    # 1. 加载有害数据
    # 1.1 AdvBench
    try:
        print("  - 加载 advbench...")
        with open(RAW_DATA_DIR / "advbench.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                if 'goal' in row:
                    goal = row['goal'].strip()
                    target = row.get('target', '').strip()
                    if goal and goal not in harmful_map:
                        harmful_map[goal] = (target, "simple_harmful")
    except Exception as e:
        print(f"  - 警告: 加载 advbench 失败: {e}")

    # 1.2 JailbreakBench
    try:
        print("  - 加载 jailbreakbench...")
        with open(RAW_DATA_DIR / "jailbreakbench.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                if 'Goal' in row:
                    goal = row['Goal'].strip()
                    target = row.get('Target', '').strip()
                    if goal and goal not in harmful_map:
                        harmful_map[goal] = (target, "simple_harmful")
    except Exception as e:
        print(f"  - 警告: 加载 jailbreakbench 失败: {e}")

    # 1.3 StrongReject
    try:
        print("  - 加载 strongreject...")
        with open(RAW_DATA_DIR / "strongreject.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                if 'forbidden_prompt' in row:
                    goal = row['forbidden_prompt'].strip()
                    if goal and goal not in harmful_map:
                        harmful_map[goal] = ("", "simple_harmful")
    except Exception as e:
        print(f"  - 警告: 加载 strongreject 失败: {e}")

    # 1.4 Malicious Instruct
    try:
        print("  - 加载 malicious_instruct...")
        with open(RAW_DATA_DIR / "malicious_instruct.txt", 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if line and line not in harmful_map:
                    harmful_map[line] = ("", "simple_harmful")
    except Exception as e:
        print(f"  - 警告: 加载 malicious_instruct 失败: {e}")

    # 1.5 TDC2023
    print("  - 加载 tdc2023...")
    for filename in ["tdc2023_dev_behaviors.json", "tdc2023_test_behaviors.json"]:
        try:
            with open(RAW_DATA_DIR / filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for goal in tqdm(data):
                        goal = goal.strip()
                        if goal and goal not in harmful_map:
                            harmful_map[goal] = ("", "simple_harmful")
        except Exception as e:
            print(f"  - 警告: 加载 {filename} 失败: {e}")


    # 1.6 WildJailbreak
    try:
        print("  - 加载 wildjailbreak_harmful...")
        wildjailbreak = load_dataset("allenai/wildjailbreak", "train", streaming=True)['train']
        for item in tqdm(wildjailbreak):
            if item['data_type'] == "vanilla_harmful" and item.get('vanilla'):
                harmful_map[item['vanilla']] = ("", "vanilla_harmful")
            elif item['data_type'] == "adversarial_harmful" and item.get('adversarial'):
                harmful_map[item['adversarial']] = ("", "adversarial_harmful")
    except Exception as e:
        print(f"  - 警告: 加载 wildjailbreak 失败: {e}")

    # 2. 加载无害数据
    benign_map = {}
    
    # 2.1 Alpaca
    try:
        print("  - 加载 alpaca...")
        with open(RAW_DATA_DIR / "alpaca.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in tqdm(data):
                if 'instruction' in item:
                    benign_map[item['instruction']] = "simple_benign"
    except Exception as e:
        print(f"  - 警告: 加载 alpaca 失败: {e}")
    
    # 2.2 OR-Bench-80k
    try:
        print("  - 加载 OR-Bench-80k...")
        orbench = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
        for item in tqdm(orbench):
            benign_map[item['prompt']] = "OR_benign"
    except Exception as e:
        print(f"  - 警告: 加载 OR-Bench-80k 失败: {e}")

    # 2.3 WildJailbreak
    try:
        print("  - 加载 wildjailbreak_benign...")
        wildjailbreak = load_dataset("allenai/wildjailbreak", "train", streaming=True)['train']
        for item in tqdm(wildjailbreak):
            if item['data_type'] == "vanilla_benign" and item.get('vanilla'):
                benign_map[item['vanilla']] = "vanilla_benign"
            elif item['data_type'] == "adversarial_benign" and item.get('adversarial'):
                benign_map[item['adversarial']] = "adversarial_benign"
    except Exception as e:
        print(f"  - 警告: 加载 wildjailbreak 失败: {e}")

    # 格式化并保存
    harmful_data = [{"instruction": instr, "target": target_type[0], "type": target_type[1]} for instr, target_type in sorted(harmful_map.items())]
    benign_data = [{"instruction": instr, "type": type} for instr, type in sorted(benign_map.items())]

    with open(LOCAL_HARMFUL_PATH, 'w', encoding='utf-8') as f:
        json.dump(harmful_data, f, indent=2, ensure_ascii=False)
    with open(LOCAL_BENIGN_PATH, 'w', encoding='utf-8') as f:
        json.dump(benign_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 已同步 {len(harmful_data)} 条有害数据和 {len(benign_data)} 条无害数据至 {BASE_DIR}")


def create_splits(force = False):
    """
    读取本地的 harmful.json 和 benign.json，生成 train/val/test 划分 (80/10/10)。
    """
    get_data()
    LOCAL_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    is_exists = True   
    for ht in HARMTYPES:
        for sp in SPLITS:
            path = LOCAL_SPLITS_DIR / f"{ht}_{sp}.json"
            if not path.exists():
                is_exists = False

    if is_exists and not force:
        print("✓ 数据划分已存在，跳过创建")
        return
            
    
    print("正在生成数据划分 (80/10/10)...")
    random.seed(42)

    def _split_and_save(data: List[Dict], prefix: str):
        random.shuffle(data)
        n = len(data)

        if prefix == 'harmful':
            simple_harmful = [d for d in data if d['type'] == 'simple_harmful']
            adversarial_harmful = [d for d in data if d['type'] == 'adversarial_harmful']
            vanilla_harmful = [d for d in data if d['type'] == 'vanilla_harmful']
            data_splits = [simple_harmful, adversarial_harmful, vanilla_harmful]
        else:
            vanilla_benign = [d for d in data if d['type'] == 'vanilla_benign']
            simple_benign = [d for d in data if d['type'] == 'simple_benign']
            adversarial_benign = [d for d in data if d['type'] == 'adversarial_benign']
            or_benign = [d for d in data if d['type'] == 'OR_benign']
            data_splits = [vanilla_benign, simple_benign, adversarial_benign, or_benign]

        splits = {'train': [], 'val': [], 'test': []}
        for data_split in data_splits:
            splits['train'] += data_split[:int(len(data_split) * 0.8)]
            splits['val'] += data_split[int(len(data_split) * 0.8):int(len(data_split) * 0.9)]
            splits['test'] += data_split[int(len(data_split) * 0.9):]
        
        for sp_name, sp_data in splits.items():
            path = LOCAL_SPLITS_DIR / f"{prefix}_{sp_name}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(sp_data, f, indent=2, ensure_ascii=False)
        
        print(f"  - {prefix}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    if LOCAL_HARMFUL_PATH.exists():
        with open(LOCAL_HARMFUL_PATH, 'r', encoding='utf-8') as f:
            harmful_data = json.load(f)
        _split_and_save(harmful_data, 'harmful')
    
    if LOCAL_BENIGN_PATH.exists():
        with open(LOCAL_BENIGN_PATH, 'r', encoding='utf-8') as f:
            benign_data = json.load(f)

        _split_and_save(benign_data, 'harmless')

    print(f"✓ 数据划分已保存至 {LOCAL_SPLITS_DIR}")


def load_dataset_split(
    split: str, 
    seed: int = 42
) -> Union[List[Dict], List[str]]:
    """按危害类型和划分加载预处理的数据集。"""
    create_splits()
    random.seed(seed)
    
    assert split in SPLITS, f"split 必须是 {SPLITS} 之一"
    
    harmful_path = LOCAL_SPLITS_DIR / f"harmful_{split}.json"
    with open(harmful_path, 'r', encoding='utf-8') as f:
        h_d = json.load(f)
        random.shuffle(h_d)
    harmful_dataset = [d['instruction'] for d in h_d if d['type'] == 'simple_harmful']
    n = len(harmful_dataset)
    # harmful_dataset += [d['instruction'] for d in h_d if d['type'] == 'adversarial_harmful'][:n]
    # harmful_dataset += [d['instruction'] for d in h_d if d['type'] == 'vanilla_harmful'][:n]

    harmless_path = LOCAL_SPLITS_DIR / f"harmless_{split}.json"
    with open(harmless_path, 'r', encoding='utf-8') as f:
        b_d = json.load(f)
        random.shuffle(b_d)
    harmless_dataset = [d['instruction'] for d in b_d if d['type'] == 'simple_benign'][:n]
    # harmless_dataset += [d['instruction'] for d in b_d if d['type'] == 'OR_benign'][:n]
    # harmless_dataset += [d['instruction'] for d in b_d if d['type'] == 'adversarial_benign'][:n]
    # harmless_dataset += [d['instruction'] for d in b_d if d['type'] == 'vanilla_benign'][:n]
    
    return harmful_dataset, harmless_dataset


def load_and_sample_direction_data(seed = 42) -> tuple:
    """加载并采样用于提取方向的数据集。"""
    harmful_train, harmless_train = load_dataset_split(split='train', seed=seed)
    harmful_val, harmless_val = load_dataset_split(split='val', seed=seed)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def get_harmful_train(instructions_only: bool = True) -> List:
    return load_dataset_split('harmful', 'train', instructions_only=instructions_only)

def get_harmless_train(instructions_only: bool = True, limit: Optional[int] = None) -> List:
    data = load_dataset_split('harmless', 'train', instructions_only=instructions_only)
    return data[:limit] if limit is not None else data

def get_harmful_val(instructions_only: bool = True) -> List:
    return load_dataset_split('harmful', 'val', instructions_only=instructions_only)

def get_harmless_val(instructions_only: bool = True, limit: Optional[int] = None) -> List:
    data = load_dataset_split('harmless', 'val', instructions_only=instructions_only)
    return data[:limit] if limit is not None else data

def get_harmful_test(instructions_only: bool = True) -> List:
    return load_dataset_split('harmful', 'test', instructions_only=instructions_only)


def get_harmless_test(instructions_only: bool = True, limit: Optional[int] = None) -> List:
    data = load_dataset_split('harmless', 'test', instructions_only=instructions_only)
    return data[:limit] if limit is not None else data


def load_defense_data(
    benign_path: Optional[Path] = None,
    harmful_path: Optional[Path] = None,
    seed = 42
) -> tuple:
    """Load data for defense/classifier building."""
    get_data() # Ensure local data is available
    
    if benign_path is None:
        benign_path = LOCAL_BENIGN_PATH
    if harmful_path is None:
        harmful_path = LOCAL_HARMFUL_PATH
        
    with open(benign_path, 'r', encoding='utf-8') as f:
        benign_data = json.load(f)
    with open(harmful_path, 'r', encoding='utf-8') as f:
        harmful_data = json.load(f)

    random.seed(seed)
    
    def get_shuffled(data, t):
        subset = [d for d in data if d.get('type') == t]
        random.shuffle(subset)
        return subset

    simple_harmful = get_shuffled(harmful_data, 'simple_harmful')
    adversarial_harmful = get_shuffled(harmful_data, 'adversarial_harmful')
    vanilla_harmful = get_shuffled(harmful_data, 'vanilla_harmful')

    vanilla_benign = get_shuffled(benign_data, 'vanilla_benign')
    simple_benign = get_shuffled(benign_data, 'simple_benign')
    adversarial_benign = get_shuffled(benign_data, 'adversarial_benign')
    or_benign = get_shuffled(benign_data, 'OR_benign')
    
    return {
        'benign': simple_benign[:len(simple_harmful)],
        'harmful': simple_harmful
    }

def load_threshold_val_data(seed: int = 42) -> Dict[str, List[Dict]]:
    """
    加载专门用于阈值搜索的验证集。
    512 OR_benign, 512 adversarial_benign, 1024 vanilla_benign (总计 2048)
    512 adversarial_harmful, 512 vanilla_harmful (总计 1024)
    """
    get_data()
    with open(LOCAL_BENIGN_PATH, 'r', encoding='utf-8') as f:
        benign_data = json.load(f)
    with open(LOCAL_HARMFUL_PATH, 'r', encoding='utf-8') as f:
        harmful_data = json.load(f)

    random.seed(seed)
    
    def get_shuffled(data, t):
        subset = [d for d in data if d.get('type') == t]
        random.shuffle(subset)
        return subset

    # 无害验证集
    or_benign = get_shuffled(benign_data, 'OR_benign')[:128]
    simple_benign = get_shuffled(benign_data, 'simple_benign')[-256:]
    adversarial_benign = get_shuffled(benign_data, 'adversarial_benign')[:64]
    
    # 有害验证集
    adv_harmful = get_shuffled(harmful_data, 'adversarial_harmful')[:256]
    simple_harmful = get_shuffled(harmful_data, 'simple_harmful')[-256:]

    return {
        'benign': or_benign + simple_benign + adversarial_benign,
        'harmful': adv_harmful + simple_harmful
    }

if __name__ == "__main__":
    # get_data(force = True)
    create_splits(force = True)