"""
DataManager: 统一的数据管理工具。
核心思路：
1. 使用 refusal_direction/dataset 中预定义的数据切分。
2. 支持 train/val/test 三种切分。
3. 为每个 harmful 指令匹配 reference_response（如果有）。
"""

import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from refusal_direction.dataset.load_dataset import load_dataset_split


class DataManager:
    # 基础物理路径
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    INTERNAL_DATA_DIR = PROJECT_ROOT / "components_safety" / "data"
    # refusal_direction/dataset 目录
    DATASET_DIR = PROJECT_ROOT / "refusal_direction" / "dataset"
    RAW_DATA_DIR = DATASET_DIR / "raw"
    PROCESSED_DATA_DIR = DATASET_DIR / "processed"
    SPLITS_DIR = DATASET_DIR / "splits"

    def __init__(self):
        self._ref_responses_cache = {}

    def _load_reference_responses(self) -> Dict[str, str]:
        """建立 instruction -> reference_response 的全量搜索映射"""
        if self._ref_responses_cache:
            return self._ref_responses_cache

        mapping = {}
        
        # 1. 优先从 raw 目录下的 CSV 文件加载（最原始的 target）
        # AdvBench
        adv = self.RAW_DATA_DIR / "advbench.csv"
        if adv.exists():
            df = pd.read_csv(adv)
            for _, row in df.iterrows():
                mapping[str(row.get('goal', '')).strip()] = str(row.get('target', '')).strip()
        
        # JailbreakBench
        jb = self.RAW_DATA_DIR / "jailbreakbench.csv"
        if jb.exists():
            df = pd.read_csv(jb)
            for _, row in df.iterrows():
                mapping[str(row.get('Goal', '')).strip()] = str(row.get('Target', '')).strip()

        # 2. 从 processed 目录备份
        for file_path in self.PROCESSED_DATA_DIR.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        instruction = item.get('instruction', '').strip()
                        if instruction in mapping: continue
                        target = item.get('target') or item.get('reference_response') or item.get('output')
                        if instruction and target:
                            mapping[instruction] = target
            except Exception:
                pass

        self._ref_responses_cache = mapping
        return mapping

    def get_harmful_train(self) -> List[Dict]:
        """加载 Harmful 训练集"""
        return self._load_split('harmful', 'train', category="harmful_safe")

    def get_harmful_val(self) -> List[Dict]:
        """加载 Harmful 验证集"""
        return self._load_split('harmful', 'val', category="harmful_safe")

    def get_harmful_test(self) -> List[Dict]:
        """加载 Harmful 测试集"""
        return self._load_split('harmful', 'test', category="harmful_safe")

    def get_harmless_train(self, limit: Optional[int] = None) -> List[Dict]:
        """加载 Harmless 训练集"""
        data = self._load_split('harmless', 'train', category="benign")
        if limit:
            data = data[:limit]
        return data

    def get_harmless_val(self, limit: Optional[int] = None) -> List[Dict]:
        """加载 Harmless 验证集"""
        data = self._load_split('harmless', 'val', category="benign")
        if limit:
            data = data[:limit]
        return data

    def get_harmless_test(self) -> List[Dict]:
        """加载 Harmless 测试集"""
        return self._load_split('harmless', 'test', category="benign")

    def _load_split(self, harmtype: str, split: str, category: str) -> List[Dict]:
        """使用 load_dataset_split 加载数据"""
        try:
            data = load_dataset_split(harmtype=harmtype, split=split)
        except Exception as e:
            print(f"Warning: Failed to load {harmtype}_{split}: {e}")
            return []

        ref_map = self._load_reference_responses()
        results = []
        for item in data:
            instruction = item.get('instruction', '').strip()
            if not instruction:
                continue

            entry = {
                "instruction": instruction,
                "category": item.get("category") or category,
                "reference_response": ref_map.get(instruction, "Sure, I'll help you with that.")
            }
            results.append(entry)
        return results

    def _load_json(self, path: Path, category: str) -> List[Dict]:
        if not path.exists():
            print(f"Warning: File not found: {path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        ref_map = self._load_reference_responses()
        results = []
        for item in data:
            instruction = item.get('instruction', '').strip()
            if not instruction: continue
            
            entry = {
                "instruction": instruction,
                "category": item.get("category") or category,
                "reference_response": item.get("reference_response") or ref_map.get(instruction, "Sure, I'll help you with that.")
            }
            results.append(entry)
        return results

    def get_direction_pipeline_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        供 Stage 2 (Direction Pipeline) 使用的独立加载接口。
        返回: (harmful_train, harmless_train, harmful_val, harmless_val) 的指令列表。
        """
        h_train = [d['instruction'] for d in self.get_harmful_train()]
        b_train = [d['instruction'] for d in self.get_harmless_train(limit=len(h_train))]
        h_val = [d['instruction'] for d in self.get_harmful_val()]
        b_val = [d['instruction'] for d in self.get_harmless_val(limit=len(h_val))]
        
        return h_train, b_train, h_val, b_val

    def load_dataset(self, path: Path) -> List[Dict]:
        """
        通用数据集加载方法。从 JSON/JSONL 文件加载数据。
        
        Args:
            path: 数据集文件路径（支持 .json 和 .jsonl）
            
        Returns:
            List[Dict]: 数据列表，每个元素包含 'instruction' 字段
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        data = []
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    # 尝试常见的嵌套结构
                    for key in ['data', 'samples', 'items', 'prompts']:
                        if key in content:
                            data = content[key]
                            break
                    else:
                        data = [content]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # 确保每个条目有 instruction 字段
        processed = []
        for item in data:
            if isinstance(item, str):
                processed.append({'instruction': item})
            elif isinstance(item, dict):
                # 尝试多种可能的字段名
                instr = item.get('instruction') or item.get('prompt') or item.get('text') or item.get('goal')
                if instr:
                    processed.append({'instruction': instr, **item})
                else:
                    processed.append(item)
            else:
                continue
        
        return processed

def generate_consolidated_jsonl(output_dir: Optional[Path] = None):
    """一键生成供 Pipeline 使用的 .jsonl 文件"""
    dm = DataManager()
    if output_dir is None:
        output_dir = dm.INTERNAL_DATA_DIR
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 拒答数据集 (Train Harmful + Val Harmful)
    refusal_data = dm.get_harmful_train() + dm.get_harmful_val()
    with open(output_dir / "refusal.jsonl", "w", encoding="utf-8") as f:
        for d in refusal_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    # 2. 无害数据集 (Train Harmless + Val Harmless)
    benign_data = dm.get_harmless_train() + dm.get_harmless_val()
    with open(output_dir / "benign.jsonl", "w", encoding="utf-8") as f:
        for d in benign_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated datasets in {output_dir}")

if __name__ == "__main__":
    dm = DataManager()
    # 优先生成到项目内部路径 (components_safety/data)
    generate_consolidated_jsonl(dm.INTERNAL_DATA_DIR)
    
    # 同时生成一份到外部 cache 路径供 Pipeline 使用
    generate_consolidated_jsonl(Path("/root/autodl-tmp/results/data"))
