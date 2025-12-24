# -*- coding: utf-8 -*-
"""
direction_utils.py

方向向量相关的底层工具函数：加载、归一化、模式解析等。
纯函数，无副作用，可复用。
"""

from pathlib import Path
import torch

def load_direction_global(path: Path) -> torch.Tensor:
    """加载并归一化全局方向向量。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        path: 全局 direction 文件路径（通常是 direction.pt）
        
    Returns:
        归一化后的全局方向向量，shape (d_model,)
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到全局 direction 文件：{path}")
    vec = torch.load(path, map_location="cpu")
    return vec / vec.norm()


def resolve_eoi_marker(model_base) -> str:
    """从 model_base 中获取 EOI 标记。"""
    try:
        # 获取 EOI tokens，然后解码为字符串
        eoi_tokens = model_base._get_eoi_toks()
        marker = model_base.tokenizer.decode(eoi_tokens)
        return marker
    except Exception as e:
        raise ValueError(f"无法从 model_base 获取 marker: {e}") from e


def normalize_token_index(idx: int, length: int) -> int:
    """规范化token索引（支持负数）。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        idx: token 索引（支持负数，-1 表示最后一个）
        length: 序列长度
        
    Returns:
        规范化后的索引（0 到 length-1）
    """
    if idx < 0:
        idx = length + idx
    if idx < 0 or idx >= length:
        raise ValueError(f"token索引越界：{idx}（长度 {length}）")
    return idx
