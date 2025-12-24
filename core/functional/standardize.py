"""标准化器"""

from __future__ import annotations

from typing import Dict, Union, Tuple
import numpy as np

# ComponentKey is defined as Union[Tuple[str, int, int], Tuple[str, int, int, int]]
ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]


class Standardizer:
    """组件投影值的标准化器（基于 Train benign 数据）"""
    
    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: 防止除零的小常数
        """
        self.eps = eps
        self.means: Dict[ComponentKey, float] = {}
        self.stds: Dict[ComponentKey, float] = {}
    
    def fit(self, train_benign_proj: Dict[ComponentKey, np.ndarray]) -> None:
        """
        在 Train benign 数据上拟合标准化参数。
        
        Args:
            train_benign_proj: 训练集 benign 的投影数据 {component_key: np.ndarray}
        """
        for key, values in train_benign_proj.items():
            values_flat = values.flatten()
            values_flat = values_flat[np.isfinite(values_flat)]
            
            if len(values_flat) > 0:
                self.means[key] = float(np.mean(values_flat))
                self.stds[key] = float(np.std(values_flat))
            else:
                self.means[key] = 0.0
                self.stds[key] = 1.0
    
    def transform(self, proj_data: Dict[ComponentKey, np.ndarray]) -> Dict[ComponentKey, np.ndarray]:
        """
        标准化投影数据。
        
        Args:
            proj_data: 投影数据 {component_key: np.ndarray}
        
        Returns:
            标准化后的数据 {component_key: np.ndarray}
        """
        standardized = {}
        for key, values in proj_data.items():
            if key not in self.means:
                continue
            
            mu = self.means[key]
            sigma = self.stds[key] + self.eps
            
            standardized[key] = (values - mu) / sigma
        
        return standardized







