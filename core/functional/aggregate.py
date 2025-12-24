"""top-m 聚合得分"""

from __future__ import annotations

from typing import Dict, Set, Union, Tuple
import numpy as np

# ComponentKey is defined as Union[Tuple[str, int, int], Tuple[str, int, int, int]]
ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]


def compute_topm_scores(
    standardized_proj: Dict[ComponentKey, np.ndarray],
    component_subset: Set[ComponentKey],
    m_cap: int = 20,
) -> np.ndarray:
    """
    对每个样本计算 top-m 聚合得分 z(x)。
    
    对于每个样本 x，从 component_subset 中选出标准化得分最高的 m 个组件，
    计算它们的平均值作为 z(x)。
    
    Args:
        standardized_proj: 标准化后的投影数据 {component_key: np.ndarray}
        component_subset: 要使用的组件集合 S_p
        m_cap: top-m 中的 m（默认 20）
    
    Returns:
        z_scores: 每个样本的聚合得分 [n_samples]
    """
    # 筛选出在 subset 中的组件
    available_keys = [k for k in component_subset if k in standardized_proj]
    
    if len(available_keys) == 0:
        return np.array([])
    
    # 确定样本数（从第一个组件推断）
    first_key = available_keys[0]
    first_scores = standardized_proj[first_key].flatten()
    n_samples = len(first_scores)
    
    if n_samples == 0:
        return np.array([])
    
    # 收集所有组件的得分
    component_scores = []
    for key in available_keys:
        scores = standardized_proj[key].flatten()
        if len(scores) == n_samples:
            component_scores.append(scores)
        elif len(scores) > n_samples:
            component_scores.append(scores[:n_samples])
    
    if len(component_scores) == 0:
        return np.zeros(n_samples)
    
    # 堆叠成矩阵 [n_components, n_samples]
    score_matrix = np.stack(component_scores, axis=0)
    
    # 对每个样本，取 top-m 组件的平均值
    m_actual = max(1, min(int(m_cap), score_matrix.shape[0]))
    
    # 对每列（样本）排序，取最大的 m_actual 个
    topm_scores = np.partition(score_matrix, -m_actual, axis=0)[-m_actual:]
    z_scores = np.mean(topm_scores, axis=0)
    
    return z_scores







