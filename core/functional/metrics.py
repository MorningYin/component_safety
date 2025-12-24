"""ROC 曲线与 AUC 计算（纯 numpy 实现）"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ROC 曲线。
    
    Args:
        y_true: 真实标签（0 或 1）
        y_score: 预测得分（越高越可能是正类）
    
    Returns:
        (fpr, tpr, thresholds): 假正率、真正率、阈值数组
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 确保是二分类
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError(f"需要二分类标签，但得到 {len(unique_labels)} 个唯一值")
    
    # 将标签映射到 0/1
    pos_label = unique_labels[1] if unique_labels[0] == 0 else unique_labels[0]
    y_binary = (y_true == pos_label).astype(int)
    
    # 按得分降序排序
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_binary_sorted = y_binary[desc_score_indices]
    
    # 计算正负样本数
    n_pos = np.sum(y_binary)
    n_neg = len(y_binary) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        # 边界情况：只有一类
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    
    # 计算累积的 TP 和 FP
    tps = np.cumsum(y_binary_sorted)
    fps = np.arange(1, len(y_binary_sorted) + 1) - tps
    
    # 计算 TPR 和 FPR
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # 添加起点 (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    
    # 计算阈值（使用排序后的得分，并在末尾添加一个更小的值）
    thresholds = np.concatenate([y_score_sorted, [y_score_sorted[-1] - 1]])
    
    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    使用梯形法则计算 AUC（曲线下面积）。
    
    Args:
        x: x 轴坐标（如 FPR）
        y: y 轴坐标（如 TPR）
    
    Returns:
        AUC 值
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 确保按 x 排序
    if len(x) != len(y):
        raise ValueError(f"x 和 y 长度不匹配: {len(x)} vs {len(y)}")
    
    # 按 x 排序
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # 使用梯形法则计算面积
    area = np.trapz(y_sorted, x_sorted)
    
    return float(area)







