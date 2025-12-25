"""
可视化辅助函数
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def extract_z_scores(individual_results: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    从 individual_results 中提取 z_scores 和真实标签
    
    Args:
        individual_results: 评估结果列表，每个元素包含 metadata 字段
        
    Returns:
        (z_scores, y_true, threshold) 或 (None, None, None) 如果没有 z_scores
    """
    z_scores = []
    y_true = []
    threshold = None
    
    for result in individual_results:
        if "metadata" in result and "z_score" in result["metadata"]:
            z_score = result["metadata"]["z_score"]
            if isinstance(z_score, (int, float)):
                z_scores.append(float(z_score))
                
                # 提取真实标签 (harmful=1, unharmful=0)
                gt_label = result.get("gt_prompt_harmfulness", "")
                y_true.append(1 if gt_label == "harmful" else 0)
                
                # 提取阈值（通常所有样本使用相同阈值）
                if threshold is None and "z_threshold" in result["metadata"]:
                    threshold = result["metadata"]["z_threshold"]
    
    if len(z_scores) == 0:
        return None, None, None
    
    return np.array(z_scores), np.array(y_true), threshold


def compute_roc_auc(z_scores: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算 ROC 曲线和 AUC
    
    Args:
        z_scores: 预测分数
        y_true: 真实标签
        
    Returns:
        (fpr, tpr, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, z_scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_pr_auc(z_scores: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算 PR 曲线和 AUC
    
    Args:
        z_scores: 预测分数
        y_true: 真实标签
        
    Returns:
        (precision, recall, auc_score)
    """
    precision, recall, _ = precision_recall_curve(y_true, z_scores)
    auc_score = auc(recall, precision)
    return precision, recall, auc_score


def organize_results_by_category(individual_results: List[Dict], category_key: str = "category_adversarial") -> Dict[str, List[Dict]]:
    """
    按类别组织结果
    
    Args:
        individual_results: 评估结果列表
        category_key: 类别字段名（如 "category_adversarial"）
        
    Returns:
        按类别分组的字典
    """
    organized = {}
    
    for result in individual_results:
        # 尝试从不同位置提取类别信息
        category = None
        
        # 从 gt 数据中提取（如果存在）
        if "prompt" in result and hasattr(result, "prompt"):
            # 这里需要根据实际数据结构调整
            pass
        
        # 从 id 或其他字段提取
        # 对于 toxicchat，类别可能在原始数据中
        # 这里返回空字典，由调用者根据具体任务调整
        
        if category is None:
            category = "unknown"
        
        if category not in organized:
            organized[category] = []
        organized[category].append(result)
    
    return organized


