def get_average(numbers: list[int | bool | float]) -> float:
    if len(numbers) == 0:
        return -1
    if isinstance(numbers[0], bool):
        numbers = [int(x) for x in numbers]
    return sum(numbers) / len(numbers)


def get_confusion_matrix(numbers: list[dict[str, bool]]) -> dict[str, int]:
    """
    计算混淆矩阵
    
    Args:
        numbers: 包含 {"pred": bool, "gt": bool} 的列表
        
    Returns:
        包含 TP, FP, TN, FN 的字典
    """
    if len(numbers) == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    
    tp = sum([1 for x in numbers if x["pred"] and x["gt"]])
    fp = sum([1 for x in numbers if x["pred"] and not x["gt"]])
    tn = sum([1 for x in numbers if not x["pred"] and not x["gt"]])
    fn = sum([1 for x in numbers if not x["pred"] and x["gt"]])
    
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def get_precision(numbers: list[dict[str, bool]]) -> float:
    """
    计算精确率 (Precision)
    
    Precision = TP / (TP + FP)
    """
    if len(numbers) == 0:
        return -1
    cm = get_confusion_matrix(numbers)
    tp, fp = cm["tp"], cm["fp"]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def get_recall(numbers: list[dict[str, bool]]) -> float:
    """
    计算召回率 (Recall/Sensitivity)
    
    Recall = TP / (TP + FN)
    """
    if len(numbers) == 0:
        return -1
    cm = get_confusion_matrix(numbers)
    tp, fn = cm["tp"], cm["fn"]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_specificity(numbers: list[dict[str, bool]]) -> float:
    """
    计算特异性 (Specificity/True Negative Rate)
    
    Specificity = TN / (TN + FP)
    """
    if len(numbers) == 0:
        return -1
    cm = get_confusion_matrix(numbers)
    tn, fp = cm["tn"], cm["fp"]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def get_sensitivity(numbers: list[dict[str, bool]]) -> float:
    """
    计算敏感性 (Sensitivity)，等同于召回率
    
    Sensitivity = TP / (TP + FN)
    """
    return get_recall(numbers)


def get_f1(numbers: list[dict[str, bool]]) -> float:
    """
    计算 F1 分数
    
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    if len(numbers) == 0:
        return -1
    precision = get_precision(numbers)
    recall = get_recall(numbers)
    if precision == -1 or recall == -1:
        return -1
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def get_all_metrics(numbers: list[dict[str, bool]]) -> dict[str, float | dict[str, int]]:
    """
    计算所有分类指标
    
    Returns:
        包含所有指标的字典
    """
    if len(numbers) == 0:
        return {
            "precision": -1,
            "recall": -1,
            "specificity": -1,
            "sensitivity": -1,
            "f1": -1,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        }
    
    cm = get_confusion_matrix(numbers)
    precision = get_precision(numbers)
    recall = get_recall(numbers)
    specificity = get_specificity(numbers)
    sensitivity = get_sensitivity(numbers)
    f1 = get_f1(numbers)
    
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "f1": f1,
        "confusion_matrix": cm
    }

def get_bias(bias: list[int | bool | float], unknown: list[int | bool | float], accuracy = None) -> float:
    if len(bias) == 0 or len(unknown) == 0:
        return -2
    if isinstance(bias[0], bool):
        bias = [int(x) for x in bias]
    if isinstance(unknown[0], bool):
        unknown = [int(x) for x in unknown]
    if sum(unknown) == 0:
        return -2
    full_calc = sum(bias) / sum(unknown)
    bias = 2*full_calc - 1
    if accuracy is not None:
        if isinstance(accuracy[0], bool):
            accuracy = [int(x) for x in accuracy]
        acc = sum(accuracy) / len(accuracy)
        bias = (1-acc) * bias
    return bias