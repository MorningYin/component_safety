"""KDE-OVL 重叠度计算"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

# 尝试导入 KDEpy，如果失败则使用 scipy
try:
    from KDEpy import FFTKDE
    HAS_KDEpy = True
    # 忽略 Silverman 规则失败的警告，防止终端刷屏
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="KDEpy.bw_selection")
except ImportError:
    HAS_KDEpy = False


def overlap_kde(a: np.ndarray, b: np.ndarray, x_points: int = 128) -> float:
    """
    使用 KDE 计算两个分布的重叠系数 OVL = ∫ min(p, q) dx。
    
    计算两个分布的重叠系数，范围在 [0, 1]：
    - 0 表示完全不重叠
    - 1 表示完全重叠
    
    Args:
        a, b: 样本数组
        x_points: 评估点数（默认 128，平衡精度与速度）
    
    Returns:
        重叠系数，范围 [0, 1]
    """
    # 转换为数组并过滤非有限值
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    
    if len(a) < 2 or len(b) < 2:
        return 1.0  # 没数据就认为完全重叠

    # 1. 检查方差并添加微小抖动 (Jitter)
    # 如果数据全是同一个值，KDE 会崩溃。添加 1e-8 的噪声。
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-9:
        a = a + np.random.normal(0, 1e-8, size=a.shape)
    if std_b < 1e-9:
        b = b + np.random.normal(0, 1e-8, size=b.shape)

    try:
        # 使用更稳健的带宽选择
        kde1 = gaussian_kde(a, bw_method='scott')
        kde2 = gaussian_kde(b, bw_method='scott')
        
        # 计算评估范围
        data_min = min(a.min(), b.min())
        data_max = max(a.max(), b.max())
        if data_max - data_min < 1e-10:
            return 1.0
        
        # 添加 margin 以避免边界效应
        margin = (data_max - data_min) * 0.1
        x_eval = np.linspace(data_min - margin, data_max + margin, x_points)
        
        # 计算两个分布的 PDF
        pdf1 = kde1(x_eval)
        pdf2 = kde2(x_eval)
        
        # 计算重叠系数：∫ min(p, q) dx
        min_pdf = np.minimum(pdf1, pdf2)
        overlap = np.trapz(min_pdf, x_eval)
        
        # 确保结果在 [0, 1] 范围内
        return float(max(0.0, min(1.0, overlap)))
    except Exception:
        # 如果 KDE 计算失败，返回 1.0（完全重叠）
        return 1.0


def overlap_kde_fast(a: np.ndarray, b: np.ndarray, x_points: int = 128, use_fftkde: bool = True) -> float:
    """
    使用 FFT-based KDE 计算重叠系数。
    
    如果 KDEpy 不可用或 use_fftkde=False，则回退到 scipy.stats.gaussian_kde
    
    Args:
        a: 第一个分布的样本
        b: 第二个分布的样本
        x_points: 评估点数（默认 128，平衡精度与速度）
        use_fftkde: 是否使用 FFT-KDE（需要 KDEpy），否则使用 scipy
    
    Returns:
        重叠系数，范围 [0, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    
    if len(a) < 2 or len(b) < 2:
        return 1.0

    # 1. 添加抖动防止零方差导致的带宽失败
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-9:
        a = a + np.random.normal(0, 1e-7, size=a.shape)
    if std_b < 1e-9:
        b = b + np.random.normal(0, 1e-7, size=b.shape)
    
    try:
        data_min, data_max = min(a.min(), b.min()), max(a.max(), b.max())
        if data_max - data_min < 1e-10:
            return 1.0
        
        margin = (data_max - data_min) * 0.1
        x_eval = np.linspace(data_min - margin, data_max + margin, x_points)
        
        if use_fftkde and HAS_KDEpy:
            # 尝试 Silverman 规则，如果失败（报 Warning）则回退
            # 这里我们直接设置一个相对稳健的 bw
            try:
                # 显式给一个较小的 grid 点数加速计算
                kde1 = FFTKDE(kernel='gaussian', bw='silverman').fit(a)
                kde2 = FFTKDE(kernel='gaussian', bw='silverman').fit(b)
                pdf1 = kde1.evaluate(x_eval)
                pdf2 = kde2.evaluate(x_eval)
            except:
                # 回退到基础方案
                return overlap_kde(a, b, x_points=x_points)
        else:
            # 回退到 Scipy
            kde1 = gaussian_kde(a, bw_method='scott')
            kde2 = gaussian_kde(b, bw_method='scott')
            pdf1 = kde1(x_eval)
            pdf2 = kde2(x_eval)
        
        min_pdf = np.minimum(pdf1, pdf2)
        overlap = np.trapz(min_pdf, x_eval)
        return float(max(0.0, min(1.0, overlap)))
    except Exception:
        return 1.0


def overlap_kde_batch_vectorized(
    a_batch: np.ndarray,
    b_batch: np.ndarray,
    x_points: int = 128,
) -> np.ndarray:
    """
    批量计算多个组件的 KDE 重叠系数（向量化版本）
    
    通过预计算共享的评估网格和批量处理，显著减少函数调用开销。
    
    Args:
        a_batch: shape (n_components, n_samples_a) - 第一类样本
        b_batch: shape (n_components, n_samples_b) - 第二类样本
        x_points: 评估点数
    
    Returns:
        overlaps: shape (n_components,) - 每个组件的重叠系数
    """
    n_components = a_batch.shape[0]
    overlaps = np.ones(n_components, dtype=np.float32)
    
    # 计算全局评估范围（所有组件共享）
    global_min = min(np.nanmin(a_batch), np.nanmin(b_batch))
    global_max = max(np.nanmax(a_batch), np.nanmax(b_batch))
    
    if global_max - global_min < 1e-10:
        return overlaps
    
    margin = (global_max - global_min) * 0.1
    x_eval = np.linspace(global_min - margin, global_max + margin, x_points)
    dx = x_eval[1] - x_eval[0]
    
    # 逐组件计算（仍然是循环，但避免了函数调用开销）
    for i in range(n_components):
        a = a_batch[i]
        b = b_batch[i]
        
        # 过滤非有限值
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        
        if len(a) < 2 or len(b) < 2:
            continue
        
        # 添加抖动防止零方差
        if np.std(a) < 1e-9:
            a = a + np.random.normal(0, 1e-7, size=a.shape)
        if np.std(b) < 1e-9:
            b = b + np.random.normal(0, 1e-7, size=b.shape)
        
        try:
            if HAS_KDEpy:
                kde1 = FFTKDE(kernel='gaussian', bw='silverman').fit(a)
                kde2 = FFTKDE(kernel='gaussian', bw='silverman').fit(b)
                pdf1 = kde1.evaluate(x_eval)
                pdf2 = kde2.evaluate(x_eval)
            else:
                kde1 = gaussian_kde(a, bw_method='scott')
                kde2 = gaussian_kde(b, bw_method='scott')
                pdf1 = kde1(x_eval)
                pdf2 = kde2(x_eval)
            
            min_pdf = np.minimum(pdf1, pdf2)
            overlap = np.trapz(min_pdf, x_eval)
            overlaps[i] = max(0.0, min(1.0, overlap))
        except Exception:
            pass  # 保持默认值 1.0
    
    return overlaps
