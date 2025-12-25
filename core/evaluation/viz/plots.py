"""
Evaluation result visualization functions - Scientific style
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from evaluation.viz.utils import extract_z_scores, compute_roc_auc, compute_pr_auc
from evaluation.viz.layout import (
    LayoutCalculator,
    TextOverlapDetector,
    TableLayoutOptimizer,
    ContentInfoCollector
)

# Scientific style color scheme (ColorBrewer)
SCIENTIFIC_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',   # Purple-red
    'success': '#F18F01',     # Orange
    'danger': '#C73E1D',      # Red
    'harmful': '#C73E1D',     # Harmful samples - red
    'unharmful': '#2E86AB',   # Unharmful samples - blue
    'threshold': '#6A994E',   # Threshold line - green
    'baseline': '#6C757D',   # Baseline - gray
}


def setup_scientific_style():
    """
    Setup scientific-style plot configuration
    """
    # Font settings
    plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans', 'Arial']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    
    # Resolution and format
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # Lines and borders
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    
    # Style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.3',
        'axes.linewidth': 1.2,
        'grid.color': '0.8',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
    })
    
    # Color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [
        SCIENTIFIC_COLORS['primary'],
        SCIENTIFIC_COLORS['secondary'],
        SCIENTIFIC_COLORS['success'],
        SCIENTIFIC_COLORS['danger'],
    ])


# Initialize scientific style
setup_scientific_style()


def plot_confusion_matrix(
    individual_results: List[Dict],
    output_path: Path,
    task_name: str = ""
) -> None:
    """
    Plot scientific-style confusion matrix heatmap with all metrics and detailed information
    
    Args:
        individual_results: List of evaluation results
        output_path: Output path
        task_name: Task name (for title)
    """
    # Calculate confusion matrix
    tp = fp = tn = fn = 0
    
    for result in individual_results:
        if result.get("is_parsing_error", False):
            # Parsing errors are treated as prediction errors
            gt_harmful = result.get("gt_prompt_harmfulness") == "harmful"
            if gt_harmful:
                fn += 1
            else:
                tn += 1
        else:
            gt_label = result.get("gt_prompt_harmfulness", "")
            pred_label = result.get("prompt_harmfulness", "")
            
            gt_harmful = (gt_label == "harmful")
            pred_harmful = (pred_label == "harmful")
            
            if pred_harmful and gt_harmful:
                tp += 1
            elif pred_harmful and not gt_harmful:
                fp += 1
            elif not pred_harmful and not gt_harmful:
                tn += 1
            elif not pred_harmful and gt_harmful:
                fn += 1
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    total = tp + fp + tn + fn
    
    # Calculate all metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Sensitivity equals recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate percentages
    cm_percent = (cm / total * 100) if total > 0 else cm
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_confusion_matrix_info(individual_results, task_name)
    
    # 构建标题
    title = 'Confusion Matrix'
    if task_name:
        title += f' - {task_name}'
    
    # 使用新的标准化布局
    layout_calc = LayoutCalculator()
    fig, ax_main, ax_stats, ax_info = layout_calc.create_standard_figure(
        content_info, 
        plot_type='confusion_matrix',
        include_stats_table=True,
        include_info_panel=True,
        title=title
    )
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Plot main confusion matrix heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Unharmful', 'Harmful'],
        yticklabels=['Unharmful', 'Harmful'],
        ax=ax_main,
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        linewidths=1.5,
        linecolor='white',
        square=True,
        annot_kws={'fontsize': 14, 'fontweight': 'bold'}
    )
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            percent = cm_percent[i, j]
            ax_main.text(j + 0.5, i + 0.75, f'({percent:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, color='gray')
    
    ax_main.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('True Label', fontsize=14, fontweight='bold')
    # 不在 ax 上设置标题，使用 suptitle
    
    # Display detailed statistics table on the right
    stats_data = [
        ['Metric', 'Value', 'Percentage'],
        ['True Positive (TP)', f'{tp}', f'{tp/total*100:.2f}%' if total > 0 else '0%'],
        ['False Positive (FP)', f'{fp}', f'{fp/total*100:.2f}%' if total > 0 else '0%'],
        ['True Negative (TN)', f'{tn}', f'{tn/total*100:.2f}%' if total > 0 else '0%'],
        ['False Negative (FN)', f'{fn}', f'{fn/total*100:.2f}%' if total > 0 else '0%'],
        ['', '', ''],
        ['Accuracy', f'{accuracy:.4f}', f'{accuracy*100:.2f}%'],
        ['Precision', f'{precision:.4f}', f'{precision*100:.2f}%'],
        ['Recall', f'{recall:.4f}', f'{recall*100:.2f}%'],
        ['Specificity', f'{specificity:.4f}', f'{specificity*100:.2f}%'],
        ['Sensitivity', f'{sensitivity:.4f}', f'{sensitivity*100:.2f}%'],
        ['F1 Score', f'{f1:.4f}', f'{f1*100:.2f}%'],
    ]
    
    # Optimize table layout
    table_optimizer = TableLayoutOptimizer()
    table_size = table_optimizer.calculate_table_size(
        len(stats_data), len(stats_data[0]), stats_data
    )
    
    # 创建表格，使用更合理的列宽
    table = ax_stats.table(cellText=stats_data, cellLoc='center', loc='upper center',
                          colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(table_size['fontsize'])
    table.scale(1.2, 1.8)  # 调整表格缩放
    
    # Set header style
    for i in range(3):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set data row styles
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i <= 5:  # Confusion matrix section
                table[(i, j)].set_facecolor('#E8F4F8')
            else:  # Metrics section
                table[(i, j)].set_facecolor('#F0F8E8')
    
    ax_stats.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    
    # Add sample information at the bottom
    info_text = (
        f'Total Samples: {total} | '
        f'Harmful: {tp + fn} ({((tp + fn)/total*100):.1f}%) | '
        f'Unharmful: {tn + fp} ({((tn + fp)/total*100):.1f}%)'
    )
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=spacing_params['pad_inches'])
    plt.close()


def plot_roc_curve(
    individual_results: List[Dict],
    output_path: Path,
    task_name: str = ""
) -> Optional[float]:
    """
    Plot scientific-style ROC curve with detailed performance metrics
    
    Args:
        individual_results: List of evaluation results
        output_path: Output path
        task_name: Task name
        
    Returns:
        AUC score, or None if no z_scores available
    """
    z_scores, y_true, threshold = extract_z_scores(individual_results)
    
    if z_scores is None:
        return None
    
    # Calculate ROC curve
    fpr, tpr, auc_score = compute_roc_auc(z_scores, y_true)
    
    # Calculate Youden index (optimal threshold point)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_threshold = np.sort(z_scores)[optimal_idx] if len(z_scores) > optimal_idx else None
    
    # Calculate performance metrics at threshold point
    threshold_metrics = {}
    if threshold is not None:
        pred_binary = (z_scores >= threshold).astype(int)
        cm = sk_confusion_matrix(y_true, pred_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            threshold_metrics = {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'threshold': threshold
            }
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_roc_info(individual_results, threshold_metrics, task_name)
    
    # 构建标题
    title = 'ROC Curve'
    if task_name:
        title += f' - {task_name}'
    
    # 使用新的标准化布局
    layout_calc = LayoutCalculator()
    fig, ax_main, ax_stats, ax_info = layout_calc.create_standard_figure(
        content_info, 
        plot_type='roc',
        include_stats_table=True,
        include_info_panel=True,
        title=title
    )
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Plot main ROC curve
    ax_main.plot(fpr, tpr, linewidth=2.5, color=SCIENTIFIC_COLORS['primary'], 
                label=f'ROC Curve (AUC = {auc_score:.4f})', zorder=3)
    ax_main.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, 
                label='Random Classifier (AUC = 0.500)', zorder=1)
    
    # Mark optimal threshold point
    ax_main.plot(optimal_fpr, optimal_tpr, 'o', markersize=12, 
                color=SCIENTIFIC_COLORS['success'], zorder=4,
                label=f'Optimal Threshold (Youden Index)')
    ax_main.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr], 
                '--', color=SCIENTIFIC_COLORS['success'], alpha=0.5, linewidth=1)
    ax_main.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr], 
                '--', color=SCIENTIFIC_COLORS['success'], alpha=0.5, linewidth=1)
    
    # Mark current threshold point (if exists)
    if threshold_metrics:
        ax_main.plot(threshold_metrics['fpr'], threshold_metrics['tpr'], 
                    's', markersize=10, color=SCIENTIFIC_COLORS['threshold'], 
                    zorder=4, label=f"Current Threshold (threshold={threshold_metrics['threshold']:.2f})")
    
    ax_main.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    # 不在 ax 上设置标题，使用 suptitle
    
    # Find optimal legend position using overlap detector
    overlap_detector = TextOverlapDetector()
    legend_items = ['ROC Curve', 'Random Classifier', 'Optimal Threshold']
    if threshold_metrics:
        legend_items.append('Current Threshold')
    data_bounds = (0, 1, 0, 1)  # ROC curve bounds
    legend_pos = overlap_detector.find_optimal_legend_position(ax_main, legend_items, data_bounds)
    ax_main.legend(loc=legend_pos, fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    
    # Display statistics on the right
    stats_data = [
        ['Metric', 'Value'],
        ['AUC-ROC', f'{auc_score:.4f}'],
        ['', ''],
        ['Optimal Threshold', ''],
        ['FPR', f'{optimal_fpr:.4f}'],
        ['TPR', f'{optimal_tpr:.4f}'],
    ]
    
    if threshold_metrics:
        stats_data.extend([
            ['', ''],
            ['Current Threshold', ''],
            ['Threshold', f'{threshold_metrics["threshold"]:.2f}'],
            ['FPR', f'{threshold_metrics["fpr"]:.4f}'],
            ['TPR', f'{threshold_metrics["tpr"]:.4f}'],
            ['Precision', f'{threshold_metrics["precision"]:.4f}'],
        ])
    
    stats_data.extend([
        ['', ''],
        ['Sample Information', ''],
        ['Total Samples', f'{len(y_true)}'],
        ['Positive', f'{np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.1f}%)'],
        ['Negative', f'{len(y_true)-np.sum(y_true)} ({(len(y_true)-np.sum(y_true))/len(y_true)*100:.1f}%)'],
    ])
    
    table = ax_stats.table(cellText=stats_data, cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Set header style
    for i in range(2):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set data row styles
    for i in range(1, len(stats_data)):
        for j in range(2):
            if 'Optimal' in stats_data[i][0] or 'Current' in stats_data[i][0] or 'Sample' in stats_data[i][0]:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax_stats.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=10)
    ax_stats.axis('off')
    
    # Bottom information
    info_text = (
        f'AUC-ROC = {auc_score:.4f} | '
        f'Optimal Threshold: FPR={optimal_fpr:.3f}, TPR={optimal_tpr:.3f}'
    )
    if threshold_metrics:
        info_text += f' | Current Threshold: {threshold_metrics["threshold"]:.2f} (FPR={threshold_metrics["fpr"]:.3f}, TPR={threshold_metrics["tpr"]:.3f})'
    
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=spacing_params['pad_inches'])
    plt.close()
    
    return auc_score


def plot_pr_curve(
    individual_results: List[Dict],
    output_path: Path,
    task_name: str = ""
) -> Optional[float]:
    """
    Plot scientific-style PR curve with detailed performance metrics
    
    Args:
        individual_results: List of evaluation results
        output_path: Output path
        task_name: Task name
        
    Returns:
        AUC score, or None if no z_scores available
    """
    z_scores, y_true, threshold = extract_z_scores(individual_results)
    
    if z_scores is None:
        return None
    
    # Calculate PR curve
    precision, recall, auc_score = compute_pr_auc(z_scores, y_true)
    
    # Calculate baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0
    positive_ratio = baseline
    
    # Calculate optimal F1 point on the curve
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Calculate performance metrics at threshold point
    threshold_metrics = {}
    if threshold is not None:
        pred_binary = (z_scores >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        try:
            prec_thresh = precision_score(y_true, pred_binary, zero_division=0)
            rec_thresh = recall_score(y_true, pred_binary, zero_division=0)
            f1_thresh = f1_score(y_true, pred_binary, zero_division=0)
            threshold_metrics = {
                'precision': prec_thresh,
                'recall': rec_thresh,
                'f1': f1_thresh,
                'threshold': threshold
            }
        except Exception:
            pass
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_pr_info(individual_results, threshold_metrics, task_name)
    
    # Calculate optimal layout
    layout_calc = LayoutCalculator()
    fig_width, fig_height = layout_calc.calculate_figure_size(content_info)
    gridspec_params = layout_calc.calculate_gridspec_ratios(content_info, 'pr')
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Create figure with calculated size
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=gridspec_params['height_ratios'],
        width_ratios=gridspec_params['width_ratios'],
        hspace=gridspec_params['hspace'],
        wspace=gridspec_params['wspace']
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    
    # Plot main PR curve
    ax_main.plot(recall, precision, linewidth=2.5, color=SCIENTIFIC_COLORS['secondary'], 
                label=f'PR Curve (AUC = {auc_score:.4f})', zorder=3)
    ax_main.axhline(y=baseline, color=SCIENTIFIC_COLORS['baseline'], linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Baseline = {baseline:.4f}', zorder=1)
    
    # Mark optimal F1 point
    ax_main.plot(optimal_recall, optimal_precision, 'o', markersize=12, 
                color=SCIENTIFIC_COLORS['success'], zorder=4,
                label=f'Optimal F1 Point (F1 = {optimal_f1:.4f})')
    ax_main.plot([optimal_recall, optimal_recall], [0, optimal_precision], 
                '--', color=SCIENTIFIC_COLORS['success'], alpha=0.5, linewidth=1)
    ax_main.plot([0, optimal_recall], [optimal_precision, optimal_precision], 
                '--', color=SCIENTIFIC_COLORS['success'], alpha=0.5, linewidth=1)
    
    # Mark current threshold point (if exists)
    if threshold_metrics:
        ax_main.plot(threshold_metrics['recall'], threshold_metrics['precision'], 
                    's', markersize=10, color=SCIENTIFIC_COLORS['threshold'], 
                    zorder=4, label=f"Current Threshold (threshold={threshold_metrics['threshold']:.2f})")
    
    ax_main.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Precision', fontsize=14, fontweight='bold')
    title = 'Precision-Recall Curve'
    if task_name:
        title += f' - {task_name}'
    ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Find optimal legend position using overlap detector
    overlap_detector = TextOverlapDetector()
    legend_items = ['PR Curve', 'Baseline', 'Optimal F1 Point']
    if threshold_metrics:
        legend_items.append('Current Threshold')
    data_bounds = (0, 1, 0, 1)  # PR curve bounds
    legend_pos = overlap_detector.find_optimal_legend_position(ax_main, legend_items, data_bounds)
    ax_main.legend(loc=legend_pos, fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    
    # Display statistics on the right
    stats_data = [
        ['Metric', 'Value'],
        ['AUC-PR', f'{auc_score:.4f}'],
        ['Baseline', f'{baseline:.4f}'],
        ['', ''],
        ['Optimal F1 Point', ''],
        ['Precision', f'{optimal_precision:.4f}'],
        ['Recall', f'{optimal_recall:.4f}'],
        ['F1 Score', f'{optimal_f1:.4f}'],
    ]
    
    if threshold_metrics:
        stats_data.extend([
            ['', ''],
            ['Current Threshold', ''],
            ['Threshold', f'{threshold_metrics["threshold"]:.2f}'],
            ['Precision', f'{threshold_metrics["precision"]:.4f}'],
            ['Recall', f'{threshold_metrics["recall"]:.4f}'],
            ['F1 Score', f'{threshold_metrics["f1"]:.4f}'],
        ])
    
    stats_data.extend([
        ['', ''],
        ['Class Distribution', ''],
        ['Positive Ratio', f'{positive_ratio:.2%}'],
        ['Class Imbalance', 'Yes' if abs(positive_ratio - 0.5) > 0.2 else 'No'],
    ])
    
    # Optimize table layout
    table_optimizer = TableLayoutOptimizer()
    table_size = table_optimizer.calculate_table_size(
        len(stats_data), len(stats_data[0]), stats_data
    )
    
    table = ax_stats.table(cellText=stats_data, cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(table_size['fontsize'])
    table.scale(1, 2.2)
    
    # Set header style
    for i in range(2):
        table[(0, i)].set_facecolor('#A23B72')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set data row styles
    for i in range(1, len(stats_data)):
        for j in range(2):
            if 'Optimal' in stats_data[i][0] or 'Current' in stats_data[i][0] or 'Class' in stats_data[i][0]:
                table[(i, j)].set_facecolor('#F0E8F4')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax_stats.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=10)
    ax_stats.axis('off')
    
    # Bottom information
    info_text = (
        f'AUC-PR = {auc_score:.4f} | '
        f'Baseline = {baseline:.4f} | '
        f'Optimal F1: Precision={optimal_precision:.3f}, Recall={optimal_recall:.3f}, F1={optimal_f1:.3f}'
    )
    if threshold_metrics:
        info_text += f' | Current Threshold: {threshold_metrics["threshold"]:.2f} (Precision={threshold_metrics["precision"]:.3f}, Recall={threshold_metrics["recall"]:.3f})'
    
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=spacing_params['pad_inches'])
    plt.close()
    
    return auc_score


def plot_score_distribution(
    individual_results: List[Dict],
    output_path: Path,
    task_name: str = ""
) -> None:
    """
    Plot scientific-style score distribution with statistical summary and test results
    
    Args:
        individual_results: List of evaluation results
        output_path: Output path
        task_name: Task name
    """
    z_scores, y_true, threshold = extract_z_scores(individual_results)
    
    if z_scores is None:
        return
    
    # Separate harmful and unharmful sample scores
    harmful_scores = z_scores[y_true == 1]
    unharmful_scores = z_scores[y_true == 0]
    
    # Calculate statistical summary
    def calc_stats(scores, name):
        if len(scores) == 0:
            return {}
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75),
            'min': np.min(scores),
            'max': np.max(scores),
            'count': len(scores)
        }
    
    harmful_stats = calc_stats(harmful_scores, 'Harmful')
    unharmful_stats = calc_stats(unharmful_scores, 'Unharmful')
    
    # Statistical tests
    test_results = {}
    if len(harmful_scores) > 0 and len(unharmful_scores) > 0:
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.ks_2samp(harmful_scores, unharmful_scores)
            test_results['KS Test'] = {'statistic': ks_stat, 'p_value': ks_p}
        except:
            pass
        
        # t-test
        try:
            t_stat, t_p = stats.ttest_ind(harmful_scores, unharmful_scores)
            test_results['t-Test'] = {'statistic': t_stat, 'p_value': t_p}
        except:
            pass
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = stats.mannwhitneyu(harmful_scores, unharmful_scores, alternative='two-sided')
            test_results['Mann-Whitney U Test'] = {'statistic': u_stat, 'p_value': u_p}
        except:
            pass
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_distribution_info(individual_results, test_results, task_name)
    
    # Calculate optimal layout
    layout_calc = LayoutCalculator()
    fig_width, fig_height = layout_calc.calculate_figure_size(content_info)
    gridspec_params = layout_calc.calculate_gridspec_ratios(content_info, 'distribution')
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Create figure with calculated size
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=gridspec_params['height_ratios'],
        width_ratios=gridspec_params['width_ratios'],
        hspace=gridspec_params['hspace'],
        wspace=gridspec_params['wspace']
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    
    # Plot main distribution
    bins = 50
    if len(harmful_scores) > 0:
        ax_main.hist(harmful_scores, bins=bins, alpha=0.5, 
                    label=f'Harmful Samples (n={harmful_stats["count"]})', 
                    color=SCIENTIFIC_COLORS['harmful'], density=True, edgecolor='black', linewidth=0.5)
        sns.kdeplot(harmful_scores, ax=ax_main, color=SCIENTIFIC_COLORS['harmful'], 
                   linewidth=2.5, label='Harmful KDE')
        # Annotate mean and median
        ax_main.axvline(harmful_stats['mean'], color=SCIENTIFIC_COLORS['harmful'], 
                       linestyle=':', linewidth=2, alpha=0.7, label=f'Harmful Mean: {harmful_stats["mean"]:.2f}')
        ax_main.axvline(harmful_stats['median'], color=SCIENTIFIC_COLORS['harmful'], 
                       linestyle='-.', linewidth=1.5, alpha=0.5)
    
    if len(unharmful_scores) > 0:
        ax_main.hist(unharmful_scores, bins=bins, alpha=0.5, 
                    label=f'Unharmful Samples (n={unharmful_stats["count"]})', 
                    color=SCIENTIFIC_COLORS['unharmful'], density=True, edgecolor='black', linewidth=0.5)
        sns.kdeplot(unharmful_scores, ax=ax_main, color=SCIENTIFIC_COLORS['unharmful'], 
                   linewidth=2.5, label='Unharmful KDE')
        # Annotate mean and median
        ax_main.axvline(unharmful_stats['mean'], color=SCIENTIFIC_COLORS['unharmful'], 
                       linestyle=':', linewidth=2, alpha=0.7, label=f'Unharmful Mean: {unharmful_stats["mean"]:.2f}')
        ax_main.axvline(unharmful_stats['median'], color=SCIENTIFIC_COLORS['unharmful'], 
                       linestyle='-.', linewidth=1.5, alpha=0.5)
    
    # Mark threshold
    if threshold is not None:
        ax_main.axvline(x=threshold, color=SCIENTIFIC_COLORS['threshold'], 
                       linestyle='--', linewidth=2.5, 
                       label=f'Threshold: {threshold:.2f}', zorder=5)
        # Calculate classification performance at threshold
        pred_binary = (z_scores >= threshold).astype(int)
        cm = sk_confusion_matrix(y_true, pred_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            ax_main.text(threshold, ax_main.get_ylim()[1] * 0.9, 
                        f'TP={tp}, FP={fp}\nTN={tn}, FN={fn}', 
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax_main.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Density', fontsize=14, fontweight='bold')
    title = 'Score Distribution'
    if task_name:
        title += f' - {task_name}'
    ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Find optimal legend position using overlap detector
    overlap_detector = TextOverlapDetector()
    legend_items = []
    if len(harmful_scores) > 0:
        legend_items.extend(['Harmful Samples', 'Harmful KDE', 'Harmful Mean'])
    if len(unharmful_scores) > 0:
        legend_items.extend(['Unharmful Samples', 'Unharmful KDE', 'Unharmful Mean'])
    if threshold is not None:
        legend_items.append('Threshold')
    
    # Estimate data bounds for legend positioning
    all_scores = np.concatenate([harmful_scores, unharmful_scores]) if len(harmful_scores) > 0 and len(unharmful_scores) > 0 else (harmful_scores if len(harmful_scores) > 0 else unharmful_scores)
    if len(all_scores) > 0:
        data_bounds = (np.min(all_scores), np.max(all_scores), 0, np.max([np.histogram(harmful_scores, bins=50, density=True)[0].max() if len(harmful_scores) > 0 else 0,
                                                                          np.histogram(unharmful_scores, bins=50, density=True)[0].max() if len(unharmful_scores) > 0 else 0]) * 1.1)
    else:
        data_bounds = None
    
    legend_pos = overlap_detector.find_optimal_legend_position(ax_main, legend_items, data_bounds)
    ax_main.legend(loc=legend_pos, fontsize=10, framealpha=0.9, ncol=1)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    # Display statistics on the right
    stats_data = [['Statistic', 'Harmful', 'Unharmful']]
    
    if harmful_stats and unharmful_stats:
        stats_data.extend([
            ['Count', f'{harmful_stats["count"]}', f'{unharmful_stats["count"]}'],
            ['Mean', f'{harmful_stats["mean"]:.3f}', f'{unharmful_stats["mean"]:.3f}'],
            ['Median', f'{harmful_stats["median"]:.3f}', f'{unharmful_stats["median"]:.3f}'],
            ['Std Dev', f'{harmful_stats["std"]:.3f}', f'{unharmful_stats["std"]:.3f}'],
            ['Min', f'{harmful_stats["min"]:.3f}', f'{unharmful_stats["min"]:.3f}'],
            ['Max', f'{harmful_stats["max"]:.3f}', f'{unharmful_stats["max"]:.3f}'],
            ['25th Percentile', f'{harmful_stats["q25"]:.3f}', f'{unharmful_stats["q25"]:.3f}'],
            ['75th Percentile', f'{harmful_stats["q75"]:.3f}', f'{unharmful_stats["q75"]:.3f}'],
        ])
    
    if test_results:
        stats_data.append(['', '', ''])
        stats_data.append(['Statistical Test', 'Statistic', 'p-value'])
        for test_name, test_data in test_results.items():
            stats_data.append([test_name, f'{test_data["statistic"]:.4f}', f'{test_data["p_value"]:.4e}'])
    
    # Optimize table layout
    table_optimizer = TableLayoutOptimizer()
    table_size = table_optimizer.calculate_table_size(
        len(stats_data), len(stats_data[0]), stats_data
    )
    
    table = ax_stats.table(cellText=stats_data, cellLoc='center', loc='center',
                          colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(table_size['fontsize'])
    table.scale(1, 1.8)
    
    # Set header style
    for i in range(3):
        table[(0, i)].set_facecolor('#F18F01')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set data row styles
    for i in range(1, len(stats_data)):
        for j in range(3):
            if 'Statistical Test' in stats_data[i][0] or any(test in stats_data[i][0] for test in ['KS', 't', 'Mann']):
                table[(i, j)].set_facecolor('#FFF4E6')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax_stats.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=10)
    ax_stats.axis('off')
    
    # Bottom information
    info_parts = []
    if harmful_stats and unharmful_stats:
        info_parts.append(f'Harmful: μ={harmful_stats["mean"]:.2f}, σ={harmful_stats["std"]:.2f}')
        info_parts.append(f'Unharmful: μ={unharmful_stats["mean"]:.2f}, σ={unharmful_stats["std"]:.2f}')
    if test_results:
        for test_name, test_data in test_results.items():
            significance = '***' if test_data['p_value'] < 0.001 else '**' if test_data['p_value'] < 0.01 else '*' if test_data['p_value'] < 0.05 else 'ns'
            info_parts.append(f'{test_name}: p={test_data["p_value"]:.4e} {significance}')
    
    info_text = ' | '.join(info_parts)
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=spacing_params['pad_inches'])
    plt.close()


def plot_category_comparison(
    metrics: Dict,
    output_path: Path,
    task_name: str = ""
) -> None:
    """
    Plot scientific-style category metrics comparison with multi-metric support
    
    Args:
        metrics: Metrics dictionary containing category-specific metrics
        output_path: Output path
        task_name: Task name
    """
    # Extract category-related metrics, grouped by metric type
    category_data = {}
    metric_types = ['micro_acc', 'f1', 'precision', 'recall', 'specificity', 'sensitivity']
    
    # Find all category metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Extract category name and metric type
            if 'category_' in key or 'harm_cat_' in key:
                # Handle different naming formats
                if '---' in key:
                    parts = key.split('---')
                    if len(parts) == 2:
                        metric_type = parts[0].replace('category_', '').replace('harm_cat_', '').replace('_f1', '')
                        category = parts[1]
                    else:
                        continue
                elif '_f1' in key:
                    metric_type = 'f1'
                    category = key.replace('category_', '').replace('harm_cat_', '').replace('_f1', '')
                else:
                    metric_type = 'micro_acc'  # Default
                    category = key.replace('category_', '').replace('harm_cat_', '')
                
                if category not in category_data:
                    category_data[category] = {}
                if metric_type in metric_types or 'f1' in key:
                    category_data[category][metric_type if metric_type in metric_types else 'f1'] = value
    
    if len(category_data) == 0:
        return
    
    # Organize data for grouped bar chart
    categories = sorted(category_data.keys())
    available_metrics = []
    for cat_data in category_data.values():
        available_metrics.extend(cat_data.keys())
    available_metrics = sorted(list(set(available_metrics)))
    
    # If too many metrics, select only main ones
    if len(available_metrics) > 4:
        priority_metrics = ['micro_acc', 'f1', 'precision', 'recall']
        available_metrics = [m for m in priority_metrics if m in available_metrics][:4]
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_category_info(metrics, task_name)
    content_info['categories'] = len(categories)
    content_info['metric_count'] = len(available_metrics)
    
    # Calculate optimal layout
    layout_calc = LayoutCalculator()
    fig_width, fig_height = layout_calc.calculate_figure_size(content_info)
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Adjust height based on number of categories
    fig_height = max(8, len(categories) * 0.6)
    fig_width = max(12, fig_width)
    
    # Prepare data
    x = np.arange(len(categories))
    width = 0.8 / len(available_metrics) if len(available_metrics) > 0 else 0.2
    
    # Create figure with calculated size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot grouped bar chart
    colors = [SCIENTIFIC_COLORS['primary'], SCIENTIFIC_COLORS['secondary'], 
              SCIENTIFIC_COLORS['success'], SCIENTIFIC_COLORS['danger']]
    for i, metric in enumerate(available_metrics):
        values = [category_data[cat].get(metric, 0) for cat in categories]
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.barh(x + offset, values, width, label=metric, 
                      color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.01:  # Only show larger values
                ax.text(val, j + offset, f' {val:.3f}', va='center', fontsize=8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Category', fontsize=14, fontweight='bold')
    title = 'Category Metrics Comparison'
    if task_name:
        title += f' - {task_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, ncol=min(len(available_metrics), 4))
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xlim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=spacing_params['pad_inches'])
    plt.close()


def plot_per_category_metrics(
    metrics: Dict,
    output_path: Path,
    task_name: str = ""
) -> None:
    """
    Plot scientific-style per-category detailed metrics heatmap
    
    Args:
        metrics: Metrics dictionary
        output_path: Output path
        task_name: Task name
    """
    # Extract categories and metrics
    category_data = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Find category-related metrics
            if 'category_' in key or 'harm_cat_' in key:
                # Parse category and metric type
                if '---' in key:
                    parts = key.split('---')
                    if len(parts) == 2:
                        metric_type = parts[0].replace('category_', '').replace('harm_cat_', '').replace('_f1', '')
                        category = parts[1]
                        
                        if category not in category_data:
                            category_data[category] = {}
                        # Handle f1 metrics
                        if '_f1' in parts[0]:
                            category_data[category]['f1'] = value
                        else:
                            category_data[category][metric_type] = value
                elif '_f1' in key:
                    metric_type = 'f1'
                    category = key.replace('category_', '').replace('harm_cat_', '').replace('_f1', '')
                    if category not in category_data:
                        category_data[category] = {}
                    category_data[category][metric_type] = value
                else:
                    # No separator, might be direct category name
                    category = key.replace('category_', '').replace('harm_cat_', '')
                    if category not in category_data:
                        category_data[category] = {}
                    category_data[category]['micro_acc'] = value
    
    if len(category_data) == 0:
        return
    
    # Build matrix
    categories = sorted(category_data.keys())
    metric_types = sorted(set([m for cat_data in category_data.values() for m in cat_data.keys()]))
    
    # If too many metrics, select main ones
    priority_metrics = ['micro_acc', 'f1', 'precision', 'recall', 'specificity', 'sensitivity']
    metric_types = [m for m in priority_metrics if m in metric_types] or metric_types[:6]
    
    matrix = []
    for category in categories:
        row = [category_data[category].get(mt, np.nan) for mt in metric_types]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Collect content information for layout calculation
    content_info = ContentInfoCollector.collect_category_info(metrics, task_name)
    content_info['categories'] = len(categories)
    content_info['metric_count'] = len(metric_types)
    
    # Calculate optimal layout
    layout_calc = LayoutCalculator()
    base_width, base_height = layout_calc.calculate_figure_size(content_info)
    spacing_params = layout_calc.calculate_spacing(content_info)
    
    # Adjust size based on content
    fig_width = max(14, len(metric_types) * 1.8, base_width)
    fig_height = max(10, len(categories) * 0.6, base_height)
    
    # Create figure with calculated size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot heatmap - use scientific-grade colormap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',  # Reversed RdYlBu, red indicates high values
        xticklabels=metric_types,
        yticklabels=categories,
        ax=ax,
        cbar_kws={'label': 'Metric Value', 'shrink': 0.8},
        linewidths=1,
        linecolor='white',
        square=False,
        vmin=0,
        vmax=1,
        center=0.5,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    ax.set_xlabel('Metric Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Category', fontsize=14, fontweight='bold')
    title = 'Per-Category Metrics Heatmap'
    if task_name:
        title += f' - {task_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=spacing_params.get('pad_inches', 0.2))
    plt.close()


def generate_all_plots(
    individual_results: List[Dict],
    metrics: Dict,
    plots_dir: Path,
    task_name: str = ""
) -> Dict[str, Optional[float]]:
    """
    Generate all plots for a given task
    
    Args:
        individual_results: List of evaluation results
        metrics: Metrics dictionary
        plots_dir: Directory to save plots
        task_name: Task name
        
    Returns:
        Dictionary containing generated plot information (e.g., AUC scores)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_info = {}
    
    # 1. Confusion Matrix
    try:
        plot_confusion_matrix(individual_results, plots_dir / "confusion_matrix.png", task_name)
        plot_info["confusion_matrix"] = "generated"
    except Exception as e:
        print(f"Failed to generate confusion matrix: {e}")
        plot_info["confusion_matrix"] = f"error: {str(e)}"
    
    # 2. ROC Curve
    try:
        auc_roc = plot_roc_curve(individual_results, plots_dir / "roc_curve.png", task_name)
        if auc_roc is not None:
            plot_info["roc_curve"] = {"auc": auc_roc}
        else:
            plot_info["roc_curve"] = "skipped (no z_scores)"
    except Exception as e:
        print(f"Failed to generate ROC curve: {e}")
        plot_info["roc_curve"] = f"error: {str(e)}"
    
    # 3. PR Curve
    try:
        auc_pr = plot_pr_curve(individual_results, plots_dir / "pr_curve.png", task_name)
        if auc_pr is not None:
            plot_info["pr_curve"] = {"auc": auc_pr}
        else:
            plot_info["pr_curve"] = "skipped (no z_scores)"
    except Exception as e:
        print(f"Failed to generate PR curve: {e}")
        plot_info["pr_curve"] = f"error: {str(e)}"
    
    # 4. Score Distribution
    try:
        plot_score_distribution(individual_results, plots_dir / "score_distribution.png", task_name)
        plot_info["score_distribution"] = "generated"
    except Exception as e:
        print(f"Failed to generate score distribution: {e}")
        plot_info["score_distribution"] = f"error: {str(e)}"
    
    # 5. Category Comparison
    try:
        plot_category_comparison(metrics, plots_dir / "category_comparison.png", task_name)
        plot_info["category_comparison"] = "generated"
    except Exception as e:
        print(f"Failed to generate category comparison: {e}")
        plot_info["category_comparison"] = f"error: {str(e)}"
    
    # 6. Per-Category Metrics
    try:
        plot_per_category_metrics(metrics, plots_dir / "per_category_metrics.png", task_name)
        plot_info["per_category_metrics"] = "generated"
    except Exception as e:
        print(f"Failed to generate per-category metrics: {e}")
        plot_info["per_category_metrics"] = f"error: {str(e)}"
    
    return plot_info


