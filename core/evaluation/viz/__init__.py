"""
评估结果可视化模块
"""
from evaluation.viz.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_score_distribution,
    plot_category_comparison,
    plot_per_category_metrics,
    generate_all_plots,
)

from evaluation.viz.utils import (
    extract_z_scores,
    compute_roc_auc,
    compute_pr_auc,
    organize_results_by_category,
)

from evaluation.viz.layout import (
    LayoutCalculator,
    TextOverlapDetector,
    TableLayoutOptimizer,
    ContentInfoCollector,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_score_distribution",
    "plot_category_comparison",
    "plot_per_category_metrics",
    "generate_all_plots",
    "extract_z_scores",
    "compute_roc_auc",
    "compute_pr_auc",
    "organize_results_by_category",
    "LayoutCalculator",
    "TextOverlapDetector",
    "TableLayoutOptimizer",
    "ContentInfoCollector",
]


