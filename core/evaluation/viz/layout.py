"""
Automatic layout system for scientific plots
Provides adaptive layout calculation, text overlap detection, and table optimization
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle


class LayoutCalculator:
    """
    Calculate optimal layout parameters based on content information
    """
    
    # Base configuration - 增大默认尺寸
    BASE_WIDTH = 14.0  # 增大基础宽度
    BASE_HEIGHT = 10.0  # 增大基础高度
    MIN_WIDTH = 12.0
    MIN_HEIGHT = 8.0
    MAX_WIDTH = 24.0
    MAX_HEIGHT = 20.0
    
    # Spacing configuration - 增大间距
    BASE_HSPACE = 0.4
    BASE_WSPACE = 0.4
    BASE_PAD_INCHES = 0.3
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize layout calculator with optional configuration
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        if config:
            for key, value in config.items():
                if hasattr(self, key.upper()):
                    setattr(self, key.upper(), value)
    
    def create_standard_figure(
        self, 
        content_info: Dict, 
        plot_type: str = 'default',
        include_stats_table: bool = True,
        include_info_panel: bool = True,
        title: str = ""
    ) -> Tuple:
        """
        创建标准化的图表布局，解决常见的布局问题
        
        Args:
            content_info: 内容信息字典
            plot_type: 图表类型
            include_stats_table: 是否包含统计表格
            include_info_panel: 是否包含底部信息面板
            title: 图表标题
        
        Returns:
            (fig, ax_main, ax_stats, ax_info) 或 (fig, ax_main, ax_info) 取决于 include_stats_table
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        # 计算图表尺寸
        fig_width, fig_height = self.calculate_figure_size(content_info)
        
        # 根据表格行数调整高度
        table_rows = content_info.get('table_rows', 0)
        if table_rows > 12:
            fig_height = max(fig_height, 8 + table_rows * 0.3)
        
        # 创建图形
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        if include_stats_table and include_info_panel:
            # 使用更精细的网格：12 行 x 12 列
            gs = gridspec.GridSpec(
                12, 12,
                figure=fig,
                hspace=0.5,  # 增大行间距
                wspace=0.5,  # 增大列间距
                top=0.90,    # 为标题留出空间
                bottom=0.08, # 为底部信息留出空间
                left=0.08,
                right=0.95
            )
            
            # 主图区域：左侧 7 列，上方 10 行
            ax_main = fig.add_subplot(gs[0:10, 0:7])
            
            # 统计表格区域：右侧 5 列，上方 10 行
            ax_stats = fig.add_subplot(gs[0:10, 7:12])
            ax_stats.axis('off')
            
            # 底部信息面板：底部 2 行，全宽
            ax_info = fig.add_subplot(gs[10:12, :])
            ax_info.axis('off')
            
            # 如果有标题，使用 suptitle（不会遮挡内容）
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
            
            return fig, ax_main, ax_stats, ax_info
            
        elif include_stats_table:
            gs = gridspec.GridSpec(
                1, 12,
                figure=fig,
                wspace=0.5,
                top=0.90,
                bottom=0.08,
                left=0.08,
                right=0.95
            )
            ax_main = fig.add_subplot(gs[0, 0:7])
            ax_stats = fig.add_subplot(gs[0, 7:12])
            ax_stats.axis('off')
            
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
            
            return fig, ax_main, ax_stats
            
        elif include_info_panel:
            gs = gridspec.GridSpec(
                12, 1,
                figure=fig,
                hspace=0.5,
                top=0.90,
                bottom=0.08,
                left=0.08,
                right=0.95
            )
            ax_main = fig.add_subplot(gs[0:10, 0])
            ax_info = fig.add_subplot(gs[10:12, 0])
            ax_info.axis('off')
            
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
            
            return fig, ax_main, ax_info
            
        else:
            fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
            return fig, ax_main
    
    def calculate_figure_size(self, content_info: Dict) -> Tuple[float, float]:
        """
        Calculate optimal figure size based on content
        
        Args:
            content_info: Dictionary containing content information:
                - table_rows: Number of table rows
                - table_cols: Number of table columns
                - categories: Number of categories (for bar charts)
                - legend_items: Number of legend items
                - data_points: Number of data points
                - task_name_length: Length of task name
                - has_threshold: Whether threshold information exists
                - stat_tests: Number of statistical tests
                
        Returns:
            Tuple of (width, height) in inches
        """
        width = self.BASE_WIDTH
        height = self.BASE_HEIGHT
        
        # Adjust for table size - 表格行数影响更显著
        table_rows = content_info.get('table_rows', 0)
        table_cols = content_info.get('table_cols', 0)
        if table_rows > 10:
            height += (table_rows - 10) * 0.25  # 增大行高因子
        if table_rows > 15:
            width += (table_rows - 15) * 0.2
        if table_cols > 3:
            width += (table_cols - 3) * 0.6
        
        # Adjust for categories (bar charts)
        categories = content_info.get('categories', 0)
        if categories > 10:
            height += (categories - 10) * 0.35
        elif categories > 0:
            height += max(0, (categories - 5) * 0.2)
        
        # Adjust for legend items
        legend_items = content_info.get('legend_items', 0)
        if legend_items > 5:
            width += 1.5  # Reserve space for legend
        elif legend_items > 3:
            width += 0.8
        
        # Adjust for task name length
        task_name_length = content_info.get('task_name_length', 0)
        if task_name_length > 30:
            height += 0.4
        
        # Adjust for statistical tests
        stat_tests = content_info.get('stat_tests', 0)
        if stat_tests > 0:
            height += stat_tests * 0.3  # 统计测试增加高度而非宽度
        
        # Clamp to min/max
        width = max(self.MIN_WIDTH, min(self.MAX_WIDTH, width))
        height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, height))
        
        return (width, height)

    
    def calculate_gridspec_ratios(self, content_info: Dict, plot_type: str) -> Dict:
        """
        Calculate optimal GridSpec ratios for subplot layout
        
        Args:
            content_info: Content information dictionary
            plot_type: Type of plot ('confusion_matrix', 'roc', 'pr', 'distribution', etc.)
            
        Returns:
            Dictionary with 'height_ratios', 'width_ratios', 'hspace', 'wspace'
        """
        table_rows = content_info.get('table_rows', 0)
        has_info_panel = content_info.get('has_info_panel', True)
        
        if plot_type == 'confusion_matrix':
            # Main plot, stats table, info panel
            main_ratio = 0.7
            stats_ratio = 0.25
            info_ratio = 0.05
            
            if table_rows > 10:
                stats_ratio = min(0.35, 0.2 + table_rows * 0.015)
                main_ratio = 1.0 - stats_ratio - info_ratio
            
            height_ratios = [main_ratio, info_ratio]
            width_ratios = [main_ratio, stats_ratio]
            
        elif plot_type in ['roc', 'pr']:
            # Main plot, stats table, info panel
            main_ratio = 0.75
            stats_ratio = 0.25
            info_ratio = 0.05
            
            if table_rows > 8:
                stats_ratio = min(0.35, 0.2 + table_rows * 0.015)
                main_ratio = 1.0 - stats_ratio - info_ratio
            
            height_ratios = [main_ratio, info_ratio]
            width_ratios = [main_ratio, stats_ratio]
            
        elif plot_type == 'distribution':
            # Main plot, stats table, info panel
            main_ratio = 0.75
            stats_ratio = 0.25
            info_ratio = 0.05
            
            stat_tests = content_info.get('stat_tests', 0)
            if stat_tests > 0:
                stats_ratio = min(0.35, 0.2 + stat_tests * 0.03)
                main_ratio = 1.0 - stats_ratio - info_ratio
            
            height_ratios = [main_ratio, info_ratio]
            width_ratios = [main_ratio, stats_ratio]
            
        else:
            # Default layout
            height_ratios = [0.9, 0.1] if has_info_panel else [1.0]
            width_ratios = [0.75, 0.25]
        
        # Calculate spacing
        hspace = self.BASE_HSPACE
        wspace = self.BASE_WSPACE
        
        # Adjust spacing based on content density
        if table_rows > 15:
            wspace += 0.1
        
        return {
            'height_ratios': height_ratios,
            'width_ratios': width_ratios,
            'hspace': hspace,
            'wspace': wspace
        }
    
    def calculate_spacing(self, content_info: Dict) -> Dict:
        """
        Calculate spacing parameters
        
        Args:
            content_info: Content information dictionary
            
        Returns:
            Dictionary with spacing parameters
        """
        table_rows = content_info.get('table_rows', 0)
        categories = content_info.get('categories', 0)
        
        hspace = self.BASE_HSPACE
        wspace = self.BASE_WSPACE
        pad_inches = self.BASE_PAD_INCHES
        
        # Adjust spacing based on content
        if table_rows > 15:
            wspace += 0.1
        if categories > 15:
            hspace += 0.1
        
        return {
            'hspace': hspace,
            'wspace': wspace,
            'pad_inches': pad_inches
        }
    
    def estimate_text_space(self, text_list: List[str], fontsize: int = 12) -> Dict:
        """
        Estimate space required for text elements
        
        Args:
            text_list: List of text strings
            fontsize: Font size in points
            
        Returns:
            Dictionary with 'width' and 'height' estimates in inches
        """
        if not text_list:
            return {'width': 0, 'height': 0}
        
        # Rough estimation: 1 inch per 12 characters at 12pt font
        max_length = max(len(text) for text in text_list)
        width = (max_length / 12.0) * (fontsize / 12.0)
        height = len(text_list) * (fontsize / 72.0) * 1.5  # 1.5 line spacing
        
        return {'width': width, 'height': height}
    
    def estimate_table_space(self, table_rows: int, table_cols: int, 
                            fontsize: int = 10) -> Dict:
        """
        Estimate space required for table
        
        Args:
            table_rows: Number of rows
            table_cols: Number of columns
            fontsize: Font size in points
            
        Returns:
            Dictionary with 'width' and 'height' estimates in inches
        """
        # Base cell dimensions
        cell_height = (fontsize / 72.0) * 2.0  # 2x font size for padding
        cell_width = 1.5  # Base width per column
        
        width = table_cols * cell_width
        height = table_rows * cell_height
        
        return {'width': width, 'height': height}


class TextOverlapDetector:
    """
    Detect and resolve text element overlaps
    """
    
    def __init__(self, overlap_threshold: float = 0.1):
        """
        Initialize overlap detector
        
        Args:
            overlap_threshold: Minimum overlap ratio to consider as overlap (0-1)
        """
        self.overlap_threshold = overlap_threshold
    
    def detect_overlaps(self, text_elements: List[Dict]) -> List[Tuple[int, int]]:
        """
        Detect overlapping text elements
        
        Args:
            text_elements: List of dictionaries, each containing:
                - 'x', 'y': Position
                - 'width', 'height': Bounding box dimensions
                - 'text': Text content
                
        Returns:
            List of tuples (i, j) indicating overlapping pairs
        """
        overlaps = []
        
        for i in range(len(text_elements)):
            for j in range(i + 1, len(text_elements)):
                elem_i = text_elements[i]
                elem_j = text_elements[j]
                
                if self._boxes_overlap(elem_i, elem_j):
                    overlaps.append((i, j))
        
        return overlaps
    
    def _boxes_overlap(self, box1: Dict, box2: Dict) -> bool:
        """
        Check if two bounding boxes overlap
        
        Args:
            box1, box2: Dictionaries with 'x', 'y', 'width', 'height'
            
        Returns:
            True if boxes overlap
        """
        x1_min = box1['x'] - box1['width'] / 2
        x1_max = box1['x'] + box1['width'] / 2
        y1_min = box1['y'] - box1['height'] / 2
        y1_max = box1['y'] + box1['height'] / 2
        
        x2_min = box2['x'] - box2['width'] / 2
        x2_max = box2['x'] + box2['width'] / 2
        y2_min = box2['y'] - box2['height'] / 2
        y2_max = box2['y'] + box2['height'] / 2
        
        # Check for overlap
        overlap_x = not (x1_max < x2_min or x2_max < x1_min)
        overlap_y = not (y1_max < y2_min or y2_max < y1_min)
        
        if overlap_x and overlap_y:
            # Calculate overlap area
            overlap_width = min(x1_max, x2_max) - max(x1_min, x2_min)
            overlap_height = min(y1_max, y2_max) - max(y1_min, y2_min)
            overlap_area = overlap_width * overlap_height
            
            box1_area = box1['width'] * box1['height']
            box2_area = box2['width'] * box2['height']
            
            # Check if overlap exceeds threshold
            overlap_ratio = overlap_area / min(box1_area, box2_area)
            return overlap_ratio >= self.overlap_threshold
        
        return False
    
    def resolve_overlaps(self, text_elements: List[Dict], 
                        ax_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """
        Resolve text overlaps by adjusting positions
        
        Args:
            text_elements: List of text element dictionaries
            ax_bounds: Tuple of (x_min, x_max, y_min, y_max) for axis bounds
            
        Returns:
            List of adjusted text elements
        """
        x_min, x_max, y_min, y_max = ax_bounds
        adjusted = [elem.copy() for elem in text_elements]
        
        # Simple force-directed approach
        max_iterations = 10
        for iteration in range(max_iterations):
            overlaps = self.detect_overlaps(adjusted)
            if not overlaps:
                break
            
            # Move overlapping elements apart
            for i, j in overlaps:
                elem_i = adjusted[i]
                elem_j = adjusted[j]
                
                # Calculate direction vector
                dx = elem_i['x'] - elem_j['x']
                dy = elem_i['y'] - elem_j['y']
                dist = np.sqrt(dx**2 + dy**2) if (dx != 0 or dy != 0) else 1.0
                
                # Normalize and scale
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    
                    # Move elements apart
                    move_distance = 0.1
                    elem_i['x'] = np.clip(elem_i['x'] + dx * move_distance, x_min, x_max)
                    elem_i['y'] = np.clip(elem_i['y'] + dy * move_distance, y_min, y_max)
                    elem_j['x'] = np.clip(elem_j['x'] - dx * move_distance, x_min, x_max)
                    elem_j['y'] = np.clip(elem_j['y'] - dy * move_distance, y_min, y_max)
        
        return adjusted
    
    def find_optimal_legend_position(self, ax, legend_items: List, 
                                    data_bounds: Optional[Tuple] = None) -> str:
        """
        Find optimal legend position to avoid overlapping with data
        
        Args:
            ax: Matplotlib axis object
            legend_items: List of legend items
            data_bounds: Optional tuple of (x_min, x_max, y_min, y_max) for data region
            
        Returns:
            Legend position string ('best', 'upper right', 'lower left', etc.)
        """
        if data_bounds is None:
            # Use axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            data_bounds = (xlim[0], xlim[1], ylim[0], ylim[1])
        
        x_min, x_max, y_min, y_max = data_bounds
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Check data density in different regions
        # For now, use a simple heuristic based on data distribution
        # In practice, this would analyze actual data points
        
        # Default to 'best' which matplotlib handles automatically
        # But we can prefer certain positions based on data distribution
        if len(legend_items) <= 3:
            return 'best'
        elif len(legend_items) <= 5:
            return 'upper right'
        else:
            return 'center left'  # Use external position for many items


class TableLayoutOptimizer:
    """
    Optimize table layout based on content and available space
    """
    
    MIN_FONTSIZE = 7
    MAX_FONTSIZE = 12
    BASE_FONTSIZE = 10
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize table optimizer
        
        Args:
            config: Optional configuration dictionary
        """
        if config:
            for key, value in config.items():
                if hasattr(self, key.upper()):
                    setattr(self, key.upper(), value)
    
    def calculate_table_size(self, rows: int, cols: int, 
                            cell_content: Optional[List] = None) -> Dict:
        """
        Calculate optimal table size
        
        Args:
            rows: Number of rows
            cols: Number of columns
            cell_content: Optional list of cell contents for width estimation
            
        Returns:
            Dictionary with 'width', 'height', 'fontsize'
        """
        # Calculate font size based on row count
        fontsize = self.optimize_font_size(rows, cols)
        
        # Estimate cell dimensions
        cell_height = (fontsize / 72.0) * 2.2  # 2.2x for padding
        cell_width = 1.2  # Base width
        
        # Adjust width based on content if provided
        if cell_content:
            max_col_widths = []
            for col_idx in range(cols):
                col_texts = [str(row[col_idx]) for row in cell_content if col_idx < len(row)]
                if col_texts:
                    max_length = max(len(text) for text in col_texts)
                    max_col_widths.append(max_length)
            
            if max_col_widths:
                avg_width = np.mean(max_col_widths)
                cell_width = max(1.0, avg_width / 12.0 * (fontsize / 10.0))
        
        width = cols * cell_width
        height = rows * cell_height
        
        return {
            'width': width,
            'height': height,
            'fontsize': fontsize
        }
    
    def optimize_font_size(self, rows: int, cols: int, 
                          available_space: Optional[Dict] = None) -> int:
        """
        Optimize font size based on table dimensions and available space
        
        Args:
            rows: Number of rows
            cols: Number of columns
            available_space: Optional dictionary with 'width' and 'height' in inches
            
        Returns:
            Optimal font size in points
        """
        fontsize = self.BASE_FONTSIZE
        
        # Adjust based on row count
        if rows > 20:
            fontsize = max(self.MIN_FONTSIZE, self.BASE_FONTSIZE - (rows - 20) * 0.15)
        elif rows > 15:
            fontsize = max(self.MIN_FONTSIZE + 1, self.BASE_FONTSIZE - (rows - 15) * 0.1)
        
        # Adjust based on column count
        if cols > 4:
            fontsize = max(self.MIN_FONTSIZE, fontsize - (cols - 4) * 0.2)
        
        # Adjust based on available space if provided
        if available_space:
            available_height = available_space.get('height', 10)
            required_height = rows * (fontsize / 72.0) * 2.2
            if required_height > available_height * 0.9:
                fontsize = int((available_height * 0.9 / rows / 2.2) * 72)
                fontsize = max(self.MIN_FONTSIZE, min(self.MAX_FONTSIZE, fontsize))
        
        return int(max(self.MIN_FONTSIZE, min(self.MAX_FONTSIZE, fontsize)))
    
    def wrap_long_text(self, text: str, max_width: float, 
                      fontsize: int = 10) -> str:
        """
        Wrap long text to fit within maximum width
        
        Args:
            text: Text to wrap
            max_width: Maximum width in inches
            fontsize: Font size in points
            
        Returns:
            Wrapped text (may include newlines)
        """
        # Estimate characters per inch
        chars_per_inch = 12.0 * (10.0 / fontsize)
        max_chars = int(max_width * chars_per_inch)
        
        if len(text) <= max_chars:
            return text
        
        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length <= max_chars:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)


class ContentInfoCollector:
    """
    Collect content information from plot data for layout calculation
    """
    
    @staticmethod
    def collect_confusion_matrix_info(individual_results: List[Dict], 
                                     task_name: str = "") -> Dict:
        """
        Collect content information for confusion matrix
        
        Args:
            individual_results: List of evaluation results
            task_name: Task name
            
        Returns:
            Content information dictionary
        """
        # Calculate table rows (fixed for confusion matrix: header + 4 metrics + 6 performance metrics)
        table_rows = 1 + 4 + 1 + 6  # Header + TP/FP/TN/FN + separator + 6 metrics
        
        return {
            'table_rows': table_rows,
            'table_cols': 3,
            'categories': 0,
            'legend_items': 0,
            'data_points': len(individual_results),
            'task_name_length': len(task_name),
            'has_info_panel': True,
            'stat_tests': 0
        }
    
    @staticmethod
    def collect_roc_info(individual_results: List[Dict], 
                        threshold_metrics: Optional[Dict] = None,
                        task_name: str = "") -> Dict:
        """
        Collect content information for ROC curve
        
        Args:
            individual_results: List of evaluation results
            threshold_metrics: Optional threshold metrics dictionary
            task_name: Task name
            
        Returns:
            Content information dictionary
        """
        z_scores, y_true, _ = extract_z_scores(individual_results)
        if z_scores is None:
            z_scores = []
            y_true = []
        
        # Count legend items
        legend_items = 2  # ROC curve + random classifier
        if threshold_metrics:
            legend_items += 1  # Current threshold
        
        # Calculate table rows
        table_rows = 1 + 2 + 1  # Header + AUC + separator
        table_rows += 1 + 2  # Optimal threshold section
        if threshold_metrics:
            table_rows += 1 + 4  # Current threshold section
        table_rows += 1 + 3  # Sample information section
        
        return {
            'table_rows': table_rows,
            'table_cols': 2,
            'categories': 0,
            'legend_items': legend_items,
            'data_points': len(z_scores),
            'task_name_length': len(task_name),
            'has_info_panel': True,
            'has_threshold': threshold_metrics is not None,
            'stat_tests': 0
        }
    
    @staticmethod
    def collect_pr_info(individual_results: List[Dict],
                       threshold_metrics: Optional[Dict] = None,
                       task_name: str = "") -> Dict:
        """
        Collect content information for PR curve
        
        Args:
            individual_results: List of evaluation results
            threshold_metrics: Optional threshold metrics dictionary
            task_name: Task name
            
        Returns:
            Content information dictionary
        """
        z_scores, y_true, _ = extract_z_scores(individual_results)
        if z_scores is None:
            z_scores = []
            y_true = []
        
        # Count legend items
        legend_items = 2  # PR curve + baseline
        if threshold_metrics:
            legend_items += 1  # Current threshold
        
        # Calculate table rows
        table_rows = 1 + 2 + 1  # Header + AUC-PR + baseline + separator
        table_rows += 1 + 3  # Optimal F1 section
        if threshold_metrics:
            table_rows += 1 + 4  # Current threshold section
        table_rows += 1 + 2  # Class distribution section
        
        return {
            'table_rows': table_rows,
            'table_cols': 2,
            'categories': 0,
            'legend_items': legend_items,
            'data_points': len(z_scores),
            'task_name_length': len(task_name),
            'has_info_panel': True,
            'has_threshold': threshold_metrics is not None,
            'stat_tests': 0
        }
    
    @staticmethod
    def collect_distribution_info(individual_results: List[Dict],
                                test_results: Dict,
                                task_name: str = "") -> Dict:
        """
        Collect content information for score distribution plot
        
        Args:
            individual_results: List of evaluation results
            test_results: Dictionary of statistical test results
            task_name: Task name
            
        Returns:
            Content information dictionary
        """
        z_scores, y_true, _ = extract_z_scores(individual_results)
        if z_scores is None:
            z_scores = []
            y_true = []
        
        # Count legend items (harmful/unharmful histograms, KDEs, means, threshold)
        legend_items = 0
        if len(z_scores[y_true == 1]) > 0:
            legend_items += 3  # Histogram, KDE, mean
        if len(z_scores[y_true == 0]) > 0:
            legend_items += 3  # Histogram, KDE, mean
        # Threshold is optional
        
        # Calculate table rows
        table_rows = 1 + 8  # Header + 8 statistics
        if test_results:
            table_rows += 1 + 1 + len(test_results)  # Separator + header + tests
        
        return {
            'table_rows': table_rows,
            'table_cols': 3,
            'categories': 0,
            'legend_items': legend_items,
            'data_points': len(z_scores),
            'task_name_length': len(task_name),
            'has_info_panel': True,
            'stat_tests': len(test_results)
        }
    
    @staticmethod
    def collect_category_info(metrics: Dict, task_name: str = "") -> Dict:
        """
        Collect content information for category comparison plots
        
        Args:
            metrics: Metrics dictionary
            task_name: Task name
            
        Returns:
            Content information dictionary
        """
        # Extract categories
        categories = set()
        metric_count = 0
        
        for key in metrics.keys():
            if isinstance(metrics[key], (int, float)):
                if 'category_' in key or 'harm_cat_' in key:
                    if '---' in key:
                        parts = key.split('---')
                        if len(parts) == 2:
                            categories.add(parts[1])
                    elif '_f1' in key:
                        category = key.replace('category_', '').replace('harm_cat_', '').replace('_f1', '')
                        categories.add(category)
                    else:
                        category = key.replace('category_', '').replace('harm_cat_', '')
                        categories.add(category)
        
        # Count unique metric types
        metric_types = set()
        for key in metrics.keys():
            if 'category_' in key or 'harm_cat_' in key:
                if '---' in key:
                    parts = key.split('---')
                    if len(parts) == 2:
                        metric_types.add(parts[0].replace('category_', '').replace('harm_cat_', '').replace('_f1', ''))
        
        return {
            'table_rows': 0,
            'table_cols': 0,
            'categories': len(categories),
            'legend_items': len(metric_types) if metric_types else 1,
            'data_points': 0,
            'task_name_length': len(task_name),
            'has_info_panel': False,
            'stat_tests': 0,
            'metric_count': len(metric_types) if metric_types else 1
        }


# Import extract_z_scores for ContentInfoCollector
from evaluation.viz.utils import extract_z_scores

