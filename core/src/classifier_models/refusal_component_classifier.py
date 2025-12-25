"""
Refusal Component Classifier

基于 refusal component projection 方法的 prompt 分类器。
使用 LLaMA-3-8B-IT 模型的内部激活来检测有害提示词。
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, Union

import numpy as np
import torch

# 添加 LLM-Safety 路径以便导入
llm_safety_path = Path(__file__).parent.parent.parent.parent / "LLM-Safety"
sys.path.insert(0, str(llm_safety_path))

# 直接定义 ComponentKey 类型，避免导入整个模块导致依赖问题
# ComponentKey 类型定义
ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]

# 导入必要的函数（延迟导入以避免依赖问题）
def _import_projections():
    from components_safety.pipeline.data.projections import read_eoi_projection_unified
    return read_eoi_projection_unified

def _import_runner():
    from components_safety.utils.runner_util import MinimalHookRunner
    return MinimalHookRunner

def _import_model_factory():
    from refusal_direction.pipeline.model_utils.model_factory import construct_model_base
    return construct_model_base

def _import_component_probe():
    from components_safety.pipeline.refusal_components.scoring.probe_model import ComponentProbe
    return ComponentProbe

from src.classifier_models.base import (
    SafetyClassifierBase,
    SafetyClassifierOutput,
    PromptHarmfulness,
)


def component_dict_to_key(comp_dict: dict) -> ComponentKey:
    """将组件字典转换为 ComponentKey 元组"""
    module_type = comp_dict["module_type"]
    layer = comp_dict["layer"]
    eoi_k = comp_dict["eoi_k"]
    
    if "head" in comp_dict:
        return (module_type, layer, eoi_k, comp_dict["head"])
    else:
        return (module_type, layer, eoi_k)


def compute_topm_scores(
    standardized_proj: Dict[ComponentKey, np.ndarray],
    component_subset: Set[ComponentKey],
    m_cap: int = 10,
) -> np.ndarray:
    """
    对每个样本计算 top-m 聚合得分 z(x)。
    
    对于每个样本 x，从 component_subset 中选出标准化得分最高的 m 个组件，
    计算它们的平均值作为 z(x)。
    
    Args:
        standardized_proj: 标准化后的投影数据 {component_key: np.ndarray}
        component_subset: 要使用的组件集合 S_p
        m_cap: top-m 中的 m（默认 10）
    
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
    
    # 收集所有组件的得分（每个组件是一个向量 [n_samples]）
    component_scores = []
    for key in available_keys:
        scores = standardized_proj[key].flatten()
        # 确保长度一致
        if len(scores) == n_samples:
            component_scores.append(scores)
        elif len(scores) > n_samples:
            # 如果更长，截断（不应该发生，但容错）
            component_scores.append(scores[:n_samples])
    
    if len(component_scores) == 0:
        return np.zeros(n_samples)
    
    # 堆叠成矩阵 [n_components, n_samples]
    score_matrix = np.stack(component_scores, axis=0)
    
    # 对每个样本，取 top-m 组件的平均值
    m_actual = min(m_cap, score_matrix.shape[0])
    
    # 对每列（样本）排序，取最大的 m_actual 个
    topm_scores = np.partition(score_matrix, -m_actual, axis=0)[-m_actual:]
    z_scores = np.mean(topm_scores, axis=0)
    
    return z_scores


class RefusalComponentClassifier(SafetyClassifierBase):
    """
    基于 refusal component projection 的 prompt 分类器。
    
    使用 LLaMA-3-8B-IT 模型的内部激活来检测有害提示词。
    通过投影到拒答方向、标准化和 top-m 聚合来计算得分。
    """
    
    def __init__(
        self,
        config_path: str = None,
        local_model_path: str = None,
        batch_size: int = 4,
        **kwargs
    ):
        """
        Args:
            config_path: reconstruct_config.json 的路径
            local_model_path: 作为 config_path 的别名（用于兼容 eval.py 的 override_model_path 参数）
            batch_size: 批处理大小
        """
        super().__init__(batch_size, **kwargs)
        
        # local_model_path 作为 config_path 的别名（兼容 eval.py）
        if config_path is None:
            config_path = local_model_path
        if config_path is None:
            raise ValueError("config_path or local_model_path must be provided")
        
        # 加载配置
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # 提取方法类型
        self.score_method = self.config.get("score_method", "threshold")
        
        # 提取关键参数
        self.eps = float(self.config["eps"])
        self.z_threshold_star = float(self.config["z_threshold_star"])
        
        # 根据方法类型加载不同的参数
        if self.score_method == "probe":
            # 探针方法：加载探针模型
            probe_path = Path(self.config["files"]["probe_model"])
            ComponentProbe = _import_component_probe()
            self.probe = ComponentProbe.load(probe_path)
            self.p_star = None
            self.m_cap = None
            self.actual_m = None
        else:
            # 阈值方法：加载 m_cap 等参数
            self.p_star = self.config.get("p_star")
            self.m_cap = int(self.config["m_cap"])
            # actual_m: 实际使用的组件数（m_cap <= 0 时使用全部组件）
            self.actual_m = int(self.config.get("actual_m", self.m_cap))
            self.probe = None
        
        # 加载模型（延迟导入）
        model_alias = self.config["model_alias"]
        construct_model_base_fn = _import_model_factory()
        self.model_base = construct_model_base_fn(model_alias)
        
        # 加载方向向量
        direction_path = Path(self.config["files"]["direction_file"])
        self.direction = torch.load(direction_path, map_location="cpu")
        if isinstance(self.direction, dict):
            # 如果保存的是字典，尝试提取向量
            self.direction = self.direction.get("direction", list(self.direction.values())[0] if self.direction else None)
        if not isinstance(self.direction, torch.Tensor):
            raise ValueError(f"Direction must be a torch.Tensor, got {type(self.direction)}")
        
        # 加载标准化器统计信息
        standardizer_stats_path = Path(self.config["files"]["standardizer_stats_json"])
        with open(standardizer_stats_path, "r", encoding="utf-8") as f:
            standardizer_stats = json.load(f)
        
        # 构建组件 key 到 mean/std 的映射
        # standardizer_stats 是一个列表，每个元素包含 component, mean, std, eps
        self.component_means: Dict[ComponentKey, float] = {}
        self.component_stds: Dict[ComponentKey, float] = {}
        self.selected_components: Set[ComponentKey] = set()
        
        for stats in standardizer_stats:
            if isinstance(stats, dict) and "component" in stats:
                comp_dict = stats["component"]
                key = component_dict_to_key(comp_dict)
                self.component_means[key] = float(stats["mean"])
                self.component_stds[key] = float(stats["std"])
                self.selected_components.add(key)
        
        # 初始化 runner（延迟到第一次使用时）
        self.runner = None
        
        # 延迟导入函数
        self._read_eoi_projection_unified = None
        self._MinimalHookRunner = None
        
        print(f"RefusalComponentClassifier initialized:")
        print(f"  - Model: {model_alias}")
        print(f"  - Score method: {self.score_method}")
        print(f"  - Selected components: {len(self.selected_components)}")
        if self.score_method == "probe":
            print(f"  - Probe model loaded")
        else:
            print(f"  - m_cap: {self.m_cap} (actual: {self.actual_m if self.actual_m > 0 else 'all'})")
        print(f"  - z_threshold_star: {self.z_threshold_star:.4f}")
    
    def get_required_input_fields(self) -> list[str]:
        return ["prompt"]
    
    def get_output_fields(self) -> list[str]:
        return ["prompt_harmfulness"]
    
    def _initialize_runner(self):
        """延迟初始化 runner"""
        if self.runner is None:
            # 延迟导入
            if self._MinimalHookRunner is None:
                MinimalHookRunner = _import_runner()
                self._MinimalHookRunner = MinimalHookRunner
            if self._read_eoi_projection_unified is None:
                read_eoi_projection_unified = _import_projections()
                self._read_eoi_projection_unified = read_eoi_projection_unified
            
            self.runner = self._MinimalHookRunner(self.model_base)
            
            # 设置状态
            cfg = getattr(self.model_base.model, "config", None)
            n_heads = None
            if cfg is not None and hasattr(cfg, "num_attention_heads"):
                n_heads = int(cfg.num_attention_heads)
            
            self.runner.set_state(
                direction=self.direction.to(
                    device=self.model_base.model.device,
                    dtype=next(self.model_base.model.parameters()).dtype
                ),
                eoi_token_ids=self.model_base.eoi_toks,
                input_ids=None,
                model_config=cfg,
                n_heads=n_heads,
            )
            
            # 注册 reader
            for module_type in ["o_proj", "mlp"]:
                self.runner.add_reader(
                    module_type=module_type,
                    layers=None,
                    fn=self._read_eoi_projection_unified
                )
    
    @torch.inference_mode()
    def _classify_batch(self, batch: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        """
        对一批 prompt 进行分类。
        
        Args:
            batch: 包含 "prompt" 字段的字典列表
        
        Returns:
            SafetyClassifierOutput 列表
        """
        # 初始化 runner（如果需要）
        self._initialize_runner()
        
        # 提取 prompts
        prompts = [item["prompt"] for item in batch]
        
        # 清空之前的结果
        self.runner.clear()
        
        # 前向传播收集激活
        self.runner.forward(prompts, batch_size=len(prompts), show_progress=False)
        
        # 收集投影数据
        proj_data: Dict[ComponentKey, np.ndarray] = {}
        
        for (layer, module_type, name), tensor in self.runner.results._final.items():
            if not name.startswith("eoi") or not name.endswith("_proj"):
                continue
            
            # 解析名称格式：
            # - MLP: eoi{k}_proj -> (module_type, layer, eoi_k)
            # - o_proj: eoi{k}_head{h}_proj -> (module_type, layer, eoi_k, head)
            name_body = name[len("eoi"):-len("_proj")]  # 去掉 "eoi" 前缀和 "_proj" 后缀
            
            # 检查是否包含 head 信息
            if "_head" in name_body:
                # 格式: {k}_head{h}
                parts = name_body.split("_head")
                if len(parts) != 2:
                    continue
                try:
                    eoi_k = int(parts[0])
                    head = int(parts[1])
                    key: ComponentKey = (module_type, layer, eoi_k, head)
                except ValueError:
                    continue
            else:
                # 格式: {k} (MLP)
                try:
                    eoi_k = int(name_body)
                    key: ComponentKey = (module_type, layer, eoi_k)
                except ValueError:
                    continue
            
            # 只保存选定的组件
            if key in self.selected_components:
                proj_data[key] = tensor.float().cpu().numpy()
        
        # 标准化
        standardized_proj: Dict[ComponentKey, np.ndarray] = {}
        for key, values in proj_data.items():
            if key not in self.component_means:
                continue
            
            mu = self.component_means[key]
            sigma = self.component_stds[key] + self.eps
            
            standardized_proj[key] = (values - mu) / sigma
        
        # 根据方法类型计算得分
        if self.score_method == "probe":
            # 探针方法：使用探针模型预测概率
            z_scores = self.probe.predict_proba(standardized_proj)
        else:
            # 阈值方法：计算 top-m 聚合得分
            # 使用 actual_m（当 m_cap <= 0 时使用全部组件）
            effective_m = self.actual_m if self.actual_m > 0 else len(self.selected_components)
            z_scores = compute_topm_scores(
                standardized_proj,
                self.selected_components,
                m_cap=effective_m
            )
        
        # 如果 z_scores 为空，返回默认值
        if len(z_scores) == 0:
            z_scores = np.zeros(len(prompts))
        
        # 生成输出
        outputs = []
        for i, z_score in enumerate(z_scores):
            is_harmful = z_score > self.z_threshold_star
            
            output = SafetyClassifierOutput(
                prompt_harmfulness=PromptHarmfulness.HARMFUL if is_harmful else PromptHarmfulness.UNHARMFUL,
                metadata={
                    "z_score": float(z_score),
                    "z_threshold": float(self.z_threshold_star),
                }
            )
            outputs.append(output)
        
        return outputs

