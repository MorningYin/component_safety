"""投影数据采集：从模型收集组件投影值"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import numpy as np
import torch

from components_safety.utils.runner_util import MinimalHookRunner
from refusal_direction.pipeline.model_utils.model_factory import ModelBase


# 组件 key 类型：tuple (module_type, layer, eoi_k) 或 (module_type, layer, eoi_k, head)
ComponentKey = Union[Tuple[str, int, int], Tuple[str, int, int, int]]


def read_eoi_projection_unified(ctx, inp: torch.Tensor, out: torch.Tensor):
    """
    Unified reader callback for:
      - module_type == "mlp": use out[B,T,d] to compute eoi{k}_proj
      - module_type == "o_proj": use inp[B,T,d] + W_O to compute eoi{k}_head{h}_proj

    ctx.state must contain:
      - direction: Tensor[d_model]
      - eoi_token_ids: List[int]  (EOI token *sequence*, matched contiguously)
      - input_ids: Tensor[B,T]

    For o_proj, also need head count:
      - ctx.state["n_heads"] (preferred) OR ctx.state["model_config"].num_attention_heads
    """
    direction: torch.Tensor = ctx.state["direction"]          # [d]
    eoi_ids = ctx.state["eoi_token_ids"]                      # List[int]
    eoi_len = len(eoi_ids)
    input_ids: torch.Tensor = ctx.state["input_ids"]          # [B, T]

    if ctx.module_type not in ("mlp", "o_proj"):
        return  # safety: ignore other module_type if accidentally used

    # -------- choose tensor x to index EOI positions --------
    # mlp -> out ; o_proj -> inp
    x = out if ctx.module_type == "mlp" else inp
    if x.ndim != 3:
        return
    B, T, d_model = x.shape

    # -------- helper: find last EOI sequence start for each sample --------
    def find_last_eoi_start(b: int) -> int | None:
        # scan backwards for contiguous match of eoi_ids
        for t in range(T - eoi_len, -1, -1):
            ok = True
            for i, tid in enumerate(eoi_ids):
                if int(input_ids[b, t + i]) != int(tid):
                    ok = False
                    break
            if ok:
                return t
        return None

    # =========================================================================
    # Case 1) MLP: save eoi{k}_proj (no head)
    # =========================================================================
    if ctx.module_type == "mlp":
        buckets = {}  # k -> list[scalar]

        for b in range(B):
            start = find_last_eoi_start(b)
            if start is None:
                continue

            for k in range(eoi_len):
                pos = start + k
                if pos >= T:
                    break
                val = x[b, pos, :] @ direction  # scalar
                buckets.setdefault(k, []).append(val)

        for k, vals in buckets.items():
            if vals:
                ctx.save(f"eoi{k}_proj", torch.stack(vals, dim=0).detach().cpu())
        return

    # =========================================================================
    # Case 2) o_proj: save eoi{k}_head{h}_proj
    #   inp is concat-heads; use W_O slices via v_all = W^T @ direction
    # =========================================================================
    # infer n_heads
    n_heads = ctx.state.get("n_heads", None)
    if n_heads is None:
        cfg = ctx.state.get("model_config", None)
        if cfg is not None and hasattr(cfg, "num_attention_heads"):
            n_heads = int(cfg.num_attention_heads)

    if n_heads is None or n_heads <= 0 or (d_model % n_heads != 0):
        raise RuntimeError(
            f"Cannot infer n_heads for o_proj. Got n_heads={n_heads}, d_model={d_model}. "
            f"Set runner.set_state(n_heads=..., model_config=model_base.model.config)."
        )
    head_dim = d_model // n_heads

    W = getattr(ctx.module, "weight", None)
    if W is None:
        raise RuntimeError("Hooked o_proj module has no .weight; expected Linear-like module.")

    # v_all: [H, Dh]
    v_all = (W.t() @ direction).view(n_heads, head_dim)

    buckets = {}  # (k,h) -> list[scalar]

    for b in range(B):
        start = find_last_eoi_start(b)
        if start is None:
            continue

        for k in range(eoi_len):
            pos = start + k
            if pos >= T:
                break

            x_heads = x[b, pos, :].view(n_heads, head_dim)     # [H, Dh]
            proj_heads = (x_heads * v_all).sum(dim=-1)         # [H]

            # store per head
            for h in range(n_heads):
                buckets.setdefault((k, h), []).append(proj_heads[h])

    # save per (k,h)
    for (k, h), vals in buckets.items():
        if vals:
            ctx.save(f"eoi{k}_head{h}_proj", torch.stack(vals, dim=0).detach().cpu())


def collect_component_projections(
    model_base: ModelBase,
    datasets: Dict[str, list],
    batch_size: int,
    direction: torch.Tensor,
    eoi_token_ids: list[int],
) -> Dict[str, Dict[ComponentKey, np.ndarray]]:
    """
    收集所有组件的投影数据。
    
    此函数只负责"采集投影"，不做任何筛选/统计。
    
    Args:
        model_base: 模型基类（包含 model 和 _get_eoi_toks 方法）
        datasets: 数据集字典，键为 split 名称，值为样本列表（每个样本有 "instruction" 字段）
        batch_size: 批处理大小
        direction: 拒答方向向量 [d_model]
        eoi_token_ids: EOI token ID 列表
    
    Returns:
        proj_data: 字典 {split_name: {component_key: np.ndarray}}
        其中 component_key 为:
        - ('mlp', layer, eoi_k) 或
        - ('o_proj', layer, eoi_k, head)
    """
    model_base.model.eval()
    
    model_dtype = next(model_base.model.parameters()).dtype
    direction_device = direction.to(device=model_base.model.device, dtype=model_dtype)
    
    runner = MinimalHookRunner(model_base)
    
    cfg = getattr(model_base.model, "config", None)
    n_heads = None
    if cfg is not None and hasattr(cfg, "num_attention_heads"):
        n_heads = int(cfg.num_attention_heads)
    
    runner.set_state(
        direction=direction_device,
        eoi_token_ids=eoi_token_ids,
        input_ids=None,
        model_config=cfg,
        n_heads=n_heads,
    )
    
    for module_type in ["o_proj", "mlp"]:
        runner.add_reader(module_type=module_type, layers=None, fn=read_eoi_projection_unified)
    
    # 收集所有 split 的投影数据
    proj_data: Dict[str, Dict[ComponentKey, np.ndarray]] = {}
    
    for split_name, dataset in datasets.items():
        runner.clear()
        instructions = [item["instruction"] for item in dataset]
        
        # 对于 Qwen 等显存敏感模型，动态调整 batch_size
        effective_batch_size = batch_size
        if "qwen" in model_base.model_name_or_path.lower():
            # Qwen 模型可能需要更小的 batch_size
            effective_batch_size = min(batch_size, 8)
        
        runner.forward(instructions, batch_size=effective_batch_size)
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        proj_data[split_name] = {}
        
        for (layer, module_type, name), tensor in runner.results._final.items():
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
            
            proj_data[split_name][key] = tensor.float().cpu().numpy()
    
    return proj_data