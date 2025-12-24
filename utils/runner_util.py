from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from collections import defaultdict

import torch
from tqdm import tqdm


# =============================================================================
# Results (in-memory)
# =============================================================================

class InMemoryResults:
    def __init__(self):
        self._buf: Dict[Tuple[int, str, str], List[torch.Tensor]] = defaultdict(list)
        self._final: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._finalized: bool = False

    def append(self, layer: int, module_type: str, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Results only accepts torch.Tensor, got {type(value).__name__}")
        if value.ndim == 0:
            raise ValueError("Saved tensor must be batch-first (at least 1D), got scalar tensor.")
        self._buf[(layer, module_type, name)].append(value.detach().cpu())
        self._finalized = False

    def finalize(self) -> None:
        self._final.clear()
        for key, parts in self._buf.items():
            if not parts:
                continue
            try:
                self._final[key] = torch.cat(parts, dim=0)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to cat results for key={key}. "
                    f"Make sure all saved tensors have compatible shapes. Error: {e}"
                )
        self._finalized = True

    def get(self, layer: int, module_type: str, name: str) -> torch.Tensor:
        if not self._finalized:
            self.finalize()
        key = (layer, module_type, name)
        if key not in self._final:
            raise KeyError(f"No result found for (layer={layer}, module_type='{module_type}', name='{name}')")
        return self._final[key]

    def clear(self) -> None:
        self._buf.clear()
        self._final.clear()
        self._finalized = False


# =============================================================================
# Callback Context
# =============================================================================

class CallbackCtx:
    """
    回调上下文（极简增强版）：
    - layer / module_type
    - module：当前被 hook 的 nn.Module（用于 o_proj.weight 等）
    - state：runner.set_state() 写入
    - save(name, tensor)：存 batch-first 结果
    """
    def __init__(
        self,
        results: InMemoryResults,
        state: Dict[str, Any],
        *,
        layer: int,
        module_type: str,
        module: Any,
    ):
        self._results = results
        self.state = state
        self.layer = layer
        self.module_type = module_type
        self.module = module

    def save(self, name: str, value: torch.Tensor) -> None:
        self._results.append(self.layer, self.module_type, name, value)


# =============================================================================
# Hook specs
# =============================================================================

# ✅ 统一签名：fn(ctx, inp, out)
ReaderFn = Callable[[CallbackCtx, torch.Tensor, torch.Tensor], None]
IntervenerFn = Callable[[CallbackCtx, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]


@dataclass(frozen=True)
class HookSpec:
    module_type: str
    layers: Optional[Union[int, List[int]]]
    fn: Union[ReaderFn, IntervenerFn]


# =============================================================================
# Minimal Hook Runner
# =============================================================================

class MinimalHookRunner:
    def __init__(self, model_base, *, device: Optional[str] = None):
        self.model_base = model_base
        self.model = model_base.model
        self.tokenizer = model_base.tokenizer
        self.device = device or str(self.model.device)

        self._state: Dict[str, Any] = {}
        self.results = InMemoryResults()

        self._readers: List[HookSpec] = []
        self._interveners: List[HookSpec] = []

        self._handles: List[Any] = []

        # ✅ 新增 o_proj
        o_proj_modules = None
        if hasattr(model_base, "_get_o_proj_modules"):
            o_proj_modules = model_base._get_o_proj_modules()

        self._module_type_map: Dict[str, List[Any]] = {
            "block": list(getattr(model_base, "model_block_modules")),
            "attn":  list(getattr(model_base, "model_attn_modules")),
            "mlp":   list(getattr(model_base, "model_mlp_modules")),
        }
        if o_proj_modules is not None:
            self._module_type_map["o_proj"] = list(o_proj_modules)

    # -------------------------
    # Public API
    # -------------------------

    def set_state(self, **kwargs) -> None:
        self._state.update(kwargs)

    def add_reader(self, module_type: str, layers: Optional[Union[int, List[int]]], fn: ReaderFn) -> None:
        self._validate_module_type(module_type)
        self._readers.append(HookSpec(module_type=module_type, layers=layers, fn=fn))

    def add_intervener(self, module_type: str, layers: Optional[Union[int, List[int]]], fn: IntervenerFn) -> None:
        self._validate_module_type(module_type)
        self._interveners.append(HookSpec(module_type=module_type, layers=layers, fn=fn))

    @torch.no_grad()
    def forward(self, instructions: List[str], *, batch_size: int = 8, show_progress: bool = True) -> None:
        if not self._readers:
            raise RuntimeError("No readers registered. Use runner.add_reader(...) first.")

        self.results.clear()
        self._register_hooks(specs=self._readers, kind="reader")

        total_batches = (len(instructions) + batch_size - 1) // batch_size
        batch_iter = self._batch_iter(instructions, batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=total_batches, desc="Forward", unit="batch", leave=False)

        try:
            for batch in batch_iter:
                tok = self.model_base.tokenize_instructions_fn(instructions=batch)

                input_ids = tok.input_ids.to(self.model.device)
                # ✅ 你不想改 util：这里做就行
                self._state["input_ids"] = input_ids

                attention_mask = getattr(tok, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)

                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 清理显存（特别是对于 Qwen 等显存敏感模型）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            self._remove_hooks()

        self.results.finalize()

    @torch.no_grad()
    def generate(
        self,
        instructions: List[str],
        *,
        batch_size: int = 8,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> List[str]:
        if not self._interveners:
            raise RuntimeError("No interveners registered. Use runner.add_intervener(...) first.")

        outputs: List[str] = []

        self._register_hooks(specs=self._interveners, kind="intervener")
        try:
            for batch in self._batch_iter(instructions, batch_size):
                tok = self.model_base.tokenize_instructions_fn(instructions=batch)
                input_ids = tok.input_ids.to(self.model.device)

                attention_mask = getattr(tok, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)

                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                gen_new = gen[:, input_ids.shape[1]:]
                for row in gen_new:
                    outputs.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        finally:
            self._remove_hooks()

        return outputs

    def get(self, layer: int, module_type: str, name: str) -> torch.Tensor:
        return self.results.get(layer, module_type, name)

    def clear(self) -> None:
        self.results.clear()

    # -------------------------
    # Internal helpers
    # -------------------------

    def _validate_module_type(self, module_type: str) -> None:
        if module_type not in self._module_type_map:
            raise ValueError(f"Invalid module_type='{module_type}'. Must be one of {list(self._module_type_map.keys())}")

    @staticmethod
    def _batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    @staticmethod
    def _unwrap_first_tensor(obj: Any) -> torch.Tensor:
        """
        输入/输出可能是 Tensor / tuple(list) / ModelOutput。
        这里尽量拿到第一个 Tensor。
        """
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], torch.Tensor):
            return obj[0]
        raise TypeError(f"Unsupported type, expected Tensor/tuple/list. Got: {type(obj).__name__}")

    @staticmethod
    def _repack_like(original_output: Any, new_first: torch.Tensor) -> Any:
        if isinstance(original_output, torch.Tensor):
            return new_first
        if isinstance(original_output, tuple):
            return (new_first,) + original_output[1:]
        if isinstance(original_output, list):
            out = list(original_output)
            out[0] = new_first
            return out
        return new_first

    def _expand_layers(self, module_type: str, layers: Optional[Union[int, List[int]]]) -> List[int]:
        n = len(self._module_type_map[module_type])
        if layers is None:
            return list(range(n))
        if isinstance(layers, int):
            return [layers]
        if isinstance(layers, list):
            return layers
        raise TypeError("layers must be None, int, or List[int]")

    def _register_hooks(self, specs: List[HookSpec], *, kind: str) -> None:
        self._remove_hooks()

        bucket: Dict[Tuple[str, int], List[Callable]] = defaultdict(list)
        for spec in specs:
            for layer_idx in self._expand_layers(spec.module_type, spec.layers):
                bucket[(spec.module_type, layer_idx)].append(spec.fn)

        for (module_type, layer_idx), fns in bucket.items():
            modules = self._module_type_map[module_type]
            if layer_idx < 0 or layer_idx >= len(modules):
                raise IndexError(f"Layer {layer_idx} out of range for module_type='{module_type}'")
            module = modules[layer_idx]

            def make_hook(mtype: str, lidx: int, callbacks: List[Callable], module_ref: Any):
                def _hook(_module, _inp, _out):
                    # ✅ inp/out 都给 callback
                    inp_t = self._unwrap_first_tensor(_inp[0] if isinstance(_inp, tuple) else _inp)
                    out_t = self._unwrap_first_tensor(_out)

                    ctx = CallbackCtx(
                        self.results,
                        self._state,
                        layer=lidx,
                        module_type=mtype,
                        module=module_ref,
                    )

                    if kind == "reader":
                        for cb in callbacks:
                            cb(ctx, inp_t, out_t)
                        return _out

                    elif kind == "intervener":
                        current = out_t
                        for cb in callbacks:
                            maybe = cb(ctx, inp_t, current)
                            if maybe is not None:
                                if not isinstance(maybe, torch.Tensor):
                                    raise TypeError(f"Intervener must return Tensor or None, got {type(maybe).__name__}")
                                current = maybe
                        return self._repack_like(_out, current)

                    else:
                        raise ValueError(f"Invalid hook kind: {kind}")

                return _hook

            handle = module.register_forward_hook(make_hook(module_type, layer_idx, fns, module))
            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
