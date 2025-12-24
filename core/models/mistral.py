import functools
from typing import List, Callable, Optional

import torch
from torch import Tensor
from jaxtyping import Float
from transformers import AutoTokenizer, AutoModelForCausalLM

from components_safety.core.models.utils import get_orthogonalized_matrix
from components_safety.core.models.base import ModelBase


# =========================
# Common helpers (robust to model family differences)
# =========================

def _get_core_decoder(model: AutoModelForCausalLM):
    """
    Return the decoder/backbone module that holds embed_tokens + layers.
    Works for most HF causal LMs:
      - Llama/Mistral: model.model
      - Qwen-like: model.model or model.transformer
      - Some custom: model.transformer
    """
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    # Fallback: some models put layers at top-level
    return model


def _get_embed_tokens(core):
    for attr in ("embed_tokens", "wte", "tok_embeddings", "embeddings"):
        if hasattr(core, attr):
            return getattr(core, attr)
    raise AttributeError("Cannot find embedding module (embed_tokens/wte/tok_embeddings/embeddings).")


def _get_layers(core):
    for attr in ("layers", "h", "blocks", "decoder_layers"):
        if hasattr(core, attr):
            return getattr(core, attr)
    raise AttributeError("Cannot find transformer layers (layers/h/blocks/decoder_layers).")


def _get_attn(block):
    for attr in ("self_attn", "attn", "attention"):
        if hasattr(block, attr):
            return getattr(block, attr)
    raise AttributeError("Cannot find attention module (self_attn/attn/attention).")


def _get_mlp(block):
    for attr in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, attr):
            return getattr(block, attr)
    raise AttributeError("Cannot find MLP module (mlp/feed_forward/ffn).")


def _get_o_proj(attn):
    for attr in ("o_proj", "out_proj", "c_proj"):
        if hasattr(attn, attr):
            return getattr(attn, attr)
    raise AttributeError("Cannot find attention o-proj (o_proj/out_proj/c_proj).")


def _get_down_proj(mlp):
    for attr in ("down_proj", "c_proj", "proj", "out_proj"):
        if hasattr(mlp, attr):
            return getattr(mlp, attr)
    # Some implementations store it nested
    if hasattr(mlp, "dense_4h_to_h"):
        return getattr(mlp, "dense_4h_to_h")
    raise AttributeError("Cannot find MLP down-proj (down_proj/c_proj/proj/out_proj/dense_4h_to_h).")


def _safe_set_bias(linear: torch.nn.Module, bias: torch.Tensor):
    """
    Ensure `linear.bias` exists and equals `bias` (as a Parameter).
    Works even if the module was constructed with bias=False (bias=None).
    """
    bias = bias.detach()
    if getattr(linear, "bias", None) is None:
        linear.bias = torch.nn.Parameter(bias)
    else:
        linear.bias = torch.nn.Parameter(bias)


def _encode_single_token_id(tokenizer: AutoTokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer produced empty ids for text={text!r}")
    return ids[0]


# =========================
# Mistral subclass
# =========================

# Mistral Instruct classic template (v0.x):
# <s>[INST] user [/INST] assistant
# HF tokenizers often provide apply_chat_template, but we keep a deterministic fallback.
MISTRAL_CHAT_TEMPLATE = "<s>[INST] {instruction} [/INST]"
MISTRAL_CHAT_TEMPLATE_WITH_SYSTEM = "<s>[INST] {system}\n\n{instruction} [/INST]"


def _format_instruction_mistral(
    instruction: str,
    output: Optional[str] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True,
) -> str:
    if system is not None:
        prompt = MISTRAL_CHAT_TEMPLATE_WITH_SYSTEM.format(system=system, instruction=instruction)
    else:
        prompt = MISTRAL_CHAT_TEMPLATE.format(instruction=instruction)

    if include_trailing_whitespace:
        prompt = prompt + " "
    else:
        prompt = prompt.rstrip()

    if output is not None:
        prompt += output
    return prompt


def _tokenize_instructions_mistral(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: Optional[List[str]] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True,
):
    # Prefer official chat template if tokenizer provides it (more compatible across Mistral variants)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompts = []
        if outputs is None:
            for ins in instructions:
                messages = []
                if system is not None:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": ins})
                # add_generation_prompt=True ends with assistant header
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        else:
            for ins, out in zip(instructions, outputs):
                messages = []
                if system is not None:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": ins})
                messages.append({"role": "assistant", "content": out})
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                )

        if include_trailing_whitespace:
            prompts = [p + " " for p in prompts]
        else:
            prompts = [p.rstrip() for p in prompts]

        return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")

    # Fallback: manual template
    if outputs is not None:
        prompts = [
            _format_instruction_mistral(ins, output=out, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for ins, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            _format_instruction_mistral(ins, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for ins in instructions
        ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def orthogonalize_mistral_weights(model: AutoModelForCausalLM, direction: Float[Tensor, "d_model"]):
    core = _get_core_decoder(model)
    emb = _get_embed_tokens(core)
    emb.weight.data = get_orthogonalized_matrix(emb.weight.data, direction)

    layers = _get_layers(core)
    for block in layers:
        attn = _get_attn(block)
        mlp = _get_mlp(block)
        o_proj = _get_o_proj(attn)
        down_proj = _get_down_proj(mlp)

        o_proj.weight.data = get_orthogonalized_matrix(o_proj.weight.data.T, direction).T
        down_proj.weight.data = get_orthogonalized_matrix(down_proj.weight.data.T, direction).T


def act_add_mistral_weights(model: AutoModelForCausalLM, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
    core = _get_core_decoder(model)
    layers = _get_layers(core)
    block = layers[layer - 1]
    mlp = _get_mlp(block)
    down_proj = _get_down_proj(mlp)

    dtype = down_proj.weight.dtype
    device = down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    _safe_set_bias(down_proj, bias)


class MistralModel(ModelBase):
    """
    Supports mistralai/Mistral-7B, mistralai/Mistral-7B-Instruct, and similar Mistral-family models
    that use the standard decoder layout with model.model.layers[*].self_attn.o_proj and mlp.down_proj.
    """

    def _load_model(self, model_name_or_path: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_name_or_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            _tokenize_instructions_mistral,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        # Prefer tokenizer chat template when available
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": "X"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # split on the user payload "X" and take suffix as EOI marker
            suffix = text.split("X", 1)[-1]
            return self.tokenizer.encode(suffix, add_special_tokens=False)
        # Fallback to manual template suffix after instruction
        suffix = MISTRAL_CHAT_TEMPLATE.split("{instruction}")[-1]
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    def _get_refusal_toks(self):
        # Use 'I' as a stable refusal-start proxy across many English-aligned instruct models
        return [_encode_single_token_id(self.tokenizer, "I")]

    def _get_model_block_modules(self):
        core = _get_core_decoder(self.model)
        return _get_layers(core)

    def _get_attn_modules(self):
        return torch.nn.ModuleList([_get_attn(b) for b in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([_get_mlp(b) for b in self.model_block_modules])

    def _get_o_proj_modules(self):
        return torch.nn.ModuleList([_get_o_proj(_get_attn(b)) for b in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_mistral_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        return functools.partial(act_add_mistral_weights, direction=direction, coeff=coeff, layer=layer)
