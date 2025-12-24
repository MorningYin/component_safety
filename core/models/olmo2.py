
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from components_safety.core.models.utils import get_orthogonalized_matrix
from components_safety.core.models.base import ModelBase

# OLMo2 chat templates
# Reference: allenai/OLMo-2-1124-7B-Instruct
# <|endoftext|> is usually the BOS/EOS token

OLMO2_CHAT_TEMPLATE = """<|endoftext|><|user|>
{instruction}
<|assistant|>
"""

OLMO2_CHAT_TEMPLATE_WITH_SYSTEM = """<|endoftext|><|system|>
{system_prompt}
<|user|>
{instruction}
<|assistant|>
"""

OLMO2_REFUSAL_TOKS = [40] # 'I'

def format_instruction_olmo2_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    if system is not None:
        formatted_instruction = OLMO2_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = OLMO2_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_olmo2_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_olmo2_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_olmo2_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_olmo2_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_olmo2_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    # Use _safe_set_bias logic if bias might be None, but for Olmo2 it usually has bias=False?
    # Actually, Olmo2Config shows attention_bias: false. 
    # But for act_add we WANT to add bias to the linear layer to shift the activation.
    
    target_module = model.model.layers[layer-1].mlp.down_proj
    if target_module.bias is None:
        target_module.bias = torch.nn.Parameter(bias)
    else:
        target_module.bias.data = bias

class OLMo2Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_olmo2_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # The EOI is the end of the assistant header: "\n<|assistant|>\n"
        return self.tokenizer.encode(OLMO2_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return OLMO2_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_o_proj_modules(self):
        return torch.nn.ModuleList([block_module.self_attn.o_proj for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_olmo2_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_olmo2_weights, direction=direction, coeff=coeff, layer=layer)
