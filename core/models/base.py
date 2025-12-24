"""
Model Base Class - Abstract interface for all model families.

Ported from refusal_direction/pipeline/model_utils/model_base.py
"""

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

from components_safety.core.models.hooks import add_hooks


class ModelBase(ABC):
    """
    模型基类，为不同模型家族（Llama、Qwen、Gemma等）提供统一的接口。
    
    这个抽象基类定义了所有模型必须实现的接口，用于：
    1. 统一不同模型的结构差异（路径、命名等）
    2. 提供统一的文本生成和激活干预接口
    3. 简化上层代码，无需关心具体模型类型
    
    所有具体模型类（如 Llama3Model、QwenModel）都必须继承此类并实现所有抽象方法。
    """
    def __init__(self, model_name_or_path: str):
        """
        初始化模型基类。
        
        参数:
            model_name_or_path: 模型路径或名称
        """
        # 模型基本信息
        self.model_name_or_path = model_name_or_path
        
        # 核心模型组件（由子类实现加载）
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        # 指令处理相关属性
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        # 模型结构访问接口（用于激活干预）
        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        """删除模型以释放内存。"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    # ========== 抽象方法：子类必须实现 ==========
    
    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        """加载Transformers模型。"""
        pass
    
    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """加载Tokenizer。"""
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        """获取指令tokenization函数。"""
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        """获取End of Instruction tokens（指令结束标记）。"""
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        """获取Refusal tokens（拒绝标记）。"""
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        """获取Transformer blocks列表。"""
        pass

    @abstractmethod
    def _get_attn_modules(self):
        """获取Attention模块列表。"""
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        """获取MLP（前馈网络）模块列表。"""
        pass

    @abstractmethod
    def _get_o_proj_modules(self):
        """获取Attention输出投影模块列表。"""
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        """获取权重正交化函数。"""
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        """获取激活加法函数（用于激活干预）。"""
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        """
        批量生成文本补全，支持激活干预（通过hooks）。
        
        参数:
            dataset: 数据集列表，每个元素应包含 'instruction' 和 'category' 字段
            fwd_pre_hooks: 前向传播前的hook列表
            fwd_hooks: 前向传播后的hook列表
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            
        返回:
            List[Dict]: 补全结果列表
        """
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

            generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'category': categories[i + generation_idx],
                    'prompt': instructions[i + generation_idx],
                    'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

        return completions
