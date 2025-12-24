"""
Model Factory - Construct model instances based on alias.

Ported from refusal_direction/pipeline/model_utils/model_factory.py
"""

from components_safety.core.models.base import ModelBase

MODEL_ALIASES = {
    "llama-3-8b-it": "/root/autodl-tmp/Projects/LLaMA-3-8b-IT",
    "qwen-7b-chat": "Qwen/Qwen-7B-Chat",
    "qwen-14b-chat": "Qwen/Qwen-14B-Chat",
    "olmo2-7b-it": "allenai/OLMo-2-1124-7B-Instruct",
}


def construct_model_base(model_alias: str) -> ModelBase:
    """
    Construct a model base instance based on the model alias.
    
    Args:
        model_alias: Either a registered alias or a direct model path
        
    Returns:
        ModelBase: An instance of the appropriate model class
    """
    model_path = MODEL_ALIASES.get(model_alias, model_alias)
    
    if 'qwen' in model_path.lower():
        from components_safety.core.models.qwen import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from components_safety.core.models.llama3 import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from components_safety.core.models.llama2 import Llama2Model
        return Llama2Model(model_path)
    elif 'mistral' in model_path.lower():
        from components_safety.core.models.mistral import MistralModel
        return MistralModel(model_path)
    elif 'olmo' in model_path.lower() or 'olmo2' in model_path.lower():
        from components_safety.core.models.olmo2 import OLMo2Model
        return OLMo2Model(model_path)
    elif 'gemma' in model_path.lower():
        from components_safety.core.models.gemma import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path.lower():
        from components_safety.core.models.yi import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
