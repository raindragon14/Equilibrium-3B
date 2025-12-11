# Equilibrium-3B Core Model Components
# =====================================

from .equilibrium_model import Equilibrium3B
from .mamba_layer import MambaBlock
from .attention_layer import MultiHeadLatentAttention
from .moe_layer import DeepSeekMoE
from .embeddings import RotaryEmbedding, LearnablePositionalEncoding

__all__ = [
    "Equilibrium3B",
    "MambaBlock", 
    "MultiHeadLatentAttention",
    "DeepSeekMoE",
    "RotaryEmbedding",
    "LearnablePositionalEncoding"
]

# Model registry for dynamic loading
MODEL_REGISTRY = {
    "equilibrium-3b": Equilibrium3B,
    "equilibrium-3b-base": Equilibrium3B,
    "equilibrium-3b-math": Equilibrium3B,
    "equilibrium-3b-econ": Equilibrium3B,
}

def get_model(model_name: str, config: dict):
    """Factory function to get model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](config)