"""
Equilibrium-3B: Hybrid SSM-Transformer Architecture
==================================================

Core model implementation combining Mamba-2 state space models with
Multi-Head Latent Attention and Fine-Grained Mixture of Experts.

Based on 2025 research paradigms for efficient small language models.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .mamba_layer import MambaBlock
from .attention_layer import MultiHeadLatentAttention  
from .moe_layer import DeepSeekMoE
from .embeddings import RotaryEmbedding, LearnablePositionalEncoding


@dataclass
class Equilibrium3BConfig:
    """Configuration for Equilibrium-3B model."""
    
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_layers: int = 24
    num_attention_layers: int = 3  # Every 8th layer
    
    # Attention configuration
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    kv_lora_rank: int = 512  # For MLA compression
    q_lora_rank: int = 1536
    
    # MoE configuration
    num_experts: int = 64
    num_experts_per_tok: int = 2
    num_shared_experts: int = 8
    
    # Context and sequence
    max_position_embeddings: int = 131072  # 128k context
    rope_theta: float = 500000.0
    
    # Training configuration
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 4
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_layers > self.num_attention_layers
        assert self.num_layers % self.num_attention_layers == 0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Equilibrium3BLayer(nn.Module):
    """Single layer of Equilibrium-3B model."""
    
    def __init__(self, config: Equilibrium3BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Determine if this is an attention layer
        attention_interval = config.num_layers // config.num_attention_layers
        self.is_attention_layer = (layer_idx % attention_interval == 0)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if self.is_attention_layer:
            # Multi-Head Latent Attention layer
            self.self_attn = MultiHeadLatentAttention(config)
        else:
            # Mamba-2 state space layer
            self.mamba = MambaBlock(
                d_model=config.hidden_size,
                d_state=128,  # Mamba-2 state dimension
                expand=2,     # Expansion factor
                headdim=64,   # Head dimension for Mamba-2
                ngroups=1,    # Group convolution groups
            )
        
        # MoE layer (applied to all layers)
        self.mlp = DeepSeekMoE(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention or Mamba
        if self.is_attention_layer:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        else:
            # Mamba processing
            hidden_states = self.mamba(hidden_states)
            self_attn_weights = None
            present_key_value = None
        
        hidden_states = residual + hidden_states
        
        # MoE Feed forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class Equilibrium3BPreTrainedModel(nn.Module):
    """Base class for Equilibrium-3B models."""
    
    config_class = Equilibrium3BConfig
    base_model_prefix = "equilibrium3b"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Equilibrium3BLayer"]
    _skip_keys_device_placement = "past_key_values"
    
    def _init_weights(self, module):
        """Initialize weights."""
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Equilibrium3BModel(Equilibrium3BPreTrainedModel):
    """The bare Equilibrium-3B Model outputting raw hidden-states."""
    
    def __init__(self, config: Equilibrium3BConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            Equilibrium3BLayer(config, layer_idx) 
            for layer_idx in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length, _ = inputs_embeds.shape
        
        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        # 4D attention mask for transformer layers
        if attention_mask is not None and len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        hidden_states = inputs_embeds
        
        # Rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    use_cache,
                    cache_position,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return (hidden_states, all_hidden_states, all_self_attns)


class Equilibrium3B(Equilibrium3BPreTrainedModel):
    """Equilibrium-3B Model for Causal Language Modeling."""
    
    def __init__(self, config: Equilibrium3BConfig):
        super().__init__()
        self.config = config
        self.model = Equilibrium3BModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.apply(self._init_weights)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        
        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, logits) + outputs[1:]
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        
        # If we have cache, we don't need all input_ids, just the last one
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                cache_length = past_key_values[0][0].shape[2]
            else:
                cache_length = past_key_values.get_seq_length()
            
            # Keep only the unprocessed tokens
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - cache_length):]
            elif input_ids.shape[1] > cache_length:
                input_ids = input_ids[:, cache_length:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        })
        
        return model_inputs


# Model size variants
def create_equilibrium_3b_base(vocab_size: int = 32000) -> Equilibrium3B:
    """Create base 3B parameter model."""
    config = Equilibrium3BConfig(
        vocab_size=vocab_size,
        hidden_size=2560,
        intermediate_size=6912,
        num_layers=24,
        num_attention_layers=3,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    return Equilibrium3B(config)


def create_equilibrium_1_5b(vocab_size: int = 32000) -> Equilibrium3B:
    """Create smaller 1.5B parameter model for development."""
    config = Equilibrium3BConfig(
        vocab_size=vocab_size,
        hidden_size=1920,
        intermediate_size=5184,
        num_layers=18,
        num_attention_layers=3,
        num_attention_heads=24,
        num_key_value_heads=6,
    )
    return Equilibrium3B(config)


def create_equilibrium_7b(vocab_size: int = 32000) -> Equilibrium3B:
    """Create larger 7B parameter model."""
    config = Equilibrium3BConfig(
        vocab_size=vocab_size,
        hidden_size=3584,
        intermediate_size=9728,
        num_layers=32,
        num_attention_layers=4,
        num_attention_heads=28,
        num_key_value_heads=7,
    )
    return Equilibrium3B(config)