"""
Embedding Layers for Equilibrium-3B
===================================

Implements rotary position embeddings (RoPE) and learnable positional 
encodings optimized for long context (128k tokens).

Key features:
- Rotary Position Embedding (RoPE) with extended context support
- Learnable positional encodings with scaling
- Efficient implementation for long sequences
- Support for different frequency interpolation methods
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    with extensions for long context support.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,  # 128k context
        base: float = 500000.0,  # Increased base for longer context
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",  # "default", "linear", "dynamic", "yarn"
    ):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        
        # Compute frequency inverse
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # For dynamic RoPE
        if rope_type == "dynamic":
            self.register_buffer("dynamic_scale", torch.tensor(1.0), persistent=False)
        
        # Pre-compute cos/sin for efficiency
        self._set_cos_sin_cache(max_position_embeddings, device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos/sin cache for given sequence length."""
        self.max_seq_len_cached = seq_len
        
        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        
        # Apply scaling based on rope type
        if self.rope_type == "linear":
            t = t / self.scaling_factor
        elif self.rope_type == "dynamic":
            if hasattr(self, 'dynamic_scale'):
                t = t / self.dynamic_scale
        elif self.rope_type == "yarn":
            # YaRN scaling: extrapolation factor with attention scaling
            alpha = max(1.0, seq_len / self.max_position_embeddings)
            beta = max(1.0, alpha * 0.1)  # Attention temperature scaling
            t = t / (alpha * beta)
        
        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor for determining sequence length and device
            position_ids: Position indices [batch_size, seq_len]
        
        Returns:
            cos: Cosine embeddings [batch_size, seq_len, dim] or [seq_len, dim]
            sin: Sine embeddings [batch_size, seq_len, dim] or [seq_len, dim]
        """
        if x.dim() == 3:
            seq_len = x.shape[1]
        else:
            seq_len = x.shape[0]
        
        # Extend cache if necessary
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        
        # Get cos and sin for the sequence
        if position_ids is not None:
            # Use specific position IDs
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            # Use sequential positions
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    def update_dynamic_scale(self, current_seq_len: int):
        """Update dynamic scaling factor for very long sequences."""
        if self.rope_type == "dynamic" and current_seq_len > self.max_position_embeddings:
            scale = current_seq_len / self.max_position_embeddings
            self.dynamic_scale = torch.tensor(scale, device=self.inv_freq.device)
            # Recompute cache with new scale
            self._set_cos_sin_cache(current_seq_len, self.inv_freq.device)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encodings with support for long sequences.
    
    Uses chunked encoding for memory efficiency with very long contexts.
    """
    
    def __init__(
        self,
        d_model: int,
        max_position_embeddings: int = 131072,  # 128k context
        dropout: float = 0.1,
        chunk_size: int = 8192,  # Process in chunks for memory efficiency
        use_scaling: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        self.chunk_size = chunk_size
        self.use_scaling = use_scaling
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(max_position_embeddings, d_model) * 0.02
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for position embeddings
        if use_scaling:
            self.pos_scaling = nn.Parameter(torch.ones(1))
        
        # Layer normalization for position embeddings
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        input_embeddings: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: Token embeddings [batch_size, seq_len, d_model]
            position_ids: Position indices [batch_size, seq_len]
        
        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = input_embeddings.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, 
                dtype=torch.long, 
                device=input_embeddings.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Clamp position IDs to max range
        position_ids = position_ids.clamp(0, self.max_position_embeddings - 1)
        
        # Get positional embeddings
        pos_embeddings = self.position_embeddings[position_ids]  # [batch_size, seq_len, d_model]
        
        # Apply scaling if enabled
        if self.use_scaling:
            pos_embeddings = pos_embeddings * self.pos_scaling
        
        # Apply layer normalization
        pos_embeddings = self.layer_norm(pos_embeddings)
        
        # Combine token and position embeddings
        combined = input_embeddings + pos_embeddings
        
        return self.dropout(combined)
    
    def extend_positions(self, new_max_positions: int):
        """Extend position embeddings for longer sequences."""
        if new_max_positions <= self.max_position_embeddings:
            return
        
        # Initialize new positions by interpolation
        old_positions = self.position_embeddings.data
        old_length = self.max_position_embeddings
        
        # Linear interpolation for new positions
        new_positions = torch.zeros(new_max_positions, self.d_model, device=old_positions.device)
        
        # Copy existing positions
        new_positions[:old_length] = old_positions
        
        # Interpolate for new positions
        for i in range(old_length, new_max_positions):
            # Simple linear interpolation from nearest positions
            ratio = (i % old_length) / old_length
            base_idx = i // old_length
            if base_idx < old_length:
                new_positions[i] = old_positions[base_idx] * (1 - ratio) + old_positions[(base_idx + 1) % old_length] * ratio
            else:
                # For very long extensions, use random initialization
                new_positions[i] = torch.randn(self.d_model, device=old_positions.device) * 0.02
        
        # Update embeddings
        self.position_embeddings.data = new_positions
        self.max_position_embeddings = new_max_positions


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings (fixed, not learnable).
    
    More memory efficient for very long sequences as no parameters to store.
    """
    
    def __init__(
        self,
        d_model: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal position encoding
        pe = torch.zeros(max_position_embeddings, d_model)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(base) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(
        self, 
        input_embeddings: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """Add sinusoidal positional encodings to input embeddings."""
        
        batch_size, seq_len, d_model = input_embeddings.shape
        
        if position_ids is not None:
            # Use specific position IDs
            pos_embeddings = self.pe[position_ids]
        else:
            # Use sequential positions
            pos_embeddings = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        return self.dropout(input_embeddings + pos_embeddings.to(input_embeddings.dtype))


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for position encoding.
    
    Instead of adding position embeddings, modifies attention scores directly.
    Very memory efficient for long sequences.
    """
    
    def __init__(
        self,
        num_heads: int,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        
        # Compute ALiBi slopes
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes, persistent=False)
        
        # Cache bias matrix for efficiency
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2) + \
                     get_slopes_power_of_2(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2]
        
        return torch.tensor(slopes).view(num_heads, 1, 1)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            device: Target device
            
        Returns:
            Bias matrix [num_heads, seq_len, seq_len]
        """
        if self._cached_bias is None or seq_len > self._cached_seq_len:
            # Create distance matrix
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            distances = positions - positions.T
            
            # Apply slopes to distances
            bias = distances.unsqueeze(0) * self.slopes.to(device)  # [num_heads, seq_len, seq_len]
            
            # Cache for efficiency
            self._cached_bias = bias
            self._cached_seq_len = seq_len
        
        return self._cached_bias[:, :seq_len, :seq_len]


# Utility functions

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def test_embeddings():
    """Test function for embedding layers."""
    
    batch_size, seq_len, d_model = 2, 2048, 768
    num_heads = 12
    
    print("Testing Rotary Position Embeddings...")
    rope = RotaryEmbedding(d_model // num_heads, max_position_embeddings=131072)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model // num_heads)
    cos, sin = rope(x)
    
    print(f"RoPE cos shape: {cos.shape}")
    print(f"RoPE sin shape: {sin.shape}")
    
    print("\nTesting Learnable Positional Encoding...")
    pos_enc = LearnablePositionalEncoding(d_model, max_position_embeddings=131072)
    
    # Test input embeddings
    input_emb = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(input_emb)
    
    print(f"Position encoding output shape: {output.shape}")
    print(f"Position encoding parameters: {sum(p.numel() for p in pos_enc.parameters()):,}")
    
    print("\nTesting ALiBi Positional Bias...")
    alibi = ALiBiPositionalBias(num_heads, max_position_embeddings=131072)
    bias = alibi(seq_len, torch.device('cpu'))
    
    print(f"ALiBi bias shape: {bias.shape}")
    print(f"ALiBi parameters: {sum(p.numel() for p in alibi.parameters()):,}")
    
    return True


if __name__ == "__main__":
    test_embeddings()