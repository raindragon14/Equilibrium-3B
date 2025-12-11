"""
Mamba-2 Block Implementation
===========================

Structured State Space Model with selective scan mechanism,
optimized for long context and efficient processing.

Based on Mamba-2: Linear-Time Sequence Modeling with Selective State Spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using fallback implementation.")


class SelectiveScan(nn.Module):
    """
    Selective Scan mechanism for Mamba-2.
    
    Implements the core selective state space operation:
    y = C @ (A^L x_0 + sum_{i=0}^{L-1} A^{L-1-i} B @ x_i)
    """
    
    def __init__(self, d_model: int, d_state: int = 128, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Parameter projections
        self.x_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(d_model, self.d_inner, bias=True)
        
        # State space parameters (A, B, C)
        # A: Diagonal state transition matrix
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, d_state)))
        
        # B and C projections
        self.B_proj = nn.Linear(d_model, self.d_inner * d_state, bias=False)
        self.C_proj = nn.Linear(d_model, self.d_inner * d_state, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Delta projection with initialization
        self.dt_proj.bias.data = torch.rand(self.d_inner) + 1.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_inner = self.x_proj(x)  # (batch, seq_len, d_inner)
        
        # Compute delta (discrete time step)
        delta = F.softplus(self.dt_proj(x))  # (batch, seq_len, d_inner)
        
        # Compute A matrix (discrete)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_discrete = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch, seq_len, d_inner, d_state)
        
        # Compute B and C matrices
        B = self.B_proj(x).view(batch_size, seq_len, self.d_inner, self.d_state)  # (batch, seq_len, d_inner, d_state)
        C = self.C_proj(x).view(batch_size, seq_len, self.d_inner, self.d_state)  # (batch, seq_len, d_inner, d_state)
        
        # Selective scan operation
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(seq_len):
            # Update state: h = A * h + B * x
            h = A_discrete[:, i] * h + B[:, i] * x_inner[:, i:i+1].transpose(-1, -2)
            
            # Compute output: y = C * h
            y = torch.sum(C[:, i] * h, dim=-1)  # (batch, d_inner)
            outputs.append(y)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Apply SiLU activation and output projection
        y = F.silu(y)
        output = self.out_proj(y)
        
        return output


class MambaBlock(nn.Module):
    """
    Mamba-2 Block with structured state space model.
    
    Combines selective scan with efficient implementation optimizations
    for long sequence modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        
        # Use official Mamba implementation if available and requested
        if MAMBA_AVAILABLE and use_fast_path:
            try:
                self.mamba = Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                )
                self.use_native = True
            except Exception as e:
                print(f"Warning: Could not initialize native Mamba2, using fallback: {e}")
                self.use_native = False
                self.selective_scan = SelectiveScan(d_model, d_state, expand)
        else:
            self.use_native = False
            self.selective_scan = SelectiveScan(d_model, d_state, expand)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner,  # Depthwise convolution
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Layer normalization
        x_norm = self.norm(x)
        
        if self.use_native:
            # Use native Mamba implementation
            output = self.mamba(x_norm)
        else:
            # Use fallback implementation
            # Apply 1D convolution for local dependencies
            x_conv = x_norm.transpose(1, 2)  # (batch, d_model, seq_len)
            x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim to original length
            x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)
            
            # Apply selective scan
            output = self.selective_scan(x_conv)
        
        return output


class MambaResidualBlock(nn.Module):
    """
    Mamba block with residual connection and optional gating.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dropout: float = 0.0,
        use_gating: bool = True,
    ):
        super().__init__()
        
        self.mamba_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Optional gating mechanism
        if use_gating:
            self.gate = nn.Linear(d_model, d_model)
            self.gate_activation = nn.Sigmoid()
        else:
            self.gate = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        
        residual = x
        
        # Mamba processing
        mamba_output = self.mamba_block(x)
        mamba_output = self.dropout(mamba_output)
        
        # Optional gating
        if self.gate is not None:
            gate_values = self.gate_activation(self.gate(x))
            mamba_output = gate_values * mamba_output
        
        # Residual connection
        return residual + mamba_output


# Utility functions for Mamba blocks

def create_mamba_stack(
    d_model: int,
    num_layers: int,
    d_state: int = 128,
    expand: int = 2,
    headdim: int = 64,
    dropout: float = 0.0,
) -> nn.ModuleList:
    """Create a stack of Mamba blocks."""
    
    return nn.ModuleList([
        MambaResidualBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])


def test_mamba_block():
    """Test function for Mamba block."""
    
    batch_size, seq_len, d_model = 2, 512, 768
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test basic Mamba block
    mamba = MambaBlock(d_model=d_model)
    output = mamba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in mamba.parameters()):,}")
    
    # Test residual block
    mamba_res = MambaResidualBlock(d_model=d_model)
    output_res = mamba_res(x)
    
    print(f"Residual output shape: {output_res.shape}")
    print(f"Residual parameters: {sum(p.numel() for p in mamba_res.parameters()):,}")
    
    return True


if __name__ == "__main__":
    test_mamba_block()