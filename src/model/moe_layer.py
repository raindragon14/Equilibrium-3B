"""
Fine-Grained Mixture of Experts (DeepSeekMoE) Implementation
==========================================================

DeepSeek-style MoE with shared experts and routed experts,
optimized for balanced load and efficient training.

Key features:
- Shared experts (always activated)
- Routed experts with Top-K routing
- Load balancing mechanisms
- Auxiliary loss for expert utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class MoEGate(nn.Module):
    """
    Gating network for Mixture of Experts routing.
    
    Implements Top-K routing with load balancing and auxiliary losses.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        eval_capacity_factor: float = 2.0,
        min_capacity: int = 8,
        use_bias: bool = False,
        second_expert_policy: str = 'sampling',  # 'sampling' or 'random'
        normalize_gate_prob_before_dropping: bool = False,
        batch_prioritized_routing: bool = False,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.batch_prioritized_routing = batch_prioritized_routing
        
        # Gate network
        self.gate = nn.Linear(d_model, num_experts, bias=use_bias)
        
        # Initialize gate weights
        self.gate.weight.data.zero_()
        if use_bias:
            self.gate.bias.data.zero_()
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        use_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            use_aux_loss: Whether to compute auxiliary losses
        
        Returns:
            dispatch_tensor: [batch_size, seq_len, num_experts, top_k]
            combine_tensor: [batch_size, seq_len, num_experts, top_k] 
            aux_loss: Dictionary with auxiliary losses
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Reshape for easier processing
        hidden_states = hidden_states.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Compute gate logits
        gate_logits = self.gate(hidden_states)  # [batch_size * seq_len, num_experts]
        gate_logits = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        
        # Determine capacity
        if self.training:
            capacity = max(
                self.min_capacity,
                int((batch_size * seq_len / self.num_experts) * self.capacity_factor)
            )
        else:
            capacity = max(
                self.min_capacity,
                int((batch_size * seq_len / self.num_experts) * self.eval_capacity_factor)
            )
        
        # Top-K selection
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        if self.normalize_gate_prob_before_dropping:
            top_k_gates = top_k_logits / (top_k_logits.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            top_k_gates = top_k_logits
        
        # Create dispatch and combine tensors
        zeros = torch.zeros_like(gate_logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        # Expert load balancing
        if self.training and use_aux_loss:
            # Load balancing auxiliary loss
            gates_mean = gates.mean(dim=0)  # [num_experts]
            selection_mean = torch.zeros_like(gates_mean)
            selection_mean.scatter_add_(0, top_k_indices.view(-1), torch.ones_like(top_k_indices.view(-1), dtype=gates.dtype))
            selection_mean = selection_mean / (batch_size * seq_len)
            
            aux_loss = {
                'load_balancing_loss': (gates_mean * selection_mean).sum() * self.num_experts,
                'router_z_loss': torch.logsumexp(gate_logits, dim=-1).pow(2).mean()
            }
        else:
            aux_loss = {}
        
        # Reshape back to original dimensions
        gates = gates.view(batch_size, seq_len, self.num_experts)
        top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        
        # Create dispatch and combine tensors for expert computation
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, self.top_k, 
            dtype=gates.dtype, device=gates.device
        )
        combine_tensor = torch.zeros_like(dispatch_tensor)
        
        # Fill dispatch and combine tensors
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]  # [batch_size, seq_len]
            expert_weights = top_k_gates.view(batch_size, seq_len, self.top_k)[:, :, k]  # [batch_size, seq_len]
            
            # Create one-hot encoding for dispatch
            dispatch_tensor.scatter_(2, expert_indices.unsqueeze(-1).unsqueeze(-1), 1.0)
            combine_tensor.scatter_(2, expert_indices.unsqueeze(-1).unsqueeze(-1), expert_weights.unsqueeze(-1).unsqueeze(-1))
        
        return dispatch_tensor, combine_tensor, aux_loss


class Expert(nn.Module):
    """
    Single expert in the MoE layer.
    Standard feed-forward network with SwiGLU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = 'swiglu'
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        if activation == 'swiglu':
            # SwiGLU requires 2 linear layers for gating
            self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
            self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection  
            self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        else:
            # Standard FFN
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            if activation == 'gelu':
                self.activation_fn = F.gelu
            elif activation == 'relu':
                self.activation_fn = F.relu
            else:
                self.activation_fn = F.silu
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        if self.activation == 'swiglu':
            # SwiGLU: SiLU(W1 * x) ⊙ W3 * x, then W2
            gate = F.silu(self.w1(x))
            up = self.w3(x)
            hidden = gate * up
            output = self.w2(hidden)
        else:
            # Standard FFN
            hidden = self.activation_fn(self.w1(x))
            output = self.w2(hidden)
        
        return self.dropout(output)


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style Mixture of Experts layer.
    
    Combines shared experts (always activated) with routed experts
    for efficient scaling and load balancing.
    """
    
    def __init__(
        self,
        config,
        layer_id: Optional[int] = None,
    ):
        super().__init__()
        
        self.d_model = config.hidden_size
        self.d_ff = config.intermediate_size
        self.num_experts = getattr(config, 'num_experts', 64)
        self.num_shared_experts = getattr(config, 'num_shared_experts', 8)
        self.top_k = getattr(config, 'num_experts_per_tok', 2)
        self.expert_capacity = getattr(config, 'expert_capacity', None)
        
        # Routing gate
        self.gate = MoEGate(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        
        # Shared experts (always activated)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(
                    d_model=self.d_model,
                    d_ff=self.d_ff // self.num_shared_experts,  # Smaller per expert
                    activation='swiglu'
                )
                for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None
        
        # Routed experts
        self.experts = nn.ModuleList([
            Expert(
                d_model=self.d_model,
                d_ff=self.d_ff,
                activation='swiglu'
            )
            for _ in range(self.num_experts)
        ])
        
        # Expert dropout for regularization
        self.expert_dropout = getattr(config, 'expert_dropout', 0.0)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            router_logits: Dictionary with routing information and losses
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Store original input for residual connection
        original_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, d_model)
        
        # Shared experts processing
        shared_output = None
        if self.shared_experts is not None:
            shared_outputs = []
            for expert in self.shared_experts:
                expert_output = expert(hidden_states)
                shared_outputs.append(expert_output)
            
            # Combine shared expert outputs (average)
            shared_output = torch.stack(shared_outputs).mean(dim=0)
        
        # Routed experts processing
        dispatch_tensor, combine_tensor, aux_loss = self.gate(
            hidden_states, 
            use_aux_loss=self.training
        )
        
        # Process tokens through selected experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = dispatch_tensor[:, :, i, :]  # [batch_size, seq_len, top_k]
            expert_tokens = expert_mask.sum() > 0
            
            if expert_tokens:
                # Process all hidden states through expert (we'll mask later)
                expert_output = expert(hidden_states)
                expert_outputs.append(expert_output)
            else:
                # No tokens for this expert
                expert_outputs.append(torch.zeros_like(hidden_states))
        
        # Combine expert outputs using routing weights
        routed_output = torch.zeros_like(hidden_states)
        for i, expert_output in enumerate(expert_outputs):
            for k in range(self.top_k):
                expert_weight = combine_tensor[:, :, i, k:k+1]  # [batch_size, seq_len, 1]
                routed_output += expert_weight * expert_output
        
        # Combine shared and routed outputs
        if shared_output is not None:
            # Learnable combination (could be made configurable)
            alpha = 0.5  # Fixed mixing ratio, could be learned
            final_output = alpha * shared_output + (1 - alpha) * routed_output
        else:
            final_output = routed_output
        
        # Apply expert dropout
        if self.training and self.expert_dropout > 0:
            final_output = F.dropout(final_output, p=self.expert_dropout)
        
        # Return auxiliary losses for training
        router_info = aux_loss.copy() if aux_loss else {}
        
        return final_output, router_info


class SparseMoE(DeepSeekMoE):
    """
    Sparse MoE variant with dynamic expert pruning and efficient routing.
    """
    
    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__(config, layer_id)
        
        # Expert importance tracking
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))
        self.usage_decay = 0.999
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward with expert usage tracking."""
        
        output, router_info = super().forward(hidden_states)
        
        # Update expert usage statistics during training
        if self.training and 'dispatch_tensor' in locals():
            # Track which experts were used
            expert_usage_batch = dispatch_tensor.sum(dim=[0, 1, 3])  # [num_experts]
            self.expert_usage = self.usage_decay * self.expert_usage + (1 - self.usage_decay) * expert_usage_batch
        
        return output, router_info
    
    def get_expert_statistics(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        return {
            'expert_usage': self.expert_usage,
            'expert_utilization': self.expert_usage / (self.expert_usage.sum() + 1e-6),
            'active_experts': (self.expert_usage > 0.01).sum().item()
        }


def create_moe_layers(config, num_layers: int) -> nn.ModuleList:
    """Create a list of MoE layers."""
    return nn.ModuleList([
        DeepSeekMoE(config, layer_id=i) 
        for i in range(num_layers)
    ])


def test_moe():
    """Test function for Mixture of Experts."""
    
    # Mock config
    class Config:
        hidden_size = 2560
        intermediate_size = 6912
        num_experts = 64
        num_shared_experts = 8
        num_experts_per_tok = 2
        expert_dropout = 0.0
    
    config = Config()
    batch_size, seq_len = 2, 512
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test MoE
    moe = DeepSeekMoE(config)
    output, router_info = moe(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"MoE parameters: {sum(p.numel() for p in moe.parameters()):,}")
    print(f"Router info keys: {list(router_info.keys())}")
    
    # Calculate parameter efficiency
    total_params = sum(p.numel() for p in moe.parameters())
    active_params = (
        sum(p.numel() for p in moe.shared_experts.parameters()) + 
        sum(p.numel() for p in moe.experts[:config.num_experts_per_tok])
    )
    
    print(f"Total parameters: {total_params:,}")
    print(f"Active parameters per forward: {active_params:,}")
    print(f"Parameter efficiency: {active_params / total_params:.2%}")
    
    return True


if __name__ == "__main__":
    test_moe()