"""
COSMOS Optimizers: Muon + SOAP + Hybrid Implementation
=====================================================

Schedule-free optimizers combining:
- Muon: Matrix-wise preconditioning for 2D parameters (weights)
- SOAP: Shampoo-style optimization for high-dimensional parameters
- COSMOS: Unified hybrid approach

Based on "Schedule-Free Optimization in LLM Training" and latest 2025 research.
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
from typing import Any, Dict, Optional, Tuple, Union, List
import warnings


class Muon(Optimizer):
    """
    Muon optimizer for matrix parameters (weights).
    
    Implements matrix-wise preconditioning with efficient Newton-Schulz iteration
    for computing matrix square roots.
    
    Based on "Shampoo: Preconditioned Stochastic Tensor Optimization" with
    optimizations for LLM training.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend: str = "newtonschulz5",  # or "newtonschulz3", "svd"
        backend_steps: int = 5,
        rank_deficiency_threshold: float = 1e-10,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
            rank_deficiency_threshold=rank_deficiency_threshold,
        )
        super().__init__(params, defaults)
    
    def _newton_schulz_iteration(self, A: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Compute A^(-1/2) using Newton-Schulz iteration.
        
        More numerically stable than SVD for large matrices.
        """
        device, dtype = A.device, A.dtype
        
        # Initialize with scaled identity
        I = torch.eye(A.size(0), device=device, dtype=dtype)
        
        # Compute initial scaling
        trace_A = torch.trace(A)
        if trace_A <= 0:
            return I
        
        # Initial guess: Y_0 = (3/2) * (trace(A)/n) * I / A
        alpha = 1.5 * trace_A / A.size(0)
        Y = alpha * torch.inverse(A + self.defaults['rank_deficiency_threshold'] * I)
        
        # Newton-Schulz iterations: Y_{k+1} = Y_k * (3*I - A*Y_k*Y_k) / 2
        for _ in range(steps):
            AY = torch.matmul(A, Y)
            AYY = torch.matmul(AY, Y)
            Y = 0.5 * torch.matmul(Y, 3.0 * I - AYY)
        
        return Y
    
    def _svd_inverse_sqrt(self, A: torch.Tensor) -> torch.Tensor:
        """Compute A^(-1/2) using SVD (more accurate but slower)."""
        U, S, V = torch.svd(A)
        
        # Threshold small eigenvalues
        threshold = self.defaults['rank_deficiency_threshold']
        S_inv_sqrt = torch.where(S > threshold, 1.0 / torch.sqrt(S), 0.0)
        
        return torch.matmul(U * S_inv_sqrt.unsqueeze(0), V.t())
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if p.dim() >= 2:  # Only for matrix parameters
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                        # Initialize preconditioning matrices
                        state['left_preconditioner'] = torch.eye(p.size(0), device=p.device, dtype=grad.dtype)
                        state['right_preconditioner'] = torch.eye(p.size(-1), device=p.device, dtype=grad.dtype)
                        state['G_left'] = torch.zeros(p.size(0), p.size(0), device=p.device, dtype=grad.dtype)
                        state['G_right'] = torch.zeros(p.size(-1), p.size(-1), device=p.device, dtype=grad.dtype)
                    else:
                        # For 1D parameters, use simple momentum
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                if p.dim() >= 2:  # Matrix parameters - use Muon
                    momentum_buf = state['momentum_buffer']
                    
                    # Reshape gradient for matrix operations
                    if p.dim() > 2:
                        original_shape = p.shape
                        grad = grad.view(grad.size(0), -1)
                        p_reshaped = p.data.view(p.size(0), -1)
                        momentum_buf = momentum_buf.view(momentum_buf.size(0), -1)
                    else:
                        original_shape = None
                        p_reshaped = p.data
                    
                    # Update second moment estimates (Fisher information approximation)
                    state['G_left'] += torch.matmul(grad, grad.t())
                    state['G_right'] += torch.matmul(grad.t(), grad)
                    
                    # Compute preconditioning matrices
                    if group['backend'] == 'svd':
                        left_prec = self._svd_inverse_sqrt(state['G_left'] / state['step'])
                        right_prec = self._svd_inverse_sqrt(state['G_right'] / state['step'])
                    else:
                        steps = group['backend_steps']
                        left_prec = self._newton_schulz_iteration(state['G_left'] / state['step'], steps)
                        right_prec = self._newton_schulz_iteration(state['G_right'] / state['step'], steps)
                    
                    # Apply preconditioning: P^(-1/2) @ grad @ Q^(-1/2)
                    preconditioned_grad = torch.matmul(torch.matmul(left_prec, grad), right_prec)
                    
                    # Momentum update
                    momentum_buf.mul_(group['momentum']).add_(preconditioned_grad)
                    
                    if group['nesterov']:
                        update = preconditioned_grad + group['momentum'] * momentum_buf
                    else:
                        update = momentum_buf
                    
                    # Reshape back if necessary
                    if original_shape is not None:
                        update = update.view(original_shape)
                        momentum_buf = momentum_buf.view(original_shape)
                    
                    # Update parameters
                    p.data.add_(update, alpha=-group['lr'])
                    
                else:  # 1D parameters - use simple momentum
                    momentum_buf = state['momentum_buffer']
                    momentum_buf.mul_(group['momentum']).add_(grad)
                    
                    if group['nesterov']:
                        update = grad + group['momentum'] * momentum_buf
                    else:
                        update = momentum_buf
                    
                    p.data.add_(update, alpha=-group['lr'])
        
        return loss


class SOAP(Optimizer):
    """
    SOAP optimizer for high-dimensional parameters.
    
    Implements Shampoo-style optimization with efficient approximations
    for very high-dimensional parameter spaces (embeddings, large linear layers).
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        shampoo_beta: float = 0.99,
        update_freq: int = 100,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=weight_decay,
            shampoo_beta=shampoo_beta,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    # Shampoo preconditioning for high-dim parameters
                    if p.numel() > 10000:  # Threshold for using Shampoo
                        # For very large parameters, use factorized approximation
                        if p.dim() >= 2:
                            state['left_stats'] = torch.eye(p.size(0), device=p.device, dtype=grad.dtype) * group['eps']
                            state['right_stats'] = torch.eye(p.size(-1), device=p.device, dtype=grad.dtype) * group['eps']
                        else:
                            state['diag_stats'] = torch.ones_like(p.data) * group['eps']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Shampoo preconditioning for large parameters
                if 'left_stats' in state and state['step'] % group['update_freq'] == 0:
                    # Update statistics matrices
                    if p.dim() >= 2:
                        grad_2d = grad.view(grad.size(0), -1)
                        state['left_stats'].mul_(group['shampoo_beta']).addmm_(
                            grad_2d, grad_2d.t(), alpha=1 - group['shampoo_beta']
                        )
                        state['right_stats'].mul_(group['shampoo_beta']).addmm_(
                            grad_2d.t(), grad_2d, alpha=1 - group['shampoo_beta']
                        )
                        
                        # Compute preconditioned update
                        try:
                            left_inv = torch.inverse(state['left_stats'] + group['eps'] * torch.eye(p.size(0), device=p.device))
                            right_inv = torch.inverse(state['right_stats'] + group['eps'] * torch.eye(p.size(-1), device=p.device))
                            
                            preconditioned_grad = torch.matmul(torch.matmul(left_inv, grad_2d), right_inv)
                            preconditioned_grad = preconditioned_grad.view_as(grad)
                        except:
                            # Fallback to diagonal preconditioning
                            preconditioned_grad = grad / (exp_avg_sq.sqrt() + group['eps'])
                    else:
                        state['diag_stats'].mul_(group['shampoo_beta']).addcmul_(
                            grad, grad, value=1 - group['shampoo_beta']
                        )
                        preconditioned_grad = grad / (state['diag_stats'].sqrt() + group['eps'])
                else:
                    # Standard Adam-style update
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    preconditioned_grad = exp_avg / bias_correction1 / denom
                
                # Update parameters
                p.data.add_(preconditioned_grad, alpha=-group['lr'])
        
        return loss


class COSMOS(Optimizer):
    """
    COSMOS: Unified hybrid optimizer combining Muon and SOAP.
    
    Automatically selects appropriate optimizer based on parameter characteristics:
    - Muon for 2D matrix parameters (most weight matrices)
    - SOAP for high-dimensional parameters (embeddings, very large matrices)
    - Adam for small 1D parameters (biases, layer norms)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        muon_lr: float = 0.02,
        soap_lr: float = 0.001,
        adam_lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        muon_momentum: float = 0.95,
        matrix_threshold: int = 1000,  # Threshold for using matrix optimizers
        **kwargs
    ):
        
        # Validate parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(
            lr=lr,
            muon_lr=muon_lr,
            soap_lr=soap_lr, 
            adam_lr=adam_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            muon_momentum=muon_momentum,
            matrix_threshold=matrix_threshold,
        )
        super().__init__(params, defaults)
        
        # Initialize sub-optimizers
        self._init_sub_optimizers()
    
    def _init_sub_optimizers(self):
        """Initialize sub-optimizers based on parameter characteristics."""
        
        muon_params = []
        soap_params = []
        adam_params = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() >= 2 and p.numel() <= group['matrix_threshold'] ** 2:
                    # Medium-sized 2D parameters -> Muon
                    muon_params.append(p)
                elif p.numel() > group['matrix_threshold']:
                    # Large parameters -> SOAP
                    soap_params.append(p)
                else:
                    # Small 1D parameters -> Adam
                    adam_params.append(p)
        
        # Create sub-optimizers
        self.muon_optimizer = Muon(
            muon_params,
            lr=group['muon_lr'],
            momentum=group['muon_momentum'],
        ) if muon_params else None
        
        self.soap_optimizer = SOAP(
            soap_params,
            lr=group['soap_lr'],
            betas=group['betas'],
            eps=group['eps'],
            weight_decay=group['weight_decay'],
        ) if soap_params else None
        
        self.adam_optimizer = torch.optim.AdamW(
            adam_params,
            lr=group['adam_lr'],
            betas=group['betas'],
            eps=group['eps'],
            weight_decay=group['weight_decay'],
        ) if adam_params else None
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step using appropriate sub-optimizer."""
        
        losses = []
        
        # Step each sub-optimizer
        if self.muon_optimizer is not None:
            loss = self.muon_optimizer.step(closure)
            if loss is not None:
                losses.append(loss)
        
        if self.soap_optimizer is not None:
            loss = self.soap_optimizer.step(closure)
            if loss is not None:
                losses.append(loss)
        
        if self.adam_optimizer is not None:
            loss = self.adam_optimizer.step(closure)
            if loss is not None:
                losses.append(loss)
        
        return losses[0] if losses else None
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all sub-optimizers."""
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none)
        if self.soap_optimizer is not None:
            self.soap_optimizer.zero_grad(set_to_none)
        if self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get state dictionary from all sub-optimizers."""
        state_dict = {
            'param_groups': self.param_groups,
        }
        
        if self.muon_optimizer is not None:
            state_dict['muon'] = self.muon_optimizer.state_dict()
        if self.soap_optimizer is not None:
            state_dict['soap'] = self.soap_optimizer.state_dict()
        if self.adam_optimizer is not None:
            state_dict['adam'] = self.adam_optimizer.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load state dictionary for all sub-optimizers."""
        self.param_groups = state_dict['param_groups']
        
        if 'muon' in state_dict and self.muon_optimizer is not None:
            self.muon_optimizer.load_state_dict(state_dict['muon'])
        if 'soap' in state_dict and self.soap_optimizer is not None:
            self.soap_optimizer.load_state_dict(state_dict['soap'])
        if 'adam' in state_dict and self.adam_optimizer is not None:
            self.adam_optimizer.load_state_dict(state_dict['adam'])


def test_optimizers():
    """Test function for COSMOS optimizers."""
    
    # Create test parameters of different sizes
    params = [
        torch.randn(512, 2560, requires_grad=True),      # Medium matrix -> Muon
        torch.randn(32000, 2560, requires_grad=True),    # Large matrix -> SOAP  
        torch.randn(2560, requires_grad=True),           # 1D parameter -> Adam
    ]
    
    print("Testing individual optimizers...")
    
    # Test Muon
    muon = Muon([params[0]], lr=0.02)
    print(f"Muon optimizer created for parameter shape: {params[0].shape}")
    
    # Test SOAP
    soap = SOAP([params[1]], lr=0.001)
    print(f"SOAP optimizer created for parameter shape: {params[1].shape}")
    
    # Test COSMOS
    cosmos = COSMOS(params, lr=0.01)
    print(f"COSMOS optimizer created for {len(params)} parameters")
    
    # Test optimization step
    for i, p in enumerate(params):
        p.grad = torch.randn_like(p) * 0.01  # Simulate gradients
    
    loss = cosmos.step()
    print(f"COSMOS step completed, loss: {loss}")
    
    # Check parameter counts for each sub-optimizer
    muon_params = len(list(cosmos.muon_optimizer.param_groups[0]['params'])) if cosmos.muon_optimizer else 0
    soap_params = len(list(cosmos.soap_optimizer.param_groups[0]['params'])) if cosmos.soap_optimizer else 0
    adam_params = len(list(cosmos.adam_optimizer.param_groups[0]['params'])) if cosmos.adam_optimizer else 0
    
    print(f"Parameter distribution - Muon: {muon_params}, SOAP: {soap_params}, Adam: {adam_params}")
    
    return True


if __name__ == "__main__":
    test_optimizers()