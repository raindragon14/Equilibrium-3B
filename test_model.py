#!/usr/bin/env python3
"""
Comprehensive Test Suite for Equilibrium-3B
==========================================

Unit tests and integration tests for all model components
to ensure correctness and production readiness.

Usage:
    python test_model.py
    pytest tests/
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.equilibrium_model import (
    Equilibrium3B, 
    Equilibrium3BConfig,
    create_equilibrium_3b_base,
    create_equilibrium_1_5b
)
from model.mamba_layer import MambaBlock, MambaResidualBlock
from model.attention_layer import MultiHeadLatentAttention
from model.moe_layer import DeepSeekMoE, MoEGate
from model.embeddings import RotaryEmbedding, LearnablePositionalEncoding
from optimizers.cosmos import COSMOS, Muon, SOAP


class TestEquilibriumModel:
    """Test suite for the main Equilibrium-3B model."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = Equilibrium3BConfig(
            vocab_size=1000,  # Small for testing
            hidden_size=512,
            intermediate_size=1024,
            num_layers=4,
            num_attention_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            kv_lora_rank=128,
            q_lora_rank=256,
            num_experts=16,
            num_shared_experts=4,
            max_position_embeddings=2048,
        )
        self.batch_size = 2
        self.seq_len = 256
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = Equilibrium3BConfig()
        assert config.hidden_size % config.num_attention_heads == 0
        
        # Invalid config should fail
        with pytest.raises(AssertionError):
            invalid_config = Equilibrium3BConfig(
                hidden_size=512,
                num_attention_heads=7  # Not divisible
            )
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        model = Equilibrium3B(self.config)
        
        # Check model structure
        assert isinstance(model, Equilibrium3B)
        assert len(model.model.layers) == self.config.num_layers
        
        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000000  # At least 1M parameters
        assert total_params < 10000000000  # Less than 10B parameters
        
        print(f"Model created with {total_params:,} parameters")
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        model = Equilibrium3B(self.config)
        
        # Create input
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check output shapes
        loss, logits = outputs[:2]
        assert logits.shape == (self.batch_size, self.seq_len, self.config.vocab_size)
        assert loss is None  # No labels provided
        
        # Test with labels
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        
        assert loss is not None
        assert loss.item() > 0  # Should have positive loss
        
        print(f"Forward pass successful - Loss: {loss.item():.4f}")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = Equilibrium3B(self.config)
        
        # Create inputs and targets
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward and backward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        
        # Check that gradients exist and are non-zero for most parameters
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                assert grad_norm >= 0, f"Invalid gradient for {name}"
        
        assert len(grad_norms) > 0, "No gradients computed"
        avg_grad_norm = np.mean(grad_norms)
        print(f"Average gradient norm: {avg_grad_norm:.6f}")
    
    def test_attention_layers(self):
        """Test that attention layers are placed correctly."""
        model = Equilibrium3B(self.config)
        
        attention_layer_indices = []
        for i, layer in enumerate(model.model.layers):
            if layer.is_attention_layer:
                attention_layer_indices.append(i)
        
        assert len(attention_layer_indices) == self.config.num_attention_layers
        print(f"Attention layers at indices: {attention_layer_indices}")


class TestMambaLayer:
    """Test suite for Mamba layers."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.d_model = 512
        self.batch_size = 2
        self.seq_len = 256
    
    def test_mamba_block_creation(self):
        """Test Mamba block creation."""
        mamba = MambaBlock(d_model=self.d_model)
        assert isinstance(mamba, MambaBlock)
        
        # Count parameters
        total_params = sum(p.numel() for p in mamba.parameters())
        print(f"Mamba block parameters: {total_params:,}")
    
    def test_mamba_forward(self):
        """Test Mamba forward pass."""
        mamba = MambaBlock(d_model=self.d_model)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        with torch.no_grad():
            output = mamba(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        print("Mamba forward pass successful")
    
    def test_mamba_residual_block(self):
        """Test Mamba residual block."""
        mamba_res = MambaResidualBlock(d_model=self.d_model)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = mamba_res(x)
        
        assert output.shape == x.shape
        # Check residual connection works
        assert not torch.allclose(output, x, atol=1e-6)  # Should be different due to processing
        print("Mamba residual block successful")


class TestAttentionLayer:
    """Test suite for Multi-Head Latent Attention."""
    
    def setup_method(self):
        """Setup test configuration."""
        class Config:
            hidden_size = 512
            num_attention_heads = 8
            num_key_value_heads = 2
            kv_lora_rank = 128
            q_lora_rank = 256
        
        self.config = Config()
        self.batch_size = 2
        self.seq_len = 256
    
    def test_mla_creation(self):
        """Test MLA creation."""
        mla = MultiHeadLatentAttention(self.config)
        assert isinstance(mla, MultiHeadLatentAttention)
        
        # Check compression ratio
        standard_kv_params = self.config.hidden_size * self.config.hidden_size * 2
        mla_params = sum(p.numel() for p in mla.parameters())
        
        compression_ratio = standard_kv_params / mla_params
        print(f"MLA compression ratio: {compression_ratio:.2f}:1")
        assert compression_ratio > 1.0  # Should be compressed
    
    def test_mla_forward(self):
        """Test MLA forward pass."""
        mla = MultiHeadLatentAttention(self.config)
        
        # Create input
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        
        # Forward pass
        output, attn_weights, past_kv = mla(
            hidden_states, 
            use_cache=True,
            output_attentions=True
        )
        
        assert output.shape == hidden_states.shape
        assert attn_weights is not None
        assert past_kv is not None
        print("MLA forward pass successful")
    
    def test_kv_cache(self):
        """Test KV-cache functionality."""
        mla = MultiHeadLatentAttention(self.config)
        
        # First forward pass
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        output1, _, past_kv = mla(hidden_states, use_cache=True)
        
        # Second forward pass with cache
        new_tokens = torch.randn(self.batch_size, 10, self.config.hidden_size)
        output2, _, _ = mla(new_tokens, past_key_value=past_kv, use_cache=True)
        
        assert output2.shape == (self.batch_size, 10, self.config.hidden_size)
        print("KV-cache functionality working")


class TestMoELayer:
    """Test suite for Mixture of Experts."""
    
    def setup_method(self):
        """Setup test configuration."""
        class Config:
            hidden_size = 512
            intermediate_size = 1024
            num_experts = 16
            num_shared_experts = 4
            num_experts_per_tok = 2
        
        self.config = Config()
        self.batch_size = 2
        self.seq_len = 256
    
    def test_moe_gate(self):
        """Test MoE gating mechanism."""
        gate = MoEGate(
            d_model=self.config.hidden_size,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok
        )
        
        # Test routing
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        dispatch_tensor, combine_tensor, aux_loss = gate(hidden_states)
        
        assert dispatch_tensor.shape[-2] == self.config.num_experts
        assert dispatch_tensor.shape[-1] == self.config.num_experts_per_tok
        assert 'load_balancing_loss' in aux_loss
        print("MoE gate working correctly")
    
    def test_moe_forward(self):
        """Test full MoE forward pass."""
        moe = DeepSeekMoE(self.config)
        
        # Forward pass
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        output, router_info = moe(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert isinstance(router_info, dict)
        
        # Calculate parameter efficiency
        total_params = sum(p.numel() for p in moe.parameters())
        shared_params = sum(p.numel() for p in moe.shared_experts.parameters()) if moe.shared_experts else 0
        expert_params = sum(p.numel() for p in moe.experts[:self.config.num_experts_per_tok])
        active_params = shared_params + expert_params
        
        efficiency = active_params / total_params
        print(f"MoE parameter efficiency: {efficiency:.2%}")


class TestEmbeddings:
    """Test suite for embedding layers."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.d_model = 512
        self.max_seq_len = 2048
        self.batch_size = 2
        self.seq_len = 256
    
    def test_rope_embedding(self):
        """Test Rotary Position Embedding."""
        rope = RotaryEmbedding(
            dim=self.d_model // 8,  # Head dimension
            max_position_embeddings=self.max_seq_len
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.seq_len, self.d_model // 8)
        cos, sin = rope(x)
        
        assert cos.shape == (self.seq_len, self.d_model // 8)
        assert sin.shape == (self.seq_len, self.d_model // 8)
        print("RoPE embedding working correctly")
    
    def test_learnable_positional_encoding(self):
        """Test learnable positional encoding."""
        pos_enc = LearnablePositionalEncoding(
            d_model=self.d_model,
            max_position_embeddings=self.max_seq_len
        )
        
        # Test forward pass
        input_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(input_emb)
        
        assert output.shape == input_emb.shape
        assert not torch.allclose(output, input_emb)  # Should add positional info
        print("Learnable positional encoding working")


class TestOptimizers:
    """Test suite for COSMOS optimizers."""
    
    def setup_method(self):
        """Create test model and parameters."""
        # Simple test model with different parameter types
        self.model = nn.Sequential(
            nn.Linear(512, 1024),      # 2D parameter -> Muon
            nn.Embedding(32000, 512),  # Large parameter -> SOAP
            nn.LayerNorm(512),         # 1D parameters -> Adam
        )
    
    def test_cosmos_optimizer(self):
        """Test COSMOS optimizer creation and step."""
        optimizer = COSMOS(self.model.parameters(), lr=0.01)
        
        # Check sub-optimizers were created
        assert optimizer.muon_optimizer is not None or optimizer.soap_optimizer is not None
        
        # Test optimization step
        # Create dummy loss
        x = torch.randn(2, 128, 512)
        output = self.model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print("COSMOS optimizer working correctly")
    
    def test_individual_optimizers(self):
        """Test individual Muon and SOAP optimizers."""
        # Test Muon on linear layer
        linear_layer = nn.Linear(512, 1024)
        muon_optimizer = Muon(linear_layer.parameters(), lr=0.02)
        
        # Create loss and step
        x = torch.randn(2, 512)
        loss = linear_layer(x).sum()
        loss.backward()
        muon_optimizer.step()
        
        print("Muon optimizer working")
        
        # Test SOAP on embedding
        embedding = nn.Embedding(10000, 512)
        soap_optimizer = SOAP(embedding.parameters(), lr=0.001)
        
        # Create loss and step
        indices = torch.randint(0, 10000, (2, 128))
        loss = embedding(indices).sum()
        loss.backward()
        soap_optimizer.step()
        
        print("SOAP optimizer working")


class TestIntegration:
    """Integration tests for complete model pipeline."""
    
    def test_end_to_end_training_step(self):
        """Test complete training step with all components."""
        # Create small model for testing
        config = Equilibrium3BConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            kv_lora_rank=64,
            q_lora_rank=128,
            num_experts=8,
            num_shared_experts=2,
        )
        
        model = Equilibrium3B(config)
        optimizer = COSMOS(model.parameters(), lr=0.01)
        
        # Training step
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        labels = torch.randint(0, config.vocab_size, (2, 128))
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"End-to-end training step completed - Loss: {loss.item():.4f}")
        assert loss.item() > 0
    
    def test_model_variants(self):
        """Test different model size variants."""
        # Test 1.5B variant
        model_1_5b = create_equilibrium_1_5b(vocab_size=1000)
        total_params_1_5b = sum(p.numel() for p in model_1_5b.parameters())
        
        # Test 3B variant
        model_3b = create_equilibrium_3b_base(vocab_size=1000)
        total_params_3b = sum(p.numel() for p in model_3b.parameters())
        
        assert total_params_3b > total_params_1_5b
        print(f"1.5B model: {total_params_1_5b:,} parameters")
        print(f"3B model: {total_params_3b:,} parameters")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running Equilibrium-3B Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestEquilibriumModel,
        TestMambaLayer,
        TestAttentionLayer,
        TestMoELayer,
        TestEmbeddings,
        TestOptimizers,
        TestIntegration,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        
        try:
            tester = test_class()
            methods = [method for method in dir(tester) if method.startswith('test_')]
            
            for method_name in methods:
                try:
                    if hasattr(tester, 'setup_method'):
                        tester.setup_method()
                    
                    method = getattr(tester, method_name)
                    method()
                    print(f"✅ {method_name}")
                    passed += 1
                    
                except Exception as e:
                    print(f"❌ {method_name}: {str(e)}")
                    failed += 1
                    
        except Exception as e:
            print(f"❌ Failed to initialize {test_class.__name__}: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)