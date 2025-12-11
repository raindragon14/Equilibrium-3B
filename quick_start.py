"""
Quick Start Guide for Equilibrium-3B Development
===============================================

This script provides a quick demonstration of the Equilibrium-3B model
capabilities and serves as a development playground.
"""

import torch
import torch.nn as nn
from src.model import Equilibrium3B, Equilibrium3BConfig
from src.optimizers import COSMOS
import time


def demo_model_creation():
    """Demonstrate model creation and basic properties."""
    print("🔧 Creating Equilibrium-3B Model...")
    
    # Create a development-sized config
    config = Equilibrium3BConfig(
        vocab_size=32000,
        hidden_size=1280,      # Smaller than full 2560
        intermediate_size=3456, # Smaller than full 6912
        num_layers=12,         # Smaller than full 24
        num_attention_layers=2, # Every 6th layer instead of 8th
        num_attention_heads=16, # Smaller than full 32
        num_key_value_heads=4,  # Smaller than full 8
        kv_lora_rank=256,      # Smaller than full 512
        q_lora_rank=768,       # Smaller than full 1536
        num_experts=32,        # Smaller than full 64
        num_shared_experts=4,  # Smaller than full 8
        max_position_embeddings=32768,  # 32k context for dev
    )
    
    model = Equilibrium3B(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params / 1e9:.2f}B parameters")
    
    return model, config


def demo_forward_pass(model, config):
    """Demonstrate forward pass and inference."""
    print("\n🚀 Testing Forward Pass...")
    
    batch_size, seq_len = 2, 512
    
    # Create sample input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        inference_time = time.time() - start_time
    
    loss, logits = outputs[:2]
    
    print(f"✅ Forward pass completed!")
    print(f"   Inference time: {inference_time:.3f}s")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test with labels for training
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss, logits = outputs[:2]
    print(f"   Training loss: {loss.item():.4f}")
    
    return outputs


def demo_optimizer():
    """Demonstrate COSMOS optimizer."""
    print("\n⚡ Testing COSMOS Optimizer...")
    
    # Create model
    model, config = demo_model_creation()
    
    # Create optimizer
    optimizer = COSMOS(
        model.parameters(),
        lr=0.01,
        muon_lr=0.02,
        soap_lr=0.001,
        weight_decay=0.01
    )
    
    print(f"✅ COSMOS optimizer created!")
    
    # Check parameter distribution
    muon_params = len(list(optimizer.muon_optimizer.param_groups[0]['params'])) if optimizer.muon_optimizer else 0
    soap_params = len(list(optimizer.soap_optimizer.param_groups[0]['params'])) if optimizer.soap_optimizer else 0
    adam_params = len(list(optimizer.adam_optimizer.param_groups[0]['params'])) if optimizer.adam_optimizer else 0
    
    print(f"   Muon parameters: {muon_params}")
    print(f"   SOAP parameters: {soap_params}")
    print(f"   Adam parameters: {adam_params}")
    
    return optimizer


def demo_training_step():
    """Demonstrate a complete training step."""
    print("\n🎯 Testing Training Step...")
    
    # Setup
    model, config = demo_model_creation()
    optimizer = COSMOS(model.parameters(), lr=0.01)
    
    # Training data
    batch_size, seq_len = 4, 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.train()
    
    # Training step
    start_time = time.time()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs[0]
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Optimization step
    optimizer.step()
    
    step_time = time.time() - start_time
    
    print(f"✅ Training step completed!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Step time: {step_time:.3f}s")
    print(f"   Throughput: {batch_size * seq_len / step_time:.0f} tokens/sec")


def demo_generation():
    """Demonstrate text generation capabilities."""
    print("\n📝 Testing Text Generation...")
    
    model, config = demo_model_creation()
    model.eval()
    
    # Simple greedy generation
    batch_size = 1
    prompt_len = 20
    max_new_tokens = 50
    
    # Create prompt
    input_ids = torch.randint(1, 100, (batch_size, prompt_len))  # Avoid padding token
    
    generated_ids = input_ids.clone()
    
    print(f"   Generating {max_new_tokens} new tokens...")
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids=generated_ids)
            logits = outputs[1]
            
            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if we hit end token (assuming token 2 is EOS)
            if next_token.item() == 2:
                break
    
    print(f"✅ Generation completed!")
    print(f"   Input length: {prompt_len}")
    print(f"   Generated length: {generated_ids.shape[1] - prompt_len}")
    print(f"   Total sequence length: {generated_ids.shape[1]}")


def demo_memory_usage():
    """Demonstrate memory usage analysis."""
    print("\n💾 Analyzing Memory Usage...")
    
    model, config = demo_model_creation()
    
    # Calculate model memory
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    print(f"✅ Memory analysis completed!")
    print(f"   Parameter memory: {param_size / 1024**2:.1f} MB")
    print(f"   Buffer memory: {buffer_size / 1024**2:.1f} MB")
    print(f"   Total model memory: {(param_size + buffer_size) / 1024**2:.1f} MB")
    
    # Estimate activation memory for different sequence lengths
    for seq_len in [1024, 8192, 32768]:
        batch_size = 1
        hidden_size = config.hidden_size
        
        # Rough estimation of activation memory
        activation_memory = (
            batch_size * seq_len * hidden_size * 4 *  # Hidden states (float32)
            config.num_layers * 3  # Roughly 3x for gradients and intermediate activations
        )
        
        print(f"   Estimated activation memory ({seq_len} tokens): {activation_memory / 1024**2:.1f} MB")


def main():
    """Run all demonstrations."""
    print("🌟 Equilibrium-3B Quick Start Demo")
    print("=" * 50)
    
    try:
        # Core functionality demos
        demo_model_creation()
        demo_optimizer()
        demo_training_step()
        demo_generation()
        demo_memory_usage()
        
        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Run full test suite: python test_model.py")
        print("2. Start development training: python training/pretrain.py --config configs/dev_config.yaml")
        print("3. Set up development environment: python setup_dev.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that src/ is in your Python path")
        print("3. Run the test suite to identify issues: python test_model.py")


if __name__ == "__main__":
    main()