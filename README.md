# Equilibrium-3B: Specialized Math & Econ SLM
> A 3B parameter State Space Model (SSM) hybrid designed for high-precision economic causal inference and formal mathematical derivation.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Model Architecture](https://img.shields.io/badge/Arch-Mamba2_Hybrid_MoE-violet)](./ARCHITECTURE.md)
[![Optimizer](https://img.shields.io/badge/Optimizer-Muon_Schedule--Free-green)](./src/optimizers/)

## 🚀 Abstract
Equilibrium-3B is an experimental Small Language Model built from scratch in late 2025. It moves away from standard Transformer monoliths, employing a **Hybrid Mamba-2 + Transformer** architecture with **Fine-Grained MoE**. Trained with **Muon optimizer** and aligned using **GRPO (Group Relative Policy Optimization)**, it aims to bridge the "reasoning gap" in sub-7B models.

## ✨ Key Technical Innovations
* **Hybrid Core:** 7:1 Mamba-to-Attention ratio for 128k context length with linear complexity.
* **DeepSeekMoE Style:** 64 fine-grained experts with sigmoid-based routing for domain specialization.
* **Reasoning-First Alignment:** Post-trained via GRPO (System 2 thinking) without a critic model.
* **Memory Efficiency:** Multi-Head Latent Attention (MLA) & ZeroQAT (4-bit training aware).

## 📊 Benchmarks (Preliminary)
| Benchmark | Equilibrium-3B | Llama-3-8B | Mistral-7B |
|-----------|----------------|------------|------------|
| MATH      |                | 51.0%      | ...        |
| EconBench |                | ...        | ...        |
