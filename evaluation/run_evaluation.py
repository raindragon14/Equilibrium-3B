#!/usr/bin/env python3
"""
Equilibrium-3B Evaluation Runner
===============================

Automated evaluation script for running all 2025 benchmark standards:
- AIME 2025: Mathematical reasoning at Olympiad level
- EconAgentBench: Economic simulation and causal inference  
- Causal Reasoning: Scientific causal analysis
- SWE-bench Economic: Economic programming tasks
- TruthfulQA: Hallucination resistance

Usage:
    # Run all benchmarks
    python run_evaluation.py --model checkpoints/equilibrium-3b-final.pt
    
    # Run specific benchmarks
    python run_evaluation.py --model checkpoints/equilibrium-3b-final.pt --benchmarks aime_2025 econ_agent
    
    # Generate detailed report
    python run_evaluation.py --model checkpoints/equilibrium-3b-final.pt --output results/evaluation_report.json
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.benchmarks import BenchmarkSuite, BenchmarkResult
from model.equilibrium import Equilibrium3B


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Equilibrium-3B Evaluation Runner")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Equilibrium-3B model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/equilibrium-3b",
        help="Path to tokenizer"
    )
    
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["aime_2025", "econ_agent", "causal_reasoning", "swe_econ", "truthful_qa"],
        help="Benchmarks to run"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (for quick testing)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (auto/cpu/cuda)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "auto") -> torch.nn.Module:
    """Load Equilibrium-3B model from checkpoint."""
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {checkpoint_path} on {device}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint or use defaults
    model_config = checkpoint.get('model_config', {
        'vocab_size': 65536,
        'hidden_size': 2560,
        'num_layers': 24,
        'num_attention_layers': 3,
        'num_experts': 64,
        'use_mla': True,
        'max_position_embeddings': 128000
    })
    
    # Initialize model
    model = Equilibrium3B(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def run_evaluation(args):
    """Run benchmark evaluation."""
    
    print("Equilibrium-3B Evaluation Suite (2025 Standards)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print()
    
    # Load model
    try:
        model = load_model(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Initialize benchmark suite
    try:
        benchmark_suite = BenchmarkSuite(args.tokenizer)
    except Exception as e:
        print(f"Error initializing benchmarks: {e}")
        return 1
    
    # Filter benchmarks if specified
    if args.benchmarks != ["all"]:
        benchmark_suite.benchmarks = {
            k: v for k, v in benchmark_suite.benchmarks.items() 
            if k in args.benchmarks
        }
    
    # Run evaluation
    print("Starting evaluation...")
    start_time = datetime.now()
    
    try:
        results = benchmark_suite.run_all_benchmarks(model, args.max_samples)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    end_time = datetime.now()
    evaluation_duration = end_time - start_time
    
    # Generate report
    report = benchmark_suite.generate_report(results)
    print("\\n" + report)
    
    # Calculate summary statistics
    total_samples = sum(r.total_samples for r in results.values())
    total_correct = sum(r.correct_samples for r in results.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    print(f"\\nEvaluation completed in {evaluation_duration}")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_samples})")
    
    # Save results to JSON if requested
    if args.output:
        output_data = {
            "evaluation_info": {
                "model_path": args.model,
                "tokenizer_path": args.tokenizer,
                "benchmarks_run": list(results.keys()),
                "max_samples": args.max_samples,
                "evaluation_time": start_time.isoformat(),
                "duration_seconds": evaluation_duration.total_seconds(),
                "device": args.device
            },
            "summary": {
                "overall_accuracy": overall_accuracy,
                "total_samples": total_samples,
                "total_correct": total_correct,
                "benchmark_count": len(results)
            },
            "benchmark_results": {
                name: {
                    "score": result.score,
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                    "metadata": result.metadata
                }
                for name, result in results.items()
            },
            "detailed_results": {
                name: result.detailed_results
                for name, result in results.items()
                if result.detailed_results
            }
        }
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\\nResults saved to {args.output}")
    
    # Performance targets check
    targets = {
        "aime_2025": 0.60,
        "econ_agent": 0.80,
        "causal_reasoning": 0.70,
        "swe_econ": 0.40,
        "truthful_qa": 0.65
    }
    
    targets_met = 0
    for name, result in results.items():
        if name in targets and result.score >= targets[name]:
            targets_met += 1
    
    print(f"\\nPerformance targets met: {targets_met}/{len([b for b in args.benchmarks if b in targets])}")
    
    # Return success if majority of targets met
    return 0 if targets_met >= len(targets) * 0.6 else 1


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        exit_code = run_evaluation(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()