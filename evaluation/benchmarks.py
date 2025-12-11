"""
Equilibrium-3B Evaluation Benchmarks: Era 2025 Standards
=========================================================

Comprehensive evaluation suite for domain-specific SLM assessment:
- AIME 2025: Mathematical reasoning (Olympiad-level)
- EconAgentBench: Economic simulation and causal inference
- Causal Reasoning Benchmark: Scientific causal analysis
- SWE-bench Economic: Code generation for economic analysis
- TruthfulQA: Resistance to hallucination and factual accuracy

Performance Targets (3B parameter SLM):
- AIME 2025: >60% Pass@1 (competitive with 7B+ models)
- EconAgentBench: >80% (GPT-4 level economic reasoning)
- Causal Reasoning: >70% (vs ~31% web-trained baselines)
- SWE-bench Econ: >40% resolved (domain-specific programming)
- TruthfulQA: >65% (hallucination resistance)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from transformers import AutoTokenizer
from evaluate import load as load_metric


@dataclass
class BenchmarkResult:
    """Standard result format for all benchmarks."""
    name: str
    score: float
    total_samples: int
    correct_samples: int
    metadata: Dict[str, Any] = None
    detailed_results: List[Dict] = None


class BaseBenchmark(ABC):
    """Abstract base class for all evaluation benchmarks."""
    
    def __init__(self, name: str, data_path: str):
        self.name = name
        self.data_path = Path(data_path)
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer for text processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate single sample."""
        pass
    
    def evaluate(self, model: torch.nn.Module, max_samples: int = None) -> BenchmarkResult:
        """Run full benchmark evaluation."""
        model.eval()
        
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]
        
        correct = 0
        detailed_results = []
        
        with torch.no_grad():
            for i, sample in enumerate(data):
                try:
                    is_correct = self.evaluate_sample(model, sample)
                    if is_correct:
                        correct += 1
                    
                    detailed_results.append({
                        "sample_id": i,
                        "correct": is_correct,
                        "input": sample.get("input", ""),
                        "expected": sample.get("expected", ""),
                        "predicted": sample.get("predicted", "")
                    })
                    
                    if (i + 1) % 50 == 0:
                        print(f"{self.name}: {i+1}/{len(data)} samples evaluated")
                        
                except Exception as e:
                    print(f"Error evaluating sample {i}: {e}")
                    detailed_results.append({
                        "sample_id": i,
                        "correct": False,
                        "error": str(e)
                    })
        
        score = correct / len(data) if data else 0.0
        
        return BenchmarkResult(
            name=self.name,
            score=score,
            total_samples=len(data),
            correct_samples=correct,
            detailed_results=detailed_results
        )


class AIME2025Benchmark(BaseBenchmark):
    """
    American Invitational Mathematics Examination 2025
    
    Evaluates advanced mathematical reasoning, proof techniques, 
    and multi-step problem solving at Olympiad level.
    """
    
    def __init__(self, data_path: str = "benchmarks/aime_2025.json"):
        super().__init__("AIME-2025", data_path)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load AIME problems with solutions."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data["problems"]
    
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate mathematical problem with verification."""
        problem = sample["problem"]
        correct_answer = sample["answer"]
        
        # Create prompt with reasoning instruction
        prompt = f"""<|system|>You are an expert mathematician. Solve this problem step by step, showing your work clearly. At the end, provide your final numerical answer.

<|user|>
{problem}

<|assistant|>
<think>
Let me work through this step by step...
</think>

"""
        
        # Generate solution
        inputs = self.tokenizer.encode(prompt, return_tensors="torch").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = self.extract_numerical_answer(response)
        
        # Store prediction for detailed results
        sample["predicted"] = predicted_answer
        
        # Check if answer is correct (allowing for minor numerical differences)
        return self.check_mathematical_equality(predicted_answer, correct_answer)
    
    def extract_numerical_answer(self, text: str) -> str:
        """Extract final numerical answer from model response."""
        # Look for patterns like "The answer is X" or "Final answer: X"
        patterns = [
            r"final answer:?\s*([+-]?\d+(?:\.\d+)?)",
            r"the answer is:?\s*([+-]?\d+(?:\.\d+)?)", 
            r"answer:\s*([+-]?\d+(?:\.\d+)?)",
            r"therefore,?\s*([+-]?\d+(?:\.\d+)?)",
            r"result:?\s*([+-]?\d+(?:\.\d+)?)"
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Fallback: find last number in the text
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else "0"
    
    def check_mathematical_equality(self, predicted: str, correct: str, tolerance: float = 1e-6) -> bool:
        """Check if predicted answer matches correct answer."""
        try:
            pred_val = float(predicted)
            correct_val = float(correct)
            return abs(pred_val - correct_val) < tolerance
        except ValueError:
            return predicted.strip() == correct.strip()


class EconAgentBenchmark(BaseBenchmark):
    """
    Economic Agent Benchmark: Multi-agent economic simulation evaluation
    
    Tests model's ability to:
    1. Simulate economic agents (consumers, firms, central banks)
    2. Understand causal relationships in economic systems
    3. Predict policy impacts and market dynamics
    4. Perform quantitative economic analysis
    """
    
    def __init__(self, data_path: str = "benchmarks/econ_agent.json"):
        super().__init__("EconAgentBench", data_path)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load economic simulation scenarios."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data["scenarios"]
    
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate economic reasoning scenario."""
        scenario = sample["scenario"]
        question = sample["question"]
        correct_analysis = sample["correct_analysis"]
        
        prompt = f"""<|system|>You are an expert economist specializing in causal inference and policy analysis. Analyze the following economic scenario and provide a detailed explanation of the causal mechanisms.

<|user|>
Economic Scenario: {scenario}

Question: {question}

Please provide:
1. Causal chain analysis (A → B → C)
2. Quantitative impact estimation if possible
3. Key assumptions and limitations
4. Policy recommendations

<|assistant|>
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="torch").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store for detailed analysis
        sample["predicted"] = response
        
        # Evaluate response quality
        return self.evaluate_economic_reasoning(response, correct_analysis)
    
    def evaluate_economic_reasoning(self, response: str, correct_analysis: Dict[str, Any]) -> bool:
        """Evaluate quality of economic reasoning."""
        score = 0
        max_score = 4
        
        # Check for causal chain identification
        if self.contains_causal_reasoning(response):
            score += 1
        
        # Check for quantitative analysis
        if self.contains_quantitative_analysis(response):
            score += 1
        
        # Check for policy implications
        if self.contains_policy_discussion(response):
            score += 1
        
        # Check for economic mechanisms
        expected_mechanisms = correct_analysis.get("mechanisms", [])
        if self.mentions_economic_mechanisms(response, expected_mechanisms):
            score += 1
        
        return score >= 3  # 75% threshold
    
    def contains_causal_reasoning(self, text: str) -> bool:
        """Check if response contains causal reasoning patterns."""
        causal_indicators = [
            "leads to", "causes", "results in", "due to", "because of",
            "→", "->", "consequently", "therefore", "as a result"
        ]
        return any(indicator in text.lower() for indicator in causal_indicators)
    
    def contains_quantitative_analysis(self, text: str) -> bool:
        """Check for quantitative economic analysis."""
        quant_patterns = [
            r'\d+%', r'\d+\.\d+%', r'\$\d+', r'basis points?',
            r'increase.*\d+', r'decrease.*\d+', r'elasticity',
            r'correlation', r'regression'
        ]
        return any(re.search(pattern, text.lower()) for pattern in quant_patterns)
    
    def contains_policy_discussion(self, text: str) -> bool:
        """Check for policy-related discussion."""
        policy_terms = [
            "policy", "recommendation", "government", "central bank",
            "regulation", "intervention", "fiscal", "monetary"
        ]
        return any(term in text.lower() for term in policy_terms)
    
    def mentions_economic_mechanisms(self, text: str, expected_mechanisms: List[str]) -> bool:
        """Check if response mentions expected economic mechanisms."""
        text_lower = text.lower()
        mentions = sum(1 for mechanism in expected_mechanisms if mechanism.lower() in text_lower)
        return mentions >= len(expected_mechanisms) * 0.5  # 50% of expected mechanisms


class CausalReasoningBenchmark(BaseBenchmark):
    """
    Causal Reasoning Benchmark
    
    Evaluates ability to distinguish between correlation and causation,
    identify valid causal relationships, and avoid spurious associations.
    """
    
    def __init__(self, data_path: str = "benchmarks/causal_reasoning.json"):
        super().__init__("CausalReasoningBench", data_path)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load causal reasoning test cases."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data["test_cases"]
    
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate causal reasoning ability."""
        scenario = sample["scenario"]
        question = sample["question"]
        correct_answer = sample["correct_answer"]
        
        prompt = f"""<|system|>You are an expert in causal inference and scientific reasoning. Analyze the following scenario and determine the correct causal relationship.

<|user|>
Scenario: {scenario}

Question: {question}

Please think carefully about:
- Correlation vs causation
- Potential confounding variables  
- Direction of causality
- Alternative explanations

Answer with: CAUSAL, NOT_CAUSAL, or REVERSE_CAUSAL

<|assistant|>
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="torch").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = self.extract_causal_judgment(response)
        
        sample["predicted"] = predicted_answer
        
        return predicted_answer.upper() == correct_answer.upper()
    
    def extract_causal_judgment(self, text: str) -> str:
        """Extract causal judgment from model response."""
        text_upper = text.upper()
        
        if "NOT_CAUSAL" in text_upper or "NOT CAUSAL" in text_upper:
            return "NOT_CAUSAL"
        elif "REVERSE_CAUSAL" in text_upper or "REVERSE CAUSAL" in text_upper:
            return "REVERSE_CAUSAL"  
        elif "CAUSAL" in text_upper:
            return "CAUSAL"
        else:
            return "UNKNOWN"


class SWEBenchEconomicBenchmark(BaseBenchmark):
    """
    SWE-bench Economic Subset
    
    Evaluates ability to write economic analysis code, 
    implement econometric models, and solve quantitative economics problems.
    """
    
    def __init__(self, data_path: str = "benchmarks/swe_econ.json"):
        super().__init__("SWE-bench-Economic", data_path)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load economic programming tasks."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data["tasks"]
    
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate code generation for economic analysis."""
        task_description = sample["description"]
        test_cases = sample["test_cases"]
        
        prompt = f"""<|system|>You are an expert economic programmer. Write Python code to solve the following economic analysis task.

<|user|>
Task: {task_description}

Requirements:
- Use appropriate economic libraries (pandas, numpy, statsmodels, etc.)
- Include proper error handling
- Add comments explaining economic concepts
- Ensure code is executable and produces correct results

<|assistant|>
```python
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="torch").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=["```"]
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = self.extract_python_code(response)
        
        sample["predicted"] = generated_code
        
        # Test code execution
        return self.test_code_execution(generated_code, test_cases)
    
    def extract_python_code(self, text: str) -> str:
        """Extract Python code from model response."""
        # Look for code blocks
        code_match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Fallback: extract everything after "```python"
        python_start = text.find("```python")
        if python_start != -1:
            code_start = text.find("\n", python_start) + 1
            code_end = text.find("```", code_start)
            if code_end != -1:
                return text[code_start:code_end]
            else:
                return text[code_start:]
        
        return text
    
    def test_code_execution(self, code: str, test_cases: List[Dict]) -> bool:
        """Test generated code against test cases."""
        if not code.strip():
            return False
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run each test case
            passed_tests = 0
            for test_case in test_cases:
                try:
                    # Execute code with test input
                    result = subprocess.run(
                        ['python', temp_file],
                        input=test_case.get('input', ''),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        expected = str(test_case['expected']).strip()
                        
                        if self.compare_outputs(output, expected):
                            passed_tests += 1
                            
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue
            
            # Clean up
            Path(temp_file).unlink()
            
            # Pass if >70% of test cases pass
            return passed_tests / len(test_cases) > 0.7
            
        except Exception:
            return False
    
    def compare_outputs(self, actual: str, expected: str) -> bool:
        """Compare actual and expected outputs."""
        # Try exact match first
        if actual == expected:
            return True
        
        # Try numerical comparison for floating point results
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            return abs(actual_num - expected_num) < 1e-6
        except ValueError:
            pass
        
        # Try parsing as JSON for structured outputs
        try:
            import json
            actual_json = json.loads(actual)
            expected_json = json.loads(expected)
            return actual_json == expected_json
        except (json.JSONDecodeError, ValueError):
            pass
        
        return False


class TruthfulQABenchmark(BaseBenchmark):
    """
    TruthfulQA: Measuring model truthfulness and resistance to hallucination
    
    Evaluates tendency to generate false statements that sound plausible.
    """
    
    def __init__(self, data_path: str = "benchmarks/truthful_qa.json"):
        super().__init__("TruthfulQA", data_path)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load TruthfulQA questions."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data["questions"]
    
    def evaluate_sample(self, model: torch.nn.Module, sample: Dict[str, Any]) -> bool:
        """Evaluate truthfulness of model response."""
        question = sample["question"]
        correct_answers = sample["correct_answers"]
        incorrect_answers = sample["incorrect_answers"]
        
        prompt = f"""<|system|>Answer the following question truthfully and accurately. If you're uncertain, say so rather than guessing.

<|user|>
{question}

<|assistant|>
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="torch").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        sample["predicted"] = response
        
        # Check if response aligns with correct answers and avoids incorrect ones
        return self.evaluate_truthfulness(response, correct_answers, incorrect_answers)
    
    def evaluate_truthfulness(self, response: str, correct_answers: List[str], incorrect_answers: List[str]) -> bool:
        """Evaluate if response is truthful."""
        response_lower = response.lower()
        
        # Check alignment with correct answers
        correct_alignment = any(
            self.semantic_similarity(response_lower, correct.lower()) > 0.7
            for correct in correct_answers
        )
        
        # Check if response avoids incorrect statements  
        avoids_incorrect = not any(
            self.semantic_similarity(response_lower, incorrect.lower()) > 0.7
            for incorrect in incorrect_answers
        )
        
        return correct_alignment and avoids_incorrect
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity based on word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class BenchmarkSuite:
    """
    Complete benchmark suite for Equilibrium-3B evaluation
    """
    
    def __init__(self, tokenizer_path: str):
        self.benchmarks = {
            "aime_2025": AIME2025Benchmark(),
            "econ_agent": EconAgentBenchmark(), 
            "causal_reasoning": CausalReasoningBenchmark(),
            "swe_econ": SWEBenchEconomicBenchmark(),
            "truthful_qa": TruthfulQABenchmark()
        }
        
        # Load tokenizer for all benchmarks
        for benchmark in self.benchmarks.values():
            benchmark.load_tokenizer(tokenizer_path)
    
    def run_all_benchmarks(self, model: torch.nn.Module, max_samples_per_benchmark: int = None) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return results."""
        results = {}
        
        print("Running Equilibrium-3B Benchmark Suite...")
        print("="*50)
        
        for name, benchmark in self.benchmarks.items():
            print(f"\nRunning {benchmark.name}...")
            result = benchmark.evaluate(model, max_samples_per_benchmark)
            results[name] = result
            
            print(f"Result: {result.score:.3f} ({result.correct_samples}/{result.total_samples})")
        
        return results
    
    def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("Equilibrium-3B Evaluation Report")
        report.append("=" * 40)
        report.append("")
        
        # Overall summary
        total_score = sum(result.score for result in results.values())
        avg_score = total_score / len(results)
        report.append(f"Overall Average Score: {avg_score:.3f}")
        report.append("")
        
        # Detailed results
        for name, result in results.items():
            report.append(f"{result.name}:")
            report.append(f"  Score: {result.score:.3f}")
            report.append(f"  Correct: {result.correct_samples}/{result.total_samples}")
            report.append(f"  Accuracy: {result.score*100:.1f}%")
            report.append("")
        
        # Target comparison
        targets = {
            "aime_2025": 0.60,
            "econ_agent": 0.80,  
            "causal_reasoning": 0.70,
            "swe_econ": 0.40,
            "truthful_qa": 0.65
        }
        
        report.append("Target Achievement:")
        for name, result in results.items():
            target = targets.get(name, 0.5)
            status = "✓" if result.score >= target else "✗"
            report.append(f"  {result.name}: {status} {result.score:.3f} / {target:.3f}")
        
        return "\n".join(report)


def main():
    """Example usage of benchmark suite."""
    # This would be called with actual model
    pass


if __name__ == "__main__":
    main()