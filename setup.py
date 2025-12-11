"""
Equilibrium-3B Package Setup
============================

Setup configuration for Equilibrium-3B: Era 2025 SLM Paradigm

Features:
- Hybrid SSM-Transformer architecture
- Schedule-free COSMOS optimization  
- ZeroQAT training-aware quantization
- Synthetic data pipeline with verification
- Domain specialization for Math & Economics
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="equilibrium-3b",
    version="2025.1.0",
    description="Era 2025 Small Language Model with Hybrid SSM-Transformer Architecture",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    author="Equilibrium Research Team",
    author_email="research@equilibrium-ai.org",
    url="https://github.com/raindragon14/Equilibrium-3B",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.10",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "jupyter>=1.0.0"
        ],
        "eval": [
            "evaluate>=0.4.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.8.0",
            "plotly>=5.17.0"
        ],
        "deploy": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "docker>=6.1.0"
        ]
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    
    keywords=[
        "artificial-intelligence",
        "machine-learning", 
        "natural-language-processing",
        "transformer",
        "mamba",
        "state-space-models",
        "mixture-of-experts",
        "small-language-models",
        "mathematical-reasoning",
        "economic-modeling"
    ],
    
    entry_points={
        "console_scripts": [
            "equilibrium-train=training.pretrain:main",
            "equilibrium-eval=evaluation.run_evaluation:main",
            "equilibrium-serve=deploy.serve:main"
        ]
    },
    
    include_package_data=True,
    package_data={
        "equilibrium": [
            "configs/*.yaml",
            "benchmarks/*.json"
        ]
    },
    
    project_urls={
        "Documentation": "https://equilibrium-3b.readthedocs.io/",
        "Source": "https://github.com/raindragon14/Equilibrium-3B",
        "Tracker": "https://github.com/raindragon14/Equilibrium-3B/issues",
        "Research Paper": "https://arxiv.org/abs/2025.XXXX"
    }
)