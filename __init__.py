"""
LLM Evaluation - Korean Benchmark Evaluation Tool

Evaluate LLMs on Korean benchmarks using vLLM or OpenAI API.

Backends:
- vLLM: Local GPU inference with efficient batching
- OpenAI API: Completions API with loglikelihood support
"""

from .core import EvaluationResult, get_kst_timestamp
from .evaluator import Evaluator, save_results, update_leaderboard, parse_leaderboard
from .api_evaluator import APIEvaluator
from .chat_evaluator import ChatEvaluator
from .dataset_configs import DatasetLoader, get_available_datasets

__version__ = "0.1.0"
__all__ = [
    # Backends
    "Evaluator",       # vLLM
    "APIEvaluator",    # OpenAI Completions API
    "ChatEvaluator",   # OpenAI Chat API
    # Core
    "EvaluationResult",
    "get_kst_timestamp",
    # Dataset
    "DatasetLoader",
    "get_available_datasets",
    # Utils
    "save_results",
    "update_leaderboard",
    "parse_leaderboard",
]
