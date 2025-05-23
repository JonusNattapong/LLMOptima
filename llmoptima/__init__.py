"""
LLMOptima: Next-generation LLM Optimization Engine
"""

from .core import LLMOptima
from .optimizers import QuantizationOptimizer, PruningOptimizer, DistillationOptimizer, InferenceOptimizer
from .utils import OptimizationMetrics, OptimizationResult
from .cost import LLMCostCalculator

__version__ = "0.1.0"
__author__ = "JonusNattapong"
__email__ = "jonus@llmoptima.ai"
__description__ = "Next-generation LLM Optimization Engine"

__all__ = [
    "LLMOptima",
    "LLMCostCalculator", 
    "OptimizationMetrics",
    "QuantizationOptimizer",
    "PruningOptimizer",
    "DistillationOptimizer",
    "InferenceOptimizer",
]
