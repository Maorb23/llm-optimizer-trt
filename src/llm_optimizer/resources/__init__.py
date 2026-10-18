"""Resource management utilities for LLM optimizer.

This package provides centralized abstractions for GPU resource management
and model memory calculations, eliminating code duplication across the codebase.
"""

from llm_optimizer.resources.gpu_manager import GPUResourceManager
from llm_optimizer.resources.memory_calculator import ModelMemoryCalculator
from llm_optimizer.resources.types import GPUResources, MemoryBreakdown, MemoryLimits

__all__ = [
    "GPUResourceManager",
    "GPUResources",
    "ModelMemoryCalculator",
    "MemoryBreakdown",
    "MemoryLimits"
]
