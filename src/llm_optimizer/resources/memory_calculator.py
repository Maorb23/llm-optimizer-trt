"""Centralized model memory calculations."""

from typing import TYPE_CHECKING, Optional

from llm_optimizer.common import get_precision_bytes_per_param
from llm_optimizer.resources.types import MemoryBreakdown

if TYPE_CHECKING:
    from llm_optimizer.common import ModelConfig


class ModelMemoryCalculator:
    """Centralized model memory calculations.

    This class consolidates all memory calculation logic that was previously
    scattered across performance.py, common.py, and tuning.py, providing
    consistent and accurate memory estimates.
    """

    def __init__(self, overhead_factor: float = 1.2):
        """Initialize memory calculator.

        Args:
            overhead_factor: Multiplicative factor for memory overhead (default 1.2 = 20%)
        """
        self.overhead_factor = overhead_factor

    def calculate_model_memory(
        self,
        model_config: "ModelConfig",
        precision: str
    ) -> int:
        """Calculate base model memory requirements.

        Args:
            model_config: Model configuration
            precision: Model precision (e.g., "fp16", "fp8", "int8")

        Returns:
            Model memory in bytes
        """
        bytes_per_param = get_precision_bytes_per_param(precision)
        model_memory_bytes = model_config.num_params * bytes_per_param

        return model_memory_bytes

    def calculate_kv_cache_memory(
        self,
        model_config: "ModelConfig",
        sequence_length: int,
        batch_size: int,
        precision: str
    ) -> int:
        """Calculate KV cache memory for given sequence length and batch.

        Args:
            model_config: Model configuration
            sequence_length: Maximum sequence length
            batch_size: Batch size
            precision: Precision for KV cache (often different from model precision)

        Returns:
            KV cache memory in bytes
        """
        # KV cache typically uses same precision as activations
        bytes_per_element = get_precision_bytes_per_param(precision)

        # For MQA/GQA, use actual number of KV heads
        kv_heads = getattr(model_config, 'num_kv_heads', model_config.num_heads)
        head_dim = model_config.hidden_dim // model_config.num_heads

        # KV cache size formula:
        # 2 * num_layers * kv_heads * head_dim * seq_len * batch_size * bytes_per_element
        kv_cache_bytes = (
            2 *  # K and V
            model_config.num_layers *
            kv_heads *
            head_dim *
            sequence_length *
            batch_size *
            bytes_per_element
        )

        return int(kv_cache_bytes)

    def calculate_activation_memory(
        self,
        model_config: "ModelConfig",
        batch_size: int,
        sequence_length: int,
        precision: str
    ) -> int:
        """Calculate activation memory requirements.

        Args:
            model_config: Model configuration
            batch_size: Batch size
            sequence_length: Sequence length
            precision: Precision for activations

        Returns:
            Activation memory in bytes
        """
        bytes_per_element = get_precision_bytes_per_param(precision)

        # Activation memory includes:
        # - Input/output tensors
        # - Intermediate activations (roughly 2-4x hidden dim per layer)
        # - Attention scores

        # Simplified calculation - more detailed version would consider
        # specific architecture details
        activation_multiplier = 4  # Conservative estimate

        activation_bytes = (
            batch_size *
            sequence_length *
            model_config.hidden_dim *
            model_config.num_layers *
            activation_multiplier *
            bytes_per_element
        )

        return int(activation_bytes)

    def calculate_total_memory_needed(
        self,
        model_config: "ModelConfig",
        batch_size: int,
        sequence_length: int,
        model_precision: str,
        kv_precision: Optional[str] = None
    ) -> MemoryBreakdown:
        """Calculate total memory requirements with detailed breakdown.

        Args:
            model_config: Model configuration
            batch_size: Batch size
            sequence_length: Maximum sequence length
            model_precision: Precision for model weights
            kv_precision: Precision for KV cache (defaults to model_precision)

        Returns:
            MemoryBreakdown with detailed memory requirements
        """
        if kv_precision is None:
            kv_precision = model_precision

        # Calculate individual components
        model_memory = self.calculate_model_memory(model_config, model_precision)
        kv_cache_memory = self.calculate_kv_cache_memory(
            model_config, sequence_length, batch_size, kv_precision
        )
        activation_memory = self.calculate_activation_memory(
            model_config, batch_size, sequence_length, kv_precision
        )

        # Calculate overhead
        subtotal = model_memory + kv_cache_memory + activation_memory
        overhead = int(subtotal * (self.overhead_factor - 1))

        return MemoryBreakdown(
            model_memory_bytes=model_memory,
            kv_cache_per_token_bytes=kv_cache_memory // (batch_size * sequence_length),
            activation_memory_bytes=activation_memory,
            overhead_bytes=overhead
        )

    def estimate_max_batch_size(
        self,
        model_config: "ModelConfig",
        available_memory_bytes: int,
        sequence_length: int,
        precision: str
    ) -> int:
        """Estimate maximum batch size that fits in available memory.

        Args:
            model_config: Model configuration
            available_memory_bytes: Available GPU memory in bytes
            sequence_length: Maximum sequence length
            precision: Model precision

        Returns:
            Maximum batch size
        """
        # Start with model memory (fixed cost)
        model_memory = self.calculate_model_memory(model_config, precision)
        remaining_memory = available_memory_bytes - model_memory

        if remaining_memory <= 0:
            return 0

        # Calculate per-batch memory requirements
        # (KV cache + activations scale with batch size)
        kv_per_batch = self.calculate_kv_cache_memory(
            model_config, sequence_length, 1, precision
        )
        activation_per_batch = self.calculate_activation_memory(
            model_config, 1, sequence_length, precision
        )

        memory_per_batch = int((kv_per_batch + activation_per_batch) * self.overhead_factor)

        if memory_per_batch <= 0:
            return 1

        max_batch_size = int(remaining_memory // memory_per_batch)
        return max(1, max_batch_size)

    def estimate_max_sequence_length(
        self,
        model_config: "ModelConfig",
        available_memory_bytes: int,
        batch_size: int,
        precision: str
    ) -> int:
        """Estimate maximum sequence length that fits in available memory.

        Args:
            model_config: Model configuration
            available_memory_bytes: Available GPU memory in bytes
            batch_size: Batch size
            precision: Model precision

        Returns:
            Maximum sequence length
        """
        # Start with model memory (fixed cost)
        model_memory = self.calculate_model_memory(model_config, precision)
        remaining_memory = available_memory_bytes - model_memory

        if remaining_memory <= 0:
            return 0

        # Calculate memory per token (mainly KV cache)
        bytes_per_element = get_precision_bytes_per_param(precision)
        kv_heads = getattr(model_config, 'num_kv_heads', model_config.num_heads)
        head_dim = model_config.hidden_dim // model_config.num_heads

        # Memory per token in KV cache
        kv_memory_per_token_bytes = (
            2 *  # K and V
            model_config.num_layers *
            kv_heads *
            head_dim *
            batch_size *
            bytes_per_element
        )

        # Add activation memory scaling (simplified)
        activation_per_token_bytes = (
            batch_size *
            model_config.hidden_dim *
            model_config.num_layers *
            4 *  # activation multiplier
            bytes_per_element
        )

        total_per_token = int((kv_memory_per_token_bytes + activation_per_token_bytes) * self.overhead_factor)

        if total_per_token <= 0:
            return 1

        max_seq_length = int(remaining_memory // total_per_token)
        return max(1, max_seq_length)

