"""Centralized GPU resource management."""

from typing import TYPE_CHECKING, Optional

from llm_optimizer.predefined.gpus import get_gpu_specs, get_precision_tflops
from llm_optimizer.resources.types import GPUResources, MemoryLimits

if TYPE_CHECKING:
    from llm_optimizer.common import ModelConfig


class GPUResourceManager:
    """Centralized GPU resource calculations and management.

    This class consolidates all GPU-related calculations that were previously
    scattered across multiple modules, providing a single source of truth
    for GPU resource management.
    """

    def __init__(self):
        """Initialize GPU resource manager."""
        pass

    def get_total_resources(
        self,
        num_gpus: int,
        gpu_name: str,
        precision: str
    ) -> GPUResources:
        """Get aggregated GPU resources with caching.

        Args:
            num_gpus: Number of GPUs to aggregate
            gpu_name: Name of the GPU model (e.g., "H100", "A100")
            precision: Precision level (e.g., "fp16", "fp8", "int8")

        Returns:
            GPUResources object with aggregated resources
        """
        # Get base GPU specifications
        gpu_specs = get_gpu_specs(gpu_name)

        # Calculate per-GPU resources
        return GPUResources(
            per_gpu_tflops=get_precision_tflops(gpu_name, precision),
            per_gpu_memory_bytes=int(gpu_specs["VRAM_GB"] * 1024**3),  # Convert GB to bytes
            per_gpu_bandwidth_bytes_per_sec=int(gpu_specs["Memory_Bandwidth_GBs"] * 1024**3),  # Convert GB/s to bytes/s
            num_gpus=num_gpus,
            gpu_name=gpu_name,
            precision=precision
        )

    def calculate_memory_limits(
        self,
        gpu_resources: GPUResources,
        memory_utilization: float = 0.9,
        reserved_memory_gb: Optional[float] = None
    ) -> MemoryLimits:
        """Calculate memory constraints for given GPU resources.

        Args:
            gpu_resources: GPU resources to calculate limits for
            memory_utilization: Fraction of GPU memory to utilize (0-1)
            reserved_memory_gb: Memory to reserve for system/framework overhead

        Returns:
            MemoryLimits object with calculated constraints
        """
        # Calculate total available memory in bytes
        total_available_bytes = int(gpu_resources.total_memory_bytes * memory_utilization)

        # Calculate reserved memory if not specified
        if reserved_memory_gb is None:
            # Reserve 10% of available memory or 2GB per GPU, whichever is larger
            reserved_memory_bytes = max(
                int(total_available_bytes * 0.1),
                int(2.0 * 1024**3 * gpu_resources.num_gpus)  # 2GB per GPU in bytes
            )
        else:
            reserved_memory_bytes = int(reserved_memory_gb * 1024**3)

        # Model memory limit (typically 40-50% of available memory)
        max_model_memory_bytes = int(total_available_bytes * 0.45)

        # KV cache gets the remaining memory after model and reserves
        max_kv_cache_bytes = total_available_bytes - max_model_memory_bytes - reserved_memory_bytes

        return MemoryLimits(
            max_model_memory_bytes=max_model_memory_bytes,
            max_kv_cache_bytes=max_kv_cache_bytes,
            max_total_memory_bytes=total_available_bytes,
            reserved_memory_bytes=reserved_memory_bytes
        )

    def estimate_concurrency_limits(
        self,
        model_config: "ModelConfig",
        gpu_resources: GPUResources,
        memory_limits: MemoryLimits,
        avg_sequence_length: int = 1024
    ) -> int:
        """Estimate maximum concurrent requests based on memory constraints.

        Args:
            model_config: Model configuration
            gpu_resources: Available GPU resources
            memory_limits: Memory constraints
            avg_sequence_length: Average expected sequence length

        Returns:
            Maximum number of concurrent requests
        """
        # Calculate KV cache size per sequence
        # This is a simplified calculation - actual implementation should
        # match the ModelMemoryCalculator's KV cache calculation
        bytes_per_element = 2 if "fp16" in gpu_resources.precision else 1

        kv_cache_per_seq_bytes = (
            2 *  # K and V
            model_config.num_layers *
            model_config.num_heads *
            (model_config.hidden_dim // model_config.num_heads) *
            avg_sequence_length *
            bytes_per_element
        )

        if kv_cache_per_seq_bytes <= 0:
            return 1

        # Calculate max sequences that fit in available KV cache memory
        max_sequences = int(memory_limits.available_for_kv_cache_bytes // kv_cache_per_seq_bytes)

        # Apply safety factor
        max_sequences = int(max_sequences * 0.8)

        return max(1, max_sequences)


    def get_compute_memory_ratio(self, gpu_resources: GPUResources) -> float:
        """Calculate compute to memory bandwidth ratio.

        This ratio is useful for determining whether a workload is
        compute-bound or memory-bound.

        Args:
            gpu_resources: GPU resources to analyze

        Returns:
            Ratio of TFLOPS to memory bandwidth
        """
        # Convert TFLOPS to FLOPS
        flops = gpu_resources.total_tflops * 1e12
        bandwidth_bytes_per_sec = gpu_resources.total_bandwidth_bytes_per_sec

        return flops / bandwidth_bytes_per_sec

    def is_compute_bound(
        self,
        gpu_resources: GPUResources,
        arithmetic_intensity: float
    ) -> bool:
        """Determine if a workload is compute-bound or memory-bound.

        Args:
            gpu_resources: GPU resources
            arithmetic_intensity: Ops per byte for the workload

        Returns:
            True if compute-bound, False if memory-bound
        """
        compute_memory_ratio = self.get_compute_memory_ratio(gpu_resources)
        return arithmetic_intensity > compute_memory_ratio
