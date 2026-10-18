"""Data types for resource management."""

from dataclasses import asdict, dataclass


@dataclass
class GPUResources:
    """GPU resources for multi-GPU systems."""

    per_gpu_tflops: float
    per_gpu_memory_bytes: int
    per_gpu_bandwidth_bytes_per_sec: int
    num_gpus: int
    gpu_name: str
    precision: str

    @property
    def total_tflops(self) -> float:
        """Total TFLOPS across all GPUs."""
        return self.per_gpu_tflops * self.num_gpus

    @property
    def total_memory_bytes(self) -> int:
        """Total memory in bytes across all GPUs."""
        return self.per_gpu_memory_bytes * self.num_gpus

    @property
    def total_bandwidth_bytes_per_sec(self) -> int:
        """Total bandwidth in bytes/sec across all GPUs."""
        return self.per_gpu_bandwidth_bytes_per_sec * self.num_gpus


@dataclass
class MemoryBreakdown:
    """Detailed breakdown of memory requirements."""

    model_memory_bytes: int
    kv_cache_per_token_bytes: int
    activation_memory_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        """Total memory requirement in bytes."""
        return (
            self.model_memory_bytes +
            self.kv_cache_per_token_bytes +
            self.activation_memory_bytes +
            self.overhead_bytes
        )

    def scale_kv_cache(self, num_tokens: int) -> int:
        """Calculate total KV cache memory for given number of tokens."""
        return self.kv_cache_per_token_bytes * num_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary with total included."""
        result = asdict(self)
        result['total_bytes'] = self.total_bytes
        return result


@dataclass
class MemoryLimits:
    """Memory constraints for model serving."""

    max_model_memory_bytes: int
    max_kv_cache_bytes: int
    max_total_memory_bytes: int
    reserved_memory_bytes: int

    @property
    def available_for_kv_cache_bytes(self) -> int:
        """Memory available for KV cache after model and reserves."""
        return max(0, self.max_total_memory_bytes - self.max_model_memory_bytes - self.reserved_memory_bytes)

    @property
    def max_concurrent_requests(self) -> int:
        """Estimate max concurrent requests based on memory limits."""
        # This is a simplified calculation - actual implementation should
        # consider sequence length and other factors
        if self.max_kv_cache_bytes <= 0:
            return 1
        # Rough estimate: assume average KV cache per request (100MB)
        avg_kv_per_request_bytes = 100 * 1024 * 1024  # 100MB in bytes
        return max(1, int(self.available_for_kv_cache_bytes // avg_kv_per_request_bytes))
