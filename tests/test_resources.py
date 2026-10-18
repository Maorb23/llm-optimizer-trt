"""Unit tests for resource management components."""

import unittest
from unittest.mock import patch

from llm_optimizer.common import ModelConfig
from llm_optimizer.resources import (
    GPUResourceManager,
    GPUResources,
    MemoryBreakdown,
    MemoryLimits,
    ModelMemoryCalculator,
)


class TestGPUResources(unittest.TestCase):
    """Test GPUResources data class."""

    def test_gpu_resources_creation(self):
        """Test creating GPUResources instance."""
        resources = GPUResources(
            per_gpu_tflops=50.0,
            per_gpu_memory_bytes=80 * 1024**3,  # 80GB in bytes
            per_gpu_bandwidth_bytes_per_sec=2000 * 1024**3,  # 2000GB/s in bytes/s
            num_gpus=2,
            gpu_name="A100",
            precision="fp16"
        )

        self.assertEqual(resources.per_gpu_tflops, 50.0)
        self.assertEqual(resources.per_gpu_memory_bytes, 80 * 1024**3)
        self.assertEqual(resources.per_gpu_bandwidth_bytes_per_sec, 2000 * 1024**3)
        self.assertEqual(resources.num_gpus, 2)
        self.assertEqual(resources.gpu_name, "A100")
        self.assertEqual(resources.precision, "fp16")

    def test_total_properties(self):
        """Test total property calculations."""
        resources = GPUResources(
            per_gpu_tflops=50.0,
            per_gpu_memory_bytes=40 * 1024**3,  # 40GB in bytes
            per_gpu_bandwidth_bytes_per_sec=1000 * 1024**3,  # 1000GB/s in bytes/s
            num_gpus=4,
            gpu_name="H100",
            precision="fp8"
        )

        self.assertEqual(resources.total_tflops, 200.0)
        self.assertEqual(resources.total_memory_bytes, 160 * 1024**3)
        self.assertEqual(resources.total_bandwidth_bytes_per_sec, 4000 * 1024**3)


class TestMemoryBreakdown(unittest.TestCase):
    """Test MemoryBreakdown data class."""

    def test_memory_breakdown_creation(self):
        """Test creating MemoryBreakdown instance."""
        breakdown = MemoryBreakdown(
            model_memory_bytes=10 * 1024**3,  # 10GB in bytes
            kv_cache_per_token_bytes=1024,     # 1KB per token
            activation_memory_bytes=2 * 1024**3,  # 2GB in bytes
            overhead_bytes=int(1.5 * 1024**3)     # 1.5GB in bytes
        )

        self.assertEqual(breakdown.model_memory_bytes, 10 * 1024**3)
        self.assertEqual(breakdown.kv_cache_per_token_bytes, 1024)
        self.assertEqual(breakdown.activation_memory_bytes, 2 * 1024**3)
        self.assertEqual(breakdown.overhead_bytes, int(1.5 * 1024**3))

    def test_total_memory_calculation(self):
        """Test total memory property."""
        breakdown = MemoryBreakdown(
            model_memory_bytes=10 * 1024**3,
            kv_cache_per_token_bytes=1024,
            activation_memory_bytes=2 * 1024**3,
            overhead_bytes=int(1.5 * 1024**3)
        )

        expected_total = 10 * 1024**3 + 1024 + 2 * 1024**3 + int(1.5 * 1024**3)
        self.assertEqual(breakdown.total_bytes, expected_total)

    def test_scale_kv_cache(self):
        """Test KV cache scaling."""
        breakdown = MemoryBreakdown(
            model_memory_bytes=10 * 1024**3,
            kv_cache_per_token_bytes=1024,  # 1KB per token
            activation_memory_bytes=2 * 1024**3,
            overhead_bytes=int(1.5 * 1024**3)
        )

        self.assertEqual(breakdown.scale_kv_cache(1000), 1024 * 1000)  # 1MB
        self.assertEqual(breakdown.scale_kv_cache(2000), 1024 * 2000)  # 2MB

    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = MemoryBreakdown(
            model_memory_bytes=10 * 1024**3,
            kv_cache_per_token_bytes=1024,
            activation_memory_bytes=2 * 1024**3,
            overhead_bytes=int(1.5 * 1024**3)
        )

        result = breakdown.to_dict()
        self.assertEqual(result['model_memory_bytes'], 10 * 1024**3)
        self.assertEqual(result['kv_cache_per_token_bytes'], 1024)
        self.assertEqual(result['activation_memory_bytes'], 2 * 1024**3)
        self.assertEqual(result['overhead_bytes'], int(1.5 * 1024**3))
        expected_total = 10 * 1024**3 + 1024 + 2 * 1024**3 + int(1.5 * 1024**3)
        self.assertEqual(result['total_bytes'], expected_total)


class TestMemoryLimits(unittest.TestCase):
    """Test MemoryLimits data class."""

    def test_memory_limits_creation(self):
        """Test creating MemoryLimits instance."""
        limits = MemoryLimits(
            max_model_memory_bytes=40 * 1024**3,  # 40GB in bytes
            max_kv_cache_bytes=30 * 1024**3,      # 30GB in bytes
            max_total_memory_bytes=80 * 1024**3,  # 80GB in bytes
            reserved_memory_bytes=10 * 1024**3    # 10GB in bytes
        )

        self.assertEqual(limits.max_model_memory_bytes, 40 * 1024**3)
        self.assertEqual(limits.max_kv_cache_bytes, 30 * 1024**3)
        self.assertEqual(limits.max_total_memory_bytes, 80 * 1024**3)
        self.assertEqual(limits.reserved_memory_bytes, 10 * 1024**3)

    def test_available_for_kv_cache(self):
        """Test available KV cache calculation."""
        limits = MemoryLimits(
            max_model_memory_bytes=40 * 1024**3,
            max_kv_cache_bytes=35 * 1024**3,
            max_total_memory_bytes=80 * 1024**3,
            reserved_memory_bytes=10 * 1024**3
        )

        # 80 - 40 - 10 = 30 GB available
        self.assertEqual(limits.available_for_kv_cache_bytes, 30 * 1024**3)

    def test_available_for_kv_cache_negative(self):
        """Test available KV cache when over limit."""
        limits = MemoryLimits(
            max_model_memory_bytes=60 * 1024**3,
            max_kv_cache_bytes=30 * 1024**3,
            max_total_memory_bytes=80 * 1024**3,
            reserved_memory_bytes=25 * 1024**3
        )

        # 80 - 60 - 25 = -5, should return 0
        self.assertEqual(limits.available_for_kv_cache_bytes, 0)


class TestGPUResourceManager(unittest.TestCase):
    """Test GPUResourceManager class."""

    @patch('llm_optimizer.resources.gpu_manager.get_gpu_specs')
    @patch('llm_optimizer.resources.gpu_manager.get_precision_tflops')
    def test_get_total_resources(self, mock_tflops, mock_specs):
        """Test getting total GPU resources."""
        mock_specs.return_value = {
            "VRAM_GB": 80,
            "Memory_Bandwidth_GBs": 2000
        }
        mock_tflops.return_value = 50.0

        manager = GPUResourceManager()
        resources = manager.get_total_resources(2, "A100", "fp16")

        self.assertEqual(resources.total_tflops, 100.0)  # 2 * 50
        self.assertEqual(resources.total_memory_bytes, 160 * 1024**3)  # 2 * 80GB in bytes
        self.assertEqual(resources.total_bandwidth_bytes_per_sec, 4000 * 1024**3)  # 2 * 2000GB/s in bytes/s
        self.assertEqual(resources.num_gpus, 2)
        self.assertEqual(resources.gpu_name, "A100")
        self.assertEqual(resources.precision, "fp16")

    @patch('llm_optimizer.resources.gpu_manager.get_gpu_specs')
    @patch('llm_optimizer.resources.gpu_manager.get_precision_tflops')
    def test_multiple_calls(self, mock_tflops, mock_specs):
        """Test that multiple calls work correctly without caching."""
        mock_specs.return_value = {
            "VRAM_GB": 80,
            "Memory_Bandwidth_GBs": 2000
        }
        mock_tflops.return_value = 50.0

        manager = GPUResourceManager()

        # First call
        resources1 = manager.get_total_resources(2, "A100", "fp16")

        # Second call with same parameters
        resources2 = manager.get_total_resources(2, "A100", "fp16")

        # Should have same values but be different objects (no caching)
        self.assertEqual(resources1, resources2)
        self.assertTrue(resources1 is not resources2)

        # Mock should be called twice
        self.assertEqual(mock_specs.call_count, 2)
        self.assertEqual(mock_tflops.call_count, 2)

    def test_calculate_memory_limits(self):
        """Test memory limit calculations."""
        manager = GPUResourceManager()

        gpu_resources = GPUResources(
            per_gpu_tflops=100.0,
            per_gpu_memory_bytes=80 * 1024**3,  # 80GB in bytes
            per_gpu_bandwidth_bytes_per_sec=2000 * 1024**3,  # 2000GB/s in bytes/s
            num_gpus=1,
            gpu_name="A100",
            precision="fp16"
        )

        limits = manager.calculate_memory_limits(
            gpu_resources,
            memory_utilization=0.9
        )

        # Total available: 80GB * 0.9 = 72GB in bytes
        self.assertEqual(limits.max_total_memory_bytes, int(72 * 1024**3))

        # Model memory: 72GB * 0.45 = 32.4GB in bytes
        self.assertEqual(limits.max_model_memory_bytes, int(72 * 0.45 * 1024**3))

        # Reserved: max(72GB * 0.1, 2.0GB * 1) = 7.2GB in bytes
        self.assertEqual(limits.reserved_memory_bytes, int(7.2 * 1024**3))

        # KV cache: 72GB - 32.4GB - 7.2GB = 32.4GB in bytes (approximately)
        expected_kv_cache = int(72 * 1024**3) - int(72 * 0.45 * 1024**3) - int(7.2 * 1024**3)
        self.assertEqual(limits.max_kv_cache_bytes, expected_kv_cache)

    def test_estimate_concurrency_limits(self):
        """Test concurrency limit estimation."""
        manager = GPUResourceManager()

        gpu_resources = GPUResources(
            per_gpu_tflops=100.0,
            per_gpu_memory_bytes=80 * 1024**3,  # 80GB in bytes
            per_gpu_bandwidth_bytes_per_sec=2000 * 1024**3,  # 2000GB/s in bytes/s
            num_gpus=1,
            gpu_name="A100",
            precision="fp16"
        )

        memory_limits = MemoryLimits(
            max_model_memory_bytes=30 * 1024**3,  # 30GB in bytes
            max_kv_cache_bytes=40 * 1024**3,      # 40GB in bytes
            max_total_memory_bytes=80 * 1024**3,  # 80GB in bytes
            reserved_memory_bytes=10 * 1024**3    # 10GB in bytes
        )

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        max_concurrent = manager.estimate_concurrency_limits(
            model_config,
            gpu_resources,
            memory_limits,
            avg_sequence_length=1024
        )

        # Should return a reasonable positive number
        self.assertGreater(max_concurrent, 0)
        self.assertLess(max_concurrent, 1000)  # Sanity check

    def test_compute_memory_ratio(self):
        """Test compute to memory ratio calculation."""
        manager = GPUResourceManager()

        gpu_resources = GPUResources(
            per_gpu_tflops=100.0,  # 100 TFLOPS = 100e12 FLOPS
            per_gpu_memory_bytes=80 * 1024**3,  # 80GB in bytes
            per_gpu_bandwidth_bytes_per_sec=2000 * 1024**3,  # 2000GB/s in bytes/s = 2e12 bytes/s
            num_gpus=1,
            gpu_name="A100",
            precision="fp16"
        )

        ratio = manager.get_compute_memory_ratio(gpu_resources)

        # 100e12 / (2000 * 1024^3) = approximately 46.57
        expected_ratio = 100e12 / (2000 * 1024**3)
        self.assertAlmostEqual(ratio, expected_ratio, places=2)

    def test_is_compute_bound(self):
        """Test compute-bound detection."""
        manager = GPUResourceManager()

        gpu_resources = GPUResources(
            per_gpu_tflops=100.0,
            per_gpu_memory_bytes=80 * 1024**3,  # 80GB in bytes
            per_gpu_bandwidth_bytes_per_sec=2000 * 1024**3,  # 2000GB/s in bytes/s
            num_gpus=1,
            gpu_name="A100",
            precision="fp16"
        )

        # Arithmetic intensity > compute/memory ratio means compute-bound
        self.assertTrue(manager.is_compute_bound(gpu_resources, 60.0))
        self.assertFalse(manager.is_compute_bound(gpu_resources, 40.0))


class TestModelMemoryCalculator(unittest.TestCase):
    """Test ModelMemoryCalculator class."""

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_model_memory_numeric(self, mock_bytes):
        """Test model memory calculation with numeric params."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        memory_bytes = calculator.calculate_model_memory(model_config, "fp16")

        # 7e9 * 2 bytes = 14,000,000,000 bytes
        self.assertEqual(memory_bytes, 14_000_000_000)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_model_memory_small(self, mock_bytes):
        """Test model memory calculation with smaller model."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=1_000_000_000,  # 1 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        memory_bytes = calculator.calculate_model_memory(model_config, "fp16")

        # 1e9 * 2 bytes = 2,000,000,000 bytes
        self.assertEqual(memory_bytes, 2_000_000_000)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_kv_cache_memory(self, mock_bytes):
        """Test KV cache memory calculation."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        kv_memory_bytes = calculator.calculate_kv_cache_memory(
            model_config,
            sequence_length=2048,
            batch_size=16,
            precision="fp16"
        )

        # Verify the calculation
        # 2 * 32 * 32 * 128 * 2048 * 16 * 2
        expected = 2 * 32 * 32 * 128 * 2048 * 16 * 2
        self.assertEqual(kv_memory_bytes, expected)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_kv_cache_with_mqa(self, mock_bytes):
        """Test KV cache calculation with Multi-Query Attention."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            num_kv_heads=8  # MQA with 8 KV heads
        )

        kv_memory_bytes = calculator.calculate_kv_cache_memory(
            model_config,
            sequence_length=2048,
            batch_size=16,
            precision="fp16"
        )

        # Should use num_kv_heads (8) instead of num_heads (32)
        expected = 2 * 32 * 8 * 128 * 2048 * 16 * 2
        self.assertEqual(kv_memory_bytes, expected)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_activation_memory(self, mock_bytes):
        """Test activation memory calculation."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        activation_bytes = calculator.calculate_activation_memory(
            model_config,
            batch_size=16,
            sequence_length=2048,
            precision="fp16"
        )

        # 16 * 2048 * 4096 * 32 * 4 * 2
        expected = 16 * 2048 * 4096 * 32 * 4 * 2
        self.assertEqual(activation_bytes, expected)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_calculate_total_memory_needed(self, mock_bytes):
        """Test total memory calculation with breakdown."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator(overhead_factor=1.2)

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        breakdown = calculator.calculate_total_memory_needed(
            model_config,
            batch_size=16,
            sequence_length=2048,
            model_precision="fp16"
        )

        # Verify components are calculated
        self.assertGreater(breakdown.model_memory_bytes, 0)
        self.assertGreater(breakdown.kv_cache_per_token_bytes, 0)
        self.assertGreater(breakdown.activation_memory_bytes, 0)
        self.assertGreater(breakdown.overhead_bytes, 0)

        # Verify overhead is 20% of subtotal
        subtotal = (
            breakdown.model_memory_bytes +
            breakdown.kv_cache_per_token_bytes * 16 * 2048 +
            breakdown.activation_memory_bytes
        )
        self.assertLess(abs(breakdown.overhead_bytes - subtotal * 0.2), 1000)  # Allow 1KB tolerance for int rounding

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_multiple_calls(self, mock_bytes):
        """Test that multiple calls work correctly without caching."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator()

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        # First call
        breakdown1 = calculator.calculate_total_memory_needed(
            model_config,
            batch_size=16,
            sequence_length=2048,
            model_precision="fp16"
        )

        # Second call with same parameters
        breakdown2 = calculator.calculate_total_memory_needed(
            model_config,
            batch_size=16,
            sequence_length=2048,
            model_precision="fp16"
        )

        # Should have same values but be different objects (no caching)
        self.assertEqual(breakdown1, breakdown2)
        self.assertTrue(breakdown1 is not breakdown2)

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_estimate_max_batch_size(self, mock_bytes):
        """Test maximum batch size estimation."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator(overhead_factor=1.0)  # No overhead for simplicity

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        max_batch = calculator.estimate_max_batch_size(
            model_config,
            available_memory_bytes=80 * 1024**3,  # 80GB in bytes
            sequence_length=2048,
            precision="fp16"
        )

        # Should return a reasonable batch size
        self.assertGreater(max_batch, 0)
        self.assertLess(max_batch, 1000)  # Sanity check

    @patch('llm_optimizer.resources.memory_calculator.get_precision_bytes_per_param')
    def test_estimate_max_sequence_length(self, mock_bytes):
        """Test maximum sequence length estimation."""
        mock_bytes.return_value = 2  # fp16

        calculator = ModelMemoryCalculator(overhead_factor=1.0)  # No overhead for simplicity

        model_config = ModelConfig(
            num_params=7_000_000_000,  # 7 billion as int
            num_layers=32,
            hidden_dim=4096,
            num_heads=32
        )

        max_seq = calculator.estimate_max_sequence_length(
            model_config,
            available_memory_bytes=80 * 1024**3,  # 80GB in bytes
            batch_size=16,
            precision="fp16"
        )

        # Should return a reasonable sequence length
        self.assertGreater(max_seq, 0)
        self.assertLess(max_seq, 100000)  # Sanity check


if __name__ == "__main__":
    unittest.main()
