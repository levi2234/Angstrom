"""
Memory monitoring utilities for tracking memory usage during video processing.
"""

import psutil
import os
import time
import torch
from typing import Dict, List
import gc


class MemoryMonitor:
    """Monitor memory usage during video processing."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_history: List[Dict[str, float]] = []
        self.start_time = time.time()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = self.process.memory_info()

        # GPU memory (if available)
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        return {
            'process_rss': process_memory.rss / (1024 * 1024),  # MB
            'process_vms': process_memory.vms / (1024 * 1024),  # MB
            'system_available': system_memory.available / (1024 * 1024),  # MB
            'system_total': system_memory.total / (1024 * 1024),  # MB
            'system_percent': system_memory.percent,  # %
            'gpu_memory': gpu_memory,  # MB
            'timestamp': time.time() - self.start_time
        }

    def record_memory(self, label: str = ""):
        """Record current memory usage with optional label."""
        memory_info = self.get_memory_usage()
        memory_info['label'] = label
        self.memory_history.append(memory_info)

    def clear_memory(self):
        """Clear memory and garbage collect."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_summary(self) -> Dict[str, float]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}

        # Calculate statistics
        rss_values = [entry['process_rss'] for entry in self.memory_history]
        gpu_values = [entry['gpu_memory'] for entry in self.memory_history]

        return {
            'max_rss': max(rss_values),
            'min_rss': min(rss_values),
            'avg_rss': sum(rss_values) / len(rss_values),
            'max_gpu': max(gpu_values),
            'min_gpu': min(gpu_values),
            'avg_gpu': sum(gpu_values) / len(gpu_values),
            'total_time': time.time() - self.start_time
        }

    def print_memory_summary(self):
        """Print memory usage summary."""
        summary = self.get_memory_summary()
        if not summary:
            print("No memory data recorded.")
            return

        print("\n=== Memory Usage Summary ===")
        print(f"Total processing time: {summary['total_time']:.2f} seconds")
        print("Process memory (RSS):")
        print(f"  Max: {summary['max_rss']:.1f} MB")
        print(f"  Min: {summary['min_rss']:.1f} MB")
        print(f"  Avg: {summary['avg_rss']:.1f} MB")

        if torch.cuda.is_available():
            print("GPU memory:")
            print(f"  Max: {summary['max_gpu']:.1f} MB")
            print(f"  Min: {summary['min_gpu']:.1f} MB")
            print(f"  Avg: {summary['avg_gpu']:.1f} MB")

    def check_memory_threshold(self, threshold_mb: float = 1000) -> bool:
        """Check if memory usage exceeds threshold."""
        memory_info = self.get_memory_usage()
        return memory_info['process_rss'] > threshold_mb


def estimate_video_memory_requirements(frame_count: int,
                                       width: int,
                                       height: int,
                                       channels: int = 1,
                                       dtype: str = 'float32') -> Dict[str,
                                                                       float]:
    """Estimate memory requirements for video processing.

    Args:
        frame_count (int): Number of frames
        width (int): Frame width
        height (int): Frame height
        channels (int): Number of channels
        dtype (str): Data type ('float32', 'float64', 'uint8')

    Returns:
        Dict[str, float]: Memory requirements in MB
    """
    # Bytes per element for different dtypes
    dtype_bytes = {
        'float32': 4,
        'float64': 8,
        'uint8': 1,
        'int32': 4
    }

    bytes_per_pixel = dtype_bytes.get(dtype, 4)
    total_pixels = frame_count * width * height * channels
    raw_memory = total_pixels * bytes_per_pixel / (1024 * 1024)  # MB

    # Estimate processing overhead
    # - Original video: 1x
    # - Pyramid coefficients: 3x (complex numbers + multiple levels)
    # - Phase/amplitude: 2x
    # - Temporal filtering: 1x
    # - Reconstruction: 1x
    processing_overhead = 8.0

    estimated_memory = raw_memory * processing_overhead

    return {'raw_memory': raw_memory, 'estimated_total': estimated_memory, 'frames_per_gb': (
        1024 * 1024) / (width * height * channels * bytes_per_pixel * processing_overhead)}


def get_safe_chunk_size(available_memory_mb: float, frame_memory_mb: float,
                        safety_factor: float = 0.8) -> int:
    """Calculate safe chunk size for processing.

    Args:
        available_memory_mb (float): Available memory in MB
        frame_memory_mb (float): Memory per frame in MB
        safety_factor (float): Safety factor (0.0-1.0)

    Returns:
        int: Safe chunk size
    """
    safe_memory = available_memory_mb * safety_factor
    max_frames = int(safe_memory / frame_memory_mb)
    return max(1, min(max_frames, 100))  # Between 1 and 100 frames


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        monitor.record_memory("Before function call")

        try:
            result = func(*args, **kwargs)
            monitor.record_memory("After function call")
            return result
        except Exception as e:
            monitor.record_memory("After exception")
            raise e
        finally:
            monitor.print_memory_summary()

    return wrapper
