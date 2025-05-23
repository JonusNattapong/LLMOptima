"""
Optimization Metrics & Benchmarking System
Author: @JonusNattapong

Comprehensive metrics tracking for LLM optimization performance
"""

import time
import psutil
import platform
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some metrics will be disabled")

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    logger.warning("NVIDIA ML Python not available - GPU metrics will be disabled")


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    inference_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class AccuracyMetrics:
    """Accuracy and quality metrics"""
    perplexity: float = 0.0
    bleu_score: float = 0.0
    rouge_score: Dict[str, float] = field(default_factory=dict)
    accuracy_score: float = 0.0
    semantic_similarity: float = 0.0
    coherence_score: float = 0.0


@dataclass
class OptimizationMetrics:
    """Complete optimization metrics"""
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_level: str = "unknown"
    model_name: str = "unknown"


class MetricsCollector:
    """
    Advanced metrics collection and benchmarking system
    
    Features:
    - Real-time performance monitoring
    - GPU/CPU utilization tracking
    - Memory usage profiling
    - Accuracy validation
    - Benchmark comparisons
    - Historical trend analysis
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics_history: List[OptimizationMetrics] = []
        self.benchmark_data: Dict[str, Any] = {}
        self.system_info = self._collect_system_info()
        
        logger.info("📊 MetricsCollector initialized")
        logger.info(f"🖥️  System: {self.system_info['platform']} {self.system_info['architecture']}")
        logger.info(f"💾 RAM: {self.system_info['memory_gb']:.1f} GB")
        if self.system_info.get('gpu_info'):
            logger.info(f"🎮 GPU: {self.system_info['gpu_info']}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context"""
        
        info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_cores': psutil.cpu_count(),
            'cpu_threads': psutil.cpu_count(logical=True),
        }
        
        # GPU information
        if NVIDIA_ML_AVAILABLE:
            try:
                gpu_count = nvml.nvmlDeviceGetCount()
                if gpu_count > 0:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    info['gpu_info'] = {
                        'name': gpu_name,
                        'memory_gb': memory_info.total / (1024**3),
                        'count': gpu_count
                    }
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
        
        return info
    
    def benchmark_model_performance(self,
                                  model: Any,
                                  test_inputs: List[str],
                                  optimization_level: str = "unknown",
                                  model_name: str = "unknown") -> OptimizationMetrics:
        """
        Comprehensive model performance benchmark
        
        Args:
            model: Model to benchmark
            test_inputs: List of test input strings
            optimization_level: Optimization level applied
            model_name: Name of the model
            
        Returns:
            Complete optimization metrics
        """
        
        logger.info(f"🏃 Starting performance benchmark for {model_name}")
        
        # Initialize metrics
        performance_metrics = PerformanceMetrics()
        accuracy_metrics = AccuracyMetrics()
        
        # Measure inference performance
        performance_metrics = self._benchmark_inference_performance(
            model, test_inputs, performance_metrics
        )
        
        # Measure resource utilization
        performance_metrics = self._measure_resource_utilization(performance_metrics)
        
        # Measure model size
        performance_metrics.model_size_mb = self._measure_model_size(model)
        
        # Measure accuracy (if possible)
        # accuracy_metrics = self._measure_accuracy(model, test_inputs, accuracy_metrics)
        
        # Create complete metrics object
        metrics = OptimizationMetrics(
            performance=performance_metrics,
            accuracy=accuracy_metrics,
            optimization_level=optimization_level,
            model_name=model_name
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        logger.info("✅ Performance benchmark completed")
        self._print_benchmark_results(metrics)
        
        return metrics
    
    def _benchmark_inference_performance(self,
                                       model: Any,
                                       test_inputs: List[str],
                                       metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Benchmark inference performance"""
        
        logger.info("⚡ Measuring inference performance...")
        
        inference_times = []
        total_tokens = 0
        
        # Warmup runs
        for _ in range(3):
            if test_inputs:
                self._run_inference(model, test_inputs[0])
        
        # Actual benchmark runs
        for input_text in test_inputs[:10]:  # Limit to 10 samples for speed
            start_time = time.time()
            output = self._run_inference(model, input_text)
            end_time = time.time()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            # Estimate token count (rough approximation)
            token_count = len(input_text.split()) + len(str(output).split())
            total_tokens += token_count
        
        if inference_times:
            metrics.inference_time_ms = sum(inference_times) / len(inference_times)
            metrics.latency_p50_ms = sorted(inference_times)[len(inference_times)//2]
            metrics.latency_p95_ms = sorted(inference_times)[int(len(inference_times)*0.95)]
            metrics.latency_p99_ms = sorted(inference_times)[int(len(inference_times)*0.99)]
            
            total_time_sec = sum(inference_times) / 1000
            metrics.throughput_tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0
        
        return metrics
    
    def _run_inference(self, model: Any, input_text: str) -> Any:
        """Run inference on model (placeholder)"""
        
        # TODO: Implement actual model inference
        # This will depend on the model type and framework
        
        # For now, simulate inference
        time.sleep(0.01)  # Simulate processing time
        return f"Generated response for: {input_text[:50]}..."
    
    def _measure_resource_utilization(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Measure current resource utilization"""
        
        # CPU utilization
        metrics.cpu_utilization_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics.memory_usage_mb = (memory.total - memory.available) / (1024**2)
        
        # GPU utilization (if available)
        if NVIDIA_ML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization_percent = utilization.gpu
                
                # GPU memory
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_usage_mb = memory_info.used / (1024**2)
                
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        return metrics
    
    def _measure_model_size(self, model: Any) -> float:
        """Measure model size in MB"""
        
        # TODO: Implement actual model size measurement
        # This will depend on the model type
        
        # For now, return a placeholder
        return 1000.0  # 1GB placeholder
    
    def compare_optimization_results(self,
                                   baseline_metrics: OptimizationMetrics,
                                   optimized_metrics: OptimizationMetrics) -> Dict[str, float]:
        """
        Compare optimization results between baseline and optimized models
        
        Returns:
            Dictionary with improvement ratios
        """
        
        logger.info("📊 Comparing optimization results...")
        
        baseline_perf = baseline_metrics.performance
        optimized_perf = optimized_metrics.performance
        
        comparison = {}
        
        # Speed improvements
        if baseline_perf.inference_time_ms > 0:
            comparison['speed_improvement'] = baseline_perf.inference_time_ms / optimized_perf.inference_time_ms
        
        if baseline_perf.throughput_tokens_per_sec > 0:
            comparison['throughput_improvement'] = optimized_perf.throughput_tokens_per_sec / baseline_perf.throughput_tokens_per_sec
        
        # Memory improvements
        if baseline_perf.memory_usage_mb > 0:
            comparison['memory_reduction'] = 1 - (optimized_perf.memory_usage_mb / baseline_perf.memory_usage_mb)
        
        if baseline_perf.gpu_memory_usage_mb > 0:
            comparison['gpu_memory_reduction'] = 1 - (optimized_perf.gpu_memory_usage_mb / baseline_perf.gpu_memory_usage_mb)
        
        # Model size reduction
        if baseline_perf.model_size_mb > 0:
            comparison['size_reduction'] = 1 - (optimized_perf.model_size_mb / baseline_perf.model_size_mb)
        
        # Accuracy retention
        if baseline_metrics.accuracy.accuracy_score > 0:
            comparison['accuracy_retention'] = optimized_metrics.accuracy.accuracy_score / baseline_metrics.accuracy.accuracy_score
        
        logger.info("✅ Optimization comparison completed")
        self._print_comparison_results(comparison)
        
        return comparison
    
    def _print_benchmark_results(self, metrics: OptimizationMetrics):
        """Print beautiful benchmark results"""
        
        perf = metrics.performance
        
        print("\n" + "="*60)
        print(f"📊 Benchmark Results - {metrics.model_name}")
        print("="*60)
        
        print(f"⚡ Performance Metrics:")
        print(f"   🚀 Inference Time: {perf.inference_time_ms:.2f} ms")
        print(f"   📈 Throughput: {perf.throughput_tokens_per_sec:.1f} tokens/sec")
        print(f"   ⏱️  Latency P95: {perf.latency_p95_ms:.2f} ms")
        
        print(f"\n💾 Resource Usage:")
        print(f"   💻 CPU: {perf.cpu_utilization_percent:.1f}%")
        print(f"   🧠 Memory: {perf.memory_usage_mb:.1f} MB")
        if perf.gpu_memory_usage_mb > 0:
            print(f"   🎮 GPU Memory: {perf.gpu_memory_usage_mb:.1f} MB")
            print(f"   🎮 GPU Util: {perf.gpu_utilization_percent:.1f}%")
        
        print(f"\n📦 Model Size: {perf.model_size_mb:.1f} MB")
        print(f"🎯 Optimization Level: {metrics.optimization_level}")
        print("="*60 + "\n")
    
    def _print_comparison_results(self, comparison: Dict[str, float]):
        """Print comparison results"""
        
        print("\n" + "="*60)
        print("🔍 Optimization Comparison Results")
        print("="*60)
        
        if 'speed_improvement' in comparison:
            print(f"⚡ Speed Improvement: {comparison['speed_improvement']:.1f}x")
        
        if 'throughput_improvement' in comparison:
            print(f"📈 Throughput Improvement: {comparison['throughput_improvement']:.1f}x")
        
        if 'memory_reduction' in comparison:
            print(f"💾 Memory Reduction: {comparison['memory_reduction']*100:.1f}%")
        
        if 'size_reduction' in comparison:
            print(f"📦 Size Reduction: {comparison['size_reduction']*100:.1f}%")
        
        if 'accuracy_retention' in comparison:
            print(f"🎯 Accuracy Retention: {comparison['accuracy_retention']*100:.1f}%")
        
        print("="*60 + "\n")
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file"""
        
        export_data = {
            'system_info': self.system_info,
            'metrics_history': [
                {
                    'performance': metrics.performance.__dict__,
                    'accuracy': metrics.accuracy.__dict__,
                    'timestamp': metrics.timestamp,
                    'optimization_level': metrics.optimization_level,
                    'model_name': metrics.model_name
                }
                for metrics in self.metrics_history
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Metrics exported to {filepath}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        # Calculate aggregated statistics
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        
        avg_inference_time = sum(m.performance.inference_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.performance.throughput_tokens_per_sec for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.performance.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        report = {
            'summary': {
                'total_benchmarks': len(self.metrics_history),
                'avg_inference_time_ms': avg_inference_time,
                'avg_throughput_tokens_per_sec': avg_throughput,
                'avg_memory_usage_mb': avg_memory,
            },
            'system_info': self.system_info,
            'latest_metrics': recent_metrics[-1].__dict__ if recent_metrics else None,
            'report_timestamp': datetime.now().isoformat()
        }
        
        return report
