"""
Core implementation of the LLMOptima engine
"""

import os
import logging
import torch
from typing import Dict, Optional, Any, Union
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import gc
import psutil
import numpy as np
import time

from .optimizers import (
    QuantizationOptimizer,
    PruningOptimizer, 
    DistillationOptimizer,
    InferenceOptimizer
)
from .utils import OptimizationMetrics, OptimizationResult

logger = logging.getLogger(__name__)

class LLMOptima:
    """Main optimization engine for Large Language Models"""
    
    def __init__(self, model_path: str, optimization_level: str = 'balanced'):
        """
        Initialize the LLMOptima engine
        
        Args:
            model_path: Path to the model or model identifier
            optimization_level: Level of optimization aggressiveness 
                                ('conservative', 'balanced', 'aggressive')
        """
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.optimizers = {
            'quantization': QuantizationOptimizer(),
            'pruning': PruningOptimizer(), 
            'distillation': DistillationOptimizer(),
            'inference': InferenceOptimizer()
        }
        self.metrics = OptimizationMetrics()
        
        logger.info(f"Initialized LLMOptima for model: {model_path}")
        logger.info(f"Optimization level: {optimization_level}")
    
    def optimize_model(self, target_metrics: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Complete model optimization pipeline
        
        Args:
            target_metrics: Target optimization metrics, such as speed improvement,
                           size reduction, and accuracy retention
        
        Returns:
            OptimizationResult object with optimization results
        """
        logger.info("🚀 Starting LLMOptima optimization...")
        
        # Step 1: Model Analysis
        analysis = self._analyze_model()
        
        # Step 2: Optimization Strategy Selection
        strategy = self._select_optimization_strategy(analysis, target_metrics)
        
        # Step 3: Apply Optimizations
        optimized_model = self._apply_optimizations(strategy)
        
        # Step 4: Validation & Benchmarking
        results = self._validate_optimization(optimized_model)
        
        logger.info("✅ Optimization complete!")
        
        return OptimizationResult(
            original_model=self.model_path,
            optimized_model=optimized_model,
            performance_gains=results,
            strategy_used=strategy
        )
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model structure and identify optimization opportunities"""
        logger.info("Analyzing model structure...")
        
        model = self._load_model()
        analysis = {}
        
        # Get basic model info
        analysis['model_type'] = type(model).__name__
        analysis['framework'] = self._detect_framework(model)
        
        # Analyze parameters
        analysis['parameter_count'] = self._count_parameters(model)
        analysis['parameter_stats'] = self._analyze_parameter_distribution(model)
        
        # Memory usage
        analysis['memory_usage'] = self._measure_memory_usage(model)
        
        # Layer analysis
        analysis['layer_structure'] = self._analyze_layer_structure(model)
        
        # Inference analysis
        analysis['inference_profile'] = self._profile_inference(model)
        
        # Identify potential bottlenecks
        analysis['optimization_opportunities'] = self._identify_optimization_opportunities(model, analysis)
        
        # Clean up model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return analysis
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the framework used by the model"""
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            return 'huggingface'
        elif isinstance(model, torch.nn.Module):
            return 'pytorch'
        else:
            return 'unknown'
    
    def _count_parameters(self, model: Any) -> Dict[str, int]:
        """Count total, trainable, and non-trainable parameters"""
        if isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            return {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': non_trainable_params
            }
        else:
            return {'total': 0, 'trainable': 0, 'non_trainable': 0}
    
    def _analyze_parameter_distribution(self, model: Any) -> Dict[str, Any]:
        """Analyze weight distribution statistics"""
        stats = {}
        
        if isinstance(model, torch.nn.Module):
            # Sample parameters from different layer types for analysis
            attention_weights = []
            ffn_weights = []
            embedding_weights = []
            other_weights = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    sample = param.data.flatten()[:1000].cpu().numpy()
                    
                    if 'attention' in name or 'attn' in name:
                        attention_weights.extend(sample)
                    elif 'ffn' in name or 'feed_forward' in name or 'mlp' in name:
                        ffn_weights.extend(sample)
                    elif 'embed' in name:
                        embedding_weights.extend(sample)
                    else:
                        other_weights.extend(sample)
            
            # Calculate statistics for each parameter group
            stats['attention'] = self._calculate_weight_stats(attention_weights)
            stats['ffn'] = self._calculate_weight_stats(ffn_weights)
            stats['embedding'] = self._calculate_weight_stats(embedding_weights)
            stats['other'] = self._calculate_weight_stats(other_weights)
            
        return stats
    
    def _calculate_weight_stats(self, weights: list) -> Dict[str, float]:
        """Calculate statistics for a group of weights"""
        if not weights:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'sparsity': 0}
        
        weights = np.array(weights)
        sparsity = np.count_nonzero(np.abs(weights) < 1e-6) / len(weights)
        
        return {
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'sparsity': float(sparsity)
        }
    
    def _measure_memory_usage(self, model: Any) -> Dict[str, float]:
        """Measure memory usage of the model"""
        result = {'cpu_mb': 0, 'gpu_mb': 0}
        
        # Measure CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory_before = process.memory_info().rss / 1024 / 1024
        
        # Measure GPU memory if available
        gpu_memory_before = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Force a garbage collection to get accurate readings
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get memory after GC
        cpu_memory_after = process.memory_info().rss / 1024 / 1024
        gpu_memory_after = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
        
        result['cpu_mb'] = cpu_memory_after
        result['gpu_mb'] = gpu_memory_after
        
        return result
    
    def _analyze_layer_structure(self, model: Any) -> Dict[str, Any]:
        """Analyze the layer structure of the model"""
        layer_stats = {
            'total_layers': 0,
            'layer_types': {},
            'attention_heads': 0,
            'hidden_size': 0
        }
        
        if hasattr(model, 'config'):
            config = model.config
            # Get model architecture details from config
            layer_stats['hidden_size'] = getattr(config, 'hidden_size', 0)
            layer_stats['attention_heads'] = getattr(config, 'num_attention_heads', 0)
            layer_stats['num_layers'] = getattr(config, 'num_hidden_layers', 0)
            layer_stats['vocab_size'] = getattr(config, 'vocab_size', 0)
        
        # Count layer types
        if isinstance(model, torch.nn.Module):
            for name, module in model.named_modules():
                module_type = type(module).__name__
                
                if module_type in layer_stats['layer_types']:
                    layer_stats['layer_types'][module_type] += 1
                else:
                    layer_stats['layer_types'][module_type] = 1
                
                layer_stats['total_layers'] += 1
        
        return layer_stats
    
    def _profile_inference(self, model: Any) -> Dict[str, Any]:
        """Profile inference characteristics"""
        inference_profile = {
            'latency_ms': 0,
            'throughput': 0,
            'memory_utilization': 0
        }
        
        if not isinstance(model, torch.nn.Module):
            return inference_profile
        
        # Create a sample input
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except:
            logger.warning("Could not load tokenizer. Using random inputs for profiling.")
        
        # Try to move model to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_device = next(model.parameters()).device
        
        # Only move if needed to avoid unnecessary copies
        if model_device != device:
            try:
                model = model.to(device)
            except Exception as e:
                logger.warning(f"Could not move model to {device}: {e}")
                device = model_device
        
        # Create input data
        if tokenizer:
            sample_text = "This is a sample text to profile the model's inference performance."
            inputs = tokenizer(sample_text, return_tensors="pt").to(device)
        else:
            # Random inputs
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                vocab_size = model.config.vocab_size
                seq_length = 32
                batch_size = 1
                inputs = {
                    'input_ids': torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
                }
            else:
                logger.warning("Could not create appropriate inputs for profiling.")
                return inference_profile
        
        # Warm-up run
        with torch.no_grad():
            try:
                _ = model(**inputs)
            except Exception as e:
                logger.warning(f"Warm-up inference failed: {e}")
                return inference_profile
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model(**inputs)
        end_time = time.time()
        
        # Calculate latency
        inference_profile['latency_ms'] = (end_time - start_time) * 1000
        
        # Get memory utilization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            inference_profile['memory_utilization'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Try to estimate throughput (tokens per second)
        seq_length = inputs['input_ids'].size(1) if 'input_ids' in inputs else 32
        inference_profile['throughput'] = seq_length / (end_time - start_time)
        
        return inference_profile
    
    def _identify_optimization_opportunities(self, model: Any, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Identify optimization opportunities based on model analysis"""
        opportunities = {}
        
        # Check for quantization opportunities
        param_stats = analysis.get('parameter_stats', {})
        weight_ranges = [stats.get('max', 0) - stats.get('min', 0) 
                         for group, stats in param_stats.items()]
        
        if weight_ranges:
            avg_range = sum(weight_ranges) / len(weight_ranges)
            opportunities['quantization'] = min(1.0, avg_range / 10.0)
        
        # Check for pruning opportunities
        sparsity_values = [stats.get('sparsity', 0) for group, stats in param_stats.items()]
        if sparsity_values:
            natural_sparsity = sum(sparsity_values) / len(sparsity_values)
            opportunities['pruning'] = min(1.0, natural_sparsity + 0.3)
        
        # Check for attention optimization opportunities
        if analysis.get('layer_structure', {}).get('attention_heads', 0) > 0:
            opportunities['attention_optimization'] = 0.8
        
        # Check for KV cache optimization opportunities
        if 'attention' in str(type(model)).lower() or 'transformer' in str(type(model)).lower():
            opportunities['kv_cache_optimization'] = 0.9
        
        # Check for batching optimization opportunities
        if analysis.get('inference_profile', {}).get('latency_ms', 0) > 100:
            opportunities['batching_optimization'] = 0.7
        
        return opportunities
    
    def _select_optimization_strategy(self, analysis: Dict[str, Any], 
                                      target_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Select optimal combination of optimization techniques"""
        logger.info("Selecting optimization strategy...")
        
        # Default strategies based on optimization level
        strategies = {
            'conservative': {
                'quantization': {'method': 'int8', 'target_compression': 0.3},
                'pruning': {'method': 'magnitude', 'sparsity_target': 0.3},
                'inference': {'techniques': ['kv_cache']}
            },
            'balanced': {
                'quantization': {'method': 'mixed_precision', 'target_compression': 0.5},
                'pruning': {'method': 'structured', 'sparsity_target': 0.5},
                'inference': {'techniques': ['kv_cache', 'attention', 'batching']}
            },
            'aggressive': {
                'quantization': {'method': 'int4', 'target_compression': 0.7},
                'pruning': {'method': 'lottery_ticket', 'sparsity_target': 0.7},
                'inference': {'techniques': ['kv_cache', 'attention', 'batching', 'speculative']}
            }
        }
        
        # Use selected strategy based on optimization level
        strategy = strategies.get(self.optimization_level, strategies['balanced']).copy()
        
        # Adjust strategy based on analysis
        opportunities = analysis.get('optimization_opportunities', {})
        
        # Adjust quantization method based on model characteristics
        param_count = analysis.get('parameter_count', {}).get('total', 0)
        if param_count > 0:
            billions = param_count / 1e9
            
            # For very large models, use more aggressive quantization
            if billions > 20 and self.optimization_level != 'conservative':
                strategy['quantization']['method'] = 'int4'
            # For small models, use less aggressive quantization
            elif billions < 1 and self.optimization_level == 'aggressive':
                strategy['quantization']['method'] = 'int8'
        
        # Adjust pruning based on natural sparsity
        natural_sparsity = opportunities.get('pruning', 0.0)
        if natural_sparsity > 0.5 and strategy.get('pruning', {}).get('sparsity_target', 0) < 0.7:
            strategy['pruning']['sparsity_target'] = min(0.8, natural_sparsity + 0.2)
        
        # Adjust inference optimizations based on identified opportunities
        inference_techniques = strategy.get('inference', {}).get('techniques', [])
        
        if opportunities.get('attention_optimization', 0) > 0.7 and 'attention' not in inference_techniques:
            inference_techniques.append('attention')
            
        if opportunities.get('kv_cache_optimization', 0) > 0.7 and 'kv_cache' not in inference_techniques:
            inference_techniques.append('kv_cache')
            
        strategy['inference']['techniques'] = inference_techniques
        
        # Adjust strategy based on target metrics if provided
        if target_metrics:
            if 'speed_improvement' in target_metrics:
                target_speed = target_metrics['speed_improvement']
                # For higher speed targets, use more aggressive techniques
                if target_speed > 5.0 and self.optimization_level != 'aggressive':
                    strategy['quantization']['method'] = 'int4'
                    strategy['pruning']['sparsity_target'] = min(0.8, strategy['pruning']['sparsity_target'] + 0.1)
                    
                    if 'speculative' not in strategy['inference']['techniques']:
                        strategy['inference']['techniques'].append('speculative')
            
            if 'size_reduction' in target_metrics:
                target_size = target_metrics['size_reduction']
                # For higher size reduction targets, adjust quantization and pruning
                if target_size > 0.7:
                    strategy['quantization']['target_compression'] = min(0.9, target_size)
                    strategy['pruning']['sparsity_target'] = min(0.9, target_size)
            
            if 'accuracy_retention' in target_metrics:
                accuracy_retention = target_metrics['accuracy_retention']
                # For higher accuracy retention, use less aggressive techniques
                if accuracy_retention > 0.97:
                    if strategy['quantization']['method'] == 'int4':
                        strategy['quantization']['method'] = 'int8'
                    strategy['pruning']['sparsity_target'] = max(0.3, strategy['pruning']['sparsity_target'] - 0.2)
            
        # Add hardware-specific optimizations if available
        if hasattr(self, 'hardware_profile'):
            strategy['hardware_profile'] = self.hardware_profile
            
        return strategy
    
    def _apply_optimizations(self, strategy: Dict[str, Any]) -> Any:
        """Apply the selected optimization techniques"""
        logger.info("Applying optimizations...")
        
        model = self._load_model()
        
        # Apply quantization if in strategy
        if 'quantization' in strategy:
            logger.info(f"Applying quantization with method: {strategy['quantization'].get('method')}")
            quantization_result = self.optimizers['quantization'].optimize(
                model, **strategy['quantization']
            )
            model = quantization_result['model']
            
            # Store metrics
            self.metrics.update_optimization_metrics('quantization', {
                'compression_ratio': quantization_result.get('compression_ratio', 0),
                'accuracy_retention': quantization_result.get('accuracy_retention', 0),
                'speed_improvement': quantization_result.get('speed_improvement', 0)
            })
        
        # Apply pruning if in strategy
        if 'pruning' in strategy:
            logger.info(f"Applying pruning with method: {strategy['pruning'].get('method')}")
            pruning_result = self.optimizers['pruning'].optimize(
                model, **strategy['pruning']
            )
            model = pruning_result['model']
            
            # Store metrics
            self.metrics.update_optimization_metrics('pruning', {
                'sparsity_achieved': pruning_result.get('sparsity_achieved', 0),
                'size_reduction': pruning_result.get('size_reduction', 0),
                'speed_improvement': pruning_result.get('speed_improvement', 0)
            })
        
        # Apply inference optimizations if in strategy
        if 'inference' in strategy:
            logger.info(f"Applying inference optimizations: {strategy['inference'].get('techniques')}")
            inference_result = self.optimizers['inference'].optimize_inference(
                model, techniques=strategy['inference'].get('techniques', [])
            )
            model = inference_result
        
        return model
    
    def _validate_optimization(self, optimized_model: Any) -> Dict[str, float]:
        """Validate optimization results and measure performance gains"""
        logger.info("Validating optimization and measuring performance...")
        
        # Load original model for comparison
        original_model = self._load_model()
        
        # Set baseline metrics
        baseline_metrics = self.metrics.set_baseline(original_model)
        
        # Measure optimized metrics
        optimized_metrics = self.metrics.measure_optimized(optimized_model)
        
        # Calculate gains
        gains = self.metrics.calculate_gains()
        
        # Clean up to free memory
        del original_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return gains
    
    def _load_model(self) -> Any:
        """Load the model from the specified path"""
        logger.info(f"Loading model from {self.model_path}...")
        
        try:
            # Try to load as a HuggingFace model first
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                logger.info("Loaded causal language model from HuggingFace")
                return model
            except:
                try:
                    model = AutoModel.from_pretrained(self.model_path)
                    logger.info("Loaded base model from HuggingFace")
                    return model
                except:
                    pass
            
            # Try to load as a PyTorch model
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                model = torch.load(self.model_path)
                logger.info("Loaded PyTorch model")
                return model
            
            # Try as a directory containing model files
            if os.path.isdir(self.model_path):
                # Check for common model files
                for filename in os.listdir(self.model_path):
                    if filename.endswith('.pt') or filename.endswith('.pth'):
                        model = torch.load(os.path.join(self.model_path, filename))
                        logger.info(f"Loaded PyTorch model from {filename}")
                        return model
            
            raise ValueError(f"Could not load model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def update_hardware_profile(self, hardware_profile: Dict[str, Any]) -> None:
        """Update hardware profile for optimization decisions"""
        self.hardware_profile = hardware_profile
        logger.info(f"Updated hardware profile: {hardware_profile}")
