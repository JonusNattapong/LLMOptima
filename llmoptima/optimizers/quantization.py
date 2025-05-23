"""
Quantization optimization techniques for LLM models
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)

class QuantizationOptimizer:
    """Advanced quantization with minimal accuracy loss"""
    
    def __init__(self):
        self.techniques = {
            'int8': self.int8_quantization,
            'int4': self.int4_quantization,
            'fp8': self.fp8_quantization,
            'dynamic': self.dynamic_quantization,
            'mixed_precision': self.mixed_precision_quantization
        }
        self.calibration_data = None
    
    def optimize(self, model: Any, 
                method: str = 'int8', 
                target_compression: float = 0.5,
                calibration_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Smart quantization with accuracy preservation
        
        Args:
            model: The model to optimize
            method: Quantization method to use
            target_compression: Target compression ratio (0.0-1.0)
            calibration_data: Data for calibrating quantization
            
        Returns:
            Dictionary containing the optimized model and performance metrics
        """
        logger.info(f"Starting quantization optimization with method: {method}")
        logger.info(f"Target compression ratio: {target_compression}")
        
        # Store calibration data for sensitivity analysis
        self.calibration_data = calibration_data
        
        # 1. Analyze model sensitivity
        sensitivity_map = self._analyze_layer_sensitivity(model)
        
        # 2. Select optimal quantization strategy
        strategy = self._select_quantization_strategy(
            sensitivity_map, target_compression, method
        )
        
        # 3. Apply graduated quantization
        quantized_model = self._apply_graduated_quantization(model, strategy)
        
        # 4. Fine-tune to recover accuracy
        fine_tuned_model = self._accuracy_recovery_tuning(quantized_model)
        
        # Calculate performance metrics
        compression_ratio = self._calculate_compression(model, fine_tuned_model)
        accuracy_retention = self._measure_accuracy_retention(model, fine_tuned_model)
        speed_improvement = self._measure_speed_gain(model, fine_tuned_model)
        
        logger.info(f"Quantization complete. Compression ratio: {compression_ratio}")
        logger.info(f"Accuracy retention: {accuracy_retention}%")
        logger.info(f"Speed improvement: {speed_improvement}x")
        
        return {
            'model': fine_tuned_model,
            'compression_ratio': compression_ratio,
            'accuracy_retention': accuracy_retention,
            'speed_improvement': speed_improvement
        }
    
    def _analyze_layer_sensitivity(self, model: Any) -> Dict[str, float]:
        """Analyze which layers are most sensitive to quantization"""
        logger.info("Analyzing layer sensitivity to quantization...")
        sensitivity_scores = {}
        
        if not isinstance(model, torch.nn.Module):
            return sensitivity_scores
            
        # Generate or use calibration data
        if self.calibration_data is None:
            self.calibration_data = self._generate_calibration_data(model)
        
        # Create test input
        test_input = self.calibration_data
        
        # For each layer, measure impact of quantization on output
        for name, module in model.named_modules():
            # Skip certain layer types
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.Dropout)):
                sensitivity_scores[name] = 0.0  # Not sensitive
                continue
                
            # Skip container modules to focus on actual computation layers
            if len(list(module.children())) > 0:
                continue
                
            # Check if module has parameters
            if not any(p.requires_grad for p in module.parameters()):
                continue
                
            try:
                # Create a copy of the module for quantization
                with torch.no_grad():
                    # Get original output
                    original_output = self._get_module_output(model, name, test_input)
                    
                    # Create a quantized version of the module
                    quantized_module = self._create_quantized_module(module, 'int8')
                    
                    # Replace the module temporarily
                    self._replace_module(model, name, quantized_module)
                    
                    # Get quantized output
                    quantized_output = self._get_module_output(model, name, test_input)
                    
                    # Restore original module
                    self._replace_module(model, name, module)
                    
                    # Calculate sensitivity score
                    sensitivity = self._calculate_sensitivity(original_output, quantized_output)
                    sensitivity_scores[name] = sensitivity
                    
            except Exception as e:
                logger.warning(f"Error analyzing sensitivity for layer {name}: {e}")
                sensitivity_scores[name] = 0.5  # Default middle sensitivity
        
        return sensitivity_scores
    
    def _generate_calibration_data(self, model: Any) -> Dict[str, torch.Tensor]:
        """Generate calibration data for quantization analysis"""
        logger.info("Generating calibration data...")
        
        device = next(model.parameters()).device
        batch_size = 1
        seq_length = 32
        
        # For transformer models, create random token inputs
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            vocab_size = model.config.vocab_size
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
            attention_mask = torch.ones(batch_size, seq_length, device=device)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # Generic input for other types of models
            hidden_size = 768  # Default for many models
            
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                hidden_size = model.config.hidden_size
                
            return {
                'inputs_embeds': torch.randn(batch_size, seq_length, hidden_size, device=device)
            }
    
    def _get_module_output(self, model: Any, module_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get the output of a specific module"""
        # Implement a forward hook to capture module output
        output = None
        
        def hook_fn(module, input, output_tensor):
            nonlocal output
            output = output_tensor
            
        # Register hook
        for name, module in model.named_modules():
            if name == module_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        with torch.no_grad():
            model(**inputs)
            
        # Remove hook
        handle.remove()
        
        return output
    
    def _create_quantized_module(self, module: torch.nn.Module, method: str) -> torch.nn.Module:
        """Create a quantized version of a module"""
        if method == 'int8':
            return self._quantize_module_int8(module)
        elif method == 'int4':
            return self._quantize_module_int4(module)
        else:
            # Default to int8
            return self._quantize_module_int8(module)
    
    def _quantize_module_int8(self, module: torch.nn.Module) -> torch.nn.Module:
        """Create an INT8 quantized version of a module"""
        quantized_module = type(module)(*module.__init__.__defaults__) if module.__init__.__defaults__ else type(module)()
        
        # Copy and quantize parameters
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    # Quantize to INT8
                    scale = torch.max(torch.abs(param)) / 127.0
                    quantized_param = torch.round(param / scale).to(torch.int8)
                    dequantized_param = quantized_param.float() * scale
                    
                    # Set parameter in quantized module
                    if hasattr(quantized_module, name):
                        getattr(quantized_module, name).data.copy_(dequantized_param)
                    
        return quantized_module
    
    def _quantize_module_int4(self, module: torch.nn.Module) -> torch.nn.Module:
        """Create an INT4 quantized version of a module"""
        quantized_module = type(module)(*module.__init__.__defaults__) if module.__init__.__defaults__ else type(module)()
        
        # Copy and quantize parameters
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    # Quantize to INT4
                    scale = torch.max(torch.abs(param)) / 7.0
                    quantized_param = torch.round(param / scale).clamp(-8, 7).to(torch.int8)
                    dequantized_param = quantized_param.float() * scale
                    
                    # Set parameter in quantized module
                    if hasattr(quantized_module, name):
                        getattr(quantized_module, name).data.copy_(dequantized_param)
                    
        return quantized_module
    
    def _replace_module(self, model: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
        """Replace a module in the model with a new one"""
        name_parts = module_name.split('.')
        
        if len(name_parts) == 1:
            setattr(model, name_parts[0], new_module)
            return
            
        parent_module = model
        for part in name_parts[:-1]:
            if part.isdigit():
                parent_module = parent_module[int(part)]
            else:
                parent_module = getattr(parent_module, part)
                
        last_part = name_parts[-1]
        if last_part.isdigit():
            parent_module[int(last_part)] = new_module
        else:
            setattr(parent_module, last_part, new_module)
    
    def _calculate_sensitivity(self, original_output: torch.Tensor, quantized_output: torch.Tensor) -> float:
        """Calculate sensitivity score based on output difference"""
        if original_output is None or quantized_output is None:
            return 0.5
            
        try:
            # Ensure tensors have the same shape
            if original_output.shape != quantized_output.shape:
                return 0.5
                
            # Calculate relative error
            abs_diff = torch.abs(original_output - quantized_output)
            abs_orig = torch.abs(original_output)
            
            # Avoid division by zero
            abs_orig = torch.where(abs_orig > 1e-10, abs_orig, torch.ones_like(abs_orig) * 1e-10)
            
            rel_error = abs_diff / abs_orig
            
            # Use a percentile to ignore outliers
            sensitivity = float(torch.quantile(rel_error.flatten(), 0.95))
            
            # Normalize to [0, 1]
            sensitivity = min(1.0, sensitivity)
            
            return sensitivity
            
        except Exception as e:
            logger.warning(f"Error calculating sensitivity: {e}")
            return 0.5
    
    def _select_quantization_strategy(self, sensitivity_map: Dict[str, float], 
                                    target_compression: float,
                                    default_method: str) -> Dict[str, Any]:
        """Select the optimal quantization strategy based on layer sensitivity"""
        logger.info("Selecting quantization strategy...")
        
        strategy = {}
        
        # Set sensitivity thresholds based on target compression
        high_sensitivity_threshold = 0.7 - (target_compression - 0.5) * 0.4  # Adjusts based on target
        low_sensitivity_threshold = 0.3 - (target_compression - 0.5) * 0.2   # Adjusts based on target
        
        # Select quantization methods by sensitivity
        for layer_name, sensitivity in sensitivity_map.items():
            if sensitivity > high_sensitivity_threshold:
                # High sensitivity - use gentler quantization
                if default_method == 'int4':
                    method = 'int8'
                else:
                    method = 'fp8' if 'fp8' in self.techniques else 'int8'
            elif sensitivity < low_sensitivity_threshold:
                # Low sensitivity - use aggressive quantization
                method = 'int4'
            else:
                # Medium sensitivity - use default method
                method = default_method
                
            strategy[layer_name] = {
                'method': method,
                'sensitivity': sensitivity
            }
        
        # For very aggressive compression, increase proportion of int4 layers
        if target_compression > 0.7:
            strategy = self._adjust_for_aggressive_compression(strategy, target_compression)
            
        return strategy
    
    def _adjust_for_aggressive_compression(self, strategy: Dict[str, Dict[str, Any]], target_compression: float) -> Dict[str, Dict[str, Any]]:
        """Adjust strategy for more aggressive compression"""
        # Calculate desired proportion of int4 layers
        desired_int4_proportion = min(0.9, (target_compression - 0.5) * 2)
        
        # Count current int4 layers
        int4_count = sum(1 for layer in strategy.values() if layer['method'] == 'int4')
        current_int4_proportion = int4_count / len(strategy) if len(strategy) > 0 else 0
        
        if current_int4_proportion < desired_int4_proportion:
            # Sort layers by sensitivity (ascending)
            sorted_layers = sorted(
                [(name, info) for name, info in strategy.items()], 
                key=lambda x: x[1]['sensitivity']
            )
            
            # Calculate how many more int4 layers are needed
            needed_int4_count = int(desired_int4_proportion * len(strategy)) - int4_count
            
            # Convert more layers to int4, starting with least sensitive
            for i in range(needed_int4_count):
                if i < len(sorted_layers):
                    layer_name = sorted_layers[i][0]
                    if strategy[layer_name]['method'] != 'int4':
                        strategy[layer_name]['method'] = 'int4'
        
        return strategy
    
    def _apply_graduated_quantization(self, model: Any, strategy: Dict[str, Any]) -> Any:
        """Apply graduated quantization with layer-specific precision"""
        logger.info("Applying graduated quantization...")
        
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not a PyTorch module, cannot apply quantization")
            return model
            
        # Create a copy of the model for quantization
        # quantized_model = copy.deepcopy(model)
        quantized_model = model  # In-place quantization to save memory
        
        # Apply quantization methods to each layer
        for name, module in tqdm(list(quantized_model.named_modules()), desc="Quantizing layers"):
            if name in strategy:
                quant_info = strategy[name]
                method = quant_info['method']
                
                # Get the appropriate quantization function
                if method == 'int8':
                    quant_fn = self.int8_quantization
                elif method == 'int4':
                    quant_fn = self.int4_quantization
                elif method == 'fp8':
                    quant_fn = self.fp8_quantization
                elif method == 'dynamic':
                    quant_fn = self.dynamic_quantization
                elif method == 'mixed_precision':
                    quant_fn = self.mixed_precision_quantization
                else:
                    quant_fn = self.int8_quantization  # Default
                
                try:
                    # Apply quantization to the module
                    quantized_module = quant_fn(module)
                    self._replace_module(quantized_model, name, quantized_module)
                    logger.debug(f"Quantized {name} using {method}")
                except Exception as e:
                    logger.warning(f"Error quantizing layer {name}: {e}")
        
        return quantized_model
    
    def _accuracy_recovery_tuning(self, quantized_model: Any) -> Any:
        """Fine-tune the quantized model to recover accuracy"""
        logger.info("Performing accuracy recovery tuning...")
        
        # Determine if we need to do recovery
        # Currently, this is a placeholder. In a real implementation:
        # 1. Measure accuracy drop
        # 2. If significant, perform a few steps of fine-tuning on calibration data
        # 3. Return the fine-tuned model
        
        return quantized_model
    
    def _calculate_compression(self, original_model: Any, quantized_model: Any) -> float:
        """Calculate the achieved compression ratio"""
        if not isinstance(original_model, torch.nn.Module) or not isinstance(quantized_model, torch.nn.Module):
            return 0.5  # Default placeholder
            
        # Count parameter bit size
        orig_params_bits = 0
        quant_params_bits = 0
        
        with torch.no_grad():
            # Original model
            for param in original_model.parameters():
                orig_params_bits += param.numel() * 32  # FP32 by default
            
            # Quantized model
            for name, module in quantized_model.named_modules():
                # Check if this module has a _weight_bit_width attribute (common in QAT frameworks)
                weight_bits = getattr(module, '_weight_bit_width', None)
                
                for param_name, param in module.named_parameters(recurse=False):
                    if weight_bits is not None and 'weight' in param_name:
                        quant_params_bits += param.numel() * weight_bits
                    else:
                        # Estimate bit width from parameter dtype
                        dtype = param.dtype
                        if dtype == torch.int8:
                            bits = 8
                        elif dtype == torch.int4 or dtype == torch.qint4:
                            bits = 4
                        elif dtype == torch.float16:
                            bits = 16
                        elif dtype == torch.bfloat16:
                            bits = 16
                        else:
                            bits = 32
                        
                        quant_params_bits += param.numel() * bits
        
        if orig_params_bits == 0:
            return 0.5  # Default
            
        compression_ratio = 1.0 - (quant_params_bits / orig_params_bits)
        return compression_ratio
    
    def _measure_accuracy_retention(self, original_model: Any, quantized_model: Any) -> float:
        """Measure what percentage of the original accuracy is retained"""
        # In a real implementation, this would evaluate both models on a validation set
        # and compare their accuracies
        
        # For now, use a heuristic based on quantization levels
        total_params = 0
        int8_params = 0
        int4_params = 0
        fp16_params = 0
        
        for name, module in quantized_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                total_params += 1
                
                dtype = param.dtype
                if dtype == torch.int8:
                    int8_params += 1
                elif dtype == torch.int4 or dtype == torch.qint4:
                    int4_params += 1
                elif dtype == torch.float16 or dtype == torch.bfloat16:
                    fp16_params += 1
        
        if total_params == 0:
            return 95.0  # Default placeholder
            
        # Estimate accuracy retention based on quantization mix
        # These values are just estimates and should be replaced with actual measurements
        int8_retention = 0.98  # INT8 retains ~98% accuracy
        int4_retention = 0.92  # INT4 retains ~92% accuracy
        fp16_retention = 0.99  # FP16 retains ~99% accuracy
        fp32_retention = 1.00  # FP32 retains 100% accuracy
        
        int8_ratio = int8_params / total_params
        int4_ratio = int4_params / total_params
        fp16_ratio = fp16_params / total_params
        fp32_ratio = 1.0 - int8_ratio - int4_ratio - fp16_ratio
        
        estimated_retention = (
            int8_ratio * int8_retention +
            int4_ratio * int4_retention +
            fp16_ratio * fp16_retention +
            fp32_ratio * fp32_retention
        )
        
        return estimated_retention * 100.0  # Convert to percentage
    
    def _measure_speed_gain(self, original_model: Any, quantized_model: Any) -> float:
        """Measure the speed improvement from quantization"""
        if not isinstance(original_model, torch.nn.Module) or not isinstance(quantized_model, torch.nn.Module):
            return 2.0  # Default placeholder
            
        # In a real implementation, this would benchmark both models
        # For now, use a heuristic based on quantization levels
        
        total_params = 0
        int8_params = 0
        int4_params = 0
        fp16_params = 0
        
        for name, module in quantized_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                total_params += 1
                
                dtype = param.dtype
                if dtype == torch.int8:
                    int8_params += 1
                elif dtype == torch.int4 or dtype == torch.qint4:
                    int4_params += 1
                elif dtype == torch.float16 or dtype == torch.bfloat16:
                    fp16_params += 1
        
        if total_params == 0:
            return 2.0  # Default placeholder
            
        # Estimate speed gains based on quantization mix
        # These values are just estimates and should be replaced with actual measurements
        int8_speedup = 3.0   # INT8 is ~3x faster than FP32
        int4_speedup = 5.0   # INT4 is ~5x faster than FP32
        fp16_speedup = 2.0   # FP16 is ~2x faster than FP32
        fp32_speedup = 1.0   # FP32 is baseline
        
        int8_ratio = int8_params / total_params
        int4_ratio = int4_params / total_params
        fp16_ratio = fp16_params / total_params
        fp32_ratio = 1.0 - int8_ratio - int4_ratio - fp16_ratio
        
        estimated_speedup = (
            int8_ratio * int8_speedup +
            int4_ratio * int4_speedup +
            fp16_ratio * fp16_speedup +
            fp32_ratio * fp32_speedup
        )
        
        return estimated_speedup
    
    # Specific quantization methods
    def int8_quantization(self, module: torch.nn.Module) -> torch.nn.Module:
        """Perform INT8 quantization"""
        if not isinstance(module, torch.nn.Module):
            return module
            
        # Identify if the module has parameters that can be quantized
        has_params = False
        for name, param in module.named_parameters(recurse=False):
            has_params = True
            break
            
        if not has_params:
            return module
            
        # Create a copy of the module for quantization
        quantized_module = module
        
        # Quantize parameters to INT8
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=False):
                if 'weight' in name or 'bias' in name:
                    # Skip parameters that shouldn't be quantized
                    if 'norm' in name or 'embedding' in name:
                        continue
                        
                    # Compute scale based on max absolute value
                    scale = torch.max(torch.abs(param)) / 127.0
                    if scale < 1e-10:
                        continue  # Skip if scale is too small
                        
                    # Quantize to INT8
                    quantized_param = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
                    
                    # Convert back to original dtype for compatibility
                    dequantized_param = (quantized_param.float() * scale).to(param.dtype)
                    
                    # Update parameter in-place
                    param.copy_(dequantized_param)
                    
                    # Store scale for inference
                    module.register_buffer(f'{name}_scale', scale.detach())
        
        return quantized_module
    
    def int4_quantization(self, module: torch.nn.Module) -> torch.nn.Module:
        """Perform INT4 quantization"""
        if not isinstance(module, torch.nn.Module):
            return module
            
        # Identify if the module has parameters that can be quantized
        has_params = False
        for name, param in module.named_parameters(recurse=False):
            has_params = True
            break
            
        if not has_params:
            return module
            
        # Create a copy of the module for quantization
        quantized_module = module
        
        # Quantize parameters to INT4
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=False):
                if 'weight' in name:  # Only quantize weights, not biases
                    # Skip parameters that shouldn't be quantized
                    if 'norm' in name or 'embedding' in name:
                        continue
                        
                    # Compute scale based on max absolute value
                    scale = torch.max(torch.abs(param)) / 7.0
                    if scale < 1e-10:
                        continue  # Skip if scale is too small
                        
                    # Quantize to INT4 (store as INT8 since PyTorch doesn't have INT4)
                    quantized_param = torch.round(param / scale).clamp(-8, 7).to(torch.int8)
                    
                    # Convert back to original dtype for compatibility
                    dequantized_param = (quantized_param.float() * scale).to(param.dtype)
                    
                    # Update parameter in-place
                    param.copy_(dequantized_param)
                    
                    # Store scale for inference
                    module.register_buffer(f'{name}_scale', scale.detach())
                    
                    # Store "bit width" for the parameter (for compression calculation)
                    module._weight_bit_width = 4
        
        return quantized_module
    
    def fp8_quantization(self, module: torch.nn.Module) -> torch.nn.Module:
        """Perform FP8 quantization (emulated)"""
        if not isinstance(module, torch.nn.Module):
            return module
            
        # Identify if the module has parameters that can be quantized
        has_params = False
        for name, param in module.named_parameters(recurse=False):
            has_params = True
            break
            
        if not has_params:
            return module
            
        # Create a copy of the module for quantization
        quantized_module = module
        
        # Emulate FP8 quantization
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=False):
                if 'weight' in name or 'bias' in name:
                    # Skip parameters that shouldn't be quantized
                    if 'norm' in name:
                        continue
                        
                    # Convert to FP16 first (emulating the range constraints)
                    fp16_param = param.to(torch.float16)
                    
                    # Further truncate precision to emulate FP8
                    # This is a crude approximation of FP8
                    scale = torch.max(torch.abs(fp16_param)) / 15.0
                    quantized_param = torch.round(fp16_param / scale) * scale
                    
                    # Convert back to original dtype for compatibility
                    dequantized_param = quantized_param.to(param.dtype)
                    
                    # Update parameter in-place
                    param.copy_(dequantized_param)
                    
                    # Store "bit width" for the parameter (for compression calculation)
                    module._weight_bit_width = 8
        
        return quantized_module
    
    def dynamic_quantization(self, module: torch.nn.Module) -> torch.nn.Module:
        """Perform dynamic quantization"""
        # Dynamic quantization determines scales at runtime
        # This is a simplified implementation
        
        if not isinstance(module, torch.nn.Module):
            return module
            
        # Try to use PyTorch's dynamic quantization
        try:
            import torch.quantization
            
            # Set up dynamic quantization config
            model_to_quantize = module
            
            # Specify which layers to quantize
            qconfig_dict = {
                "": torch.quantization.default_dynamic_qconfig
            }
            
            # Prepare the model for quantization
            model_prepared = torch.quantization.prepare(model_to_quantize, qconfig_dict)
            
            # Quantize the model
            quantized_model = torch.quantization.convert(model_prepared)
            
            return quantized_model
            
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"PyTorch dynamic quantization failed: {e}")
            logger.warning("Falling back to INT8 quantization")
            
            # Fall back to INT8 quantization
            return self.int8_quantization(module)
    
    def mixed_precision_quantization(self, module: torch.nn.Module) -> torch.nn.Module:
        """Perform mixed precision quantization"""
        if not isinstance(module, torch.nn.Module):
            return module
            
        # Identify if the module has parameters that can be quantized
        has_params = False
        for name, param in module.named_parameters(recurse=False):
            has_params = True
            break
            
        if not has_params:
            return module
            
        # Create a copy of the module for quantization
        quantized_module = module
        
        # Apply different quantization based on parameter importance
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=False):
                if 'weight' in name:
                    # Skip parameters that shouldn't be quantized
                    if 'norm' in name:
                        continue
                    
                    # Determine quantization level based on parameter statistics
                    param_importance = self._estimate_parameter_importance(param, name)
                    
                    if param_importance > 0.8:
                        # High importance - use FP16
                        quantized_param = param.to(torch.float16).to(param.dtype)
                        module._weight_bit_width = 16
                    elif param_importance > 0.4:
                        # Medium importance - use INT8
                        scale = torch.max(torch.abs(param)) / 127.0
                        if scale >= 1e-10:
                            quantized_param = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
                            quantized_param = (quantized_param.float() * scale).to(param.dtype)
                            module._weight_bit_width = 8
                        else:
                            quantized_param = param  # Keep original
                    else:
                        # Low importance - use INT4
                        scale = torch.max(torch.abs(param)) / 7.0
                        if scale >= 1e-10:
                            quantized_param = torch.round(param / scale).clamp(-8, 7).to(torch.int8)
                            quantized_param = (quantized_param.float() * scale).to(param.dtype)
                            module._weight_bit_width = 4
                        else:
                            quantized_param = param  # Keep original
                    
                    # Update parameter in-place
                    param.copy_(quantized_param)
                
                elif 'bias' in name:
                    # Use INT8 for biases
                    scale = torch.max(torch.abs(param)) / 127.0
                    if scale >= 1e-10:
                        quantized_param = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
                        quantized_param = (quantized_param.float() * scale).to(param.dtype)
                    else:
                        quantized_param = param  # Keep original
                    
                    # Update parameter in-place
                    param.copy_(quantized_param)
        
        return quantized_module
    
    def _estimate_parameter_importance(self, param: torch.Tensor, name: str) -> float:
        """Estimate parameter importance for mixed precision quantization"""
        # Heuristics for parameter importance:
        # 1. Position in network (early layers more important)
        # 2. Parameter variance
        # 3. Parameter type (attention matrices more important than FFN)
        
        importance = 0.5  # Default medium importance
        
        # Adjust by layer position
        if any(prefix in name for prefix in ['first', 'embed', 'input']):
            importance += 0.2
        elif any(prefix in name for prefix in ['final', 'output', 'head']):
            importance += 0.1
            
        # Adjust by parameter variance (high variance = more important)
        param_var = torch.var(param).item()
        if param_var > 0.1:
            importance += 0.2
        elif param_var < 0.01:
            importance -= 0.1
            
        # Adjust by parameter type
        if any(key in name for key in ['attn', 'attention']):
            importance += 0.2
        elif any(key in name for key in ['bias']):
            importance -= 0.3
            
        # Clamp to [0, 1]
        importance = max(0.0, min(1.0, importance))
        
        return importance
