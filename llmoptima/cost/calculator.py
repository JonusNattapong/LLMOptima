"""
Cost calculation and ROI analysis for LLM optimization
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMCostCalculator:
    """Calculate actual LLM deployment costs and optimization savings"""
    
    def __init__(self):
        self.pricing = {
            'gpt4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'claude': {'input': 0.015, 'output': 0.075},
            'llama2': {'compute': 0.0008},  # per hour GPU
            'llama3': {'compute': 0.0012},  # per hour GPU
            'mistral': {'compute': 0.0007}  # per hour GPU
        }
    
    def calculate_optimization_savings(self, 
                                     model_config: Dict[str, Any], 
                                     usage_pattern: Dict[str, Any],
                                     optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate real cost savings from optimization
        
        Args:
            model_config: Model configuration (name, size, etc.)
            usage_pattern: Usage pattern (requests per day, tokens, etc.)
            optimization_results: Results from optimization
            
        Returns:
            Dictionary with cost analysis and savings
        """
        logger.info(f"Calculating cost savings for model: {model_config.get('name', 'unknown')}")
        
        # Original costs
        original_costs = self._calculate_original_costs(model_config, usage_pattern)
        
        # Optimized costs  
        optimized_costs = self._calculate_optimized_costs(
            model_config, usage_pattern, optimization_results
        )
        
        # Calculate savings and ROI
        monthly_savings = original_costs['monthly'] - optimized_costs['monthly']
        yearly_savings = original_costs['yearly'] - optimized_costs['yearly']
        
        if original_costs['monthly'] > 0:
            percentage_saved = (monthly_savings / original_costs['monthly']) * 100
        else:
            percentage_saved = 0.0
            
        optimization_cost = optimization_results.get('optimization_cost', 0)
        
        if monthly_savings > 0:
            payback_period_months = optimization_cost / monthly_savings
        else:
            payback_period_months = float('inf')
            
        savings = {
            'monthly_savings': monthly_savings,
            'yearly_savings': yearly_savings,
            'percentage_saved': percentage_saved,
            'payback_period_months': payback_period_months
        }
        
        roi_12_months = self._calculate_roi(savings, 12)
        
        logger.info(f"Monthly savings: ${monthly_savings:,.2f}")
        logger.info(f"Percentage saved: {percentage_saved:.2f}%")
        logger.info(f"ROI (12 months): {roi_12_months:.2f}%")
        
        return {
            'original_costs': original_costs,
            'optimized_costs': optimized_costs,
            'savings': savings,
            'roi_12_months': roi_12_months
        }
    
    def _calculate_original_costs(self, 
                               model_config: Dict[str, Any], 
                               usage_pattern: Dict[str, Any]) -> Dict[str, float]:
        """Calculate original costs based on model and usage"""
        logger.info("Calculating original costs...")
        
        model_name = model_config.get('name', '').lower()
        model_type = None
        
        # Determine model type for pricing
        for key in self.pricing.keys():
            if key in model_name:
                model_type = key
                break
                
        if model_type is None:
            logger.warning(f"Unknown model type: {model_name}, using default pricing")
            model_type = 'llama2'  # Default
        
        # Calculate costs based on model type and usage pattern
        daily_requests = usage_pattern.get('requests_per_day', 0)
        avg_input_tokens = usage_pattern.get('avg_input_tokens', 0)
        avg_output_tokens = usage_pattern.get('avg_output_tokens', 0)
        
        daily_cost = 0.0
        
        if 'gpt' in model_type or 'claude' in model_type:
            # API-based pricing
            pricing = self.pricing[model_type]
            daily_input_cost = (daily_requests * avg_input_tokens / 1000) * pricing.get('input', 0)
            daily_output_cost = (daily_requests * avg_output_tokens / 1000) * pricing.get('output', 0)
            daily_cost = daily_input_cost + daily_output_cost
        else:
            # Compute-based pricing
            pricing = self.pricing[model_type]
            # Estimate compute hours based on tokens processed
            total_tokens = daily_requests * (avg_input_tokens + avg_output_tokens)
            # Assume processing speed based on model size
            tokens_per_hour = self._estimate_tokens_per_hour(model_config)
            compute_hours = total_tokens / tokens_per_hour
            daily_cost = compute_hours * pricing.get('compute', 0)
        
        # Calculate monthly and yearly costs
        monthly_cost = daily_cost * 30
        yearly_cost = monthly_cost * 12
        
        return {
            'daily': daily_cost,
            'monthly': monthly_cost,
            'yearly': yearly_cost
        }
    
    def _calculate_optimized_costs(self, 
                                model_config: Dict[str, Any], 
                                usage_pattern: Dict[str, Any],
                                optimization_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimized costs based on optimization results"""
        logger.info("Calculating optimized costs...")
        
        # Get optimization improvements
        speed_improvement = optimization_results.get('speed_improvement', 1.0)
        
        # For compute-based models, faster inference = lower cost
        # For API-based models, costs might be the same unless using a smaller model
        
        # Calculate original costs first
        original_costs = self._calculate_original_costs(model_config, usage_pattern)
        
        model_name = model_config.get('name', '').lower()
        
        # API-based models (costs likely stay the same)
        if 'gpt' in model_name or 'claude' in model_name:
            # If using a smaller model after distillation, adjust accordingly
            if 'distillation' in optimization_results.get('techniques_applied', []):
                cost_reduction_factor = 0.7  # Assume 30% cost reduction from distillation
                optimized_costs = {
                    'daily': original_costs['daily'] * cost_reduction_factor,
                    'monthly': original_costs['monthly'] * cost_reduction_factor,
                    'yearly': original_costs['yearly'] * cost_reduction_factor
                }
            else:
                # No cost reduction for API models without model replacement
                optimized_costs = original_costs.copy()
        else:
            # Compute-based models (faster = cheaper)
            cost_reduction_factor = 1.0 / speed_improvement
            optimized_costs = {
                'daily': original_costs['daily'] * cost_reduction_factor,
                'monthly': original_costs['monthly'] * cost_reduction_factor,
                'yearly': original_costs['yearly'] * cost_reduction_factor
            }
        
        return optimized_costs
    
    def _estimate_tokens_per_hour(self, model_config: Dict[str, Any]) -> float:
        """Estimate tokens per hour based on model size"""
        model_size = model_config.get('size', '7B')
        
        # Extract numeric part of model size
        if isinstance(model_size, str):
            try:
                size_value = float(model_size.replace('B', '').strip())
            except ValueError:
                size_value = 7.0
        else:
            size_value = float(model_size)
        
        # Very rough estimate of tokens per hour based on model size
        # These values would need to be calibrated with real-world data
        if size_value <= 3:
            return 1000000  # 1M tokens/hour for small models
        elif size_value <= 7:
            return 500000   # 500K tokens/hour for medium models
        elif size_value <= 20:
            return 200000   # 200K tokens/hour for large models
        else:
            return 100000   # 100K tokens/hour for very large models
    
    def _calculate_roi(self, savings: Dict[str, float], months: int) -> float:
        """Calculate ROI over a given time period"""
        if 'optimization_cost' not in savings or savings['optimization_cost'] == 0:
            # Assume a default optimization cost if not provided
            optimization_cost = 5000.0
        else:
            optimization_cost = savings['optimization_cost']
        
        if months <= 0:
            return 0.0
            
        monthly_savings = savings.get('monthly_savings', 0.0)
        total_savings = monthly_savings * months
        
        if optimization_cost == 0:
            return float('inf')
            
        roi = ((total_savings - optimization_cost) / optimization_cost) * 100
        
        return roi
    
    def estimate_optimization_cost(self, model_config: Dict[str, Any], 
                                optimization_level: str = 'balanced') -> float:
        """
        Estimate the cost of performing optimization
        
        Args:
            model_config: Model configuration (name, size, etc.)
            optimization_level: Level of optimization effort
            
        Returns:
            Estimated cost in USD
        """
        model_size = model_config.get('size', '7B')
        
        # Extract numeric part of model size
        if isinstance(model_size, str):
            try:
                size_value = float(model_size.replace('B', '').strip())
            except ValueError:
                size_value = 7.0
        else:
            size_value = float(model_size)
        
        # Base cost factors
        if optimization_level == 'conservative':
            base_cost = 2000.0
        elif optimization_level == 'balanced':
            base_cost = 5000.0
        elif optimization_level == 'aggressive':
            base_cost = 10000.0
        else:
            base_cost = 5000.0
        
        # Scale based on model size
        if size_value <= 3:
            size_factor = 0.5
        elif size_value <= 7:
            size_factor = 1.0
        elif size_value <= 20:
            size_factor = 2.0
        else:
            size_factor = 3.0
        
        estimated_cost = base_cost * size_factor
        
        logger.info(f"Estimated optimization cost: ${estimated_cost:,.2f}")
        
        return estimated_cost
