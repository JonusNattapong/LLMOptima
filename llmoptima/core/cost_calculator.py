"""
LLM Cost Calculator & ROI Engine
Author: @JonusNattapong

Calculate real-world cost savings and ROI from LLM optimization
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown structure"""
    compute_cost: float
    memory_cost: float
    storage_cost: float
    network_cost: float
    total_cost: float


@dataclass
class UsagePattern:
    """Usage pattern for cost calculation"""
    requests_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    peak_concurrency: int
    model_type: str
    deployment_type: str = "cloud"  # cloud, on-premise, hybrid


@dataclass
class CostSavings:
    """Cost savings calculation result"""
    monthly_savings: float
    yearly_savings: float
    percentage_saved: float
    payback_period_months: float
    roi_12_months: float
    roi_24_months: float
    total_cost_reduction: float


class LLMCostCalculator:
    """
    Advanced cost calculator for LLM deployment and optimization
    
    Features:
    - Real-time cloud pricing (AWS, Azure, GCP)
    - Token-based cost calculation
    - Hardware utilization modeling
    - ROI analysis with multiple scenarios
    - Cost optimization recommendations
    """
    
    def __init__(self):
        """Initialize cost calculator with latest pricing data"""
        
        # Updated pricing data (May 2024)
        self.cloud_pricing = {
            'openai': {
                'gpt-4-turbo': {'input': 0.01, 'output': 0.03},  # per 1K tokens
                'gpt-4': {'input': 0.03, 'output': 0.06},
                'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            },
            'anthropic': {
                'claude-3-opus': {'input': 0.015, 'output': 0.075},
                'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
                'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            },
            'google': {
                'gemini-pro': {'input': 0.0005, 'output': 0.0015},
                'gemini-ultra': {'input': 0.01, 'output': 0.03},
            }
        }
        
        # Hardware pricing (per hour)
        self.hardware_pricing = {
            'aws': {
                'p4d.24xlarge': {'cost_per_hour': 32.77, 'gpu_memory': 320, 'gpus': 8},  # A100
                'p3.8xlarge': {'cost_per_hour': 12.24, 'gpu_memory': 64, 'gpus': 4},     # V100
                'g5.12xlarge': {'cost_per_hour': 5.67, 'gpu_memory': 96, 'gpus': 4},     # A10G
            },
            'azure': {
                'NC96ads_A100_v4': {'cost_per_hour': 36.19, 'gpu_memory': 320, 'gpus': 4},
                'NC24ads_A100_v4': {'cost_per_hour': 9.05, 'gpu_memory': 80, 'gpus': 1},
            },
            'gcp': {
                'a2-ultragpu-8g': {'cost_per_hour': 33.22, 'gpu_memory': 320, 'gpus': 8},
                'a2-highgpu-4g': {'cost_per_hour': 16.61, 'gpu_memory': 160, 'gpus': 4},
            }
        }
        
        # Model size requirements (GB)
        self.model_memory_requirements = {
            'llama2-7b': {'fp16': 14, 'int8': 7, 'int4': 3.5},
            'llama2-13b': {'fp16': 26, 'int8': 13, 'int4': 6.5},
            'llama2-70b': {'fp16': 140, 'int8': 70, 'int4': 35},
            'mixtral-8x7b': {'fp16': 90, 'int8': 45, 'int4': 22.5},
            'gpt-4': {'fp16': 3200, 'int8': 1600, 'int4': 800},  # Estimated
        }
        
        logger.info("💰 LLM Cost Calculator initialized with latest pricing data")
    
    def calculate_optimization_savings(self,
                                     model_config: Dict[str, Any],
                                     usage_pattern: UsagePattern,
                                     optimization_results: Dict[str, float],
                                     deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive cost savings from optimization
        
        Args:
            model_config: Model configuration and specifications
            usage_pattern: Expected usage patterns
            optimization_results: Results from LLMOptima optimization
            deployment_config: Deployment configuration (cloud provider, region, etc.)
            
        Returns:
            Comprehensive cost analysis with savings breakdown
        """
        
        logger.info("📊 Calculating optimization cost savings...")
        
        # Calculate original costs
        original_costs = self._calculate_deployment_costs(
            model_config, usage_pattern, deployment_config
        )
        
        # Calculate optimized costs
        optimized_config = self._apply_optimization_to_config(
            model_config, optimization_results
        )
        
        optimized_costs = self._calculate_deployment_costs(
            optimized_config, usage_pattern, deployment_config
        )
        
        # Calculate savings
        savings = self._calculate_savings_metrics(original_costs, optimized_costs)
        
        # Generate recommendations
        recommendations = self._generate_cost_recommendations(
            model_config, usage_pattern, optimization_results
        )
        
        result = {
            'original_costs': original_costs,
            'optimized_costs': optimized_costs,
            'savings': savings,
            'optimization_results': optimization_results,
            'recommendations': recommendations,
            'calculation_date': datetime.now().isoformat(),
            'model_config': model_config,
            'usage_pattern': usage_pattern.__dict__
        }
        
        self._print_cost_analysis(result)
        return result
    
    def _calculate_deployment_costs(self,
                                  model_config: Dict[str, Any],
                                  usage_pattern: UsagePattern,
                                  deployment_config: Optional[Dict[str, Any]] = None) -> CostBreakdown:
        """Calculate deployment costs based on configuration"""
        
        model_name = model_config.get('name', 'llama2-7b')
        precision = model_config.get('precision', 'fp16')
        
        # Get memory requirements
        memory_gb = self.model_memory_requirements.get(model_name, {}).get(precision, 14)
        
        if usage_pattern.deployment_type == "cloud_api":
            return self._calculate_api_costs(model_config, usage_pattern)
        else:
            return self._calculate_infrastructure_costs(
                memory_gb, usage_pattern, deployment_config
            )
    
    def _calculate_api_costs(self,
                           model_config: Dict[str, Any],
                           usage_pattern: UsagePattern) -> CostBreakdown:
        """Calculate costs for cloud API usage"""
        
        provider = model_config.get('provider', 'openai')
        model_name = model_config.get('api_model', 'gpt-3.5-turbo')
        
        pricing = self.cloud_pricing.get(provider, {}).get(model_name, {
            'input': 0.001, 'output': 0.002
        })
        
        # Calculate monthly token usage
        daily_input_tokens = usage_pattern.requests_per_day * usage_pattern.avg_input_tokens
        daily_output_tokens = usage_pattern.requests_per_day * usage_pattern.avg_output_tokens
        
        monthly_input_tokens = daily_input_tokens * 30
        monthly_output_tokens = daily_output_tokens * 30
        
        # Calculate costs
        input_cost = (monthly_input_tokens / 1000) * pricing['input']
        output_cost = (monthly_output_tokens / 1000) * pricing['output']
        
        total_cost = input_cost + output_cost
        
        return CostBreakdown(
            compute_cost=total_cost,
            memory_cost=0,
            storage_cost=0,
            network_cost=0,
            total_cost=total_cost
        )
    
    def _calculate_infrastructure_costs(self,
                                      memory_gb: float,
                                      usage_pattern: UsagePattern,
                                      deployment_config: Optional[Dict[str, Any]] = None) -> CostBreakdown:
        """Calculate infrastructure costs for self-hosted deployment"""
        
        provider = deployment_config.get('provider', 'aws') if deployment_config else 'aws'
        
        # Select appropriate instance type based on memory requirements
        instance_type = self._select_optimal_instance(memory_gb, provider)
        instance_pricing = self.hardware_pricing[provider][instance_type]
        
        # Calculate utilization based on request pattern
        peak_hours_per_day = 12  # Assume 12 hours peak usage
        utilization_factor = min(1.0, usage_pattern.peak_concurrency / 10)  # Scale with concurrency
        
        # Monthly costs calculation
        hours_per_month = 30 * 24
        compute_cost = instance_pricing['cost_per_hour'] * hours_per_month * utilization_factor
        
        # Additional costs
        storage_cost = 100  # $100/month for storage
        network_cost = 50   # $50/month for network
        memory_cost = 0     # Included in compute
        
        total_cost = compute_cost + storage_cost + network_cost
        
        return CostBreakdown(
            compute_cost=compute_cost,
            memory_cost=memory_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            total_cost=total_cost
        )
    
    def _select_optimal_instance(self, memory_gb: float, provider: str) -> str:
        """Select optimal instance type based on memory requirements"""
        
        instances = self.hardware_pricing[provider]
        
        # Find the smallest instance that can fit the model
        suitable_instances = [
            (name, config) for name, config in instances.items()
            if config['gpu_memory'] >= memory_gb
        ]
        
        if not suitable_instances:
            # If model doesn't fit in any single instance, return the largest
            return max(instances.keys(), key=lambda x: instances[x]['gpu_memory'])
        
        # Return the most cost-effective suitable instance
        return min(suitable_instances, key=lambda x: x[1]['cost_per_hour'])[0]
    
    def _apply_optimization_to_config(self,
                                    model_config: Dict[str, Any],
                                    optimization_results: Dict[str, float]) -> Dict[str, Any]:
        """Apply optimization results to model configuration"""
        
        optimized_config = model_config.copy()
        
        # Apply quantization effect on memory
        if 'size_reduction' in optimization_results:
            reduction_factor = optimization_results['size_reduction']
            original_name = model_config.get('name', 'llama2-7b')
            
            # Update precision based on size reduction
            if reduction_factor >= 0.7:
                optimized_config['precision'] = 'int4'
            elif reduction_factor >= 0.4:
                optimized_config['precision'] = 'int8'
            else:
                optimized_config['precision'] = 'fp16'
        
        return optimized_config
    
    def _calculate_savings_metrics(self,
                                 original_costs: CostBreakdown,
                                 optimized_costs: CostBreakdown) -> CostSavings:
        """Calculate detailed savings metrics"""
        
        monthly_savings = original_costs.total_cost - optimized_costs.total_cost
        yearly_savings = monthly_savings * 12
        percentage_saved = (monthly_savings / original_costs.total_cost) * 100 if original_costs.total_cost > 0 else 0
        
        # Assume optimization cost of $5000 (one-time)
        optimization_cost = 5000
        payback_period = optimization_cost / monthly_savings if monthly_savings > 0 else float('inf')
        
        roi_12_months = ((yearly_savings - optimization_cost) / optimization_cost) * 100 if optimization_cost > 0 else 0
        roi_24_months = ((yearly_savings * 2 - optimization_cost) / optimization_cost) * 100 if optimization_cost > 0 else 0
        
        return CostSavings(
            monthly_savings=monthly_savings,
            yearly_savings=yearly_savings,
            percentage_saved=percentage_saved,
            payback_period_months=payback_period,
            roi_12_months=roi_12_months,
            roi_24_months=roi_24_months,
            total_cost_reduction=monthly_savings
        )
    
    def _generate_cost_recommendations(self,
                                     model_config: Dict[str, Any],
                                     usage_pattern: UsagePattern,
                                     optimization_results: Dict[str, float]) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        # Deployment type recommendations
        if usage_pattern.deployment_type == "cloud_api":
            if usage_pattern.requests_per_day > 50000:
                recommendations.append(
                    "💡 Consider self-hosted deployment for >50k requests/day to reduce costs by 60-80%"
                )
        
        # Optimization level recommendations
        speed_improvement = optimization_results.get('speed_improvement', 1.0)
        if speed_improvement < 3.0:
            recommendations.append(
                "⚡ Increase optimization level to 'aggressive' for better cost savings"
            )
        
        # Scaling recommendations
        if usage_pattern.peak_concurrency > 100:
            recommendations.append(
                "📈 Implement auto-scaling to optimize costs during low-traffic periods"
            )
        
        # Model size recommendations
        model_name = model_config.get('name', '')
        if '70b' in model_name and optimization_results.get('accuracy_retention', 1.0) > 0.95:
            recommendations.append(
                "🎯 Consider model distillation to a smaller model for additional 50-70% cost savings"
            )
        
        return recommendations
    
    def _print_cost_analysis(self, analysis: Dict[str, Any]):
        """Print beautiful cost analysis summary"""
        
        print("\n" + "="*70)
        print("💰 LLM Cost Analysis & Savings Report")
        print("="*70)
        
        original = analysis['original_costs']
        optimized = analysis['optimized_costs']
        savings = analysis['savings']
        
        print(f"📊 Original Monthly Costs:")
        print(f"   💻 Compute: ${original.compute_cost:,.2f}")
        print(f"   💾 Storage: ${original.storage_cost:,.2f}")
        print(f"   🌐 Network: ${original.network_cost:,.2f}")
        print(f"   📋 Total: ${original.total_cost:,.2f}")
        
        print(f"\n⚡ Optimized Monthly Costs:")
        print(f"   💻 Compute: ${optimized.compute_cost:,.2f}")
        print(f"   💾 Storage: ${optimized.storage_cost:,.2f}")
        print(f"   🌐 Network: ${optimized.network_cost:,.2f}")
        print(f"   📋 Total: ${optimized.total_cost:,.2f}")
        
        print(f"\n🎉 Cost Savings:")
        print(f"   💵 Monthly: ${savings.monthly_savings:,.2f}")
        print(f"   📈 Yearly: ${savings.yearly_savings:,.2f}")
        print(f"   📊 Percentage: {savings.percentage_saved:.1f}%")
        print(f"   ⏰ Payback: {savings.payback_period_months:.1f} months")
        print(f"   🚀 ROI (12mo): {savings.roi_12_months:.0f}%")
        
        if analysis.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   {rec}")
        
        print("="*70 + "\n")
    
    def estimate_enterprise_savings(self,
                                  company_size: str,
                                  ai_maturity: str,
                                  current_spend: float) -> Dict[str, float]:
        """Estimate enterprise-wide savings potential"""
        
        # Savings multipliers based on company characteristics
        size_multipliers = {
            'startup': 1.0,
            'mid-market': 2.5,
            'enterprise': 5.0,
            'fortune500': 10.0
        }
        
        maturity_multipliers = {
            'early': 0.5,      # Just starting with AI
            'growing': 1.0,    # Moderate AI usage
            'mature': 1.5,     # Heavy AI usage
            'advanced': 2.0    # AI-first company
        }
        
        base_savings_rate = 0.65  # 65% average savings
        size_factor = size_multipliers.get(company_size, 1.0)
        maturity_factor = maturity_multipliers.get(ai_maturity, 1.0)
        
        potential_savings = current_spend * base_savings_rate * maturity_factor
        
        return {
            'monthly_potential': potential_savings,
            'yearly_potential': potential_savings * 12,
            'implementation_cost': potential_savings * 0.1,  # 10% of savings
            'roi_first_year': ((potential_savings * 12 - potential_savings * 0.1) / (potential_savings * 0.1)) * 100
        }
