"""
Full optimization example for LLMOptima
"""

import logging
from llmoptima import LLMOptima, LLMCostCalculator

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize optimizer with a model path
    optimizer = LLMOptima('meta-llama/Llama-2-7b', optimization_level='aggressive')
    
    # Set hardware profile
    optimizer.update_hardware_profile({
        'device_type': 'cuda',
        'gpu_memory': 16,  # GB
        'gpu_type': 'A100', 
        'cpu_cores': 16
    })
    
    # Run full optimization with target metrics
    results = optimizer.optimize_model(target_metrics={
        'speed_improvement': 8.0,   # 8x faster
        'size_reduction': 0.8,      # 80% smaller  
        'accuracy_retention': 0.92   # Keep 92% accuracy
    })
    
    # Print optimization results
    results.print_summary()
    
    # Save optimization report
    results.save_report('llama2_optimization_report.md')
    
    # Save optimized model
    results.save_model('optimized_llama2_model')
    
    # Calculate cost savings
    calculator = LLMCostCalculator()
    
    # Define model and usage
    model_config = {
        'name': 'llama2-7b',
        'size': '7B',
        'type': 'decoder-only'
    }
    
    usage_pattern = {
        'requests_per_day': 50000,
        'avg_input_tokens': 300,
        'avg_output_tokens': 150,
    }
    
    # Calculate savings
    savings = calculator.calculate_optimization_savings(
        model_config=model_config,
        usage_pattern=usage_pattern,
        optimization_results={
            'speed_improvement': results.performance_gains.get('speed_improvement', 5.0),
            'size_reduction': results.performance_gains.get('size_reduction', 70.0),
            'accuracy_retention': results.performance_gains.get('accuracy_retention', 95.0),
            'techniques_applied': ['quantization', 'pruning', 'inference'],
            'optimization_cost': 10000.0  # USD
        }
    )
    
    # Print savings
    print("\n=== Cost Analysis ===")
    print(f"Monthly savings: ${savings['savings']['monthly_savings']:,.2f}")
    print(f"Yearly savings: ${savings['savings']['yearly_savings']:,.2f}")
    print(f"ROI (12 months): {savings['roi_12_months']:.1f}%")
    print(f"Payback period: {savings['savings']['payback_period_months']:.1f} months")

if __name__ == "__main__":
    main()
