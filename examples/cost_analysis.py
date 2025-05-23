"""
Cost analysis example for LLMOptima
"""

from llmoptima import LLMCostCalculator

def main():
    # Initialize cost calculator
    calculator = LLMCostCalculator()
    
    # Define your model configuration
    model_config = {
        'name': 'llama2-70b',
        'size': '70B',
        'type': 'decoder-only'
    }
    
    # Define your usage pattern
    usage_pattern = {
        'requests_per_day': 10000,
        'avg_input_tokens': 500,
        'avg_output_tokens': 200,
    }
    
    # Example optimization results
    # In real usage, you would get this from LLMOptima.optimize_model()
    optimization_results = {
        'speed_improvement': 5.0,
        'size_reduction': 70.0,
        'accuracy_retention': 95.0,
        'techniques_applied': ['quantization', 'pruning', 'inference'],
        'optimization_cost': 15000.0  # USD
    }
    
    # Calculate savings
    savings = calculator.calculate_optimization_savings(
        model_config=model_config,
        usage_pattern=usage_pattern,
        optimization_results=optimization_results
    )
    
    # Print the results
    print("\n" + "=" * 50)
    print("LLMOptima Cost Analysis")
    print("=" * 50)
    
    print(f"\nOriginal Monthly Cost: ${savings['original_costs']['monthly']:,.2f}")
    print(f"Optimized Monthly Cost: ${savings['optimized_costs']['monthly']:,.2f}")
    print(f"Monthly Savings: ${savings['savings']['monthly_savings']:,.2f}")
    print(f"Yearly Savings: ${savings['savings']['yearly_savings']:,.2f}")
    print(f"Cost Reduction: {savings['savings']['percentage_saved']:.1f}%")
    print(f"Payback Period: {savings['savings']['payback_period_months']:.1f} months")
    print(f"ROI (12 months): {savings['roi_12_months']:.1f}%")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
