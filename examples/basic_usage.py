"""
Basic usage example for LLMOptima
"""

from llmoptima import LLMOptima

def main():
    # Initialize optimizer with a model path or identifier
    optimizer = LLMOptima('meta-llama/Llama-2-7b', optimization_level='balanced')
    
    # Run optimization with target metrics
    results = optimizer.optimize_model(target_metrics={
        'speed_improvement': 5.0,   # 5x faster
        'size_reduction': 0.7,      # 70% smaller  
        'accuracy_retention': 0.95   # Keep 95% accuracy
    })
    
    # Print optimization results
    results.print_summary()
    
    # Save optimization report
    results.save_report('optimization_report.md')
    
    # Save optimized model
    results.save_model('optimized_model')

if __name__ == "__main__":
    main()
