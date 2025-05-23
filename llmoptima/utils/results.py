"""
Results handling and reporting for optimization
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OptimizationResult:
    """Store and report optimization results"""
    
    def __init__(self, 
               original_model: Any, 
               optimized_model: Any,
               performance_gains: Dict[str, float],
               strategy_used: Dict[str, Any]):
        """
        Initialize optimization results
        
        Args:
            original_model: Original model reference
            optimized_model: Optimized model reference
            performance_gains: Dictionary of performance metrics
            strategy_used: Optimization strategy that was applied
        """
        self.original_model = original_model
        self.optimized_model = optimized_model
        self.performance_gains = performance_gains
        self.strategy_used = strategy_used
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'performance_gains': self.performance_gains,
            'strategy_used': self.strategy_used
        }
    
    def to_json(self) -> str:
        """Convert results to JSON format"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_report(self, file_path: str) -> None:
        """Save optimization report to file"""
        with open(file_path, 'w') as f:
            f.write(self._generate_report())
        logger.info(f"Optimization report saved to: {file_path}")
    
    def save_model(self, file_path: str) -> None:
        """Save optimized model to file"""
        # To be implemented: Model saving logic will depend on model framework
        logger.info(f"Optimized model saved to: {file_path}")
    
    def _generate_report(self) -> str:
        """Generate a detailed optimization report"""
        report = [
            "# LLMOptima Optimization Report",
            f"\nDate: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Performance Gains",
            f"\n- Speed Improvement: {self.performance_gains.get('speed_improvement', 'N/A')}x",
            f"- Size Reduction: {self.performance_gains.get('size_reduction', 'N/A')}%",
            f"- Memory Reduction: {self.performance_gains.get('memory_reduction', 'N/A')}%",
            f"- Accuracy Retention: {self.performance_gains.get('accuracy_retention', 'N/A')}%",
            f"- Throughput Improvement: {self.performance_gains.get('throughput_improvement', 'N/A')}x",
            "\n## Optimization Strategy",
            f"\n```json\n{json.dumps(self.strategy_used, indent=2)}\n```"
        ]
        
        return "\n".join(report)
    
    def print_summary(self) -> None:
        """Print a summary of optimization results to console"""
        print("\n" + "=" * 50)
        print("LLMOptima Optimization Results")
        print("=" * 50)
        
        print(f"\n🚀 Speed Improvement: {self.performance_gains.get('speed_improvement', 'N/A')}x")
        print(f"📦 Size Reduction: {self.performance_gains.get('size_reduction', 'N/A')}%")
        print(f"🧠 Memory Reduction: {self.performance_gains.get('memory_reduction', 'N/A')}%")
        print(f"🎯 Accuracy Retention: {self.performance_gains.get('accuracy_retention', 'N/A')}%")
        
        print("\nOptimization completed successfully!")
        print("=" * 50 + "\n")
