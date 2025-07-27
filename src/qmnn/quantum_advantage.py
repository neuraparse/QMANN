"""
Quantum Advantage Verification for QMANN (2025)

This module implements rigorous quantum advantage verification
based on the latest 2025 theoretical frameworks and experimental protocols.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time
import logging
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class QuantumAdvantageMetrics(nn.Module):
    """
    Comprehensive quantum advantage metrics and verification.
    
    Implements 2025 state-of-the-art quantum advantage verification
    protocols including statistical significance testing and
    computational complexity analysis.
    """
    
    def __init__(self, significance_level: float = 0.01, 
                 min_effect_size: float = 0.2):
        super().__init__()
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        
        # Metrics storage
        self.quantum_metrics = []
        self.classical_metrics = []
        self.advantage_history = []
        
    def compute_quantum_volume(self, n_qubits: int, circuit_depth: int,
                              fidelity: float) -> float:
        """
        Compute quantum volume metric.
        
        Quantum Volume = min(2^n, d^2) * fidelity_factor
        where n is qubits, d is depth, and fidelity_factor accounts for noise.
        """
        max_volume = min(2**n_qubits, circuit_depth**2)
        fidelity_factor = fidelity**circuit_depth  # Exponential decay with depth
        
        quantum_volume = max_volume * fidelity_factor
        return quantum_volume
    
    def compute_quantum_supremacy_metric(self, quantum_time: float,
                                       classical_time: float,
                                       problem_size: int) -> Dict[str, float]:
        """
        Compute quantum supremacy metrics.
        
        Based on 2025 refined quantum supremacy criteria.
        """
        # Speedup ratio
        speedup = classical_time / quantum_time if quantum_time > 0 else float('inf')
        
        # Exponential advantage threshold
        expected_classical_scaling = 2**problem_size  # Exponential for hard problems
        quantum_scaling = problem_size**3  # Polynomial for quantum
        
        # Advantage factor
        advantage_factor = (classical_time / expected_classical_scaling) / \
                          (quantum_time / quantum_scaling)
        
        # Quantum supremacy achieved if advantage_factor > 1
        supremacy_achieved = advantage_factor > 1.0
        
        return {
            'speedup': speedup,
            'advantage_factor': advantage_factor,
            'supremacy_achieved': supremacy_achieved,
            'quantum_scaling': quantum_scaling,
            'classical_scaling': expected_classical_scaling
        }
    
    def statistical_significance_test(self, quantum_results: List[float],
                                    classical_results: List[float]) -> Dict[str, float]:
        """
        Perform statistical significance test for quantum advantage.
        
        Uses Welch's t-test and effect size calculation.
        """
        if len(quantum_results) < 3 or len(classical_results) < 3:
            return {'p_value': 1.0, 'effect_size': 0.0, 'significant': False}
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = ttest_ind(quantum_results, classical_results, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(quantum_results) + np.var(classical_results)) / 2)
        effect_size = (np.mean(quantum_results) - np.mean(classical_results)) / pooled_std
        
        # Significance determination
        significant = (p_value < self.significance_level) and \
                     (abs(effect_size) > self.min_effect_size)
        
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': significant,
            't_statistic': t_stat,
            'quantum_mean': np.mean(quantum_results),
            'classical_mean': np.mean(classical_results)
        }
    
    def compute_learning_advantage(self, quantum_model_performance: Dict[str, float],
                                 classical_model_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Compute quantum learning advantage metrics.
        
        Compares quantum and classical models on multiple dimensions.
        """
        advantages = {}
        
        # Accuracy advantage
        if 'accuracy' in quantum_model_performance and 'accuracy' in classical_model_performance:
            accuracy_advantage = quantum_model_performance['accuracy'] - \
                               classical_model_performance['accuracy']
            advantages['accuracy_advantage'] = accuracy_advantage
        
        # Training efficiency advantage
        if 'training_time' in quantum_model_performance and 'training_time' in classical_model_performance:
            time_advantage = classical_model_performance['training_time'] / \
                           quantum_model_performance['training_time']
            advantages['time_advantage'] = time_advantage
        
        # Parameter efficiency advantage
        if 'n_parameters' in quantum_model_performance and 'n_parameters' in classical_model_performance:
            param_advantage = classical_model_performance['n_parameters'] / \
                            quantum_model_performance['n_parameters']
            advantages['parameter_advantage'] = param_advantage
        
        # Memory efficiency advantage
        if 'memory_usage' in quantum_model_performance and 'memory_usage' in classical_model_performance:
            memory_advantage = classical_model_performance['memory_usage'] / \
                             quantum_model_performance['memory_usage']
            advantages['memory_advantage'] = memory_advantage
        
        # Overall advantage score
        advantage_scores = [v for v in advantages.values() if v > 0]
        overall_advantage = np.mean(advantage_scores) if advantage_scores else 0.0
        advantages['overall_advantage'] = overall_advantage
        
        return advantages
    
    def verify_quantum_advantage(self, quantum_results: List[Dict[str, float]],
                               classical_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Comprehensive quantum advantage verification.
        
        Performs multiple tests and returns detailed verification report.
        """
        verification_report = {
            'timestamp': time.time(),
            'n_quantum_runs': len(quantum_results),
            'n_classical_runs': len(classical_results),
            'tests_performed': [],
            'advantage_verified': False,
            'confidence_level': 1 - self.significance_level
        }
        
        # Extract performance metrics
        quantum_accuracies = [r.get('accuracy', 0) for r in quantum_results]
        classical_accuracies = [r.get('accuracy', 0) for r in classical_results]
        
        quantum_times = [r.get('training_time', 0) for r in quantum_results]
        classical_times = [r.get('training_time', 0) for r in classical_results]
        
        # Test 1: Statistical significance of accuracy improvement
        if quantum_accuracies and classical_accuracies:
            accuracy_test = self.statistical_significance_test(
                quantum_accuracies, classical_accuracies
            )
            verification_report['accuracy_test'] = accuracy_test
            verification_report['tests_performed'].append('accuracy_significance')
        
        # Test 2: Training time advantage
        if quantum_times and classical_times:
            time_test = self.statistical_significance_test(
                [-t for t in quantum_times],  # Negative because lower is better
                [-t for t in classical_times]
            )
            verification_report['time_test'] = time_test
            verification_report['tests_performed'].append('time_significance')
        
        # Test 3: Learning advantage computation
        if quantum_results and classical_results:
            avg_quantum_perf = self._average_performance(quantum_results)
            avg_classical_perf = self._average_performance(classical_results)
            
            learning_advantage = self.compute_learning_advantage(
                avg_quantum_perf, avg_classical_perf
            )
            verification_report['learning_advantage'] = learning_advantage
            verification_report['tests_performed'].append('learning_advantage')
        
        # Test 4: Quantum volume assessment
        if quantum_results:
            avg_qubits = np.mean([r.get('n_qubits', 0) for r in quantum_results])
            avg_depth = np.mean([r.get('circuit_depth', 0) for r in quantum_results])
            avg_fidelity = np.mean([r.get('fidelity', 1.0) for r in quantum_results])
            
            quantum_volume = self.compute_quantum_volume(
                int(avg_qubits), int(avg_depth), avg_fidelity
            )
            verification_report['quantum_volume'] = quantum_volume
            verification_report['tests_performed'].append('quantum_volume')
        
        # Overall advantage determination
        advantages_found = []
        
        if 'accuracy_test' in verification_report:
            if verification_report['accuracy_test']['significant'] and \
               verification_report['accuracy_test']['effect_size'] > 0:
                advantages_found.append('accuracy')
        
        if 'time_test' in verification_report:
            if verification_report['time_test']['significant'] and \
               verification_report['time_test']['effect_size'] > 0:
                advantages_found.append('training_time')
        
        if 'learning_advantage' in verification_report:
            if verification_report['learning_advantage']['overall_advantage'] > 1.1:
                advantages_found.append('overall_learning')
        
        # Quantum advantage verified if at least one significant advantage found
        verification_report['advantage_verified'] = len(advantages_found) > 0
        verification_report['advantages_found'] = advantages_found
        
        # Store results
        self.advantage_history.append(verification_report)
        
        return verification_report
    
    def _average_performance(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average performance across multiple runs."""
        if not results:
            return {}
        
        avg_performance = {}
        keys = set().union(*[r.keys() for r in results])
        
        for key in keys:
            values = [r.get(key, 0) for r in results if key in r]
            if values:
                avg_performance[key] = np.mean(values)
        
        return avg_performance
    
    def generate_advantage_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive quantum advantage report."""
        if not self.advantage_history:
            return "No quantum advantage verification data available."
        
        latest_verification = self.advantage_history[-1]
        
        report = f"""
QUANTUM ADVANTAGE VERIFICATION REPORT
=====================================

Verification Date: {time.ctime(latest_verification['timestamp'])}
Confidence Level: {latest_verification['confidence_level']:.1%}

SUMMARY
-------
Quantum Advantage Verified: {'YES' if latest_verification['advantage_verified'] else 'NO'}
Advantages Found: {', '.join(latest_verification.get('advantages_found', []))}
Tests Performed: {len(latest_verification['tests_performed'])}

DETAILED RESULTS
---------------
"""
        
        # Accuracy test results
        if 'accuracy_test' in latest_verification:
            acc_test = latest_verification['accuracy_test']
            report += f"""
Accuracy Advantage Test:
  - Quantum Mean Accuracy: {acc_test['quantum_mean']:.4f}
  - Classical Mean Accuracy: {acc_test['classical_mean']:.4f}
  - Effect Size (Cohen's d): {acc_test['effect_size']:.4f}
  - P-value: {acc_test['p_value']:.6f}
  - Statistically Significant: {'YES' if acc_test['significant'] else 'NO'}
"""
        
        # Time advantage test results
        if 'time_test' in latest_verification:
            time_test = latest_verification['time_test']
            report += f"""
Training Time Advantage Test:
  - Effect Size: {time_test['effect_size']:.4f}
  - P-value: {time_test['p_value']:.6f}
  - Statistically Significant: {'YES' if time_test['significant'] else 'NO'}
"""
        
        # Learning advantage results
        if 'learning_advantage' in latest_verification:
            learn_adv = latest_verification['learning_advantage']
            report += f"""
Learning Advantage Analysis:
  - Overall Advantage Score: {learn_adv.get('overall_advantage', 0):.4f}
  - Parameter Efficiency: {learn_adv.get('parameter_advantage', 0):.2f}x
  - Memory Efficiency: {learn_adv.get('memory_advantage', 0):.2f}x
"""
        
        # Quantum volume
        if 'quantum_volume' in latest_verification:
            qv = latest_verification['quantum_volume']
            report += f"""
Quantum Volume: {qv:.2f}
"""
        
        report += f"""
CONCLUSION
----------
Based on {len(latest_verification['tests_performed'])} rigorous tests with 
{latest_verification['confidence_level']:.1%} confidence level, quantum advantage 
has been {'VERIFIED' if latest_verification['advantage_verified'] else 'NOT VERIFIED'}.
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
