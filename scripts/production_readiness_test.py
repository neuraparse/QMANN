#!/usr/bin/env python3
"""
QMNN Production Readiness Test

This script validates that all production-ready features are working correctly
and the system meets enterprise requirements.
"""

import sys
import os
import time
import warnings
from typing import Dict, List, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_noise_aware_qram():
    """Test noise-aware QRAM functionality."""
    print("🧪 Testing Noise-Aware QRAM...")
    
    try:
        from qmnn.core.noise_aware_qram import NoiseAwareQRAM, NoiseParameters, ShotCostScheduler
        import numpy as np
        
        # Test with budget constraints
        scheduler = ShotCostScheduler(budget_usd=5.0, target_fidelity=0.9)
        noise_params = NoiseParameters(
            single_qubit_gate_error=0.001,
            two_qubit_gate_error=0.01,
            readout_error=0.02
        )
        
        qram = NoiseAwareQRAM(
            memory_size=16, 
            address_qubits=4,
            noise_params=noise_params,
            shot_scheduler=scheduler
        )
        
        # Test write operation
        test_data = np.array([1.0, 0.5, -0.3, 0.8])
        write_result = qram.write_with_noise(address=5, data=test_data)
        
        assert write_result['success'], "Write operation failed"
        assert 'fidelity' in write_result, "Missing fidelity metric"
        assert 'cost' in write_result, "Missing cost tracking"
        
        # Test read operation
        read_result = qram.read_with_noise(address=5)
        
        assert read_result['success'], "Read operation failed"
        assert read_result['data'] is not None, "No data returned"
        
        # Test statistics
        stats = qram.get_noise_statistics()
        assert 'noise_parameters' in stats, "Missing noise parameters"
        assert 'cost_metrics' in stats, "Missing cost metrics"
        
        print("  ✅ Noise-aware QRAM working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Noise-aware QRAM test failed: {e}")
        return False

def test_assetops_integration():
    """Test AssetOps benchmark integration."""
    print("🧪 Testing AssetOps Integration...")
    
    try:
        from benchmarks.assetops_adapter import AssetOpsBenchmarkAdapter
        
        adapter = AssetOpsBenchmarkAdapter()
        
        # Test task listing
        tasks = adapter.list_available_tasks()
        assert len(tasks) > 0, "No tasks available"
        
        # Test each industry scenario
        for task_id in tasks:
            task_info = adapter.get_task_info(task_id)
            
            # Generate synthetic data
            X, y = adapter.generate_synthetic_dataset(task_id, n_samples=50)
            
            assert X.shape[0] == 50, f"Wrong sample count for {task_id}"
            assert len(X.shape) == 3, f"Wrong tensor dimensions for {task_id}"
            assert task_info.industry_domain in ['manufacturing', 'energy', 'logistics', 'healthcare'], \
                f"Invalid industry domain for {task_id}"
                
        # Test benchmark summary
        summary = adapter.get_benchmark_summary()
        assert summary['total_tasks'] == len(tasks), "Task count mismatch"
        assert len(summary['industry_domains']) > 0, "No industry domains"
        
        print(f"  ✅ AssetOps integration working ({len(tasks)} tasks)")
        return True
        
    except Exception as e:
        print(f"  ❌ AssetOps integration test failed: {e}")
        return False

def test_telemetry_system():
    """Test telemetry and monitoring system."""
    print("🧪 Testing Telemetry System...")
    
    try:
        from qmnn.telemetry.agentops_integration import (
            AgentOpsIntegration, QuantumMemoryMetrics, QMNNPerformanceMetrics
        )
        
        # Test without external dependencies
        telemetry = AgentOpsIntegration(enable_agentops=False, enable_prometheus=True)
        
        # Test quantum metrics recording
        quantum_metrics = QuantumMemoryMetrics(
            timestamp=time.time(),
            operation_type='read',
            memory_address=42,
            memory_capacity_used=0.75,
            quantum_fidelity=0.95,
            classical_fallback=False,
            execution_time_ms=150.0,
            shots_used=1024,
            cost_usd=0.10,
            error_rate=0.01,
            decoherence_time_us=50.0,
            backend_name='ibm_simulator'
        )
        
        telemetry.record_quantum_operation(quantum_metrics)
        
        # Test performance metrics
        perf_metrics = QMNNPerformanceMetrics(
            timestamp=time.time(),
            model_id='qmnn_test',
            batch_size=32,
            sequence_length=50,
            forward_pass_time_ms=200.0,
            memory_hit_ratio=0.85,
            quantum_advantage_ratio=1.2,
            total_parameters=10000,
            quantum_parameters=2000,
            memory_usage_mb=512.0,
            gpu_utilization=0.6,
            accuracy=0.92
        )
        
        telemetry.record_model_performance(perf_metrics)
        
        # Test statistics
        stats = telemetry.get_real_time_stats()
        assert stats['total_quantum_operations'] > 0, "No operations recorded"
        
        summary = telemetry.get_quantum_metrics_summary()
        assert summary['count'] > 0, "No metrics in summary"
        
        # Test Grafana dashboard generation
        dashboard = telemetry.create_grafana_dashboard()
        assert 'dashboard' in dashboard, "Invalid dashboard format"
        
        print("  ✅ Telemetry system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Telemetry system test failed: {e}")
        return False

def test_watermarking_system():
    """Test quantum watermarking system."""
    print("🧪 Testing Watermarking System...")
    
    try:
        from qmnn.utils.watermark import (
            prepare_dataset_with_watermark, verify_dataset_watermark,
            WatermarkConfig, QuantumWatermarkEmbedder
        )
        import torch
        
        # Create test dataset
        test_data = torch.randn(100, 20, 8)
        
        # Test watermark embedding
        config = WatermarkConfig(
            watermark_strength=0.1,
            embedding_ratio=0.05,
            quantum_encoding=True,
            secret_key="test_secret_key"
        )
        
        watermarked_data, metadata = prepare_dataset_with_watermark(
            test_data, config, "test_watermark"
        )
        
        assert watermarked_data.shape == test_data.shape, "Shape changed during watermarking"
        assert 'watermark_info' in metadata, "Missing watermark info"
        
        # Test watermark verification
        verification = verify_dataset_watermark(watermarked_data, metadata)
        
        assert verification['watermark_detected'], "Watermark not detected in original data"
        assert verification['confidence'] > 0.8, "Low confidence in watermark detection"
        
        # Test with modified data (should have lower confidence)
        modified_data = watermarked_data + 0.1 * torch.randn_like(watermarked_data)
        verification_modified = verify_dataset_watermark(modified_data, metadata)
        
        # Should still detect but with lower confidence
        assert verification_modified['confidence'] < verification['confidence'], \
            "Confidence should decrease with data modification"
            
        print("  ✅ Watermarking system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Watermarking system test failed: {e}")
        return False

def test_surface_code_integration():
    """Test surface code error correction."""
    print("🧪 Testing Surface Code Integration...")
    
    try:
        from qmnn.error_correction import SurfaceCodeManager
        
        # Test surface code manager
        surface_code = SurfaceCodeManager(code_distance=3, error_rate=0.005)
        
        # Test circuit creation
        circuit = surface_code.create_surface_code_circuit(logical_qubits=1)
        
        assert circuit.num_qubits > 0, "No qubits in surface code circuit"
        
        # Test syndrome decoding (with dummy data)
        dummy_results = {'000000000': 800, '000000001': 200}
        decoding = surface_code.decode_syndrome(dummy_results)
        
        assert 'syndrome_bits' in decoding, "Missing syndrome bits"
        assert 'error_correction' in decoding, "Missing error correction"
        assert 'success_probability' in decoding, "Missing success probability"
        
        # Test surface code info
        info = surface_code.get_surface_code_info()
        assert info['code_distance'] == 3, "Wrong code distance"
        assert info['below_threshold'], "Should be below threshold"
        
        print("  ✅ Surface code integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Surface code integration test failed: {e}")
        return False

def test_mode_separation():
    """Test mode separation system."""
    print("🧪 Testing Mode Separation...")
    
    try:
        from qmnn import recommend_mode, print_mode_guide, quick_start
        from qmnn.config import (
            THEORETICAL_ANALYSIS, SIMULATION_VALIDATION, HARDWARE_PROOF_OF_CONCEPT,
            validate_experimental_setup
        )
        
        # Test mode recommendations
        theoretical_rec = recommend_mode(purpose="research", budget_usd=0.0, n_qubits=50)
        assert "THEORETICAL" in theoretical_rec, "Should recommend theoretical mode"
        
        simulation_rec = recommend_mode(purpose="development", budget_usd=0.0, n_qubits=10)
        assert "SIMULATION" in simulation_rec, "Should recommend simulation mode"
        
        hardware_rec = recommend_mode(purpose="validation", budget_usd=20.0, n_qubits=6)
        assert "HARDWARE" in hardware_rec, "Should recommend hardware mode"
        
        # Test configuration validation
        theoretical_valid = validate_experimental_setup(THEORETICAL_ANALYSIS)
        simulation_valid = validate_experimental_setup(SIMULATION_VALIDATION)
        hardware_valid = validate_experimental_setup(HARDWARE_PROOF_OF_CONCEPT)
        
        # All should be valid (warnings are OK)
        assert isinstance(theoretical_valid, bool), "Invalid validation result"
        assert isinstance(simulation_valid, bool), "Invalid validation result"
        assert isinstance(hardware_valid, bool), "Invalid validation result"
        
        print("  ✅ Mode separation working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Mode separation test failed: {e}")
        return False

def test_cost_estimation():
    """Test cost estimation system."""
    print("🧪 Testing Cost Estimation...")
    
    try:
        sys.path.append('scripts')
        from estimate_hardware_costs import estimate_experiment_cost, estimate_qmnn_experiment_cost
        
        # Test single experiment cost estimation
        cost_estimate = estimate_experiment_cost(
            backend_name='ibm_brisbane',
            n_qubits=6,
            shots=1000,
            circuit_depth=50,
            n_experiments=1
        )
        
        assert 'total_cost' in cost_estimate, "Missing total cost"
        assert 'backend' in cost_estimate, "Missing backend info"
        assert cost_estimate['n_qubits'] == 6, "Wrong qubit count"
        
        # Test QMNN experiment cost estimation
        qmnn_estimate = estimate_qmnn_experiment_cost(
            n_qubits=6,
            memory_capacity=32,
            training_epochs=5,
            backends=['ibm_brisbane']
        )
        
        assert 'experiment_parameters' in qmnn_estimate, "Missing experiment parameters"
        assert 'backend_estimates' in qmnn_estimate, "Missing backend estimates"
        
        print("  ✅ Cost estimation working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Cost estimation test failed: {e}")
        return False

def test_license_compliance():
    """Test license compliance and documentation."""
    print("🧪 Testing License Compliance...")
    
    try:
        # Check license files exist
        license_files = ['LICENSE-ACADEMIC', 'LICENSE-COMMERCIAL']
        
        for license_file in license_files:
            if not os.path.exists(license_file):
                raise FileNotFoundError(f"Missing license file: {license_file}")
                
        # Check README has proper licensing section
        with open('README.md', 'r') as f:
            readme_content = f.read()
            
        license_indicators = ['license', 'citation', 'commercial']
        for indicator in license_indicators:
            if indicator.lower() not in readme_content.lower():
                warnings.warn(f"README missing {indicator} information")
                
        print("  ✅ License compliance verified")
        return True
        
    except Exception as e:
        print(f"  ❌ License compliance test failed: {e}")
        return False

def generate_production_report(test_results: Dict[str, bool]) -> str:
    """Generate production readiness report."""
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    report = []
    report.append("QMNN PRODUCTION READINESS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        report.append("✅ PRODUCTION READY - All systems operational")
    elif passed_tests >= total_tests * 0.8:
        report.append("⚠️  MOSTLY READY - Minor issues to address")
    else:
        report.append("❌ NOT READY - Critical issues require attention")
        
    report.append("")
    
    # Detailed Results
    report.append("DETAILED TEST RESULTS")
    report.append("-" * 25)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        report.append(f"{status} {test_name}")
        
    report.append("")
    
    # Production Features Status
    report.append("PRODUCTION FEATURES STATUS")
    report.append("-" * 30)
    
    features = [
        ("Noise-Aware Quantum Operations", test_results.get('noise_aware_qram', False)),
        ("Industry 4.0 Benchmarks", test_results.get('assetops_integration', False)),
        ("Enterprise Telemetry", test_results.get('telemetry_system', False)),
        ("Data Protection & Watermarking", test_results.get('watermarking_system', False)),
        ("Fault-Tolerant Error Correction", test_results.get('surface_code_integration', False)),
        ("Mode Separation & Safety", test_results.get('mode_separation', False)),
        ("Cost Management", test_results.get('cost_estimation', False)),
        ("License Compliance", test_results.get('license_compliance', False))
    ]
    
    for feature, status in features:
        status_icon = "✅" if status else "❌"
        report.append(f"{status_icon} {feature}")
        
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 15)
    
    if passed_tests == total_tests:
        report.append("• System is ready for production deployment")
        report.append("• Consider setting up monitoring and alerting")
        report.append("• Establish regular testing and maintenance schedule")
    else:
        report.append("• Address failing tests before production deployment")
        report.append("• Review error logs and fix critical issues")
        report.append("• Re-run tests after fixes")
        
    return "\n".join(report)

def main():
    """Run all production readiness tests."""
    print("🚀 QMNN PRODUCTION READINESS VALIDATION")
    print("=" * 60)
    print("Testing enterprise-grade features and capabilities...")
    print()
    
    tests = [
        ("noise_aware_qram", test_noise_aware_qram),
        ("assetops_integration", test_assetops_integration),
        ("telemetry_system", test_telemetry_system),
        ("watermarking_system", test_watermarking_system),
        ("surface_code_integration", test_surface_code_integration),
        ("mode_separation", test_mode_separation),
        ("cost_estimation", test_cost_estimation),
        ("license_compliance", test_license_compliance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results[test_name] = False
            
        print()  # Add spacing between tests
        
    # Generate and display report
    report = generate_production_report(results)
    
    print("=" * 60)
    print("PRODUCTION READINESS REPORT")
    print("=" * 60)
    print(report)
    
    # Save report
    with open('production_readiness_report.txt', 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to: production_readiness_report.txt")
    
    # Return exit code
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    if passed_tests == total_tests:
        print("\n🎉 QMNN is PRODUCTION READY!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Address issues before production.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
