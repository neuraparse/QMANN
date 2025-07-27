"""
AgentOps Telemetry Integration for QMNN

This module provides comprehensive telemetry and observability
for quantum memory operations, enabling monitoring and optimization
of QMNN performance in production environments.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False


@dataclass
class QuantumMemoryMetrics:
    """Quantum memory operation metrics."""
    timestamp: float
    operation_type: str  # "read", "write", "search"
    memory_address: int
    memory_capacity_used: float  # Percentage
    quantum_fidelity: float
    classical_fallback: bool
    execution_time_ms: float
    shots_used: int
    cost_usd: float
    error_rate: float
    decoherence_time_us: float
    backend_name: str


@dataclass
class QMNNPerformanceMetrics:
    """QMNN model performance metrics."""
    timestamp: float
    model_id: str
    batch_size: int
    sequence_length: int
    forward_pass_time_ms: float
    memory_hit_ratio: float
    quantum_advantage_ratio: float  # Quantum vs classical performance
    total_parameters: int
    quantum_parameters: int
    memory_usage_mb: float
    gpu_utilization: float
    accuracy: Optional[float] = None
    loss: Optional[float] = None


class PrometheusMetrics:
    """Prometheus metrics collector for QMNN."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics."""
        self.registry = registry or prometheus_client.REGISTRY
        
        # Quantum memory metrics
        self.quantum_operations_total = Counter(
            'qmnn_quantum_operations_total',
            'Total quantum memory operations',
            ['operation_type', 'backend', 'success'],
            registry=self.registry
        )
        
        self.quantum_fidelity = Histogram(
            'qmnn_quantum_fidelity',
            'Quantum operation fidelity',
            ['operation_type', 'backend'],
            registry=self.registry
        )
        
        self.quantum_execution_time = Histogram(
            'qmnn_quantum_execution_time_seconds',
            'Quantum operation execution time',
            ['operation_type', 'backend'],
            registry=self.registry
        )
        
        self.quantum_cost = Counter(
            'qmnn_quantum_cost_usd_total',
            'Total quantum operation cost in USD',
            ['backend'],
            registry=self.registry
        )
        
        self.memory_hit_ratio = Gauge(
            'qmnn_memory_hit_ratio',
            'Quantum memory hit ratio',
            ['model_id'],
            registry=self.registry
        )
        
        # QMNN model metrics
        self.model_forward_time = Histogram(
            'qmnn_model_forward_time_seconds',
            'QMNN model forward pass time',
            ['model_id'],
            registry=self.registry
        )
        
        self.quantum_advantage_ratio = Gauge(
            'qmnn_quantum_advantage_ratio',
            'Quantum vs classical performance ratio',
            ['model_id'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'qmnn_model_accuracy',
            'QMNN model accuracy',
            ['model_id', 'dataset'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'qmnn_memory_usage_bytes',
            'QMNN memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.decoherence_events = Counter(
            'qmnn_decoherence_events_total',
            'Total decoherence events',
            ['backend', 'severity'],
            registry=self.registry
        )


class AgentOpsIntegration:
    """
    AgentOps integration for QMNN telemetry and monitoring.
    
    Provides comprehensive observability for quantum memory operations
    and QMNN model performance.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 enable_prometheus: bool = True,
                 enable_agentops: bool = True):
        """
        Initialize AgentOps integration.
        
        Args:
            api_key: AgentOps API key
            enable_prometheus: Enable Prometheus metrics
            enable_agentops: Enable AgentOps integration
        """
        self.api_key = api_key
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_agentops = enable_agentops and AGENTOPS_AVAILABLE
        
        # Initialize metrics collectors
        if self.enable_prometheus:
            self.prometheus_metrics = PrometheusMetrics()
        else:
            self.prometheus_metrics = None
            
        # Initialize AgentOps
        if self.enable_agentops and api_key:
            try:
                agentops.init(api_key=api_key)
                self.agentops_session = agentops.start_session()
            except Exception as e:
                warnings.warn(f"Failed to initialize AgentOps: {e}")
                self.enable_agentops = False
        else:
            self.agentops_session = None
            
        # Metrics storage
        self.quantum_metrics_buffer = deque(maxlen=10000)
        self.performance_metrics_buffer = deque(maxlen=10000)
        
        # Real-time statistics
        self.stats = {
            'total_quantum_operations': 0,
            'total_cost_usd': 0.0,
            'average_fidelity': 0.0,
            'memory_hit_ratio': 0.0,
            'quantum_advantage_ratio': 1.0,
            'decoherence_events': 0,
            'classical_fallbacks': 0
        }
        
        # Background metrics processing
        self._metrics_lock = threading.Lock()
        self._start_background_processing()
        
    def record_quantum_operation(self, metrics: QuantumMemoryMetrics):
        """Record quantum memory operation metrics."""
        with self._metrics_lock:
            self.quantum_metrics_buffer.append(metrics)
            
            # Update real-time stats
            self.stats['total_quantum_operations'] += 1
            self.stats['total_cost_usd'] += metrics.cost_usd
            
            if metrics.classical_fallback:
                self.stats['classical_fallbacks'] += 1
                
            # Update running averages
            self._update_running_averages()
            
        # Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.quantum_operations_total.labels(
                operation_type=metrics.operation_type,
                backend=metrics.backend_name,
                success=str(not metrics.classical_fallback)
            ).inc()
            
            self.prometheus_metrics.quantum_fidelity.labels(
                operation_type=metrics.operation_type,
                backend=metrics.backend_name
            ).observe(metrics.quantum_fidelity)
            
            self.prometheus_metrics.quantum_execution_time.labels(
                operation_type=metrics.operation_type,
                backend=metrics.backend_name
            ).observe(metrics.execution_time_ms / 1000.0)
            
            self.prometheus_metrics.quantum_cost.labels(
                backend=metrics.backend_name
            ).inc(metrics.cost_usd)
            
        # AgentOps integration
        if self.enable_agentops and self.agentops_session:
            try:
                agentops.record_action(
                    action_type="quantum_memory_operation",
                    params={
                        "operation_type": metrics.operation_type,
                        "backend": metrics.backend_name,
                        "fidelity": metrics.quantum_fidelity,
                        "cost_usd": metrics.cost_usd,
                        "shots_used": metrics.shots_used
                    },
                    returns={"success": not metrics.classical_fallback}
                )
            except Exception as e:
                warnings.warn(f"AgentOps recording failed: {e}")
                
    def record_model_performance(self, metrics: QMNNPerformanceMetrics):
        """Record QMNN model performance metrics."""
        with self._metrics_lock:
            self.performance_metrics_buffer.append(metrics)
            
        # Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.model_forward_time.labels(
                model_id=metrics.model_id
            ).observe(metrics.forward_pass_time_ms / 1000.0)
            
            self.prometheus_metrics.memory_hit_ratio.labels(
                model_id=metrics.model_id
            ).set(metrics.memory_hit_ratio)
            
            self.prometheus_metrics.quantum_advantage_ratio.labels(
                model_id=metrics.model_id
            ).set(metrics.quantum_advantage_ratio)
            
            if metrics.accuracy is not None:
                self.prometheus_metrics.model_accuracy.labels(
                    model_id=metrics.model_id,
                    dataset="current"
                ).set(metrics.accuracy)
                
            self.prometheus_metrics.memory_usage.labels(
                component="model"
            ).set(metrics.memory_usage_mb * 1024 * 1024)
            
        # AgentOps integration
        if self.enable_agentops and self.agentops_session:
            try:
                agentops.record_action(
                    action_type="model_inference",
                    params={
                        "model_id": metrics.model_id,
                        "batch_size": metrics.batch_size,
                        "sequence_length": metrics.sequence_length
                    },
                    returns={
                        "forward_time_ms": metrics.forward_pass_time_ms,
                        "memory_hit_ratio": metrics.memory_hit_ratio,
                        "quantum_advantage": metrics.quantum_advantage_ratio,
                        "accuracy": metrics.accuracy
                    }
                )
            except Exception as e:
                warnings.warn(f"AgentOps recording failed: {e}")
                
    def record_decoherence_event(self, backend: str, severity: str, 
                                details: Dict[str, Any]):
        """Record quantum decoherence event."""
        with self._metrics_lock:
            self.stats['decoherence_events'] += 1
            
        if self.prometheus_metrics:
            self.prometheus_metrics.decoherence_events.labels(
                backend=backend,
                severity=severity
            ).inc()
            
        if self.enable_agentops and self.agentops_session:
            try:
                agentops.record_action(
                    action_type="decoherence_event",
                    params={
                        "backend": backend,
                        "severity": severity,
                        **details
                    }
                )
            except Exception as e:
                warnings.warn(f"AgentOps recording failed: {e}")
                
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics."""
        with self._metrics_lock:
            return self.stats.copy()
            
    def get_quantum_metrics_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Get summary of recent quantum metrics."""
        with self._metrics_lock:
            recent_metrics = list(self.quantum_metrics_buffer)[-last_n:]
            
        if not recent_metrics:
            return {}
            
        # Calculate statistics
        fidelities = [m.quantum_fidelity for m in recent_metrics]
        execution_times = [m.execution_time_ms for m in recent_metrics]
        costs = [m.cost_usd for m in recent_metrics]
        
        return {
            'count': len(recent_metrics),
            'average_fidelity': sum(fidelities) / len(fidelities),
            'average_execution_time_ms': sum(execution_times) / len(execution_times),
            'total_cost_usd': sum(costs),
            'classical_fallback_rate': sum(1 for m in recent_metrics if m.classical_fallback) / len(recent_metrics),
            'backends_used': list(set(m.backend_name for m in recent_metrics)),
            'operation_types': list(set(m.operation_type for m in recent_metrics))
        }
        
    def export_metrics_json(self, filepath: str):
        """Export metrics to JSON file."""
        with self._metrics_lock:
            quantum_metrics = [asdict(m) for m in self.quantum_metrics_buffer]
            performance_metrics = [asdict(m) for m in self.performance_metrics_buffer]
            
        export_data = {
            'timestamp': time.time(),
            'stats': self.stats,
            'quantum_metrics': quantum_metrics,
            'performance_metrics': performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
    def _update_running_averages(self):
        """Update running average statistics."""
        if not self.quantum_metrics_buffer:
            return
            
        recent_metrics = list(self.quantum_metrics_buffer)[-100:]  # Last 100 operations
        
        # Average fidelity
        fidelities = [m.quantum_fidelity for m in recent_metrics if not m.classical_fallback]
        if fidelities:
            self.stats['average_fidelity'] = sum(fidelities) / len(fidelities)
            
        # Memory hit ratio (simplified calculation)
        successful_ops = [m for m in recent_metrics if not m.classical_fallback]
        if recent_metrics:
            self.stats['memory_hit_ratio'] = len(successful_ops) / len(recent_metrics)
            
    def _start_background_processing(self):
        """Start background thread for metrics processing."""
        def process_metrics():
            while True:
                time.sleep(60)  # Process every minute
                try:
                    self._update_running_averages()
                    
                    # AgentOps session heartbeat
                    if self.enable_agentops and self.agentops_session:
                        agentops.record_action(
                            action_type="heartbeat",
                            params={"stats": self.get_real_time_stats()}
                        )
                except Exception as e:
                    warnings.warn(f"Background metrics processing failed: {e}")
                    
        thread = threading.Thread(target=process_metrics, daemon=True)
        thread.start()
        
    def create_grafana_dashboard(self) -> Dict[str, Any]:
        """Create Grafana dashboard configuration for QMNN metrics."""
        dashboard = {
            "dashboard": {
                "title": "QMNN Quantum Memory Monitoring",
                "panels": [
                    {
                        "title": "Quantum Operations Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(qmnn_quantum_operations_total[5m])",
                                "legendFormat": "{{operation_type}} - {{backend}}"
                            }
                        ]
                    },
                    {
                        "title": "Quantum Fidelity",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "qmnn_quantum_fidelity",
                                "legendFormat": "{{operation_type}} - {{backend}}"
                            }
                        ]
                    },
                    {
                        "title": "Memory Hit Ratio",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "qmnn_memory_hit_ratio",
                                "legendFormat": "{{model_id}}"
                            }
                        ]
                    },
                    {
                        "title": "Quantum Cost",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(qmnn_quantum_cost_usd_total[1h])",
                                "legendFormat": "{{backend}}"
                            }
                        ]
                    },
                    {
                        "title": "Decoherence Events",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(qmnn_decoherence_events_total[5m])",
                                "legendFormat": "{{backend}} - {{severity}}"
                            }
                        ]
                    }
                ]
            }
        }
        
        return dashboard
        
    def __del__(self):
        """Cleanup AgentOps session."""
        if self.enable_agentops and self.agentops_session:
            try:
                agentops.end_session("Success")
            except:
                pass
