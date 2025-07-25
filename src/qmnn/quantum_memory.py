"""
Quantum Memory Components for QMNN

This module contains quantum memory implementations extracted from core.py
for better modularity and maintainability.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import warnings
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from .core import QRAM, QuantumMemory


class QuantumMemoryManager:
    """
    High-level manager for quantum memory operations.
    
    Provides unified interface for different quantum memory types
    and handles memory allocation, optimization, and monitoring.
    """
    
    def __init__(self, memory_configs: List[Dict]):
        """
        Initialize quantum memory manager.
        
        Args:
            memory_configs: List of memory configuration dictionaries
        """
        self.memories = []
        self.memory_types = []
        self.total_capacity = 0
        
        for config in memory_configs:
            memory_type = config.get('type', 'qram')
            
            if memory_type == 'qram':
                memory = QRAM(
                    memory_size=config['capacity'],
                    address_qubits=config['address_qubits'],
                    max_data_qubits=config.get('max_data_qubits', 8),
                    use_amplitude_encoding=config.get('use_amplitude_encoding', True)
                )
            elif memory_type == 'quantum_memory':
                memory = QuantumMemory(
                    capacity=config['capacity'],
                    embedding_dim=config['embedding_dim'],
                    max_qubits=config.get('max_qubits', 16),
                    use_amplitude_encoding=config.get('use_amplitude_encoding', True)
                )
            else:
                raise ValueError(f"Unknown memory type: {memory_type}")
                
            self.memories.append(memory)
            self.memory_types.append(memory_type)
            self.total_capacity += config['capacity']
            
    def allocate_memory(self, size: int, memory_type: str = 'auto') -> int:
        """
        Allocate memory from available quantum memories.
        
        Args:
            size: Required memory size
            memory_type: Preferred memory type or 'auto'
            
        Returns:
            Memory index allocated
        """
        if memory_type == 'auto':
            # Find best fit memory
            for i, memory in enumerate(self.memories):
                if hasattr(memory, 'effective_capacity'):
                    available = memory.effective_capacity - memory.stored_count
                elif hasattr(memory, 'memory_size'):
                    available = memory.memory_size - memory._memory_usage * memory.memory_size
                else:
                    available = 0
                    
                if available >= size:
                    return i
        else:
            # Find specific memory type
            for i, mtype in enumerate(self.memory_types):
                if mtype == memory_type:
                    return i
                    
        raise RuntimeError(f"Cannot allocate {size} units of {memory_type} memory")
        
    def get_memory_stats(self) -> Dict[str, Union[int, float, Dict]]:
        """Get comprehensive memory statistics."""
        stats = {
            'total_memories': len(self.memories),
            'total_capacity': self.total_capacity,
            'memory_details': []
        }
        
        total_used = 0
        for i, (memory, mtype) in enumerate(zip(self.memories, self.memory_types)):
            if hasattr(memory, 'get_memory_info'):
                memory_info = memory.get_memory_info()
            elif hasattr(memory, 'capacity_bound'):
                memory_info = memory.capacity_bound()
            else:
                memory_info = {'type': mtype}
                
            stats['memory_details'].append({
                'index': i,
                'type': mtype,
                'info': memory_info
            })
            
            if 'capacity_utilization' in memory_info:
                total_used += memory_info['capacity_utilization']
                
        stats['average_utilization'] = total_used / len(self.memories) if self.memories else 0
        
        return stats
        
    def optimize_memory_layout(self) -> None:
        """Optimize memory layout for better performance."""
        for memory in self.memories:
            if hasattr(memory, 'defragment'):
                memory.defragment()
            elif hasattr(memory, 'optimize_memory'):
                memory.optimize_memory()
                
    def validate_quantum_constraints(self) -> Dict[str, bool]:
        """Validate quantum hardware constraints across all memories."""
        all_constraints = {}
        
        for i, memory in enumerate(self.memories):
            if hasattr(memory, 'validate_quantum_constraints'):
                constraints = memory.validate_quantum_constraints()
                for key, value in constraints.items():
                    constraint_key = f"memory_{i}_{key}"
                    all_constraints[constraint_key] = value
                    
        # Overall system constraints
        total_qubits = sum(
            memory.get_hardware_requirements().get('total_qubits', 0)
            for memory in self.memories
            if hasattr(memory, 'get_hardware_requirements')
        )
        
        all_constraints['system_total_qubits_feasible'] = total_qubits <= 100
        all_constraints['system_memory_count_reasonable'] = len(self.memories) <= 10
        
        return all_constraints


class AdaptiveQuantumMemory:
    """
    Adaptive quantum memory that adjusts encoding and capacity based on usage patterns.
    """
    
    def __init__(self, initial_capacity: int = 64, max_capacity: int = 512,
                 adaptation_threshold: float = 0.8):
        """
        Initialize adaptive quantum memory.
        
        Args:
            initial_capacity: Starting memory capacity
            max_capacity: Maximum allowed capacity
            adaptation_threshold: Usage threshold for capacity expansion
        """
        self.initial_capacity = initial_capacity
        self.max_capacity = max_capacity
        self.adaptation_threshold = adaptation_threshold
        
        # Start with basic quantum memory
        self.current_memory = QuantumMemory(
            capacity=initial_capacity,
            embedding_dim=32,
            max_qubits=12
        )
        
        # Usage tracking
        self.access_patterns = []
        self.adaptation_history = []
        
    def store(self, key: np.ndarray, value: np.ndarray) -> int:
        """Store data with adaptive capacity management."""
        # Check if expansion is needed
        current_usage = self.current_memory.memory_usage()
        
        if current_usage > self.adaptation_threshold:
            self._expand_capacity()
            
        # Store in current memory
        try:
            address = self.current_memory.store_embedding(key, value)
            self._record_access('store', address)
            return address
        except RuntimeError as e:
            if "capacity exceeded" in str(e).lower():
                self._expand_capacity()
                return self.current_memory.store_embedding(key, value)
            else:
                raise
                
    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """Retrieve data with access pattern tracking."""
        result = self.current_memory.retrieve_embedding(query)
        self._record_access('retrieve', query)
        return result
        
    def _expand_capacity(self) -> None:
        """Expand memory capacity if possible."""
        current_capacity = self.current_memory.effective_capacity
        new_capacity = min(current_capacity * 2, self.max_capacity)
        
        if new_capacity > current_capacity:
            # Create new memory with expanded capacity
            old_memory = self.current_memory
            
            self.current_memory = QuantumMemory(
                capacity=new_capacity,
                embedding_dim=old_memory.embedding_dim,
                max_qubits=old_memory.max_qubits
            )
            
            # Migrate data (simplified - in practice would need proper migration)
            self.adaptation_history.append({
                'old_capacity': current_capacity,
                'new_capacity': new_capacity,
                'trigger_usage': old_memory.memory_usage()
            })
            
            warnings.warn(f"Quantum memory expanded from {current_capacity} to {new_capacity}")
        else:
            warnings.warn(f"Cannot expand memory beyond maximum capacity {self.max_capacity}")
            
    def _record_access(self, operation: str, data) -> None:
        """Record access patterns for optimization."""
        self.access_patterns.append({
            'operation': operation,
            'timestamp': len(self.access_patterns),
            'data_hash': hash(str(data)) if hasattr(data, '__str__') else 0
        })
        
        # Keep only recent access patterns
        if len(self.access_patterns) > 1000:
            self.access_patterns = self.access_patterns[-500:]
            
    def get_adaptation_stats(self) -> Dict:
        """Get statistics about memory adaptation."""
        return {
            'current_capacity': self.current_memory.effective_capacity,
            'current_usage': self.current_memory.memory_usage(),
            'adaptations_count': len(self.adaptation_history),
            'adaptation_history': self.adaptation_history,
            'access_patterns_count': len(self.access_patterns)
        }
