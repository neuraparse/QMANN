"""
Classical Controller Components for QMNN

This module contains classical neural network components that interface
with quantum memory and quantum layers in the QMNN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ClassicalEncoder(nn.Module):
    """
    Classical encoder that prepares data for quantum processing.
    
    Handles input normalization, dimensionality reduction, and
    quantum-compatible encoding.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Quantum normalization layer
        self.quantum_norm = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Tanh()  # Normalize to [-1, 1] for quantum encoding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input for quantum processing.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Quantum-ready encoded tensor [batch_size, seq_len, output_dim]
        """
        # Apply encoding layers
        encoded = self.layers(x)
        
        # Normalize for quantum compatibility
        quantum_ready = self.quantum_norm(encoded)
        
        return quantum_ready


class LSTMController(nn.Module):
    """
    LSTM-based controller for managing quantum memory operations.
    
    Generates read/write commands and manages temporal dependencies
    in quantum memory access patterns.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension adjustment for bidirectional LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Memory control heads
        self.read_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.write_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # Memory operation gates
        self.read_gate = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
            nn.Sigmoid()
        )
        
        self.write_gate = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM controller.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            hidden: Optional hidden state tuple
            
        Returns:
            Tuple of (lstm_output, read_commands, write_commands, gates, new_hidden)
        """
        # LSTM forward pass
        lstm_output, new_hidden = self.lstm(x, hidden)
        
        # Generate memory operation commands
        read_commands = self.read_head(lstm_output)
        write_commands = self.write_head(lstm_output)
        
        # Generate operation gates
        read_gates = self.read_gate(lstm_output)
        write_gates = self.write_gate(lstm_output)
        
        gates = torch.cat([read_gates, write_gates], dim=-1)
        
        return lstm_output, read_commands, write_commands, gates, new_hidden


class ClassicalDecoder(nn.Module):
    """
    Classical decoder that processes quantum outputs for final predictions.
    
    Handles quantum-to-classical conversion and final output generation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_classes: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Quantum-to-classical conversion
        self.quantum_converter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Main decoder layers
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output head
        if num_classes is not None:
            # Classification head
            self.output_head = nn.Linear(hidden_dim // 2, num_classes)
        else:
            # Regression head
            self.output_head = nn.Linear(hidden_dim // 2, output_dim)
            
    def forward(self, quantum_output: torch.Tensor, 
                classical_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode quantum output to final predictions.
        
        Args:
            quantum_output: Output from quantum layers [batch_size, seq_len, input_dim]
            classical_context: Optional classical context for fusion
            
        Returns:
            Final predictions [batch_size, seq_len, output_dim]
        """
        # Convert quantum output to classical representation
        classical_repr = self.quantum_converter(quantum_output)
        
        # Fuse with classical context if provided
        if classical_context is not None:
            # Ensure compatible dimensions
            if classical_context.shape[-1] != classical_repr.shape[-1]:
                context_proj = nn.Linear(
                    classical_context.shape[-1], 
                    classical_repr.shape[-1]
                ).to(classical_repr.device)
                classical_context = context_proj(classical_context)
                
            # Fusion via addition (could be more sophisticated)
            classical_repr = classical_repr + classical_context
            
        # Apply decoder layers
        decoded = self.decoder_layers(classical_repr)
        
        # Generate final output
        output = self.output_head(decoded)
        
        return output


class HybridController(nn.Module):
    """
    Hybrid controller that manages both classical and quantum components.
    
    Coordinates information flow between classical encoders/decoders
    and quantum memory/processing units.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Classical components
        self.encoder = ClassicalEncoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['quantum_dim'],
            dropout=config.get('dropout', 0.1)
        )
        
        self.controller = LSTMController(
            input_dim=config['quantum_dim'],
            hidden_dim=config['controller_dim'],
            num_layers=config.get('controller_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
        
        self.decoder = ClassicalDecoder(
            input_dim=config['quantum_dim'] + config['controller_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_classes=config.get('num_classes'),
            dropout=config.get('dropout', 0.1)
        )
        
        # Quantum interface parameters
        self.quantum_interface = nn.Parameter(torch.randn(config['quantum_dim']) * 0.1)
        
    def forward(self, x: torch.Tensor, quantum_processor=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid controller.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            quantum_processor: Optional quantum processing function
            
        Returns:
            Dictionary with outputs and intermediate results
        """
        batch_size, seq_len, _ = x.shape
        
        # Classical encoding
        encoded = self.encoder(x)
        
        # Controller processing
        controller_out, read_cmds, write_cmds, gates, _ = self.controller(encoded)
        
        # Quantum processing (if available)
        if quantum_processor is not None:
            quantum_out = quantum_processor(encoded, read_cmds, write_cmds)
        else:
            # Fallback to classical processing
            quantum_out = encoded * torch.sigmoid(self.quantum_interface)
            
        # Combine controller and quantum outputs
        combined = torch.cat([controller_out, quantum_out], dim=-1)
        
        # Classical decoding
        output = self.decoder(combined, controller_out)
        
        return {
            'output': output,
            'encoded': encoded,
            'controller_output': controller_out,
            'quantum_output': quantum_out,
            'read_commands': read_cmds,
            'write_commands': write_cmds,
            'gates': gates
        }
        
    def get_controller_info(self) -> Dict[str, Any]:
        """Get information about controller configuration."""
        return {
            'type': 'HybridController',
            'config': self.config,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'encoder_parameters': sum(p.numel() for p in self.encoder.parameters()),
            'controller_parameters': sum(p.numel() for p in self.controller.parameters()),
            'decoder_parameters': sum(p.numel() for p in self.decoder.parameters()),
            'quantum_interface_parameters': self.quantum_interface.numel()
        }
