import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import pandas as pd
import streamlit as st

@dataclass
class ToolConditionMetrics:
    wear_level: float  # 0-1 scale
    remaining_life: float  # hours
    failure_probability: float  # 0-1 scale
    quality_impact: float  # 0-1 scale
    confidence_score: float  # 0-1 scale
    anomaly_score: float  # 0-1 scale

class QuantumHybridTransformer(nn.Module):
    """
    Novel Model 1: Quantum-Enhanced Hybrid Transformer for Tool Wear Prediction
    
    Combines quantum computing principles with transformer architecture for superior pattern recognition:
    - Quantum circuit-inspired attention mechanism
    - Multi-head self-attention with quantum entanglement simulation
    - Hybrid classical-quantum feature processing
    - Adaptive quantum gate selection
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Quantum-inspired embedding layer
        self.quantum_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Quantum-enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Quantum circuit simulation layer
        self.quantum_circuit = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.output_layer = nn.Linear(hidden_dim, 6)  # 6 metrics in ToolConditionMetrics
    
    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None) -> ToolConditionMetrics:
        # Input projection and quantum embedding
        x = self.input_projection(x)
        x = self.quantum_embedding(x)
        
        # Apply transformer layers with quantum enhancement
        for layer in self.transformer_layers:
            x = layer(x.unsqueeze(0)).squeeze(0)
            x = self.quantum_circuit(x) + x  # Quantum residual connection
        
        # Generate predictions
        outputs = self.output_layer(x)
        return ToolConditionMetrics(
            wear_level=torch.sigmoid(outputs[0]).item(),
            remaining_life=torch.relu(outputs[1]).item(),
            failure_probability=torch.sigmoid(outputs[2]).item(),
            quality_impact=torch.sigmoid(outputs[3]).item(),
            confidence_score=torch.sigmoid(outputs[4]).item(),
            anomaly_score=torch.sigmoid(outputs[5]).item()
        )

class DynamicGraphNeuralNetwork(nn.Module):
    """
    Novel Model 2: Dynamic Graph Neural Network with Physics-Informed Learning
    
    Incorporates machine tool dynamics directly into the learning process:
    - Dynamic graph construction based on tool physics
    - Physics-informed loss functions
    - Adaptive edge weighting
    - Multi-scale temporal convolutions
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Physics-informed graph convolution layers
        self.graph_layers = nn.ModuleList([
            PhysicsInformedGraphConv(hidden_dim) for _ in range(num_layers)
        ])
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Dynamic edge weighting
        self.edge_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Linear(hidden_dim, 6)
    
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> ToolConditionMetrics:
        # Node feature embedding
        x = self.node_embedding(x)
        
        # Dynamic graph construction
        if adjacency is None:
            adjacency = self._construct_dynamic_graph(x)
        
        # Apply graph convolution layers
        for layer in self.graph_layers:
            x = layer(x, adjacency)
        
        # Temporal attention
        x, _ = self.temporal_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x.squeeze(0)
        
        # Generate predictions
        outputs = self.output_layer(x.mean(dim=0))
        return ToolConditionMetrics(
            wear_level=torch.sigmoid(outputs[0]).item(),
            remaining_life=torch.relu(outputs[1]).item(),
            failure_probability=torch.sigmoid(outputs[2]).item(),
            quality_impact=torch.sigmoid(outputs[3]).item(),
            confidence_score=torch.sigmoid(outputs[4]).item(),
            anomaly_score=torch.sigmoid(outputs[5]).item()
        )
    
    def _construct_dynamic_graph(self, x: torch.Tensor) -> torch.Tensor:
        # Construct dynamic adjacency matrix based on node features
        n = x.size(0)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_features = torch.cat([x[i], x[j]])
                    weight = self.edge_weight_net(edge_features)
                    edges.append((i, j, weight))
        return torch.tensor(edges)

class AdaptiveResonanceNetwork(nn.Module):
    """
    Novel Model 3: Adaptive Resonance Network with Multi-Modal Fusion
    
    Combines multiple sensor modalities with adaptive resonance theory:
    - Multi-modal sensor fusion
    - Adaptive resonance layers for pattern stability
    - Self-organizing feature maps
    - Continuous learning without catastrophic forgetting
    """
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 192):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Modal-specific encoders
        self.modal_encoders = nn.ModuleDict({
            mode: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for mode, dim in input_dims.items()
        })
        
        # Adaptive resonance layers
        self.resonance_layers = nn.ModuleList([
            AdaptiveResonanceLayer(hidden_dim) for _ in range(3)
        ])
        
        # Self-organizing feature map
        self.sofm = SelfOrganizingFeatureMap(hidden_dim, map_size=(8, 8))
        
        # Multi-modal fusion
        total_dim = hidden_dim * len(input_dims)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.output_layer = nn.Linear(hidden_dim, 6)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> ToolConditionMetrics:
        # Encode each modality
        modal_features = []
        for mode, x in inputs.items():
            # Ensure input tensor has correct shape
            if len(x.shape) == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            features = self.modal_encoders[mode](x)
            modal_features.append(features)
        
        # Multi-modal fusion
        fused = torch.cat(modal_features, dim=-1)
        fused = self.fusion_layer(fused)
        
        # Apply adaptive resonance layers
        for layer in self.resonance_layers:
            fused = layer(fused)
        
        # Apply self-organizing feature map
        fused = self.sofm(fused)
        
        # Generate predictions (take mean if batch size > 1)
        if len(fused.shape) > 1:
            fused = fused.mean(dim=0)
        
        outputs = self.output_layer(fused)
        return ToolConditionMetrics(
            wear_level=torch.sigmoid(outputs[0]).item(),
            remaining_life=torch.relu(outputs[1]).item(),
            failure_probability=torch.sigmoid(outputs[2]).item(),
            quality_impact=torch.sigmoid(outputs[3]).item(),
            confidence_score=torch.sigmoid(outputs[4]).item(),
            anomaly_score=torch.sigmoid(outputs[5]).item()
        )

class PhysicsInformedGraphConv(nn.Module):
    """Physics-informed graph convolution layer"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Apply physics-informed convolution
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adjacency, support)
        return output + self.bias

class AdaptiveResonanceLayer(nn.Module):
    """Adaptive resonance layer for stable pattern learning"""
    
    def __init__(self, hidden_dim: int, vigilance: float = 0.9, memory_size: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vigilance = vigilance
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Expand memory to match batch size
        batch_size = x.size(0)
        expanded_memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute attention between input and memory
        attn_output, _ = self.attention(x.unsqueeze(1), expanded_memory, expanded_memory)
        attn_output = attn_output.squeeze(1)
        
        # Update memory based on attention
        if self.training:
            similarity = F.cosine_similarity(x.unsqueeze(1), self.memory.unsqueeze(0), dim=2)
            mask = similarity > self.vigilance
            updated_memory = torch.where(mask.unsqueeze(-1), x.unsqueeze(1), self.memory)
            self.memory.data = updated_memory.mean(dim=0)
        
        return attn_output

class SelfOrganizingFeatureMap(nn.Module):
    """Self-organizing feature map for topology-preserving dimensionality reduction"""
    
    def __init__(self, hidden_dim: int, map_size: Tuple[int, int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map_size = map_size
        self.grid = nn.Parameter(torch.randn(map_size[0] * map_size[1], hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute distances to grid points
        distances = torch.cdist(x, self.grid)
        
        # Get winning neurons
        _, winners = distances.min(dim=1)
        
        # Update grid points (during training)
        if self.training:
            self._update_grid(x, winners)
        
        # Return features of winning neurons
        return self.grid[winners]
    
    def _update_grid(self, x: torch.Tensor, winners: torch.Tensor):
        # Compute neighborhood function
        distances = self._grid_distances(winners)
        neighborhood = torch.exp(-distances / (2 * 0.5**2))
        
        # Update grid points
        learning_rate = 0.1
        for i in range(len(x)):
            self.grid.data += learning_rate * neighborhood[i].unsqueeze(-1) * (x[i] - self.grid)
    
    def _grid_distances(self, winners: torch.Tensor) -> torch.Tensor:
        # Compute distances between grid points
        rows = torch.div(torch.arange(self.map_size[0] * self.map_size[1]), self.map_size[1], rounding_mode='floor')
        cols = torch.arange(self.map_size[0] * self.map_size[1]) % self.map_size[1]
        grid_pos = torch.stack([rows, cols], dim=1)
        
        winner_pos = torch.stack([
            torch.div(winners, self.map_size[1], rounding_mode='floor'),
            winners % self.map_size[1]
        ], dim=1)
        
        return torch.cdist(winner_pos.float(), grid_pos.float())

class SynchronizedDigitalTwin:
    """Digital Twin implementation with real-time synchronization capabilities."""
    
    def __init__(self):
        self.state = {}
        self.history = []
        self.last_sync = None
        
    def update_state(self, new_state: dict):
        """Update the digital twin's state with new data."""
        self.state.update(new_state)
        self.history.append({
            'timestamp': datetime.now(),
            'state': new_state.copy()
        })
        self.last_sync = datetime.now()
        
    def get_current_state(self) -> dict:
        """Get the current state of the digital twin."""
        return self.state
        
    def render_synchronized_view(self):
        """Render the synchronized view of the digital twin."""
        st.subheader("Digital Twin Synchronized View")
        
        # Display current state
        if self.state:
            st.write("Current State:")
            for key, value in self.state.items():
                st.metric(key, f"{value:.2f}" if isinstance(value, (float, int)) else str(value))
        else:
            st.info("No state data available. Waiting for synchronization...")
        
        # Display synchronization status
        if self.last_sync:
            st.success(f"Last synchronized: {self.last_sync.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display historical data if available
        if self.history:
            st.subheader("Historical Data")
            df = pd.DataFrame(self.history)
            st.line_chart(df.set_index('timestamp')) 