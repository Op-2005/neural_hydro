"""Phase 3: Minimal Invariant MPNN Implementation.

Implements a message passing neural network layer following:
- Message: m_ij = φ_m(h_i, h_j)
- Aggregate: sum over neighbors
- Update: h'_i = φ_h(h_i, m_i)

Handles isolated nodes (empty neighborhoods) by returning zero vectors.
"""

import torch
import torch.nn as nn
from typing import Dict, Set
import importlib.util
from pathlib import Path

# Import BasinGraph from phase1
_phase1_graph_module = Path(__file__).parent.parent / "phase1" / "graph_structure.py"
spec = importlib.util.spec_from_file_location("graph_structure", _phase1_graph_module)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)
BasinGraph = graph_structure.BasinGraph


class MPNNLayer(nn.Module):
    """Minimal invariant Message Passing Neural Network layer.
    
    For each node i:
    1. Compute messages from neighbors: m_ij = φ_m(h_i, h_j) for j ∈ N(i)
    2. Aggregate messages: m_i = Σ_{j∈N(i)} m_ij
    3. Update node state: h'_i = φ_h(h_i, m_i)
    
    Handles isolated nodes (N(i) = ∅) by setting m_i = 0.
    """
    
    def __init__(self, state_dim: int = 64, message_dim: int = 64, hidden_dim: int = 64):
        """
        Initialize MPNN layer.
        
        Parameters
        ----------
        state_dim : int
            Dimension of node state vectors (default: 64)
        message_dim : int
            Dimension of message vectors (default: 64)
        hidden_dim : int
            Hidden dimension for MLPs (default: 64)
        """
        super().__init__()
        self.state_dim = state_dim
        self.message_dim = message_dim
        
        # Message function: φ_m(h_i, h_j) -> m_ij
        # Takes concatenated [h_i, h_j] and outputs message
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Update function: φ_h(h_i, m_i) -> h'_i
        # Takes concatenated [h_i, m_i] and outputs updated state
        self.update_mlp = nn.Sequential(
            nn.Linear(state_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, node_states: torch.Tensor, graph: BasinGraph) -> torch.Tensor:
        """
        Perform message passing update.
        
        Parameters
        ----------
        node_states : torch.Tensor
            Node state matrix of shape [num_nodes, state_dim]
        graph : BasinGraph
            Basin graph containing edge structure
        
        Returns
        -------
        updated_states : torch.Tensor
            Updated node states of shape [num_nodes, state_dim]
        """
        num_nodes, state_dim = node_states.shape
        device = node_states.device
        
        # Create node-to-index mapping
        node_list = sorted(graph.nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Initialize updated states
        updated_states = torch.zeros_like(node_states)
        
        # Process each node
        for node in node_list:
            node_idx = node_to_idx[node]
            h_i = node_states[node_idx]  # Current state of node i
            
            # Get neighbors (parents in directed graph: upstream -> downstream)
            # For MPNN, we consider incoming edges (nodes that send messages to this node)
            neighbors = graph.get_parents(node)  # Upstream basins
            
            if len(neighbors) == 0:
                # Isolated node: no neighbors, so m_i = 0
                m_i = torch.zeros(self.message_dim, device=device)
            else:
                # Aggregate messages from neighbors
                messages = []
                for neighbor in neighbors:
                    neighbor_idx = node_to_idx[neighbor]
                    h_j = node_states[neighbor_idx]  # State of neighbor j
                    
                    # Compute message: m_ij = φ_m(h_i, h_j)
                    h_concat = torch.cat([h_i, h_j], dim=-1)
                    m_ij = self.message_mlp(h_concat)
                    messages.append(m_ij)
                
                # Aggregate: m_i = Σ_{j∈N(i)} m_ij
                m_i = torch.stack(messages, dim=0).sum(dim=0)
            
            # Update: h'_i = φ_h(h_i, m_i)
            h_m_concat = torch.cat([h_i, m_i], dim=-1)
            h_i_updated = self.update_mlp(h_m_concat)
            
            updated_states[node_idx] = h_i_updated
        
        return updated_states


class MPNNModel(nn.Module):
    """MPNN model with 1-2 message passing layers."""
    
    def __init__(self, num_layers: int = 1, state_dim: int = 64, message_dim: int = 64, hidden_dim: int = 64):
        """
        Initialize MPNN model.
        
        Parameters
        ----------
        num_layers : int
            Number of MPNN layers (1 or 2)
        state_dim : int
            Dimension of node state vectors
        message_dim : int
            Dimension of message vectors
        hidden_dim : int
            Hidden dimension for MLPs
        """
        super().__init__()
        assert num_layers in [1, 2], "num_layers must be 1 or 2"
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            MPNNLayer(state_dim, message_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, node_states: torch.Tensor, graph: BasinGraph) -> torch.Tensor:
        """
        Apply message passing layers sequentially.
        
        Parameters
        ----------
        node_states : torch.Tensor
            Initial node states [num_nodes, state_dim]
        graph : BasinGraph
            Basin graph
        
        Returns
        -------
        updated_states : torch.Tensor
            Final node states after message passing
        """
        x = node_states
        for layer in self.layers:
            x = layer(x, graph)
        return x
