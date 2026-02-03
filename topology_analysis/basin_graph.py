"""Minimal graph structure for Phase 1 experimental object.

Represents the directed basin graph with node states and time indexing.
"""

from typing import Dict, Set, List, Optional
from pathlib import Path
import numpy as np


class BasinGraph:
    """Directed graph representing basin topology.
    
    Nodes are basin IDs, edges represent parent -> child (upstream -> downstream)
    relationships. Node states are time-indexed feature vectors.
    """
    
    def __init__(self, node_set: Set[str], edge_set: Optional[Dict[str, Set[str]]] = None):
        """
        Initialize basin graph.
        
        Parameters
        ----------
        node_set : Set[str]
            Set of basin IDs (USGS gauge IDs)
        edge_set : Dict[str, Set[str]], optional
            Mapping from parent basin ID to set of child basin IDs.
            If None, creates empty edge set.
        """
        self.nodes = node_set
        self.edges = edge_set if edge_set is not None else {}
        
        # Ensure all nodes are present (even if no edges)
        for node in self.nodes:
            if node not in self.edges:
                self.edges[node] = set()
    
    @classmethod
    def from_topology_file(cls, topology_file: Path) -> "BasinGraph":
        """Load graph from edge list file."""
        nodes = set()
        edges: Dict[str, Set[str]] = {}
        
        with open(topology_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    parent, child = parts[0], parts[1]
                    nodes.add(parent)
                    nodes.add(child)
                    if parent not in edges:
                        edges[parent] = set()
                    edges[parent].add(child)
        
        return cls(nodes, edges)
    
    def get_children(self, node: str) -> Set[str]:
        """Get set of child nodes (downstream basins)."""
        return self.edges.get(node, set())
    
    def get_parents(self, node: str) -> Set[str]:
        """Get set of parent nodes (upstream basins)."""
        parents = set()
        for parent, children in self.edges.items():
            if node in children:
                parents.add(parent)
        return parents
    
    def is_leaf(self, node: str) -> bool:
        """Check if node has no children."""
        return len(self.get_children(node)) == 0
    
    def is_root(self, node: str) -> bool:
        """Check if node has no parents."""
        return len(self.get_parents(node)) == 0
    
    def num_edges(self) -> int:
        """Total number of edges."""
        return sum(len(children) for children in self.edges.values())
    
    def __repr__(self) -> str:
        return f"BasinGraph(nodes={len(self.nodes)}, edges={self.num_edges()})"


class NodeStateMatrix:
    """Time-indexed node state matrix.
    
    Stores state vectors for all nodes at each time step.
    """
    
    def __init__(self, num_nodes: int, state_dim: int, num_timesteps: int):
        """
        Initialize state matrix.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes (basins)
        state_dim : int
            Dimension of state vector per node
        num_timesteps : int
            Number of time steps
        """
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.num_timesteps = num_timesteps
        # Shape: [num_timesteps, num_nodes, state_dim]
        self.states = np.zeros((num_timesteps, num_nodes, state_dim))
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
    
    def set_node_mapping(self, node_list: List[str]):
        """Set mapping between node IDs and matrix indices."""
        self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
    
    def set_state(self, t: int, node: str, state: np.ndarray):
        """Set state vector for node at time t."""
        if node not in self.node_to_idx:
            raise ValueError(f"Node {node} not in mapping")
        idx = self.node_to_idx[node]
        self.states[t, idx, :] = state
    
    def get_state(self, t: int, node: str) -> np.ndarray:
        """Get state vector for node at time t."""
        idx = self.node_to_idx[node]
        return self.states[t, idx, :].copy()
    
    def get_all_states_at_time(self, t: int) -> np.ndarray:
        """Get state matrix for all nodes at time t. Shape: [num_nodes, state_dim]"""
        return self.states[t, :, :].copy()
