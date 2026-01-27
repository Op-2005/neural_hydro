"""Synthetic graph construction for diagnostic analysis.

Creates chain and tree graphs using existing basin node IDs.
These are diagnostic scaffolds only, not real topology.
"""

from typing import Dict, Set, List
import importlib.util
from pathlib import Path

# Import BasinGraph from phase1
_phase1_graph_module = Path(__file__).parent.parent / "phase1" / "graph_structure.py"
spec = importlib.util.spec_from_file_location("graph_structure", _phase1_graph_module)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)
BasinGraph = graph_structure.BasinGraph


def create_chain_graph(node_ids: List[str], length: int = None) -> BasinGraph:
    """
    Create a chain graph: v0 → v1 → ... → vL.
    
    Parameters
    ----------
    node_ids : List[str]
        List of basin IDs to use
    length : int, optional
        Chain length (default: use all nodes)
    
    Returns
    -------
    BasinGraph
        Chain graph with synthetic diagnostic edges
    """
    if length is None:
        length = len(node_ids)
    else:
        length = min(length, len(node_ids))
    
    edges: Dict[str, Set[str]] = {}
    for i in range(length - 1):
        parent = node_ids[i]
        child = node_ids[i + 1]
        if parent not in edges:
            edges[parent] = set()
        edges[parent].add(child)
    
    return BasinGraph(set(node_ids), edges)


def create_tree_graph(node_ids: List[str], branching_factor: int = 2, depth: int = 2) -> BasinGraph:
    """
    Create a small tree graph with specified branching factor and depth.
    
    Parameters
    ----------
    node_ids : List[str]
        List of basin IDs to use
    branching_factor : int
        Number of children per node
    depth : int
        Maximum depth of tree
    
    Returns
    -------
    BasinGraph
        Tree graph with synthetic diagnostic edges
    """
    if len(node_ids) < 2:
        return BasinGraph(set(node_ids))
    
    edges: Dict[str, Set[str]] = {}
    node_idx = 0
    
    def add_children(parent: str, current_depth: int):
        """Recursively add children to parent node."""
        nonlocal node_idx
        if current_depth >= depth or node_idx >= len(node_ids) - 1:
            return
        
        for _ in range(branching_factor):
            if node_idx >= len(node_ids) - 1:
                break
            node_idx += 1
            child = node_ids[node_idx]
            if parent not in edges:
                edges[parent] = set()
            edges[parent].add(child)
            add_children(child, current_depth + 1)
    
    # Start from root
    root = node_ids[0]
    add_children(root, 0)
    
    return BasinGraph(set(node_ids), edges)
