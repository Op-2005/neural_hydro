"""Hop distance computation for directed graphs.

Computes shortest directed path lengths from source node to all other nodes.
"""

from typing import Dict, Set, Optional
from collections import deque
import importlib.util
from pathlib import Path

# Import BasinGraph from phase1
_phase1_graph_module = Path(__file__).parent.parent / "phase1" / "graph_structure.py"
spec = importlib.util.spec_from_file_location("graph_structure", _phase1_graph_module)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)
BasinGraph = graph_structure.BasinGraph


def compute_hop_distances(graph: BasinGraph, source: str) -> Dict[str, Optional[int]]:
    """
    Compute hop distances from source node to all nodes using BFS.
    
    Parameters
    ----------
    graph : BasinGraph
        Directed graph
    source : str
        Source node ID
    
    Returns
    -------
    Dict[str, Optional[int]]
        Mapping from node ID to hop distance (None if unreachable)
    """
    if source not in graph.nodes:
        raise ValueError(f"Source node {source} not in graph")
    
    distances: Dict[str, Optional[int]] = {node: None for node in graph.nodes}
    distances[source] = 0
    
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        # Traverse outgoing edges (children)
        for child in graph.get_children(current):
            if distances[child] is None:
                distances[child] = current_dist + 1
                queue.append(child)
    
    return distances


def group_nodes_by_hop_distance(distances: Dict[str, Optional[int]]) -> Dict[int, Set[str]]:
    """
    Group nodes by their hop distance from source.
    
    Parameters
    ----------
    distances : Dict[str, Optional[int]]
        Node to hop distance mapping
    
    Returns
    -------
    Dict[int, Set[str]]
        Mapping from hop distance to set of node IDs
    """
    groups: Dict[int, Set[str]] = {}
    
    for node, dist in distances.items():
        if dist is None:
            # Unreachable nodes (infinite distance)
            continue
        if dist not in groups:
            groups[dist] = set()
        groups[dist].add(node)
    
    return groups
