"""Phase 2: Graph structure and bottleneck characterization.

Analyzes the basin graph to identify information-flow bottlenecks
independent of any neural model.
"""

from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict, deque
import importlib.util

# Import BasinGraph from phase1
_phase1_graph_module = Path(__file__).parent.parent / "phase1" / "graph_structure.py"
spec = importlib.util.spec_from_file_location("graph_structure", _phase1_graph_module)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)
BasinGraph = graph_structure.BasinGraph


class GraphAnalyzer:
    """Analyzes basin graph structure for bottleneck identification."""
    
    def __init__(self, graph: BasinGraph):
        """Initialize analyzer with a basin graph."""
        self.graph = graph
        self.components: Optional[List[Set[str]]] = None
        self.component_labels: Optional[Dict[str, int]] = None
    
    def compute_weakly_connected_components(self) -> Tuple[List[Set[str]], Dict[str, int]]:
        """
        Compute weakly connected components (ignoring edge direction).
        
        Returns
        -------
        components : List[Set[str]]
            List of component sets, each containing basin IDs in that component
        component_labels : Dict[str, int]
            Mapping from basin ID to component index
        """
        visited = set()
        components = []
        component_labels = {}
        
        def bfs_component(start_node: str, component_id: int) -> Set[str]:
            """BFS to find all nodes reachable from start (undirected)."""
            component = set()
            queue = deque([start_node])
            visited.add(start_node)
            
            while queue:
                node = queue.popleft()
                component.add(node)
                component_labels[node] = component_id
                
                # Add neighbors (both parents and children)
                neighbors = self.graph.get_children(node) | self.graph.get_parents(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return component
        
        component_id = 0
        for node in self.graph.nodes:
            if node not in visited:
                component = bfs_component(node, component_id)
                components.append(component)
                component_id += 1
        
        self.components = components
        self.component_labels = component_labels
        return components, component_labels
    
    def compute_path_metrics(self, component: Set[str]) -> Dict[str, any]:
        """
        Compute path metrics for a connected component.
        
        Returns
        -------
        metrics : dict
            - max_path_length: longest upstream -> downstream chain
            - avg_path_length: average path length (if computable)
            - roots: set of root nodes (no parents)
            - leaves: set of leaf nodes (no children)
        """
        if len(component) == 1:
            # Single node component
            node = list(component)[0]
            return {
                "max_path_length": 0,
                "avg_path_length": 0.0,
                "roots": {node},
                "leaves": {node},
                "num_paths": 0
            }
        
        # Find roots (nodes with no parents in this component)
        roots = {node for node in component if self.graph.is_root(node)}
        
        # Find leaves (nodes with no children in this component)
        leaves = {node for node in component if self.graph.is_leaf(node)}
        
        # Compute longest path from any root to any leaf
        max_path_length = 0
        all_path_lengths = []
        
        def dfs_longest_path(node: str, visited: Set[str], path_length: int) -> int:
            """DFS to find longest path from node to any leaf."""
            if self.graph.is_leaf(node):
                return path_length
            
            max_from_here = path_length
            for child in self.graph.get_children(node):
                if child in component:  # Only consider nodes in this component
                    max_from_here = max(max_from_here, 
                                       dfs_longest_path(child, visited, path_length + 1))
            return max_from_here
        
        # Find longest path from each root
        for root in roots:
            path_len = dfs_longest_path(root, set(), 0)
            max_path_length = max(max_path_length, path_len)
            if path_len > 0:
                all_path_lengths.append(path_len)
        
        # Compute all paths for average (BFS from roots to leaves)
        all_paths = []
        for root in roots:
            queue = deque([(root, 0)])  # (node, path_length)
            visited_paths = set()
            
            while queue:
                node, path_len = queue.popleft()
                
                if self.graph.is_leaf(node):
                    all_paths.append(path_len)
                    continue
                
                for child in self.graph.get_children(node):
                    if child in component:
                        state = (child, path_len + 1)
                        if state not in visited_paths:
                            visited_paths.add(state)
                            queue.append(state)
        
        avg_path_length = sum(all_paths) / len(all_paths) if all_paths else 0.0
        
        return {
            "max_path_length": max_path_length,
            "avg_path_length": avg_path_length,
            "roots": roots,
            "leaves": leaves,
            "num_paths": len(all_paths)
        }
    
    def compute_node_degrees(self) -> Dict[str, Dict[str, int]]:
        """
        Compute in-degree and out-degree for each node.
        
        Returns
        -------
        degrees : Dict[str, Dict[str, int]]
            Mapping from node ID to {"in_degree": int, "out_degree": int}
        """
        degrees = {}
        for node in self.graph.nodes:
            degrees[node] = {
                "in_degree": len(self.graph.get_parents(node)),
                "out_degree": len(self.graph.get_children(node))
            }
        return degrees
    
    def compute_path_betweenness(self, component: Set[str]) -> Dict[str, int]:
        """
        Compute qualitative betweenness: number of shortest paths passing through each node.
        
        This is a simplified version that counts paths from roots to leaves.
        
        Returns
        -------
        betweenness : Dict[str, int]
            Mapping from node ID to path count
        """
        if len(component) <= 1:
            return {node: 0 for node in component}
        
        roots = {node for node in component if self.graph.is_root(node)}
        leaves = {node for node in component if self.graph.is_leaf(node)}
        
        betweenness = {node: 0 for node in component}
        
        # Find all paths from roots to leaves
        def find_paths(node: str, target: str, visited: Set[str], path: List[str]) -> List[List[str]]:
            """Find all paths from node to target."""
            if node == target:
                return [path + [node]]
            
            paths = []
            visited.add(node)
            
            for child in self.graph.get_children(node):
                if child in component and child not in visited:
                    paths.extend(find_paths(child, target, visited.copy(), path + [node]))
            
            return paths
        
        # Count paths through each node
        for root in roots:
            for leaf in leaves:
                paths = find_paths(root, leaf, set(), [])
                for path in paths:
                    for node in path:
                        if node != root and node != leaf:  # Exclude endpoints
                            betweenness[node] += 1
        
        return betweenness
    
    def identify_bottleneck_candidates(self, 
                                     min_in_degree: int = 2,
                                     min_out_degree: int = 2,
                                     min_betweenness: int = 1) -> List[Tuple[str, Dict[str, any]]]:
        """
        Identify candidate bottleneck nodes.
        
        Parameters
        ----------
        min_in_degree : int
            Minimum in-degree to be considered a bottleneck
        min_out_degree : int
            Minimum out-degree to be considered a bottleneck
        min_betweenness : int
            Minimum betweenness count to be considered a bottleneck
        
        Returns
        -------
        bottlenecks : List[Tuple[str, Dict]]
            List of (node_id, metrics_dict) for bottleneck candidates
        """
        degrees = self.compute_node_degrees()
        bottlenecks = []
        
        # Analyze each component separately
        if self.components is None:
            self.compute_weakly_connected_components()
        
        for component in self.components:
            betweenness = self.compute_path_betweenness(component)
            
            for node in component:
                node_deg = degrees[node]
                node_bet = betweenness.get(node, 0)
                
                # Bottleneck criteria: high in-degree OR high out-degree OR high betweenness
                is_bottleneck = (
                    node_deg["in_degree"] >= min_in_degree or
                    node_deg["out_degree"] >= min_out_degree or
                    node_bet >= min_betweenness
                )
                
                if is_bottleneck:
                    bottlenecks.append((
                        node,
                        {
                            "in_degree": node_deg["in_degree"],
                            "out_degree": node_deg["out_degree"],
                            "betweenness": node_bet,
                            "component": self.component_labels[node]
                        }
                    ))
        
        return sorted(bottlenecks, key=lambda x: x[1]["betweenness"] + x[1]["in_degree"] + x[1]["out_degree"], reverse=True)
    
    def analyze(self) -> Dict[str, any]:
        """
        Perform complete graph analysis.
        
        Returns
        -------
        analysis : dict
            Complete analysis results
        """
        # Step 1: Compute connected components
        components, labels = self.compute_weakly_connected_components()
        
        # Step 2: Analyze each component
        component_metrics = []
        overall_max_depth = 0
        
        for comp_id, component in enumerate(components):
            metrics = self.compute_path_metrics(component)
            metrics["component_id"] = comp_id
            metrics["size"] = len(component)
            component_metrics.append(metrics)
            overall_max_depth = max(overall_max_depth, metrics["max_path_length"])
        
        # Step 3: Compute node degrees
        degrees = self.compute_node_degrees()
        
        # Step 4: Identify bottlenecks
        bottlenecks = self.identify_bottleneck_candidates()
        
        return {
            "num_nodes": len(self.graph.nodes),
            "num_edges": self.graph.num_edges(),
            "num_components": len(components),
            "components": component_metrics,
            "overall_max_depth": overall_max_depth,
            "degrees": degrees,
            "bottlenecks": bottlenecks
        }


def main():
    """Run Phase 2 graph analysis."""
    # Load basin IDs
    basin_file = Path(__file__).parent.parent.parent / "experiments" / "1_basin.txt"
    with open(basin_file) as f:
        basin_ids = {line.strip() for line in f if line.strip()}
    
    # Load graph (try from topology file, fallback to empty)
    topology_file = Path(__file__).parent.parent / "phase1" / "basin_topology.txt"
    if topology_file.exists():
        graph = BasinGraph.from_topology_file(topology_file)
        # Ensure all basins are in the graph
        graph.nodes.update(basin_ids)
        for node in basin_ids:
            if node not in graph.edges:
                graph.edges[node] = set()
    else:
        graph = BasinGraph(basin_ids)
    
    # Analyze
    analyzer = GraphAnalyzer(graph)
    results = analyzer.analyze()
    
    # Print summary
    print("=" * 60)
    print("Phase 2: Graph Structure & Bottleneck Characterization")
    print("=" * 60)
    print(f"\nGraph Summary:")
    print(f"  Nodes: {results['num_nodes']}")
    print(f"  Edges: {results['num_edges']}")
    print(f"  Connected Components: {results['num_components']}")
    print(f"  Maximum Depth: {results['overall_max_depth']}")
    
    print(f"\nComponent Details:")
    for comp in results['components']:
        print(f"  Component {comp['component_id']}:")
        print(f"    Size: {comp['size']} nodes")
        print(f"    Max Path Length: {comp['max_path_length']}")
        print(f"    Avg Path Length: {comp['avg_path_length']:.2f}")
        print(f"    Roots: {sorted(comp['roots'])}")
        print(f"    Leaves: {sorted(comp['leaves'])}")
    
    print(f"\nBottleneck Candidates:")
    if results['bottlenecks']:
        for node, metrics in results['bottlenecks']:
            print(f"  {node}:")
            print(f"    In-degree: {metrics['in_degree']}")
            print(f"    Out-degree: {metrics['out_degree']}")
            print(f"    Betweenness: {metrics['betweenness']}")
            print(f"    Component: {metrics['component']}")
    else:
        print("  None identified (graph may be fully disconnected or have low connectivity)")
    
    # Save summary to file
    output_file = Path(__file__).parent / "graph_analysis_summary.txt"
    with open(output_file, "w") as f:
        f.write("Phase 2: Graph Structure & Bottleneck Characterization\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Graph Summary:\n")
        f.write(f"  Nodes: {results['num_nodes']}\n")
        f.write(f"  Edges: {results['num_edges']}\n")
        f.write(f"  Connected Components: {results['num_components']}\n")
        f.write(f"  Maximum Depth: {results['overall_max_depth']}\n\n")
        
        f.write("Component Details:\n")
        for comp in results['components']:
            f.write(f"  Component {comp['component_id']}:\n")
            f.write(f"    Size: {comp['size']} nodes\n")
            f.write(f"    Max Path Length: {comp['max_path_length']}\n")
            f.write(f"    Avg Path Length: {comp['avg_path_length']:.2f}\n")
            f.write(f"    Roots: {sorted(comp['roots'])}\n")
            f.write(f"    Leaves: {sorted(comp['leaves'])}\n\n")
        
        f.write("Bottleneck Candidates:\n")
        if results['bottlenecks']:
            for node, metrics in results['bottlenecks']:
                f.write(f"  {node}:\n")
                f.write(f"    In-degree: {metrics['in_degree']}\n")
                f.write(f"    Out-degree: {metrics['out_degree']}\n")
                f.write(f"    Betweenness: {metrics['betweenness']}\n")
                f.write(f"    Component: {metrics['component']}\n\n")
        else:
            f.write("  None identified\n")
    
    print(f"\nAnalysis saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
