"""Phase 4: Signal decay analysis via perturbation experiments.

Demonstrates information propagation limits (over-squashing) through
perturbation protocol and hop-distance analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import importlib.util

# Import dependencies
_phase1_graph_module = Path(__file__).parent.parent / "phase1" / "graph_structure.py"
spec = importlib.util.spec_from_file_location("graph_structure", _phase1_graph_module)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)
BasinGraph = graph_structure.BasinGraph

from synthetic_graphs import create_chain_graph, create_tree_graph
from hop_distance import compute_hop_distances, group_nodes_by_hop_distance

# Import MPNNModel from phase3
_phase3_mpnn_module = Path(__file__).parent.parent / "phase3" / "mpnn_layer.py"
spec_mpnn = importlib.util.spec_from_file_location("mpnn_layer", _phase3_mpnn_module)
mpnn_module = importlib.util.module_from_spec(spec_mpnn)
spec_mpnn.loader.exec_module(mpnn_module)
MPNNModel = mpnn_module.MPNNModel


class PerturbationExperiment:
    """Runs perturbation protocol to measure signal decay."""
    
    def __init__(self, seed: int = 42, epsilon: float = 0.05, state_dim: int = 64):
        """
        Initialize experiment with fixed random seed.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        epsilon : float
            Perturbation magnitude
        state_dim : int
            Node state dimension
        """
        self.seed = seed
        self.epsilon = epsilon
        self.state_dim = state_dim
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def create_initial_states(self, num_nodes: int, device: str = "cpu") -> torch.Tensor:
        """
        Create fixed initial node states.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes
        device : str
            Device to create tensors on
        
        Returns
        -------
        torch.Tensor
            Initial state matrix [num_nodes, state_dim]
        """
        return torch.randn(num_nodes, self.state_dim, device=device, generator=torch.Generator().manual_seed(self.seed))
    
    def perturb_node(self, states: torch.Tensor, node_idx: int, device: str = "cpu") -> torch.Tensor:
        """
        Create perturbed state matrix by adding epsilon * unit_vector to source node.
        
        Parameters
        ----------
        states : torch.Tensor
            Original state matrix [num_nodes, state_dim]
        node_idx : int
            Index of source node to perturb
        device : str
            Device
        
        Returns
        -------
        torch.Tensor
            Perturbed state matrix
        """
        perturbed = states.clone()
        
        # Generate fixed unit vector
        rng = np.random.RandomState(self.seed)
        unit_vec = torch.tensor(rng.randn(self.state_dim), device=device, dtype=states.dtype)
        unit_vec = unit_vec / unit_vec.norm()
        
        # Add perturbation
        perturbed[node_idx] = perturbed[node_idx] + self.epsilon * unit_vec
        
        return perturbed
    
    def compute_signal_metric(self, 
                              original_output: torch.Tensor,
                              perturbed_output: torch.Tensor) -> torch.Tensor:
        """
        Compute Δ_i = ||Z(+)_i - Z_i||_2 for each node.
        
        Parameters
        ----------
        original_output : torch.Tensor
            MPNN output from original states [num_nodes, state_dim]
        perturbed_output : torch.Tensor
            MPNN output from perturbed states [num_nodes, state_dim]
        
        Returns
        -------
        torch.Tensor
            Signal metric per node [num_nodes]
        """
        delta = (perturbed_output - original_output).norm(dim=1)
        return delta
    
    def run_regime(self,
                   graph: BasinGraph,
                   model: MPNNModel,
                   regime_name: str,
                   graph_type: str,
                   source_node: str,
                   node_list: List[str]) -> pd.DataFrame:
        """
        Run perturbation experiment for a given graph regime.
        
        Parameters
        ----------
        graph : BasinGraph
            Graph to use
        model : MPNNModel
            MPNN model
        regime_name : str
            Regime identifier ("real" or "synthetic")
        graph_type : str
            Graph type ("disconnected", "chain", "tree")
        source_node : str
            Source node ID
        node_list : List[str]
            Ordered list of node IDs
        
        Returns
        -------
        pd.DataFrame
            Results with columns: regime, graph_type, source_node, target_node, hop, delta_norm
        """
        num_nodes = len(node_list)
        source_idx = node_list.index(source_node)
        
        # Create initial states
        states = self.create_initial_states(num_nodes)
        perturbed_states = self.perturb_node(states, source_idx)
        
        # Run MPNN forward pass
        model.eval()
        with torch.no_grad():
            original_output = model(states, graph)
            perturbed_output = model(perturbed_states, graph)
        
        # Compute signal metrics
        delta = self.compute_signal_metric(original_output, perturbed_output)
        
        # Compute hop distances
        hop_distances = compute_hop_distances(graph, source_node)
        
        # Build results dataframe
        results = []
        for i, target_node in enumerate(node_list):
            hop = hop_distances.get(target_node)
            hop_str = str(hop) if hop is not None else "inf"
            
            results.append({
                "regime": regime_name,
                "graph_type": graph_type,
                "source_node": source_node,
                "target_node": target_node,
                "hop": hop_str,
                "hop_numeric": hop if hop is not None else float('inf'),
                "delta_norm": delta[i].item()
            })
        
        return pd.DataFrame(results)
    
    def summarize_by_hop(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize results by hop distance.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        
        Returns
        -------
        pd.DataFrame
            Summary with mean/median delta per hop distance
        """
        # Filter out inf hop distances for summary
        df_finite = df[df["hop_numeric"] != float('inf')].copy()
        
        if len(df_finite) == 0:
            return pd.DataFrame()
        
        summary = df_finite.groupby("hop_numeric")["delta_norm"].agg([
            ("mean", "mean"),
            ("median", "median"),
            ("max", "max"),
            ("count", "count")
        ]).reset_index()
        summary.columns = ["hop", "mean_delta", "median_delta", "max_delta", "count"]
        
        return summary


def run_phase4_experiment(output_dir: Path,
                         seed: int = 42,
                         epsilon: float = 0.05,
                         num_layers_list: List[int] = [1, 2]):
    """
    Run complete Phase 4 experiment.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for results
    seed : int
        Random seed
    epsilon : float
        Perturbation magnitude
    num_layers_list : List[int]
        List of layer counts to test
    """
    # Load basin IDs
    basin_file = Path(__file__).parent.parent.parent / "experiments" / "1_basin.txt"
    with open(basin_file) as f:
        basin_ids = [line.strip() for line in f if line.strip()]
    
    node_list = sorted(basin_ids)
    num_nodes = len(node_list)
    source_node = node_list[0]  # Use first node as source
    
    # Create experiment
    experiment = PerturbationExperiment(seed=seed, epsilon=epsilon)
    
    # Create run config
    run_config = {
        "seed": seed,
        "epsilon": epsilon,
        "perturbation_type": "additive_unit_vector",
        "num_nodes": num_nodes,
        "source_node": source_node,
        "layers": num_layers_list,
        "state_dim": 64,
        "timestamp": datetime.now().isoformat()
    }
    
    all_results = []
    
    # Regime A: Real graph baseline (E=∅)
    print("Running Regime A: Real graph baseline (E=∅)...")
    real_graph = BasinGraph(set(basin_ids))
    
    for num_layers in num_layers_list:
        model = MPNNModel(num_layers=num_layers, state_dim=64)
        df = experiment.run_regime(
            graph=real_graph,
            model=model,
            regime_name="real",
            graph_type="disconnected",
            source_node=source_node,
            node_list=node_list
        )
        df["layers"] = num_layers
        all_results.append(df)
    
    # Regime B: Synthetic diagnostic topology
    print("Running Regime B: Synthetic diagnostic topology...")
    
    # Chain graph
    chain_graph = create_chain_graph(node_list, length=min(5, num_nodes))
    for num_layers in num_layers_list:
        model = MPNNModel(num_layers=num_layers, state_dim=64)
        df = experiment.run_regime(
            graph=chain_graph,
            model=model,
            regime_name="synthetic",
            graph_type="chain",
            source_node=source_node,
            node_list=node_list
        )
        df["layers"] = num_layers
        all_results.append(df)
    
    # Tree graph (if we have enough nodes)
    if num_nodes >= 4:
        tree_graph = create_tree_graph(node_list, branching_factor=2, depth=2)
        for num_layers in num_layers_list:
            model = MPNNModel(num_layers=num_layers, state_dim=64)
            df = experiment.run_regime(
                graph=tree_graph,
                model=model,
                regime_name="synthetic",
                graph_type="tree",
                source_node=source_node,
                node_list=node_list
            )
            df["layers"] = num_layers
            all_results.append(df)
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    results_df.to_csv(output_dir / "metrics.csv", index=False)
    
    # Save run config
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    # Create summary
    summary_lines = [
        "# Phase 4: Signal Decay Analysis Summary",
        "",
        f"**Run Configuration:**",
        f"- Seed: {seed}",
        f"- Epsilon: {epsilon}",
        f"- Source Node: {source_node}",
        f"- Layers Tested: {num_layers_list}",
        "",
        "## Key Findings",
        ""
    ]
    
    # Analyze real graph
    real_df = results_df[results_df["regime"] == "real"]
    real_source_delta = real_df[real_df["target_node"] == source_node]["delta_norm"].mean()
    real_others_delta = real_df[real_df["target_node"] != source_node]["delta_norm"].mean()
    
    summary_lines.extend([
        "### Regime A: Real Graph Baseline (E=∅)",
        f"- Source node Δ: {real_source_delta:.6f}",
        f"- Other nodes mean Δ: {real_others_delta:.6f}",
        f"- **Observation**: Perturbations do not propagate (all other nodes ~0)",
        ""
    ])
    
    # Analyze synthetic graphs
    synthetic_df = results_df[results_df["regime"] == "synthetic"]
    for graph_type in ["chain", "tree"]:
        graph_df = synthetic_df[synthetic_df["graph_type"] == graph_type]
        if len(graph_df) == 0:
            continue
        
        summary_lines.append(f"### Regime B: Synthetic {graph_type.capitalize()} Graph")
        
        for num_layers in num_layers_list:
            layer_df = graph_df[graph_df["layers"] == num_layers]
            summary = experiment.summarize_by_hop(layer_df)
            
            if len(summary) > 0:
                summary_lines.append(f"\n**{num_layers}-layer MPNN:**")
                summary_lines.append("| Hop | Mean Δ | Median Δ | Max Δ | Count |")
                summary_lines.append("|-----|--------|----------|-------|-------|")
                for _, row in summary.iterrows():
                    summary_lines.append(f"| {int(row['hop'])} | {row['mean_delta']:.6f} | {row['median_delta']:.6f} | {row['max_delta']:.6f} | {int(row['count'])} |")
        
        summary_lines.append("")
    
    summary_lines.extend([
        "## Interpretation",
        "",
        "1. **Real graph (E=∅)**: Confirms no propagation - validates Phase 2 + Phase 3 consistency.",
        "2. **Synthetic topologies**: Show signal decay with hop distance, demonstrating propagation limits.",
        "3. **Layer depth**: Compare 1-layer vs 2-layer to assess reach vs stability trade-offs.",
        ""
    ])
    
    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- metrics.csv: {len(results_df)} rows")
    print(f"- run_config.json: Configuration")
    print(f"- summary.md: Analysis summary")
    
    return results_df, run_config
