"""Example usage of Phase 1 experimental object."""

from pathlib import Path
from graph_structure import BasinGraph, NodeStateMatrix
import numpy as np

# Load basin IDs
basin_file = Path(__file__).parent.parent.parent / "experiments" / "1_basin.txt"
with open(basin_file) as f:
    basin_ids = {line.strip() for line in f if line.strip()}

# Create graph (empty edge set for now)
graph = BasinGraph(basin_ids)
print(f"Created graph: {graph}")
print(f"Nodes: {sorted(graph.nodes)}")
print(f"Number of edges: {graph.num_edges()}")

# Example: Check properties of a node
example_node = list(graph.nodes)[0]
print(f"\nExample node: {example_node}")
print(f"  Is root: {graph.is_root(example_node)}")
print(f"  Is leaf: {graph.is_leaf(example_node)}")
print(f"  Children: {graph.get_children(example_node)}")
print(f"  Parents: {graph.get_parents(example_node)}")

# Create state matrix
num_nodes = len(graph.nodes)
state_dim = 64  # Matches LSTM hidden size
num_timesteps = 100  # Example: 100 days

state_matrix = NodeStateMatrix(num_nodes, state_dim, num_timesteps)
state_matrix.set_node_mapping(sorted(graph.nodes))

# Example: Set and retrieve states
example_node = sorted(graph.nodes)[0]
example_state = np.random.randn(state_dim)  # Random state for demonstration
state_matrix.set_state(t=0, node=example_node, state=example_state)

retrieved_state = state_matrix.get_state(t=0, node=example_node)
print(f"\nState matrix created:")
print(f"  Shape: {state_matrix.states.shape}")
print(f"  Retrieved state shape: {retrieved_state.shape}")
print(f"  States match: {np.allclose(example_state, retrieved_state)}")

# Get all states at a time step
all_states_t0 = state_matrix.get_all_states_at_time(t=0)
print(f"  All states at t=0 shape: {all_states_t0.shape}")  # [num_nodes, state_dim]
