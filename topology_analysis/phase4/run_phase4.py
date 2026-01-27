"""Phase 4: Main execution script for signal decay analysis."""

from pathlib import Path
from datetime import datetime
from signal_decay import run_phase4_experiment
import matplotlib.pyplot as plt
import pandas as pd


def create_plot(results_df: pd.DataFrame, output_dir: Path):
    """Create hop distance vs mean delta plot."""
    synthetic_df = results_df[results_df["regime"] == "synthetic"]
    
    if len(synthetic_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for graph_type in ["chain", "tree"]:
        graph_df = synthetic_df[synthetic_df["graph_type"] == graph_type]
        if len(graph_df) == 0:
            continue
        
        ax = axes[0] if graph_type == "chain" else axes[1]
        
        for num_layers in [1, 2]:
            layer_df = graph_df[graph_df["layers"] == num_layers]
            layer_df = layer_df[layer_df["hop_numeric"] != float('inf')]
            
            if len(layer_df) == 0:
                continue
            
            summary = layer_df.groupby("hop_numeric")["delta_norm"].mean().reset_index()
            summary.columns = ["hop", "mean_delta"]
            
            ax.plot(summary["hop"], summary["mean_delta"], 
                   marker='o', label=f"{num_layers}-layer", linewidth=2)
        
        ax.set_xlabel("Hop Distance")
        ax.set_ylabel("Mean Δ")
        ax.set_title(f"{graph_type.capitalize()} Graph")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"- plot.png: Signal decay visualization")


def main():
    """Run Phase 4 experiment."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "outputs" / f"{timestamp}_phase4"
    
    print("=" * 60)
    print("Phase 4: Signal Decay Analysis")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # Run experiment
    results_df, run_config = run_phase4_experiment(
        output_dir=output_dir,
        seed=42,
        epsilon=0.05,
        num_layers_list=[1, 2]
    )
    
    # Create plot
    try:
        create_plot(results_df, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 4 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
