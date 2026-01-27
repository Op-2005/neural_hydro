"""Infer parent-child basin relationships from CAMELS attributes.

Uses heuristics based on catchment area, elevation, and spatial proximity
to construct a directed graph of upstream -> downstream relationships.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple

def load_basin_attributes(basin_ids: list, data_dir: Path) -> pd.DataFrame:
    """Load topographic attributes for specified basins."""
    topo_file = data_dir / "camels_attributes_v2.0" / "camels_topo.txt"
    df = pd.read_csv(topo_file, sep=";")
    df = df[df["gauge_id"].isin(basin_ids)].copy()
    return df.set_index("gauge_id")


def compute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute approximate distance in km using Haversine formula."""
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def infer_parent_child_relationships(df: pd.DataFrame, 
                                    max_distance_km: float = 50.0,
                                    area_ratio_threshold: float = 1.1) -> Dict[str, Set[str]]:
    """
    Infer parent -> child relationships using heuristics:
    - Child has larger or equal catchment area (accumulation)
    - Child is at lower elevation (rivers flow downhill)
    - Basins are spatially close (within max_distance_km)
    
    Returns dict mapping parent_id -> set of child_ids
    """
    parent_to_children: Dict[str, Set[str]] = {basin: set() for basin in df.index}
    
    for parent_id in df.index:
        parent_row = df.loc[parent_id]
        parent_area = parent_row["area_gages2"]
        parent_elev = parent_row["elev_mean"]
        parent_lat = parent_row["gauge_lat"]
        parent_lon = parent_row["gauge_lon"]
        
        for child_id in df.index:
            if parent_id == child_id:
                continue
                
            child_row = df.loc[child_id]
            child_area = child_row["area_gages2"]
            child_elev = child_row["elev_mean"]
            child_lat = child_row["gauge_lat"]
            child_lon = child_row["gauge_lon"]
            
            # Heuristic 1: Child area >= parent area (with threshold)
            if child_area < parent_area * area_ratio_threshold:
                continue
            
            # Heuristic 2: Child elevation < parent elevation
            if child_elev >= parent_elev:
                continue
            
            # Heuristic 3: Spatial proximity
            distance = compute_distance(parent_lat, parent_lon, child_lat, child_lon)
            if distance > max_distance_km:
                continue
            
            parent_to_children[parent_id].add(child_id)
    
    # Remove empty entries
    return {k: v for k, v in parent_to_children.items() if v}


def main():
    """Infer topology for Maine/NH basins."""
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "camels_us"
    basin_file = Path(__file__).parent.parent.parent / "experiments" / "1_basin.txt"
    
    # Load basin IDs
    with open(basin_file) as f:
        basin_ids = [line.strip() for line in f if line.strip()]
    
    # Load attributes
    df = load_basin_attributes(basin_ids, data_dir)
    
    # Infer relationships
    parent_to_children = infer_parent_child_relationships(df)
    
    # Print results
    print(f"Inferred topology for {len(basin_ids)} basins:")
    print(f"Found {sum(len(children) for children in parent_to_children.values())} parent->child edges")
    print("\nParent -> Children relationships:")
    for parent, children in sorted(parent_to_children.items()):
        print(f"  {parent} -> {sorted(children)}")
    
    # Save to file
    output_file = Path(__file__).parent / "basin_topology.txt"
    with open(output_file, "w") as f:
        f.write("# Parent -> Child relationships (upstream -> downstream)\n")
        for parent, children in sorted(parent_to_children.items()):
            for child in sorted(children):
                f.write(f"{parent} {child}\n")
    
    print(f"\nSaved topology to {output_file}")


if __name__ == "__main__":
    main()
