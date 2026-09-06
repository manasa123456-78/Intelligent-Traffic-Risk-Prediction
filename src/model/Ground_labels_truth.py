import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from global_variables_02 import CFG

def generate_ground_truth():
    print("=" * 70)
    print("PHASE 4: Generating Continuous Future Risk Ground Truth Labels")
    print("=" * 70)

    graph_path = CFG.OUTPUT_DIR / "13_graph.pt"
    graphs = torch.load(graph_path, weights_only=False)

    lookup_mod_ttc = {}
    lookup_std_ttc = {}

    for graph in tqdm(graphs, desc="Building Future Lookup Index"):
        frame = int(graph.frame)
        if graph.edge_index.shape[1] == 0:
            continue
        edge_tracks = graph.edge_tracks.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()

        for edge_id in range(edge_tracks.shape[0]):
            src_track = int(edge_tracks[edge_id, 0])
            dst_track = int(edge_tracks[edge_id, 1])
            
            ttc_std_val = float(edge_attr[edge_id, 5])
            ttc_mod_val = float(edge_attr[edge_id, 6])
            
            # -------------------------------------------------------------
            # 🚨 FIX 3: SAFEGUARD ANY RESIDUAL NaNs IN TTC 🚨
            # If TTC is NaN (no collision), cap it at 10.0 seconds
            # -------------------------------------------------------------
            if np.isnan(ttc_std_val) or np.isinf(ttc_std_val):
                ttc_std_val = CFG.TTC_CAP
            if np.isnan(ttc_mod_val) or np.isinf(ttc_mod_val):
                ttc_mod_val = CFG.TTC_CAP

            lookup_std_ttc[(frame, src_track, dst_track)] = ttc_std_val
            lookup_mod_ttc[(frame, src_track, dst_track)] = ttc_mod_val

    labelled_graphs = []
    pair_rows = []
    scene_rows = []

    for graph in tqdm(graphs, desc="Assigning Multi-Horizon Risk"):
        frame = int(graph.frame)
        num_edges = graph.edge_index.shape[1]

        if num_edges == 0:
            graph.y_pair_prob = torch.empty((0,), dtype=torch.float32)
            graph.y_pair_std_prob = torch.empty((0,), dtype=torch.float32)
            graph.y_scene_prob = torch.tensor([0.0], dtype=torch.float32)
            labelled_graphs.append(graph)
            scene_rows.append([frame, 0.0])
            continue

        edge_tracks = graph.edge_tracks.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()
        
        pair_risk_mod = np.zeros(num_edges, dtype=np.float32)
        pair_risk_std = np.zeros(num_edges, dtype=np.float32)

        for edge_id in range(num_edges):
            src = int(edge_tracks[edge_id, 0])
            dst = int(edge_tracks[edge_id, 1])

            future_min_mod = CFG.TTC_CAP
            future_min_std = CFG.TTC_CAP

            for step in range(1, CFG.T_PRED + 1):
                key = (frame + step, src, dst)
                if key in lookup_mod_ttc:
                    future_min_mod = min(future_min_mod, lookup_mod_ttc[key])
                if key in lookup_std_ttc:
                    future_min_std = min(future_min_std, lookup_std_ttc[key])

            # Continuous Exponential Mapping: Risk = exp(-TTC / tau)
            risk_mod = np.clip(np.exp(-future_min_mod / CFG.PET_CRITICAL), 0.0, 1.0)
            risk_std = np.clip(np.exp(-future_min_std / CFG.PET_CRITICAL), 0.0, 1.0)

            pair_risk_mod[edge_id] = risk_mod
            pair_risk_std[edge_id] = risk_std

            # Re-fetch TTCs for the CSV logging
            ttc_std_curr = float(edge_attr[edge_id, 5])
            ttc_mod_curr = float(edge_attr[edge_id, 6])
            if np.isnan(ttc_std_curr): ttc_std_curr = CFG.TTC_CAP
            if np.isnan(ttc_mod_curr): ttc_mod_curr = CFG.TTC_CAP

            pair_rows.append([
                frame, src, dst,
                ttc_std_curr, ttc_mod_curr,
                float(risk_mod), float(risk_std)
            ])

        # Scene Risk = Balanced Mean
        scene_risk = float(np.mean(pair_risk_mod))

        graph.y_pair_prob = torch.tensor(pair_risk_mod, dtype=torch.float32)
        graph.y_pair_std_prob = torch.tensor(pair_risk_std, dtype=torch.float32)
        graph.y_scene_prob = torch.tensor([scene_risk], dtype=torch.float32)

        labelled_graphs.append(graph)
        scene_rows.append([frame, scene_risk])

    # Save labeled dataset
    dataset_out = CFG.OUTPUT_DIR / "14_graph_dataset_labelled.pt"
    torch.save(labelled_graphs, dataset_out)

    # -------------------------------------------------------------
    # 🚨 FIX 4: ADD HEADERS TO GROUND TRUTH CSV EXPORTS 🚨
    # -------------------------------------------------------------
    pair_cols = ["frame", "source_track", "target_track", "ttc_std", "ttc_mod", "pair_risk_mod", "pair_risk_std"]
    pd.DataFrame(pair_rows, columns=pair_cols).to_csv(CFG.OUTPUT_DIR / "14_pair_ground_truth.csv", index=False)

    scene_cols = ["frame", "scene_risk"]
    pd.DataFrame(scene_rows, columns=scene_cols).to_csv(CFG.OUTPUT_DIR / "14_scene_ground_truth.csv", index=False)

    print(f"[✓] Ground Truth Generated. Labeled dataset saved to {dataset_out}")

if __name__ == "__main__":
    generate_ground_truth()
