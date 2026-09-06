import math
import torch
import time
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from torch_geometric.data import Data
from tqdm import tqdm
from global_variables_02 import CFG


def safe_to_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
        print(f"  [✓] Saved: {path}")
    except PermissionError:
        fallback = str(path).replace(".csv", f"_{int(time.time())}.csv")
        print(f"  [!] Warning: File locked in another program. Saved to fallback: {fallback}")
        df.to_csv(fallback, index=False)

        
class EdgeFeatureComputer:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.one_hot = np.eye(cfg.NUM_CLASSES, dtype=np.float32)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _ttc_standard(self, pi, vi, pj, vj, ri, rj):
        dp = pi - pj
        dv = vi - vj
        R = ri + rj
        a = np.dot(dv, dv)
        b = 2.0 * np.dot(dp, dv)
        c = np.dot(dp, dp) - R * R
        if c <= 0.0:
            return 0.0
        if a < 1e-8:
            return self.cfg.TTC_CAP
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return self.cfg.TTC_CAP
        sqrt_disc = math.sqrt(discriminant)
        t1, t2 = (-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)
        ttc = self.cfg.TTC_CAP
        if 0.0 < t1 < ttc: ttc = t1
        if 0.0 < t2 < ttc: ttc = t2
        return float(ttc)

    def _ttc_modified(self, ttc_std, speed_i, speed_j, distance, ci, cj):
        a_i = self.cfg.MAX_DECEL[ci]
        a_j = self.cfg.MAX_DECEL[cj]
        brake_i = (speed_i ** 2) / (2.0 * a_i + 1e-6)
        brake_j = (speed_j ** 2) / (2.0 * a_j + 1e-6)
        ratio = (brake_i + brake_j) / (distance + 1e-6)
        scale = 1.0 + min(ratio, 4.0)
        return float(min(ttc_std / scale, self.cfg.TTC_CAP))

    def compute(self, si, sj):
        pi, pj = np.array([si["x"], si["y"]]), np.array([sj["x"], sj["y"]])
        vi, vj = np.array([si["vx"], si["vy"]]), np.array([sj["vx"], sj["vy"]])
        ci, cj = int(si["class_id"]), int(sj["class_id"])

        delta_p = pj - pi
        distance = float(np.linalg.norm(delta_p))
        delta_v = vi - vj
        dvx, dvy = float(delta_v[0]), float(delta_v[1])
        rel_speed = float(np.linalg.norm(delta_v))

        direction = delta_p / (distance + 1e-6)
        approach_speed = float(np.dot(delta_v, direction))
        heading_diff = self.normalize_angle(si["heading"] - sj["heading"])

        speed_i = float(si["speed"])
        if speed_i > 1e-6:
            cos_theta = np.clip(np.dot(vi, direction) / speed_i, -1.0, 1.0)
            approach_angle = float(math.acos(cos_theta))
        else:
            approach_angle = 0.0

        lateral_offset = float(distance * math.sin(approach_angle))
        time_headway = min(distance / (speed_i + 1e-6), self.cfg.TTC_CAP)

        ri = self.cfg.DIMENSIONS[ci][0] / 2.0
        rj = self.cfg.DIMENSIONS[cj][0] / 2.0
        ttc_std = self._ttc_standard(pi, vi, pj, vj, ri, rj)
        ttc_mod = self._ttc_modified(ttc_std, speed_i, float(sj["speed"]), distance, ci, cj)

        # 27-D Edge Feature Array
        edge_feature = np.concatenate([
            np.array([
                distance,          # 0
                dvx,               # 1
                dvy,               # 2
                rel_speed,         # 3
                approach_speed,    # 4
                ttc_std,           # 5 (Standard TTC)
                ttc_mod,           # 6 (Modified TTC)
                approach_angle,    # 7
                heading_diff,      # 8
                lateral_offset,    # 9
                time_headway       # 10
            ], dtype=np.float32),
            self.one_hot[ci],      # 11-18 (Source One-Hot)
            self.one_hot[cj]       # 19-26 (Target One-Hot)
        ])
        return edge_feature, ttc_std, ttc_mod

def build_spatio_temporal_graphs():
    print("=" * 70)
    print("PHASE 3: Building Spatio-Temporal Interaction Graphs from PINN CSV")
    print("=" * 70)

    if not CFG.SMOOTHED_CSV.exists():
        raise FileNotFoundError(f"Missing {CFG.SMOOTHED_CSV}! Please ensure the CSV is placed in the project directory.")

    df = pd.read_csv(CFG.SMOOTHED_CSV)

    # -------------------------------------------------------------
    # 🚨 FIX 1: SANITIZE NaN VALUES BEFORE PROCESSING 🚨
    # This guarantees the PyTorch tensors won't crash with "loss=nan"
    # -------------------------------------------------------------
    df = df.dropna(subset=["x_m_smooth", "y_m_smooth", "vx", "vy", "heading"])
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)

    # Harmonize column names
    if "x_m_smooth" in df.columns:
        df["x"] = df["x_m_smooth"]
        df["y"] = df["y_m_smooth"]

    if "class_id" not in df.columns:
        cls_map = {name: i for i, name in enumerate(CFG.CLASS_NAMES)}
        df["class_id"] = df["class_name"].map(cls_map).fillna(0).astype(int)

    edge_computer = EdgeFeatureComputer(CFG)
    graph_list = []
    frames = sorted(df["frame"].unique())

    all_node_rows = []
    all_edge_rows = []

    for frame_id in tqdm(frames, desc="Graph Construction"):
        frame_df = df[df["frame"] == frame_id].reset_index(drop=True)
        num_nodes = len(frame_df)
        if num_nodes == 0:
            continue

        positions = frame_df[["x", "y"]].values.astype(np.float32)
        tree = KDTree(positions)

        node_features = []
        states = []

        for row in frame_df.itertuples(index=False):
            cid = int(row.class_id)
            w, l = CFG.DIMENSIONS.get(cid, (1.5, 3.5))
            
            # Decompose scalar acceleration along heading
            heading = float(row.heading)
            scalar_acc = float(getattr(row, "accel", 0.0))
            ax = scalar_acc * math.cos(heading)
            ay = scalar_acc * math.sin(heading)

            st = {
                "frame": int(row.frame),
                "track_id": int(row.track_id),
                "class_id": cid,
                "confidence": float(getattr(row, "conf", 1.0)),
                "x": float(row.x),
                "y": float(row.y),
                "vx": float(row.vx),
                "vy": float(row.vy),
                "ax": ax,
                "ay": ay,
                "speed": float(row.speed),
                "heading": heading,
                "width": float(w),
                "length": float(l)
            }
            states.append(st)

            # 20-D Node Feature Vector
            one_hot = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
            one_hot[cid] = 1.0
            nf = np.concatenate([
                np.array([
                    st["x"], st["y"], st["vx"], st["vy"], st["ax"], st["ay"],
                    st["speed"], st["heading"], st["width"], st["length"],
                    st["confidence"], float(st["track_id"])
                ], dtype=np.float32),
                one_hot
            ])
            node_features.append(nf)
            all_node_rows.append([int(row.frame), int(row.track_id), *nf.tolist()])

        edge_index = []
        edge_features = []
        edge_tracks = []

        for i in range(num_nodes):
            neighbours = tree.query_ball_point(positions[i], r=CFG.MAX_RADIUS)
            for j in neighbours:
                if i == j:
                    continue

                # Skip pedestrian-pedestrian interactions to isolate vehicular conflict dynamics
                if states[i]["class_id"] == 0 and states[j]["class_id"] == 0:
                    continue

                ef, _, _ = edge_computer.compute(states[i], states[j])
                edge_index.append([i, j])
                edge_features.append(ef)
                edge_tracks.append([states[i]["track_id"], states[j]["track_id"]])
                all_edge_rows.append([int(frame_id), states[i]["track_id"], states[j]["track_id"], *ef.tolist()])

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
            edge_tracks_tensor = torch.tensor(edge_tracks, dtype=torch.long)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, CFG.EDGE_DIM), dtype=torch.float32)
            edge_tracks_tensor = torch.empty((0, 2), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr)
        graph.frame = int(frame_id)
        graph.num_agents = int(num_nodes)
        graph.track_ids = torch.tensor(frame_df["track_id"].values, dtype=torch.long)
        graph.edge_tracks = edge_tracks_tensor
        graph_list.append(graph)

    # Save Graphs
    out_graph = CFG.OUTPUT_DIR / "13_graph.pt"
    torch.save(graph_list, out_graph)
    
    # -------------------------------------------------------------
    # 🚨 FIX 2: ADD HEADERS TO NODE AND EDGE CSV EXPORTS 🚨
    # -------------------------------------------------------------
    node_cols = [
        "frame", "track_id", "x", "y", "vx", "vy", "ax", "ay", 
        "speed", "heading", "width", "length", "confidence", "track_id_dup",
        "is_pedestrian", "is_cyclist", "is_motorcycle", "is_auto_rickshaw", 
        "is_car", "is_lcv", "is_bus", "is_truck"
    ]
    pd.DataFrame(all_node_rows, columns=node_cols).to_csv(CFG.OUTPUT_DIR / "10_node_features.csv", index=False)

    edge_cols = [
        "frame", "source_track_id", "target_track_id", 
        "distance", "delta_vx", "delta_vy", "relative_speed", "approach_speed",
        "ttc_standard", "ttc_modified", "approach_angle", "heading_difference",
        "lateral_offset", "time_headway",
        "src_pedestrian", "src_cyclist", "src_motorcycle", "src_auto_rickshaw", 
        "src_car", "src_lcv", "src_bus", "src_truck",
        "dst_pedestrian", "dst_cyclist", "dst_motorcycle", "dst_auto_rickshaw", 
        "dst_car", "dst_lcv", "dst_bus", "dst_truck"
    ]
    pd.DataFrame(all_edge_rows, columns=edge_cols).to_csv(CFG.OUTPUT_DIR / "12_edge_features.csv", index=False)
    
    print(f"[✓] Graph Construction Complete. Saved {len(graph_list)} PyG graphs to {out_graph}")

if __name__ == "__main__":
    build_spatio_temporal_graphs()
