import os
import pandas as pd
from config import TRAJECTORY_CSV, safe_to_csv

def save_trajectories(df_raw, path=TRAJECTORY_CSV):
    safe_to_csv(df_raw, path, index=False)
    print(f"Trajectories successfully saved to {path}")

def load_trajectories(path=TRAJECTORY_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CRITICAL ERROR: '{path}' not found! Run detection tracking first."
        )
    return pd.read_csv(path)
