import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from config import DT, safe_to_csv, SMOOTHED_CSV

class TrajectoryPINN(nn.Module):
    """Physics-Informed Neural Network for trajectory smoothing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.net(t)

def smooth_track_with_pinn(xy_raw, times_array, dt_median=0.25):
    t = torch.tensor(times_array, dtype=torch.float32).unsqueeze(1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    
    xy_mean = np.mean(xy_raw, axis=0)
    xy_std = np.std(xy_raw, axis=0) + 1e-6
    noisy_tensor = torch.tensor(xy_raw, dtype=torch.float32)
    mean_tensor = torch.tensor(xy_mean, dtype=torch.float32)
    std_tensor = torch.tensor(xy_std, dtype=torch.float32)

    model = TrajectoryPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    MAX_ACCEL = 5.0 

    for _ in range(500):
        optimizer.zero_grad()
        pred_norm = model(t_norm)
        smooth_xy = pred_norm * std_tensor + mean_tensor
        
        data_loss = torch.nn.functional.mse_loss(smooth_xy, noisy_tensor)
        
        if smooth_xy.shape[0] > 2:
            velocity = (smooth_xy[1:] - smooth_xy[:-1]) / dt_median
            acceleration = (velocity[1:] - velocity[:-1]) / dt_median
            accel_magnitude = torch.norm(acceleration, dim=1)
            physics_loss = torch.mean(torch.relu(accel_magnitude - MAX_ACCEL))
        else:
            physics_loss = torch.tensor(0.0)
            
        loss = data_loss + (0.1 * physics_loss)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_pred = model(t_norm)
        final_xy = final_pred * std_tensor + mean_tensor
        
    return final_xy.numpy()

def apply_pinn_smoothing(df_raw, output_csv=SMOOTHED_CSV):
    print("--- Physics-Informed Kinematic Smoothing (PINN) ---")
    smoothed_data = []
    grouped_tracks = list(df_raw.groupby('track_id'))

    for track_id, group in tqdm(grouped_tracks, desc="Applying Parametric PINN"):
        group = group.sort_values('time_s').copy()
        if len(group) < 5: 
            continue
        
        dt_val_series = group['time_s'].diff().fillna(DT)
        median_dt = dt_val_series.median() if not pd.isna(dt_val_series.median()) else DT
        
        xy_raw = group[['x_m', 'y_m']].values
        times_array = group['time_s'].values
        
        xy_smooth = smooth_track_with_pinn(xy_raw, times_array, dt_median=median_dt)
        
        group['x_m_smooth'] = xy_smooth[:, 0]
        group['y_m_smooth'] = xy_smooth[:, 1]
        
        dt_array = np.where(dt_val_series.values < 1e-6, DT, dt_val_series.values)
        
        group['vx'] = group['x_m_smooth'].diff() / dt_array
        group['vy'] = group['y_m_smooth'].diff() / dt_array
        group['speed'] = np.sqrt(group['vx']**2 + group['vy']**2).rolling(3, min_periods=1).mean()
        group['accel'] = group['speed'].diff() / dt_array
        group['accel'] = group['accel'].fillna(0.0).rolling(3, min_periods=1).mean()
        group['heading'] = np.arctan2(group['vy'], group['vx']).rolling(3, min_periods=1).mean()

        smoothed_data.append(group)

    df_smooth = pd.concat(smoothed_data, ignore_index=True)
    safe_to_csv(df_smooth, output_csv, index=False)
    print(f"Smoothed tracks using PINN saved. Total unique tracks: {df_smooth['track_id'].nunique()}")
    return df_smooth
