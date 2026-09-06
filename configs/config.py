import os
import time
import numpy as np
import pandas as pd

# --- Windows-safe CSV writer ---
def safe_to_csv(df, path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
        print(f"  saved: {path}")
    except PermissionError:
        fallback = path.replace('.csv', f'_{int(time.time())}.csv')
        print(f"  WARNING: '{path}' is open/locked. Saving instead to: {fallback}")
        df.to_csv(fallback, **kwargs)

# Execution Switches & File Paths
RUN_YOLO_TRACKING = True

VIDEO_PATH        = 'newest_video_busy_5min.mp4'
WARPED_VIDEO_PATH = 'newest_video_bev_full.mp4'
OUTPUT_VIDEO      = 'tracked_output_newest.mp4'
TRAJECTORY_CSV    = 'trajectories_newest.csv'
SMOOTHED_CSV      = 'smoothed_trajectories_newest.csv'
H_NPY_PATH        = 'H_out_v3.npy'

YOLO_MODEL  = '../yolov8x.pt'
CONF_THRESH = 0.30

# Spatial & Metric Constants
ISLAND_RADIUS_PX = 100.15
ISLAND_RADIUS_M  = 7.0
if ISLAND_RADIUS_M is None:
    raise ValueError("ISLAND_RADIUS_M must be set to compute metric spatial conversions.")

SCALE = ISLAND_RADIUS_PX / ISLAND_RADIUS_M  # pixels per metre
CANVAS_WIDTH  = 502
CANVAS_HEIGHT = 1420

# Kinematic Defaults
DT                  = 0.25    
PROXIMITY_M         = 15.0    
MIN_OVERLAP         = 4       
MIN_OVERLAP_SECONDS = 0.5   
NM_BAND_FRACTION    = 0.25    

# Vehicle Classes & Dimensions
VEHICLE_DIMS = {
    'pedestrian':    (0.5,  0.5),
    'motorcycle':    (0.75, 2.0),
    'auto_rickshaw': (1.3,  2.65),
    'car':           (1.5,  3.5),
    'lcv':           (1.8,  4.5),
    'bus':           (2.0,  5.0),
    'truck':         (2.0,  5.0),
    'cyclist':       (0.6,  1.8),
}
DEFAULT_DIM = (1.5, 3.5)
CLASS_NAMES = ['pedestrian', 'cyclist', 'motorcycle', 'auto_rickshaw', 'car', 'lcv', 'bus', 'truck']

# Region of Interest Polygon Coordinates
ROI_POLYGON = np.array([
    [5, 5],
    [5, 1515],
    [2587, 1515],
    [2587, 5],
])
