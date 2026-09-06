import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

def safe_to_csv(df, filepath, index=False):
    dirname = os.path.dirname(os.path.abspath(filepath))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(filepath, index=index)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION PARAMETERS
# -----------------------------------------------------------------------------
RUN_YOLO_TRACKING = True

VIDEO_PATH        = 'newest_video_busy_5min.mp4'   
WARPED_VIDEO_PATH = 'newest_video_bev_full.mp4'
OUTPUT_VIDEO      = 'tracked_output_newest.mp4'
TRAJECTORY_CSV    = 'trajectories_newest.csv'
SMOOTHED_CSV      = 'smoothed_trajectories_newest.csv'

YOLO_MODEL  = '../yolov8x.pt'   
CONF_THRESH = 0.30

DT                  = 0.25    
PROXIMITY_M         = 15.0    
MIN_OVERLAP         = 4       
MIN_OVERLAP_SECONDS = 0.5   
NM_BAND_FRACTION    = 0.25    

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

ROI_POLYGON = np.array([
    [5, 5],
    [5, 1515],
    [2587, 1515],
    [2587, 5],
])
