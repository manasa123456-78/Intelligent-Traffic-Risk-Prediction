import numpy as np

# Phase 1: Configuration Parameters
RUN_YOLO_TRACKING = True

VIDEO_PATH        = 'newest_video_busy_5min.mp4'   # busiest 5-min window (17:00-22:00) extracted from newest_video.avi
WARPED_VIDEO_PATH = 'newest_video_bev_full.mp4'
OUTPUT_VIDEO      = 'tracked_output_newest.mp4'
TRAJECTORY_CSV    = 'trajectories_newest.csv'
SMOOTHED_CSV      = 'smoothed_trajectories_newest.csv'

YOLO_MODEL  = '../yolov8x.pt'   # reuse existing weights instead of re-downloading into this folder
CONF_THRESH = 0.30

DT               = 0.25    
PROXIMITY_M      = 15.0    
MIN_OVERLAP      = 4       
MIN_OVERLAP_SECONDS = 0.5   # true minimum shared-observation duration required
NM_BAND_FRACTION = 0.25    

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
