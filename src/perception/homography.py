import os
import cv2
import numpy as np
from tqdm import tqdm

H_OUT = np.load('H_out_v3.npy')
out_width, out_height = 502, 1420   # canvas size H_OUT was built for

ISLAND_RADIUS_PX = 100.15
ISLAND_RADIUS_M  = 7.0   # real-world radius of the roundabout island, in metres (given by user)

if ISLAND_RADIUS_M is None:
    raise ValueError(
        "ISLAND_RADIUS_M is not set (Phase 2, run.py). Measure or obtain the "
        "real-world radius (metres) of the newest_video.avi roundabout island "
        "and set it above before running the pipeline -- every downstream "
        "distance/safety metric depends on this scale factor."
    )

SCALE = ISLAND_RADIUS_PX / ISLAND_RADIUS_M   # pixels per metre in the H_OUT canvas

H_viewport = H_OUT

def pixel_to_world_viewport(px, py, H=H_viewport):
    pt = np.array([[[px, py]]], dtype=np.float32)
    wpt = cv2.perspectiveTransform(pt, H)
    wx = float(wpt[0, 0, 0]) / SCALE
    wy = float(wpt[0, 0, 1]) / SCALE
    return wx, wy

# BEV Homography Video Generation
if not os.path.exists(WARPED_VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_warp = cv2.VideoWriter(WARPED_VIDEO_PATH, fourcc, fps, (out_width, out_height))
    for _ in tqdm(range(total_frames), desc="Warping Video"):
        ret, frame = cap.read()
        if not ret: break
        warped_frame = cv2.warpPerspective(frame, H_viewport, (out_width, out_height))
        out_warp.write(warped_frame)
    cap.release()
    out_warp.release()
