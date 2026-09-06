import os
import cv2
import numpy as np
from tqdm import tqdm
from config import SCALE, CANVAS_WIDTH, CANVAS_HEIGHT, H_NPY_PATH

def load_homography_matrix(path=H_NPY_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography matrix file not found at '{path}'.")
    return np.load(path)

def pixel_to_world_viewport(px, py, H, scale=SCALE):
    """Converts image pixel coordinates into metric world coordinates using homography."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    wpt = cv2.perspectiveTransform(pt, H)
    wx = float(wpt[0, 0, 0]) / scale
    wy = float(wpt[0, 0, 1]) / scale
    return wx, wy

def generate_bev_video(video_path, warped_output_path, H, width=CANVAS_WIDTH, height=CANVAS_HEIGHT):
    """Warps source video into Bird's Eye View canvas."""
    print("--- Generating Full BEV Homography Video ---")
    if os.path.exists(warped_output_path):
        print(f"Found existing warped video at '{warped_output_path}'. Skipping generation.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_warp = cv2.VideoWriter(warped_output_path, fourcc, fps, (width, height))
    
    for _ in tqdm(range(total_frames), desc="Warping Video"):
        ret, frame = cap.read()
        if not ret: 
            break
        warped_frame = cv2.warpPerspective(frame, H, (width, height))
        out_warp.write(warped_frame)
        
    cap.release()
    out_warp.release()
    print(f"Warped video saved to '{warped_output_path}'")
