import pandas as pd
import torch
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

from config import (
    VIDEO_PATH, OUTPUT_VIDEO, YOLO_MODEL, CONF_THRESH, 
    ROI_POLYGON, CLASS_NAMES
)
from mapper import COCOtoIndianMapper
from homography import pixel_to_world_viewport

def run_detection_and_tracking(H_viewport):
    """Runs YOLO object detection and BoT-SORT/ByteTrack trajectory tracking."""
    print("--- Processing Video (YOLO + Tracking) ---")
    
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_info.fps, video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
    
    zone = sv.PolygonZone(polygon=ROI_POLYGON)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=3)

    device_target = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Tracking using device: {device_target}")

    model = YOLO(YOLO_MODEL)
    tracker = sv.ByteTrack()
    mapper = COCOtoIndianMapper()
    raw_data = []

    generator = sv.get_video_frames_generator(VIDEO_PATH)

    for frame_idx, frame in enumerate(tqdm(generator, total=video_info.total_frames, desc='Tracking')):
        results = model(frame, verbose=False, conf=CONF_THRESH, device=device_target)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        mask = zone.trigger(detections=detections)
        detections = detections[mask] 

        detections = tracker.update_with_detections(detections)
        
        if len(detections) > 0:
            mapped_dets = mapper.map_classes(detections.xyxy, detections.confidence, detections.class_id)
            if len(mapped_dets) > 0:
                for i, det in enumerate(mapped_dets):
                    x1, y1, x2, y2, conf, mapped_cls = det
                    track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    
                    if track_id != -1:
                        px_c, py_b = (x1 + x2) / 2, y2
                        wx, wy = pixel_to_world_viewport(px_c, py_b, H_viewport)
                        
                        raw_data.append({
                            'frame': frame_idx,
                            'time_s': frame_idx / video_info.fps,
                            'track_id': track_id,
                            'class_id': int(mapped_cls),
                            'class_name': CLASS_NAMES[int(mapped_cls)],
                            'x_m': wx,
                            'y_m': wy
                        })
        
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        all_labels = []
        for i in range(len(detections)):
            coco_cls = int(detections.class_id[i]) if detections.class_id is not None else -1
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            if coco_cls not in mapper.lut:
                all_labels.append("Unknown")
                continue
                
            indian_cls = mapper.lut[coco_cls]
            x1, y1, x2, y2 = detections.xyxy[i]
            
            if indian_cls == 4:
                w, h = x2 - x1, y2 - y1
                area, asp = w * h, w / (h + 1e-6)
                if 400 <= area <= 4000 and 0.70 <= asp <= 1.60:
                    indian_cls = 3
            elif indian_cls == 2:
                if (x2 - x1) * (y2 - y1) > 7500:
                    indian_cls = 5
                    
            name = CLASS_NAMES[indian_cls]
            all_labels.append(f'#{tid} {name}' if tid != -1 else name)

        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=all_labels)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        out.write(annotated_frame)

    out.release()
    if not raw_data:
        raise RuntimeError("CRITICAL ERROR: YOLO found no objects inside the ROI!")

    return pd.DataFrame(raw_data)
