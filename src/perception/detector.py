import cv2
import numpy as np
import torch

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import time

from config.py import CFG
from mapper import COCOtoIndianMapper, UVHtoIndianMapper

DEVICE = CFG.DEVICE

class Detector:

    CLASS_COLORS = [

        (50,220,50),      # pedestrian

        (255,165,0),      # cyclist

        (0,0,255),        # motorcycle

        (180,40,255),     # auto_rickshaw

        (0,200,200),      # car

        (128,0,128),      # lcv

        (0,255,255),      # bus

        (0,100,255)       # truck

    ]

    # Initialization
    def __init__(self,
                 cfg=CFG):

        self.cfg = cfg

        print("-"*60)
        print("Initializing Detector")
        print("-"*60)

        # Models
        print("Loading COCO model...")

        self.coco_model = YOLO(
            cfg.COCO_WEIGHTS
        )

        print("Loading UVH model...")

        self.uvh_model = YOLO(
            cfg.UVH_WEIGHTS
        )

        if DEVICE.type == "cuda":

            self.coco_model.to(DEVICE)

            self.uvh_model.to(DEVICE)


        # Mappers
        self.coco_mapper = COCOtoIndianMapper(cfg)

        self.uvh_mapper = UVHtoIndianMapper(cfg)


        # TrackeR
        self.tracker = DeepSort(

            max_age=30,

            n_init=1,

            max_iou_distance=0.7,

            max_cosine_distance=0.3,

            nn_budget=100,

            embedder="mobilenet",

            half=(DEVICE.type=="cuda"),

            bgr=True

        )


        # ROI
        # ROI
        if cfg.ROI_POLYGON is not None:
            self.roi = np.array(cfg.ROI_POLYGON).astype(np.int32)
        else:
            self.roi = None


        # Prediction Parameters

        self.predict_args = dict(

            imgsz=cfg.YOLO_IMGSZ,

            conf=cfg.CONF_THRESH,

            iou=cfg.IOU_NMS,

            max_det=cfg.MAX_DET,

            verbose=False

        )

        # Pass the global torch.device object directly to Ultralytics YOLO
        self.predict_args["device"] = DEVICE
        
        if DEVICE.type == "cuda":
            self.predict_args["half"] = True
            
        print("Detector Ready.")
        print("="*60)


    # ROI

    def set_roi(self,
                roi_points):

        """
        roi_points:

        np.array([
            [x1,y1],
            [x2,y2],
            ...
        ])
        """

        self.roi = roi_points.astype(np.int32)


    def apply_roi(self,
                  frame):

        """
        Mask everything
        outside ROI
        """

        if self.roi is None:

            return frame

        mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        cv2.fillPoly(

            mask,

            [self.roi],

            255

        )

        roi_frame = cv2.bitwise_and(

            frame,

            frame,

            mask=mask

        )

        return roi_frame


    # COCO Detection


    def detect_coco(self, frame):

        with torch.no_grad():
            results = self.coco_model.predict(
                frame,
                **self.predict_args
            )

        if (
            len(results) == 0 or
            results[0].boxes is None or
            len(results[0].boxes) == 0
        ):
            return np.empty((0,6), dtype=np.float32)

        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()

        conf = boxes.conf.cpu().numpy()

        cls = boxes.cls.cpu().numpy().astype(int)

        return self.coco_mapper.map(
            xyxy,
            conf,
            cls
        )



    # UVH Detection


    def detect_uvh(self, frame):

        with torch.no_grad():
            results = self.uvh_model.predict(
                frame,
                **self.predict_args
            )
        if (
            len(results) == 0 or
            results[0].boxes is None or
            len(results[0].boxes) == 0
        ):
            return np.empty((0,6), dtype=np.float32)

        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()

        conf = boxes.conf.cpu().numpy()

        cls = boxes.cls.cpu().numpy().astype(int)

        return self.uvh_mapper.map(
            xyxy,
            conf,
            cls
        )



    # Merge Detections
    def merge_detections(self, coco_dets, uvh_dets):

        if len(coco_dets) == 0:
            return uvh_dets

        if len(uvh_dets) == 0:
            return coco_dets

        merged = np.vstack([coco_dets, uvh_dets])

        boxes = []
        scores = []

        for det in merged:
            x1, y1, x2, y2 = det[:4]
            boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ])
            scores.append(float(det[4]))

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.cfg.CONF_THRESH,
            self.cfg.IOU_NMS
        )

        if len(indices) == 0:
            return np.empty((0,6), dtype=np.float32)

        indices = np.array(indices).flatten()

        return merged[indices]

    # DeepSORT Tracking
    def track(self, frame, detections):

        """
        Input:
            detections:
            [x1,y1,x2,y2,conf,class]

        Output:
            [x1,y1,x2,y2,track_id,conf,class]
        """

        if len(detections) == 0:
            return np.empty((0,7), dtype=np.float32)


        # Prepare DeepSORT detections
        ds_detections = []

        for det in detections:

            x1, y1, x2, y2, conf, cls = det

            ds_detections.append(
                (
                    [float(x1),
                    float(y1),
                    float(x2-x1),
                    float(y2-y1)],
                    float(conf),
                    int(cls)
                )
            )

        tracks = self.tracker.update_tracks(
            ds_detections,
            frame=frame
        )


        # Match tracks back to detections
        tracked = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            # Extract the actual bounding box dimensions from the DeepSORT track object
            ltrb = track.to_ltrb()
            if ltrb is None:
                continue
            
            l, t, r, b = ltrb
            frame_h, frame_w = frame.shape[:2]

            l = max(0, min(l, frame_w - 1))
            t = max(0, min(t, frame_h - 1))
            r = max(0, min(r, frame_w - 1))
            b = max(0, min(b, frame_h - 1))

            best_iou = 0
            best_det = None

            for det in detections:

                dx1, dy1, dx2, dy2, conf, cls = det

                # IoU
                xx1 = max(l, dx1)
                yy1 = max(t, dy1)
                xx2 = min(r, dx2)
                yy2 = min(b, dy2)

                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)

                inter = inter_w * inter_h

                area_track = (r-l)*(b-t)
                area_det = (dx2-dx1)*(dy2-dy1)

                union = area_track + area_det - inter

                iou = inter/union if union>0 else 0

                if iou > best_iou:

                    best_iou = iou
                    best_det = det


            # Attach original class + confidence

            if best_det is None:

                continue

            _, _, _, _, conf, cls = best_det

            tracked.append([

                l,
                t,
                r,
                b,

                int(track.track_id),

                float(conf),

                int(cls)

            ])

        if len(tracked)==0:
            return np.empty((0,7), dtype=np.float32)

        return np.array(tracked,dtype=np.float32)

    # Complete Detection Pipeline
    def detect(self, frame):

        frame_roi = self.apply_roi(frame)

        coco = self.detect_coco(frame_roi)

        uvh = self.detect_uvh(frame_roi)

        detections = self.merge_detections(
            coco,
            uvh
        )

        tracks = self.track(
            frame_roi,
            detections
        )

        return tracks


if __name__ == "__main__":

    detector = Detector(CFG)

    cap = cv2.VideoCapture(CFG.VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    frame_count = 0
    print("Processing first 10 frames to initialize tracking...")
    print("=" * 60)
    
    while cap.isOpened() and frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        t0 = time.time()
        tracks = detector.detect(frame)
        dt = (time.time() - t0) * 1000
        
        print(f"Frame {frame_count} | Inference: {round(dt, 1)}ms | Tracks Found: {len(tracks)}")
        if len(tracks) > 0:
            print("Tracks details:\n", tracks[:3]) # Print up to 3 tracks
            print("-" * 30)

    cap.release()
    print("=" * 60)
    print("Warm-up test complete.")

if __name__ == "__main__":
    import numpy as np

    print("\n" + "="*60)
    print("RUNNING DETECTOR SMOKE TEST")
    print("="*60)

    detector = Detector(CFG)
    cap = cv2.VideoCapture(CFG.VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ CRITICAL ERROR: Could not open video file at: {CFG.VIDEO_PATH}")
        print("Please double check your path or mount Google Drive.")
        exit()

    print(f"✅ Video successfully opened.")
    print(f"Testing the first 10 frames for raw model detections...\n")

    for frame_idx in range(1, 11):
        ret, frame = cap.read()
        if not ret:
            print(f"⏹️ Reached end of video or failed to read at frame {frame_idx}")
            break

        # Apply ROI manually to see shape
        frame_roi = detector.apply_roi(frame)
        
        # Get raw shapes from mappers/predictions
        with torch.no_grad():
            coco_results = detector.coco_model.predict(frame_roi, **detector.predict_args)
            uvh_results = detector.uvh_model.predict(frame_roi, **detector.predict_args)
        
        coco_count = len(coco_results[0].boxes) if (coco_results and coco_results[0].boxes is not None) else 0
        uvh_count = len(uvh_results[0].boxes) if (uvh_results and uvh_results[0].boxes is not None) else 0

        # Run your actual pipeline up to merged detections
        coco_mapped = detector.detect_coco(frame_roi)
        uvh_mapped = detector.detect_uvh(frame_roi)
        merged = detector.merge_detections(coco_mapped, uvh_mapped)
        
        # Finally, run tracking
        tracks = detector.track(frame_roi, merged)

        print(f"[Frame {frame_idx:02d}] "
              f"Raw COCO Dets: {coco_count:2d} | "
              f"Raw UVH Dets: {uvh_count:2d} | "
              f"Merged: {len(merged):2d} | "
              f"Confirmed Tracks: {len(tracks):2d}")

    cap.release()
    print("="*60)
    print("SMOKE TEST COMPLETE")
    print("="*60)
