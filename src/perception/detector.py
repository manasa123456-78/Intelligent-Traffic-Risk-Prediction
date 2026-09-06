import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

class COCOtoIndianMapper:
    def __init__(self):
        self.lut = {0: 0, 1: 1, 2: 4, 3: 2, 5: 6, 7: 7}
        
    def map_classes(self, boxes, confs, cls_ids):
        out = []
        for i in range(len(cls_ids)):
            coco = int(cls_ids[i])
            if coco not in self.lut: continue
            indian = self.lut[coco]
            conf = float(confs[i])
            x1, y1, x2, y2 = map(float, boxes[i])
            
            if indian == 4:
                w, h = x2 - x1, y2 - y1
                area, asp = w * h, w / (h + 1e-6)
                if 400 <= area <= 4000 and 0.70 <= asp <= 1.60:
                    indian, conf = 3, conf * 0.85
            elif indian == 2:
                if (x2 - x1) * (y2 - y1) > 7500:
                    indian, conf = 5, conf * 0.80
                    
            out.append([x1, y1, x2, y2, conf, indian])
        return np.array(out, dtype=np.float32) if out else np.empty((0,6), np.float32)

mapper = COCOtoIndianMapper()

device_target = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO(YOLO_MODEL)
zone = sv.PolygonZone(polygon=ROI_POLYGON)

# Frame detection logic inside video loop:
# results = model(frame, verbose=False, conf=CONF_THRESH, device=device_target)[0]
# detections = sv.Detections.from_ultralytics(results)
# mask = zone.trigger(detections=detections)
# detections = detections[mask]
# mapped_dets = mapper.map_classes(detections.xyxy, detections.confidence, detections.class_id)
