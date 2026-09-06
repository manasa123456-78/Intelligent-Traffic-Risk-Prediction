import supervision as sv

tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=3)

# Frame tracking & annotation logic inside video loop:
# detections = tracker.update_with_detections(detections)
# annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
# annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
# annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=all_labels)
# annotated_frame = zone_annotator.annotate(scene=annotated_frame)
