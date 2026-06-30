# RiskPulse: A Spatio-Temporal Graph Neural Network for Future Traffic Risk Prediction

run_pipeline.py
│
├── ROI_04.py
├── detector_05.py
├── mapper_03.py
├── tracking_06.py
├── smooth_tracks_08.py
├── trajectory_08.py
├── homography_06.py
├── homography_projection_07.py
├── edgefeatures_10.py
├── GraphBuilder_11.py
└── Ground_labels_truth_12.py
        │
        ▼
14_graph_dataset_labelled.pt
        │
        ▼
training/
│
├── dataset.py
│       │
│       ▼
├── trainer.py
│       │
│       ▼
├── losses.py
│
├── metrics.py
│
├── models/
│      │
│      ├── spatial_encoder.py
│      ├── temporal_encoder.py
│      ├── prediction_heads.py
│      └── traffic_risk_model.py
│
└── train.py
        │
        ▼
checkpoint_best.pt
        │
        ▼
evaluate.py
        │
        ▼
Final Metrics


