import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # ---------------------------------------------------------
    # PATHS & DIRECTORIES
    # ---------------------------------------------------------
    PROJECT_ROOT: Path = Path("C:\\manasa_project")
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    REPORTS_DIR: Path = OUTPUT_DIR / "reports"

    # Input CSV from PINN smoothing
    SMOOTHED_CSV: Path = Path("C:\\manasa_project\\data\\csv\\smoothed_trajectories_newest.csv")
    CHECKPOINT_BEST: Path = CHECKPOINT_DIR / "checkpoint_best.pt"

    # ---------------------------------------------------------
    # HARDWARE & REPRODUCIBILITY
    # ---------------------------------------------------------
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP: bool = False
    SEED: int = 42
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True

    # ---------------------------------------------------------
    # HETEROGENEOUS INDIAN TRAFFIC CLASSES
    # ---------------------------------------------------------
    CLASS_NAMES: List[str] = field(default_factory=lambda: [
        "pedestrian", "cyclist", "motorcycle", "auto_rickshaw",
        "car", "lcv", "bus", "truck"
    ])
    NUM_CLASSES: int = 8

    # Agent Dimensions: (Width, Length in meters)
    DIMENSIONS: Dict[int, tuple] = field(default_factory=lambda: {
        0: (0.5, 0.5),    # pedestrian
        1: (0.6, 1.8),    # cyclist
        2: (0.75, 2.0),   # motorcycle
        3: (1.3, 2.65),   # auto_rickshaw
        4: (1.5, 3.5),    # car
        5: (1.8, 4.5),    # lcv
        6: (2.0, 5.0),    # bus
        7: (2.0, 5.0)     # truck
    })

    # Maximum Deceleration Limits (m/s^2) for Braking-Aware TTC
    MAX_DECEL: List[float] = field(default_factory=lambda: [
        2.5,  # pedestrian
        2.0,  # cyclist
        4.0,  # motorcycle
        3.5,  # auto_rickshaw
        3.0,  # car
        3.5,  # lcv
        2.5,  # bus
        2.0   # truck
    ])

    # ---------------------------------------------------------
    # KINEMATIC & GRAPH TOPOLOGY CONSTANTS
    # ---------------------------------------------------------
    SAMPLE_INTERVAL: float = 0.05003  # Empirical frame interval (~20 FPS)
    MAX_RADIUS: float = 15.0          # Interaction query radius (meters)
    TTC_CAP: float = 10.0             # Maximum TTC cap (seconds)
    PET_CRITICAL: float = 3.0         # Critical scaling parameter tau (seconds)

    # Feature Dimensions
    NODE_DIM: int = 20
    EDGE_DIM: int = 27

    # Spatio-Temporal Sequences
    T_OBS: int = 8                    # Observation window (frames)
    T_PRED: int = 12                  # Prediction horizon lookahead (frames)

    # GNN Architecture
    HIDDEN: int = 128
    HGT_HEADS: int = 4
    HGT_LAYERS: int = 2
    TCN_CH: int = 64
    TCN_BLOCKS: int = 4
    TCN_KERN: int = 3
    DROPOUT: float = 0.2

    # ---------------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # ---------------------------------------------------------
    EPOCHS: int = 1000
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    EARLY_STOPPING_PATIENCE: int = 50
    LR_FACTOR: float = 0.5
    LR_PATIENCE: int = 10
    MIN_LR: float = 1e-6
    PAIR_LOSS_WEIGHT: float = 1.0
    SCENE_LOSS_WEIGHT: float = 0.5

    # ---------------------------------------------------------
    # ASYNC LOCAL LLM SETTINGS
    # ---------------------------------------------------------
    LLM_TRIGGER_SCENE_RISK: float = 0.30
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

CFG = Config()

# Build prerequisite directories
for d in [CFG.OUTPUT_DIR, CFG.CHECKPOINT_DIR, CFG.PLOTS_DIR, CFG.REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)