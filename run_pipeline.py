#!/usr/bin/env python3


import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Absolute path setup
ROOT_DIR = Path(__file__).resolve().parent

# Ensure logs directory exists
LOGS_DIR = ROOT_DIR / "outputs" / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ROOT_DIR / "outputs" / "plots", exist_ok=True)
os.makedirs(ROOT_DIR / "outputs" / "checkpoints", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "pipeline_execution.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("PipelineRunner")

# Sequential execution order of all script files
PIPELINE_STAGES = [
    {
        "name": "Stage 1: Detection & Tracking",
        "script": ROOT_DIR / "src" / "perception" / "detector.py",
        "description": "Running YOLOv8 detection and ByteTrack object tracking"
    },
    {
        "name": "Stage 2: Homography & BEV Transformation",
        "script": ROOT_DIR / "src" / "perception" / "homography.py",
        "description": "Mapping pixel coordinates to metric Bird's-Eye View (BEV)"
    },
    {
        "name": "Stage 3: Trajectory Smoothing",
        "script": ROOT_DIR / "src" / "graphs" / "smoothing_trajectories.py",
        "description": "Applying PINN kinematic smoothing to BEV trajectories"
    },
    {
        "name": "Stage 4: Spatio-Temporal Graph Construction",
        "script": ROOT_DIR / "src" / "graphs" / "GraphBuilder.py",
        "description": "Building 20-D node and 27-D physics-guided edge graphs"
    },
    {
        "name": "Stage 5: Ground Truth Risk Labeling",
        "script": ROOT_DIR / "src" / "model" / "Ground_labels_truth.py",
        "description": "Generating braking-aware continuous future risk targets"
    },
    {
        "name": "Stage 6: ST-GNN Model Training",
        "script": ROOT_DIR / "src" / "model" / "trainer.py",
        "description": "Training the Spatio-Temporal Graph Neural Network"
    },
    {
        "name": "Stage 7: Evaluation & LLM Diagnostics",
        "script": ROOT_DIR / "src" / "valuation" / "evaluate_visualize_llm.py",
        "description": "Evaluating metrics, thresholding, and triggering LLM XAI"
    },
    {
        "name": "Stage 8: Publication Figure Generation",
        "script": ROOT_DIR / "src" / "valuation" / "figures.py",
        "description": "Generating publication-ready figures and ROC/PR plots"
    }
]


def execute_script(stage_info):
    """Executes a single python script via subprocess and monitors completion."""
    script_path = stage_info["script"]
    stage_name = stage_info["name"]
    description = stage_info["description"]

    if not script_path.exists():
        logger.error(f"File not found: {script_path}")
        raise FileNotFoundError(f"Missing script: {script_path}")

    logger.info("=" * 70)
    logger.info(f"STARTING: {stage_name}")
    logger.info(f"Target: {script_path.relative_to(ROOT_DIR)}")
    logger.info(f"Info: {description}")
    logger.info("=" * 70)

    start_time = time.time()

    # Run Python script using current environment executable
    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=False  # Streams stdout/stderr directly to terminal
    )

    elapsed = time.time() - start_time

    if process.returncode == 0:
        logger.info(f"SUCCESS: {stage_name} completed in {elapsed:.2f} seconds.\n")
    else:
        logger.error(f"FAILURE: {stage_name} failed with return code {process.returncode}.")
        raise RuntimeError(f"Pipeline stopped at {stage_name} ({script_path.name})")


def run_full_pipeline():
    """Runs all stages continuously from start to finish."""
    total_start = time.time()
    logger.info("Starting Full Continuous Pipeline Execution...")

    for stage in PIPELINE_STAGES:
        execute_script(stage)

    total_elapsed = time.time() - total_start
    logger.info("#" * 70)
    logger.info(f"🎉 PIPELINE EXECUTION COMPLETE!")
    logger.info(f"Total execution time: {total_elapsed / 60:.2f} minutes")
    logger.info(f"All outputs saved to: {ROOT_DIR / 'outputs'}")
    logger.info("#" * 70)


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as err:
        logger.critical(f"\nPipeline terminated unexpectedly: {err}")
        sys.exit(1)
