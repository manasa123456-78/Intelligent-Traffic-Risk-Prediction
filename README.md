# Edge-Deployable Spatio-Temporal Graph Transformer Framework for Continuous Traffic Conflict Prediction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official implementation of the research paper: **"An Edge-Deployable Spatio-Temporal Graph Transformer Framework for Continuous Multi-Agent Traffic Conflict Prediction and Explainable LLM-Driven Safety Diagnostics."**

## Overview
Proactive traffic conflict analysis is essential for Intelligent Transportation Systems (ITS). Conventional Surrogate Safety Measures (SSMs) like TTC and PET assume constant velocities and evaluate vehicle pairs in isolation, frequently failing in heterogeneous, unsignalized traffic environments. 

This project introduces an end-to-end deep learning framework that models traffic scenes as dynamic multi-relational graphs. By embedding a novel **Braking-Aware Modified TTC** into 27-dimensional edge features, a Spatio-Temporal Graph Neural Network (ST-GNN) regresses continuous future conflict risk at **64.6 FPS**. 

To resolve the "black-box" interpretability bottleneck of deep learning, the framework decouples numerical prediction from semantic reasoning by integrating a local **Large Language Model (Llama 3 8B)**. The LLM acts as an asynchronous Explainable AI (XAI) agent, synthesizing threshold-triggered high-dimensional tensors into structured JSON incident reports.

## Key Contributions
1. **Heterogeneous Graph Formulation:** 20-D node and 27-D physics-guided edge features explicit modeling multi-agent interactions across 8 road-user classes.
2. **Continuous Multi-Task Risk Regression:** Predicts gradual risk escalation on a `[0, 1]` continuous manifold over a 0.6s future horizon, achieving an $R^2$ of 0.558 and MAE of 0.1118.
3. **High-Speed Edge Deployability:** Replaces recurrent LSTMs with a 4-block Dilated 1D TCN, achieving an inference latency of 15.48 ms/frame (64.6 FPS).
4. **Agentic LLM Explainability:** A decoupled, on-premises LLM translates critical graph states (Risk $\ge 0.75$) into actionable, hallucination-free JSON safety advisories.

## System Architecture

* **Perception:** YOLOv8x + ByteTrack $\rightarrow$ Metric Bird's-Eye View (BEV) Homography.
* **Trajectory Refinement:** Physics-Informed Neural Network (PINN) kinematic smoothing.
* **AI Predictor (ST-GNN):** 2-layer Graph Transformer (Spatial) + 4-Block Dilated 1D TCN (Temporal) $\rightarrow$ Dual-Head Multi-Task Regression.
* **Explainability (XAI):** Threshold-triggered Ollama/Llama-3 LLM diagnostic generation.


## Experimental Results

Tested on 5 minutes (5,996 frames) of heterogeneous unsignalized roundabout traffic in India.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Inference Throughput** | `64.6 FPS` | Real-time ready (15.48 ms/frame latency). |
| **Pairwise R² Score** | `0.558` | Strong goodness-of-fit on continuous risk targets. |
| **Classification AUC** | `0.909` | High discrimination at early-warning threshold ($\tau = 0.3$). |
| **Recall (Sensitivity)** | `83.7%` | Successfully captures vast majority of critical encounters. |
| **Ablation (MAE Error)** | `0.22` $\to$ `0.11` | Braking-aware Modified TTC reduces prediction error by 50% vs. Standard TTC. |

### Visualizations
*(Plots generated directly from the evaluation pipeline)*
* **ROC & PR Curves:** Proves robust discrimination with minimal false alarms (`outputs/plots/Fig3_ROC_PR_Curves_Corrected.png`).
* **Spatial Attention:** The Graph Transformer actively zeroes in on converging hazards (`outputs/plots/Fig1_Attention_Heatmap.png`).
* **Multi-Horizon Early Warning:** Triggers safety alerts 1.5 seconds *before* physical encroachment (`outputs/plots/Fig2_Predictive_TimeSeries.png`).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Intelligent-Traffic-Risk-Prediction.git](https://github.com/yourusername/Intelligent-Traffic-Risk-Prediction.git)
   cd Intelligent-Traffic-Risk-Prediction

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama (For LLM Explainability):**
   * Download and install Ollama.
   * Pull the Llama 3 model: 
     ```bash
     ollama run llama3:8b
     ```

## Reproducing Results

The pipeline is entirely modular. You can execute the full flow sequentially using the master script:

```bash
python run_pipeline.py
```

**Step-by-Step Execution:**
* `src/perception/detector.py`: Extracts YOLO detections and ByteTrack IDs.
* `src/perception/homography.py`: Maps pixel coordinates to BEV metric space.
* `src/graph/trajectories_smoothing_09_11.py`: Applies PINN kinematic smoothing.
* `src/graph/GraphBuilder_13.py`: Generates the 20-D/27-D dynamic graphs.
* `src/model/Ground_labels_truth_14.py`: Calculates future exponential risk targets.
* `src/model/trainer.py`: Trains the ST-GNN architecture.
* `src/evaluation/evaluate_visualize_llm.py`: Evaluates thresholds, generates ROC curves, and triggers the LLM JSON reports.

## Academic Paper & Citation

The complete academic manuscript detailing the mathematics, network architecture, and ablation studies is available in `papers/Research_paper_1.pdf`.

If you use this code or methodology in your research, please cite:

```bibtex
@article{karri2026edge,
  title={An Edge-Deployable Spatio-Temporal Graph Transformer Framework for Continuous Multi-Agent Traffic Conflict Prediction and Explainable LLM-Driven Safety Diagnostics},
  author={Karri, Manasa Indu Sri},
  institution={Indian Institute of Technology Kharagpur},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
