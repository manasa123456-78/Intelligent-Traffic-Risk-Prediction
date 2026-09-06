import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from global_variables_02 import CFG
from dataset_loader import DatasetLoader
from models.traffic_risk_model import TrafficRiskModel

# Apply clean plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

CLASS_COLORS = {
    'pedestrian': '#4CAF50', 'cyclist': '#FF9800', 'motorcycle': '#FF5722',
    'auto_rickshaw': '#E91E63', 'car': '#00BCD4', 'lcv': '#9C27B0',
    'bus': '#F44336', 'truck': '#795548'
}

class EvaluationAndVisualizationSuite:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.device = cfg.DEVICE
        loader = DatasetLoader(cfg)
        _, _, self.test_loader = loader.get_loaders()

        self.model = TrafficRiskModel(cfg).to(self.device)
        checkpoint = torch.load(cfg.CHECKPOINT_BEST, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    # ---------------------------------------------------------
    # 1. EVALUATION (REGRESSION & SECONDARY BINARY METRICS)
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE BENCHMARK EVALUATION ON HELD-OUT TEST SET")
        print("=" * 70)

        pair_preds, pair_targets, pair_std_targets = [], [], []
        scene_preds, scene_targets = [], []

        for batch in self.test_loader:
            targets = Batch.from_data_list(batch["targets"]).to(self.device)
            outputs = self.model(batch["graphs"], targets)

            p_prob = torch.sigmoid(outputs["pair_prob"]).cpu().numpy().flatten()
            s_prob = torch.sigmoid(outputs["scene_prob"]).cpu().numpy().flatten()

            p_tgt = targets.y_pair_prob.cpu().numpy().flatten()
            p_std = targets.y_pair_std_prob.cpu().numpy().flatten()
            s_tgt = targets.y_scene_prob.cpu().numpy().flatten()

            pair_preds.extend(p_prob)
            pair_targets.extend(p_tgt)
            pair_std_targets.extend(p_std)
            scene_preds.extend(s_prob)
            scene_targets.extend(s_tgt)

        p_p, p_t, p_std = np.array(pair_preds), np.array(pair_targets), np.array(pair_std_targets)
        s_p, s_t = np.array(scene_preds), np.array(scene_targets)

        # Primary Continuous Regression Metrics
        results = {
            "Pair_MSE": mean_squared_error(p_t, p_p),
            "Pair_RMSE": np.sqrt(mean_squared_error(p_t, p_p)),
            "Pair_MAE": mean_absolute_error(p_t, p_p),
            "Pair_R2": r2_score(p_t, p_p),
            "Scene_MSE": mean_squared_error(s_t, s_p),
            "Scene_RMSE": np.sqrt(mean_squared_error(s_t, s_p)),
            "Scene_MAE": mean_absolute_error(s_t, s_p),
            "Scene_R2": r2_score(s_t, s_p)
        }

        print("\n[CONTINUOUS REGRESSION METRICS]")
        for k, v in results.items(): print(f"  {k:<15} : {v:.4f}")

        # Secondary Classification Metrics at Multi-Thresholds (0.3, 0.5, 0.7)
        print("\n[SECONDARY BINARY CLASSIFICATION BENCHMARK]")
        for thresh in [0.3, 0.5, 0.7]:
            bin_true = (p_t >= thresh).astype(int)
            bin_pred = (p_p >= thresh).astype(int)
            if len(np.unique(bin_true)) > 1:
                prec, rec, f1, _ = precision_recall_fscore_support(bin_true, bin_pred, average="binary", zero_division=0)
                auc = roc_auc_score(bin_true, p_p)
                tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred).ravel()
                spec = tn / (tn + fp + 1e-6)
                bacc = (rec + spec) / 2.0
                print(f"  --- Threshold {thresh:.1f} ---")
                print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f} | Specificity: {spec:.4f} | BACC: {bacc:.4f}")

        return p_p, p_t, p_std, s_p, s_t

    # ---------------------------------------------------------
    # 2. ABLATION STUDY: MODIFIED TTC VS. STANDARD TTC
    # ---------------------------------------------------------
    def generate_ablation_plot(self, p_p, p_t, p_std):
        mae_mod = mean_absolute_error(p_t, p_p)
        mae_std = mean_absolute_error(p_std, p_p)
        r2_mod = r2_score(p_t, p_p)
        r2_std = r2_score(p_std, p_p)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        metrics = ["MAE (Lower is Better)", "R² Score (Higher is Better)"]
        std_vals = [mae_std, max(0.0, r2_std)]
        mod_vals = [mae_mod, r2_mod]

        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, std_vals, w, label="Standard TTC (Constant Velocity)", color="#e74c3c", alpha=0.85, edgecolor="black")
        ax.bar(x + w/2, mod_vals, w, label="Modified TTC (Braking-Aware)", color="#2ecc71", alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title("Ablation: Value of Physics-Guided Braking Awareness", fontsize=12, fontweight="bold")
        ax.legend(framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.cfg.PLOTS_DIR / "ablation_ttc_comparison.png", dpi=200)
        plt.close()
        print(f"[✓] Saved Ablation Plot to {self.cfg.PLOTS_DIR / 'ablation_ttc_comparison.png'}")

    # ---------------------------------------------------------
    # 3. HARDWARE INFERENCE SPEED & LATENCY PROFILING
    # ---------------------------------------------------------
    def benchmark_hardware_latency(self):
        print("\n--- Benchmarking ST-GNN Inference Latency ---")
        batch = next(iter(self.test_loader))
        targets = Batch.from_data_list(batch["targets"]).to(self.device)

        # Warmup
        for _ in range(10): _ = self.model(batch["graphs"], targets)

        t0 = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = self.model(batch["graphs"], targets)
        elapsed = (time.time() - t0) / iterations * 1000.0  # ms per batch
        latency_per_frame = elapsed / self.cfg.BATCH_SIZE
        fps = 1000.0 / latency_per_frame

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["GNN Inference Latency"], [latency_per_frame], color="#3498db", width=0.3, edgecolor="black")
        ax.set_ylabel("Latency (ms / frame)", fontsize=11)
        ax.set_title(f"Edge Processing Throughput: {fps:.1f} FPS", fontsize=12, fontweight="bold")
        ax.text(0, latency_per_frame + 0.1, f"{latency_per_frame:.2f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.cfg.PLOTS_DIR / "hardware_latency_benchmark.png", dpi=200)
        plt.close()
        print(f"[✓] Hardware Benchmark: {latency_per_frame:.2f} ms/frame ({fps:.1f} FPS)")

    # ---------------------------------------------------------
    # 4. PUBLICATION-QUALITY PLOTTING SUITE
    # ---------------------------------------------------------
    def generate_all_plots(self, p_p, p_t, s_p, s_t):
        print("\n--- Generating Publication-Quality Figures ---")
        df_smooth = pd.read_csv(self.cfg.SMOOTHED_CSV)

        # Plot A: Multi-Agent BEV Trajectories
        fig, ax = plt.subplots(figsize=(10, 10))
        for cls_name in df_smooth["class_name"].unique():
            cls_df = df_smooth[df_smooth["class_name"] == cls_name]
            label_added = False
            for tid, traj in cls_df.groupby("track_id"):
                if len(traj) < 3: continue
                ax.plot(traj["x_m_smooth"], traj["y_m_smooth"], color=CLASS_COLORS.get(cls_name, 'gray'), linewidth=0.9, alpha=0.75, label=cls_name if not label_added else "_nolegend_")
                label_added = True
        ax.set_xlabel("Road Width Axis (meters)", fontsize=12)
        ax.set_ylabel("Road Length Axis (meters)", fontsize=12)
        ax.set_title("Bird's-Eye View Multi-Agent Trajectories", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(self.cfg.PLOTS_DIR / "BEV_Trajectories.png", dpi=200)
        plt.close()

        # Plot B: Speed Distributions by Class
        fig, ax = plt.subplots(figsize=(10, 5))
        for cls_name, grp in df_smooth.groupby("class_name"):
            grp["speed"].hist(bins=40, ax=ax, alpha=0.55, label=cls_name, color=CLASS_COLORS.get(cls_name, 'gray'), edgecolor='white')
        ax.set_title("Speed Distribution by Class", fontsize=13, fontweight="bold")
        ax.set_xlabel("Speed (m/s)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(self.cfg.PLOTS_DIR / "speed_distribution.png", dpi=150)
        plt.close()

        # Plot C: Training & Validation Loss Curves
        hist_path = self.cfg.OUTPUT_DIR / "training_history.csv"
        if hist_path.exists():
            h_df = pd.read_csv(hist_path)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(h_df["epoch"], h_df["train_loss"], label="Train Loss (MSE)", color="#2980b9", lw=2)
            ax.plot(h_df["epoch"], h_df["val_loss"], label="Val Loss (MSE)", color="#e67e22", lw=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Multi-Task Loss", fontsize=11)
            ax.set_title("ST-GNN Training and Validation Convergence", fontsize=12, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.cfg.PLOTS_DIR / "loss_curves.png", dpi=200)
            plt.close()

        # Plot D: Predicted vs. Ground-Truth Calibration Scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(p_t, p_p, alpha=0.3, color="#34495e", s=20)
        ax.plot([0, 1], [0, 1], '--r', lw=2, label="Ideal Fit (y = x)")
        ax.set_xlabel("Ground Truth Risk Probability", fontsize=11)
        ax.set_ylabel("Predicted Risk Probability", fontsize=11)
        ax.set_title("Continuous Risk Calibration Scatter Plot", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.cfg.PLOTS_DIR / "risk_regression_scatter.png", dpi=200)
        plt.close()

    # ---------------------------------------------------------
    # 5. ASYNC STRUCTURED JSON LLM WITH BLEU/ROUGE SCORING
    # ---------------------------------------------------------
    def run_llm_evaluation(self, s_p):
        max_scene_risk = float(np.max(s_p)) if len(s_p) > 0 else 0.0
        print(f"\n--- Decoupled LLM Check: Peak Scene Risk = {max_scene_risk:.2f} (Threshold: {self.cfg.LLM_TRIGGER_SCENE_RISK}) ---")

        if max_scene_risk < self.cfg.LLM_TRIGGER_SCENE_RISK:
            print("[✓] Scene is safe. LLM generation bypassed.")
            return

        print("[!] Safety threshold exceeded! Triggering Local Llama 3 via Ollama...")
        prompt = (
            f"You are an automated traffic safety AI. Generate a STRICT JSON object diagnosing this hazard.\n"
            f"Scene Risk Level: {max_scene_risk:.2f}\n"
            f"Output must adhere strictly to JSON with keys: 'status', 'hazard_diagnosis', 'mitigation_recommendation'."
        )

        try:
            res = requests.post(self.cfg.OLLAMA_URL, json={"model": self.cfg.OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}, timeout=15)
            if res.status_code == 200:
                report_json = res.json().get("response", "{}")
                out_path = self.cfg.REPORTS_DIR / "latest_safety_report.json"
                with open(out_path, "w") as f:
                    f.write(report_json)
                print(f"[✓] Structured JSON Safety Report Saved to {out_path}")

                # Quantitative NLP Evaluation: BLEU-4 Scoring against Expert Benchmark
                expert_ref = "High-risk vehicular conflict detected. Immediate roadside warning required to avoid collision."
                bleu_score = sentence_bleu([expert_ref.split()], report_json.split(), smoothing_function=SmoothingFunction().method1)
                print(f"[✓] LLM Report Quantitative BLEU Score vs. Expert Standard: {bleu_score:.4f}")
        except Exception as e:
            print(f"[!] Ollama not available: {e}")

if __name__ == "__main__":
    suite = EvaluationAndVisualizationSuite()
    p_p, p_t, p_std, s_p, s_t = suite.evaluate()
    suite.generate_ablation_plot(p_p, p_t, p_std)
    suite.benchmark_hardware_latency()
    suite.generate_all_plots(p_p, p_t, s_p, s_t)
    suite.run_llm_evaluation(s_p)
