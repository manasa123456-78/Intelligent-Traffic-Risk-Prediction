import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torch_geometric.data import Batch
from tqdm import tqdm
from global_variables_02 import CFG
from dataset_loader import DatasetLoader
from models.traffic_risk_model import TrafficRiskModel

class TrafficRiskLoss(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_pair = cfg.PAIR_LOSS_WEIGHT
        self.lambda_scene = cfg.SCENE_LOSS_WEIGHT

    def forward(self, predictions, target_batch):
        pair_prob = torch.sigmoid(predictions["pair_prob"])
        scene_prob = torch.sigmoid(predictions["scene_prob"])

        pair_target = target_batch.y_pair_prob.to(pair_prob.device, dtype=torch.float32)
        scene_target = target_batch.y_scene_prob.to(scene_prob.device, dtype=torch.float32)

        pair_loss = self.mse(pair_prob.view(-1), pair_target.view(-1)) if len(pair_target) > 0 else torch.tensor(0.0, device=pair_prob.device)
        scene_loss = self.mse(scene_prob.view(-1), scene_target.view(-1))

        return {
            "total_loss": (self.lambda_pair * pair_loss) + (self.lambda_scene * scene_loss),
            "pair_loss": pair_loss,
            "scene_loss": scene_loss
        }

class TrafficRiskTrainer:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.device = cfg.DEVICE
        print(f"Initializing Trainer on Device: {self.device}")

        loader = DatasetLoader(cfg)
        self.train_loader, self.val_loader, self.test_loader = loader.get_loaders()

        self.model = TrafficRiskModel(cfg).to(self.device)
        self.criterion = TrafficRiskLoss(cfg).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_pair, total_scene = 0.0, 0.0, 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]")

        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            target_graphs = Batch.from_data_list(batch["targets"]).to(self.device)

            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(batch["graphs"], target_graphs)
                losses = self.criterion(outputs, target_graphs)
                loss = losses["total_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_pair += losses["pair_loss"].item()
            total_scene += losses["scene_loss"].item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "pair": f"{losses['pair_loss'].item():.4f}", "scene": f"{losses['scene_loss'].item():.4f}"})

        n = max(len(self.train_loader), 1)
        return total_loss / n, total_pair / n, total_scene / n

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss, total_pair, total_scene = 0.0, 0.0, 0.0
        for batch in self.val_loader:
            target_graphs = Batch.from_data_list(batch["targets"]).to(self.device)
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(batch["graphs"], target_graphs)
                losses = self.criterion(outputs, target_graphs)
            total_loss += losses["total_loss"].item()
            total_pair += losses["pair_loss"].item()
            total_scene += losses["scene_loss"].item()

        n = max(len(self.val_loader), 1)
        return total_loss / n, total_pair / n, total_scene / n

    def train(self):
        print("\n" + "=" * 70)
        print(f"Starting ST-GNN Training (Max {self.cfg.EPOCHS} Epochs | Patience: {self.cfg.EARLY_STOPPING_PATIENCE})")
        print("=" * 70)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            tr_tot, tr_pair, tr_scene = self.train_epoch(epoch)
            va_tot, va_pair, va_scene = self.val_epoch(epoch)
            self.scheduler.step(va_tot)

            self.history.append({
                "epoch": epoch,
                "train_loss": tr_tot, "train_pair_loss": tr_pair, "train_scene_loss": tr_scene,
                "val_loss": va_tot, "val_pair_loss": va_pair, "val_scene_loss": va_scene
            })

            is_best = va_tot < self.best_val_loss
            if is_best:
                self.best_val_loss = va_tot
                self.patience_counter = 0
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_loss": va_tot}, self.cfg.CHECKPOINT_BEST)
                tag = "⭐ (Best Saved)"
            else:
                self.patience_counter += 1
                tag = f"(Patience: {self.patience_counter}/{self.cfg.EARLY_STOPPING_PATIENCE})"

            print(f"Summary Ep {epoch:03d} | Train: {tr_tot:.4f} | Val: {va_tot:.4f} {tag}")

            if self.patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] Early stopping triggered at Epoch {epoch}. Halting training.")
                break

        # Save training metrics log
        history_path = self.cfg.OUTPUT_DIR / "training_history.csv"
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        print(f"[✓] Training History saved to {history_path}")

if __name__ == "__main__":
    trainer = TrafficRiskTrainer()
    trainer.train()
