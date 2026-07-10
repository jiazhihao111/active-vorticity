import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from ..models.transformer import CausalTransformer
from ..models.memory_bank import CausalMemoryBank
from ..models.gauge_field import GaugeField as CausalGaugeField
from ..losses.causal_geometry import (
    CausalGeometryLoss,
    CombinedLoss,
    loop_back_contrastive_loss,
)
from ..losses.layered import rigid_flexible_layered_loss
from ..npnw.tokenizer import NPNWTokenizer
from ..npnw.story_generator import Story, StoryStep, step_layer_kind, step_is_violated
from ..utils.logger import setup_logger
from ..utils.metrics import discrete_curvature


class StoryDataset(Dataset):
    def __init__(self, stories: List[Story], tokenizer: NPNWTokenizer, max_seq_len: int = 64):
        self.stories = stories
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        story = self.stories[idx]
        steps_data = []
        for step in story.steps:
            steps_data.append({
                "state": step.state,
                "action": step.action,
                "causal_labels": step.causal_labels,
            })
        token_ids, step_ids = self.tokenizer.encode_story(steps_data, with_step_ids=True)
        token_ids = self.tokenizer.pad_sequence(token_ids, self.max_seq_len)
        # step_ids 与 token_ids 同长; 超出部分用 -1 哨兵, 对齐到 max_seq_len
        if len(step_ids) > self.max_seq_len:
            step_ids = step_ids[:self.max_seq_len]
        else:
            step_ids = step_ids + [-1] * (self.max_seq_len - len(step_ids))
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        # 刚柔分层掩码 (论文 §3.7 铁律八): 每 token 标注所属层与是否违规
        token_layer = torch.full((self.max_seq_len,), -1, dtype=torch.long)
        token_negative = torch.zeros(self.max_seq_len, dtype=torch.long)
        for pos, sid in enumerate(step_ids):
            if sid < 0:
                continue
            st = story.steps[sid]
            token_layer[pos] = int(step_layer_kind(st))
            token_negative[pos] = 1 if step_is_violated(st) else 0
        label = 1 if story.is_positive else 0
        violation = 0
        if story.violation_type == "physical":
            violation = 1
        elif story.violation_type == "narrative":
            violation = 2
        elif story.violation_type == "psychological":
            violation = 3
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "label": label,
            "violation_type": violation,
            "token_layer": token_layer,
            "token_negative": token_negative,
        }


class Trainer:
    def __init__(self, config: dict, model: CausalTransformer,
                 memory_bank: Optional[CausalMemoryBank] = None,
                 gauge_field: Optional["CausalGaugeField"] = None):
        self.config = config
        self.model = model
        self.memory_bank = memory_bank
        self.gauge_field = gauge_field
        self.loss_fn = CombinedLoss(config)
        # B-06/C-11: 规范场必须纳入优化器，否则 A_head/g/δ/η 永远停在随机初始化，
        # exp4 的 Wilson 环量退化为噪声，无法反映「叙事闭环/平坦化」。
        # 仅当 gauge_field 存在时才加入（baseline 训练不传 gauge_field，保持冻结）。
        opt_params = list(model.parameters())
        if self.gauge_field is not None:
            opt_params += list(self.gauge_field.parameters())
        self.optimizer = torch.optim.AdamW(
            opt_params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.logger = setup_logger("Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.memory_bank:
            self.memory_bank.to(self.device)
        if self.gauge_field:
            self.gauge_field.to(self.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
        lambda_value: float = 0.0,
        closure_lambda: float = 0.0,
        closure_contrastive_lambda: float = 0.0,
        closure_margin: float = 2.0,
        layered_rf_lambda: float = 0.0,
        rf_tau: float = 0.5,
        rf_lambda_phys: float = 1.0,
        rf_lambda_flex: float = 1.0,
        rf_margin: float = 0.3,
    ) -> Dict[str, float]:
        self.model.train()
        if self.gauge_field is not None:
            self.gauge_field.train()
        total_lm_loss = 0.0
        total_causal_loss = 0.0
        total_loss = 0.0
        total_closure_loss = 0.0
        total_closure_contrastive = 0.0
        total_rf_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            logits, hidden = self.model(input_ids, attention_mask)
            if lambda_value > 0 and self.memory_bank is not None:
                # B-04: 由规范场派生 G^curve，组合进 G^eff
                G_field = self.gauge_field(hidden) if self.gauge_field is not None else None
                G_composed, _ = self.memory_bank(hidden, G_curve=G_field)   # (B, db, db)
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)
                if pos_mask.any() and neg_mask.any():
                    Tm1 = hidden.size(1) - 1
                    db = hidden.size(-1)
                    # 正例 (h_t, h_t1) 对：展平所有时间步 -> (N, db)
                    h_t = hidden[pos_mask, :-1, :].reshape(-1, db)          # (N, db)
                    h_t1_pos = hidden[pos_mask, 1:, :].reshape(-1, db)      # (N, db)
                    h_t1_neg = hidden[neg_mask, 1:, :].reshape(-1, db)      # (M, db)
                    # memory_bank 返回每序列度量 (N_pos, db, db)，需展开到每个时间步对 (N, db, db)
                    # 以匹配 CausalGeometryLoss 的 G_composed 契约（N 与 h_t 一致）。
                    G_pos_seq = G_composed[pos_mask]                        # (N_pos, db, db)
                    G_per_pair = (G_pos_seq.unsqueeze(1)
                                  .expand(-1, Tm1, db, db)
                                  .reshape(-1, db, db))                     # (N, db, db)
                    N = h_t.size(0)
                    M = h_t1_neg.size(0)
                    if N > 0 and M > 0:
                        # 每个正例对取 K 个负例；负例池不足时随机复用（允许重复）
                        K = max(1, min(M // N, M))
                        sel = torch.randint(0, M, (N, K), device=self.device)
                        h_t1_neg_sel = h_t1_neg[sel]                 # (N, K, db)
                        indices = torch.randperm(N, device=self.device)[:min(N, 128)]
                        loss_dict = self.loss_fn(
                            logits, target_ids,
                            h_t[indices], h_t1_pos[indices], h_t1_neg_sel[indices],
                            G_composed=G_per_pair[indices],
                            lambda_value=lambda_value,
                        )
                    else:
                        loss_dict = self.loss_fn(logits, target_ids, lambda_value=lambda_value)
                else:
                    loss_dict = self.loss_fn(logits, target_ids, lambda_value=lambda_value)
            else:
                loss_dict = self.loss_fn(logits, target_ids)
            loss = loss_dict["total_loss"]
            # ---- 闭合敏感训练信号 (挽救路径实验用, 默认 closure_lambda=0 不生效) ----
            # 对正例(闭环叙事)惩罚 ‖h_T - h_0‖, 迫使模型学会「闭环叙事回到起点」.
            # 若加此信号后 exp6 出现 闭环<破缺, 说明机制可用、只是原损失未提供闭合激励;
            # 若仍无差异, 则规范场机制根本无法编码叙事闭合, C-11 应退役.
            closure_loss_val = torch.tensor(0.0, device=self.device)
            if closure_lambda > 0:
                pos_mask_local = (labels == 1)
                if pos_mask_local.any():
                    ph = hidden[pos_mask_local]                  # (N, T, db)
                    h0 = ph[:, 0, :]
                    hT = ph[:, -1, :]
                    closure_loss_val = ((hT - h0) ** 2).mean()
                    loss = loss + closure_lambda * closure_loss_val
            # ---- 回环 holonomy 对比训练信号 (实验7, §10.7 任务2) ----
            # 用标签(闭环/破缺)作监督: 正例拉平、负例推离 (差异化, 非统一压缩).
            # 仅当 gauge_field 存在且 closure_contrastive_lambda>0 时生效.
            closure_contrastive_val = torch.tensor(0.0, device=self.device)
            if closure_contrastive_lambda > 0 and self.gauge_field is not None:
                with torch.set_grad_enabled(True):
                    flat, _ = self.gauge_field.loop_back_holonomy_flatness(hidden)
                cc_loss, _ = loop_back_contrastive_loss(flat, labels, margin=closure_margin)
                closure_contrastive_val = cc_loss
                loss = loss + closure_contrastive_lambda * cc_loss
            # ---- 刚柔分层几何损失 (论文 §3.7 铁律八·九, 新分层对比体制) ----
            # 物理层: 无阈值惩罚 (须平坦); 柔性层: 带阈值 + 负例推离 (差异化信号).
            rf_val = torch.tensor(0.0, device=self.device)
            if layered_rf_lambda > 0 and self.gauge_field is not None:
                with torch.set_grad_enabled(True):
                    curv = self.gauge_field.per_transition_connection_norm(hidden)  # (B, Tm1)
                Tm1 = curv.size(1)
                layer = batch["token_layer"][:, 1:Tm1 + 1].long().to(self.device)   # (B, Tm1)
                neg = batch["token_negative"][:, 1:Tm1 + 1].bool().to(self.device)
                attn = batch["attention_mask"][:, 1:Tm1 + 1].bool().to(self.device)
                cv = curv[attn]
                ly = layer[attn]
                ng = neg[attn]
                if cv.numel() > 0:
                    rf_loss, _ = rigid_flexible_layered_loss(
                        cv, ly, ng, rf_tau, rf_lambda_phys,
                        rf_lambda_flex, rf_margin)
                    rf_val = rf_loss
                    loss = loss + layered_rf_lambda * rf_loss
            self.optimizer.zero_grad()
            loss.backward()
            clip_params = list(self.model.parameters())
            if self.gauge_field is not None:
                clip_params += list(self.gauge_field.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, self.config["training"]["grad_clip"])
            self.optimizer.step()
            total_lm_loss += loss_dict["lm_loss"].item()
            total_causal_loss += loss_dict.get("causal_loss", torch.tensor(0.0)).item()
            total_loss += loss.item()
            total_closure_loss += closure_loss_val.item()
            total_closure_contrastive += closure_contrastive_val.item()
            total_rf_loss += rf_val.item()
            num_batches += 1
        return {
            "lm_loss": total_lm_loss / max(num_batches, 1),
            "causal_loss": total_causal_loss / max(num_batches, 1),
            "closure_loss": total_closure_loss / max(num_batches, 1),
            "closure_contrastive_loss": total_closure_contrastive / max(num_batches, 1),
            "rf_loss": total_rf_loss / max(num_batches, 1),
            "total_loss": total_loss / max(num_batches, 1),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits, _ = self.model(input_ids, attention_mask)
                loss_dict = self.loss_fn(logits, target_ids)
                total_loss += loss_dict["total_loss"].item()
                num_batches += 1
        return {"val_loss": total_loss / max(num_batches, 1)}

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lambda_value: float = 0.0,
        closure_lambda: float = 0.0,
        closure_contrastive_lambda: float = 0.0,
        closure_margin: float = 2.0,
        layered_rf_lambda: float = 0.0,
        rf_tau: float = 0.5,
        rf_lambda_phys: float = 1.0,
        rf_lambda_flex: float = 1.0,
        rf_margin: float = 0.3,
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lm_loss": [], "causal_loss": [],
                   "closure_loss": [], "closure_contrastive_loss": [], "rf_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        max_epochs = self.config["training"]["max_epochs"]
        patience = self.config["training"]["patience"]
        for epoch in range(max_epochs):
            train_metrics = self.train_epoch(
                train_loader, lambda_value, closure_lambda,
                closure_contrastive_lambda, closure_margin,
                layered_rf_lambda, rf_tau, rf_lambda_phys,
                rf_lambda_flex, rf_margin,
            )
            val_metrics = self.evaluate(val_loader)
            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["lm_loss"].append(train_metrics["lm_loss"])
            history["causal_loss"].append(train_metrics["causal_loss"])
            history["closure_loss"].append(train_metrics.get("closure_loss", 0.0))
            history["closure_contrastive_loss"].append(
                train_metrics.get("closure_contrastive_loss", 0.0))
            history["rf_loss"].append(train_metrics.get("rf_loss", 0.0))
            self.logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"LM: {train_metrics['lm_loss']:.4f} | "
                f"Causal: {train_metrics['causal_loss']:.4f} | "
                f"Closure: {train_metrics.get('closure_loss', 0.0):.4f} | "
                f"ClosureContr: {train_metrics.get('closure_contrastive_loss', 0.0):.4f} | "
                f"RigidFlex: {train_metrics.get('rf_loss', 0.0):.4f}"
            )
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        return history