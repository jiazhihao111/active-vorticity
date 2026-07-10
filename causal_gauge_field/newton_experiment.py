import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.newton.causal_potential import CausalPotential
from causal_gauge_field.newton.acceleration import AccelerationLoss, BarrierLoss
from causal_gauge_field.newton.hamiltonian import HamiltonianVerifier
from causal_gauge_field.experiments.trainer import StoryDataset


class NewtonTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("NewtonTrainer")
        self.tokenizer = NPNWTokenizer()
        config["model"]["vocab_size"] = self.tokenizer.vocab_size
        self.max_seq_len = config["model"]["max_seq_len"]
        self.hidden_dim = config["model"].get("base_dim", config["model"]["d_model"])

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def train_baseline(self, train_loader, val_loader, epochs=20):
        model = CausalTransformer(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["training"]["learning_rate"])
        lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.logger.info(f"基线模型参数量: {model.count_parameters()}")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits, _ = model(input_ids, attention_mask)
                loss = lm_loss_fn(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["training"]["grad_clip"])
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"  基线 Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader):.4f}")
        return model

    def train_newton(self, train_loader, val_loader, epochs=20, lambda_accel=5.0, lambda_barrier=1.0):
        model = CausalTransformer(self.config).to(self.device)
        potential = CausalPotential(self.hidden_dim).to(self.device)
        accel_loss_fn = AccelerationLoss(margin=self.config["training"]["margin"])
        barrier_loss_fn = BarrierLoss(margin=0.5)
        all_params = list(model.parameters()) + list(potential.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.config["training"]["learning_rate"])
        lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.logger.info(f"牛顿模型参数量: {sum(p.numel() for p in all_params)}")
        for epoch in range(epochs):
            model.train()
            potential.train()
            total_loss = 0
            total_lm = 0
            total_accel = 0
            total_barrier = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                logits, hidden = model(input_ids, attention_mask)
                lm_loss = lm_loss_fn(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)
                accel_loss_val = torch.tensor(0.0, device=self.device)
                barrier_loss_val = torch.tensor(0.0, device=self.device)
                if pos_mask.any() and neg_mask.any():
                    hidden_pos = hidden[pos_mask]
                    hidden_neg = hidden[neg_mask]
                    if hidden_pos.size(1) >= 3 and hidden_neg.size(1) >= 3:
                        accel_loss_val, accel_info = accel_loss_fn(hidden_pos, hidden_neg)
                        h_t_pos = hidden_pos[:, :-1, :].reshape(-1, self.hidden_dim)
                        h_t1_pos = hidden_pos[:, 1:, :].reshape(-1, self.hidden_dim)
                        h_t_neg = hidden_neg[:, :-1, :].reshape(-1, self.hidden_dim)
                        h_t1_neg = hidden_neg[:, 1:, :].reshape(-1, self.hidden_dim)
                        min_len = min(h_t_pos.size(0), h_t_neg.size(0))
                        if min_len > 0:
                            idx = torch.randperm(min_len, device=self.device)[:min(min_len, 64)]
                            V_t = potential(h_t_pos[idx])
                            V_t1_pos = potential(h_t1_pos[idx])
                            V_t1_neg = potential(h_t1_neg[idx])
                            barrier_loss_val, _ = barrier_loss_fn(V_t, V_t1_pos, V_t1_neg)
                total_loss_val = lm_loss + lambda_accel * accel_loss_val + lambda_barrier * barrier_loss_val
                optimizer.zero_grad()
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(all_params, self.config["training"]["grad_clip"])
                optimizer.step()
                total_loss += total_loss_val.item()
                total_lm += lm_loss.item()
                total_accel += accel_loss_val.item()
                total_barrier += barrier_loss_val.item()
            n = len(train_loader)
            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"  牛顿 Epoch {epoch+1}/{epochs} "
                    f"Total: {total_loss/n:.4f} LM: {total_lm/n:.4f} "
                    f"Accel: {total_accel/n:.4f} Barrier: {total_barrier/n:.4f}"
                )
        return model, potential

    def verify_newton(self, model, potential, test_stories):
        self.logger.info("=" * 60)
        self.logger.info("牛顿版本验证: 平直空间 + 势函数 + 加速度惩罚")
        self.logger.info("=" * 60)
        model.eval()
        potential.eval()
        verifier = HamiltonianVerifier(potential)

        pos_hidden = []
        neg_hidden = []

        with torch.no_grad():
            for story in test_stories:
                steps_data = []
                for step in story.steps:
                    steps_data.append({
                        "state": step.state,
                        "action": step.action,
                        "causal_labels": step.causal_labels,
                    })
                token_ids = self.tokenizer.encode_story(steps_data)
                token_ids = self._truncate_tokens(token_ids)
                if len(token_ids) < 4:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)
                h = hidden[0]
                if story.is_positive:
                    pos_hidden.append(h)
                else:
                    neg_hidden.append(h)

        results = verifier.full_verification(pos_hidden, neg_hidden)

        accel_pos_list = []
        accel_neg_list = []
        for h in pos_hidden:
            if h.size(0) >= 3:
                v1 = h[1:] - h[:-1]
                v2 = h[2:] - h[1:-1]
                accel = v2 - v1[:-1] if v1.size(0) > 1 else v2
                accel_pos_list.extend((accel ** 2).sum(dim=-1).tolist())
        for h in neg_hidden:
            if h.size(0) >= 3:
                v1 = h[1:] - h[:-1]
                v2 = h[2:] - h[1:-1]
                accel = v2 - v1[:-1] if v1.size(0) > 1 else v2
                accel_neg_list.extend((accel ** 2).sum(dim=-1).tolist())

        if accel_pos_list and accel_neg_list:
            pos_mean = np.mean(accel_pos_list)
            neg_mean = np.mean(accel_neg_list)
            t_stat, p_val = stats.ttest_ind(accel_pos_list, accel_neg_list)
            flatter = pos_mean < neg_mean
            results["acceleration_flatness"] = {
                "accel_pos_mean": float(pos_mean),
                "accel_neg_mean": float(neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_flatter": flatter,
                "verdict": "SUPPORT" if flatter and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  加速度平坦化: pos={pos_mean:.4f}, neg={neg_mean:.4f}, "
                f"flatter={flatter}, p={p_val:.4f}"
            )

        for key in ["hamiltonian_conservation", "force_alignment", "potential_barrier"]:
            if key in results:
                v = results[key]
                main_key = [k for k in v.keys() if "mean" in k and "pos" in k.lower()]
                main_key2 = [k for k in v.keys() if "mean" in k and "neg" in k.lower()]
                self.logger.info(
                    f"  {key}: {v.get(main_key[0], 'N/A') if main_key else 'N/A'} vs "
                    f"{v.get(main_key2[0], 'N/A') if main_key2 else 'N/A'}, "
                    f"verdict={v.get('verdict', 'N/A')}"
                )

        overall = results.get("overall_verdict", {})
        self.logger.info("=" * 60)
        self.logger.info(f"牛顿版本总判决: {overall.get('verdict', 'N/A')} "
                         f"({overall.get('support_count', 0)}/{overall.get('total_count', 0)})")
        self.logger.info("=" * 60)
        return results


def main():
    config = load_config()
    config["data"]["num_stories"] = 800
    config["training"]["max_epochs"] = 20
    config["training"]["patience"] = 8

    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    logger = setup_logger("NewtonMain", config["logging"]["log_dir"], "newton_experiment.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("牛顿版本实验: 平直空间 + 势函数 + 加速度惩罚")
    logger.info("=" * 60)

    logger.info("[阶段0] 数据生成...")
    gen = StoryGenerator(config, seed=config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )
    logger.info(f"  正例: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    all_train = train_pos + train_neg
    all_val = val_pos + val_neg
    train_ds = StoryDataset(all_train, tokenizer, config["model"]["max_seq_len"])
    val_ds = StoryDataset(all_val, tokenizer, config["model"]["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    trainer = NewtonTrainer(config)

    logger.info("[阶段1] 训练基线模型...")
    baseline_model = trainer.train_baseline(train_loader, val_loader, epochs=20)

    logger.info("[阶段2] 训练牛顿版本模型...")
    newton_model, potential = trainer.train_newton(
        train_loader, val_loader, epochs=20, lambda_accel=5.0, lambda_barrier=1.0
    )

    logger.info("[阶段3] 牛顿版本验证...")
    test_stories = test_pos[:40] + test_neg[:40]
    results = trainer.verify_newton(newton_model, potential, test_stories)

    output_dir = Path(config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "Newton (flat space + potential)",
        "results": results,
    }
    report_path = output_dir / "newton_experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    md_lines = [
        "# 牛顿版本实验报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n## 框架",
        f"\n> **平直空间 + 因果势函数 + 加速度惩罚**",
        "\n---\n",
        "## 验证结果\n",
        "| 假设 | 正例均值 | 负例均值 | 方向正确 | p值 | 判决 |",
        "|------|----------|----------|----------|-----|------|",
    ]
    hypothesis_names = {
        "hamiltonian_conservation": ("哈密顿量守恒", "H_var_pos_mean", "H_var_neg_mean", "legal_more_conserved"),
        "force_alignment": ("力场对齐", "alignment_pos_mean", "alignment_neg_mean", "legal_more_aligned"),
        "potential_barrier": ("势垒分离", "delta_V_pos_mean", "delta_V_neg_mean", "legal_downhill"),
        "acceleration_flatness": ("加速度平坦化", "accel_pos_mean", "accel_neg_mean", "legal_flatter"),
    }
    for key, (name, pos_key, neg_key, dir_key) in hypothesis_names.items():
        v = results.get(key, {})
        if v:
            pos_val = v.get(pos_key, 0)
            neg_val = v.get(neg_key, 0)
            p_val = v.get("p_value", 1)
            pos_str = f"{pos_val:.4f}" if isinstance(pos_val, (int, float)) else str(pos_val)
            neg_str = f"{neg_val:.4f}" if isinstance(neg_val, (int, float)) else str(neg_val)
            p_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
            md_lines.append(
                f"| {name} | {pos_str} | {neg_str} | "
                f"{v.get(dir_key, 'N/A')} | {p_str} | {v.get('verdict', 'N/A')} |"
            )

    overall = results.get("overall_verdict", {})
    md_lines.append(f"\n**总判决**: {overall.get('verdict', 'N/A')} "
                    f"({overall.get('support_count', 0)}/{overall.get('total_count', 0)})")
    md_lines.append(f"\n**升级条件**: 若3个假设全部通过，考虑升级为爱因斯坦版本（弯曲空间+规范场）")

    md_path = output_dir / "newton_experiment_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return results


if __name__ == "__main__":
    main()