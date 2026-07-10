#!/usr/bin/env python3
"""exp10b 定理二验证（加载已训练模型）"""
import sys, os, warnings
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import torch
import numpy as np
from scipy import stats as scipy_stats

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.npnw.enhanced_generator import EnhancedClosureGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.npnw.story_generator import Story
from causal_gauge_field.utils.frenet_serret import FrenetSerretAnalyzer
from causal_gauge_field.experiments.exp10_geometric_dynamics import create_high_dim_model
from causal_gauge_field.theorems.theorem_2_conservation import TheoremConservation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJ_ROOT = Path(__file__).parent
config = load_config(PROJ_ROOT / "causal_gauge_field" / "config.yaml")

# ── 1. Load model ──
model = create_high_dim_model(config, base_dim=128, max_seq_len=256).to(DEVICE)
ckpt = torch.load(
    PROJ_ROOT / "exp10_outputs" / "exp10b_model.pt",
    map_location=DEVICE, weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {n_params:,} params")

# ── 2. Generate test data ──
gen_cfg = dict(config)
gen_cfg["data"] = dict(config["data"])
gen_cfg["data"]["enh_min_steps"] = 6
gen_cfg["data"]["enh_max_steps"] = 12
generator = EnhancedClosureGenerator(gen_cfg, seed=777)
(_, _), (_, _), (test_pos, test_neg) = generator.generate_dataset(num_stories=200)
positive_test = [s for s in test_pos if s.is_positive]
if len(positive_test) == 0:
    positive_test = test_pos[:20]
    print(f"(fallback: using all test_pos)")
print(f"Positive test stories: {len(positive_test)}")

# ── 3. Story variant generation ──
def shuffle_story_steps(story, block_size, seed=None):
    rng = np.random.RandomState(seed)
    steps = list(story.steps)
    n = len(steps)
    if n < block_size * 2:
        return story
    n_blocks = max(2, n // block_size)
    actual_block_size = n // n_blocks
    blocks = [steps[i:i + actual_block_size] for i in range(0, n, actual_block_size)]
    blocks = [b for b in blocks if len(b) > 0]
    if len(blocks) < 2:
        return story
    indices = list(range(len(blocks)))
    rng.shuffle(indices)
    shuffled_blocks = [blocks[i] for i in indices]
    shuffled_steps = []
    for block in shuffled_blocks:
        shuffled_steps.extend(block)
    return Story(
        story.story_id, shuffled_steps, story.personality,
        story.is_positive, story.violation_type, deepcopy(story.causal_graph),
    )

N = min(20, len(positive_test))
variants = {"original": [], "sentence": [], "paragraph": []}
for i in range(N):
    story = positive_test[i]
    n_steps = len(story.steps)
    variants["original"].append(story)
    variants["sentence"].append(shuffle_story_steps(story, 3, seed=42 + i))
    variants["paragraph"].append(shuffle_story_steps(story, max(8, n_steps // 4), seed=99 + i))
print(f"Variants: original={len(variants['original'])}, sentence={len(variants['sentence'])}, paragraph={len(variants['paragraph'])}")

# ── 4. Extract hidden states and tanTheta ──
tokenizer = NPNWTokenizer()
fs_analyzer = FrenetSerretAnalyzer(eps=1e-8)
MAX_SEQ_LEN = 256

def extract_tan_theta(stories):
    hidden_list = []
    for story in stories:
        try:
            steps_data = [
                {"state": s.state, "action": s.action, "causal_labels": s.causal_labels}
                for s in story.steps
            ]
            token_ids = tokenizer.encode_story(steps_data)
            if len(token_ids) > MAX_SEQ_LEN:
                token_ids = token_ids[:MAX_SEQ_LEN]
            if len(token_ids) < 4:
                continue
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                _, hidden = model(input_ids)
            hidden_list.append(hidden[0])
        except Exception:
            continue
    if not hidden_list:
        return None
    max_T = min(max(h.size(0) for h in hidden_list), MAX_SEQ_LEN)
    padded = []
    for h in hidden_list:
        T = h.size(0)
        if T < max_T:
            pad = torch.zeros(max_T - T, h.size(1), device=h.device)
            h = torch.cat([h, pad], dim=0)
        else:
            h = h[:max_T]
        padded.append(h)
    batch = torch.stack(padded, dim=0)
    result = fs_analyzer.analyze(batch, compute_chern=False)
    return result.tan_theta.flatten().detach().cpu().numpy()

print("Extracting tanTheta for 3 variants...")
tan_orig = extract_tan_theta(variants["original"])
tan_sent = extract_tan_theta(variants["sentence"])
tan_para = extract_tan_theta(variants["paragraph"])

def cv_and_mean(tan):
    if tan is None:
        return float("nan"), float("nan")
    valid = np.isfinite(tan)
    t = tan[valid]
    if len(t) < 2:
        return float("nan"), float("nan")
    m = float(np.mean(t))
    s = float(np.std(t, ddof=1))
    cv = s / m if m > 0 else float("inf")
    return cv, m

cv_orig, mean_orig = cv_and_mean(tan_orig)
cv_sent, mean_sent = cv_and_mean(tan_sent)
cv_para, mean_para = cv_and_mean(tan_para)

print(f"\nCV(original)  = {cv_orig:.4f}  mean={mean_orig:.4f}")
print(f"CV(sentence)  = {cv_sent:.4f}  mean={mean_sent:.4f}")
print(f"CV(paragraph) = {cv_para:.4f}  mean={mean_para:.4f}")

# ── 5. Per-story tanTheta means (for paired t-test) ──
def per_story_tan_mean(stories):
    means = []
    for story in stories:
        tan = extract_tan_theta([story])
        if tan is not None and len(tan) > 0:
            valid = np.isfinite(tan)
            if valid.sum() > 0:
                means.append(float(np.mean(tan[valid])))
    return np.array(means), len(means)

orig_means, n_orig = per_story_tan_mean(variants["original"])
sent_means, n_sent = per_story_tan_mean(variants["sentence"])
para_means, n_para = per_story_tan_mean(variants["paragraph"])

min_n = min(n_orig, n_sent, n_para)
orig_means = orig_means[:min_n]
sent_means = sent_means[:min_n]
para_means = para_means[:min_n]

print(f"\nPer-story paired N = {min_n}")

if min_n >= 5:
    t_s, p_s = scipy_stats.ttest_rel(sent_means, orig_means)
    p_s = float(p_s / 2)  # one-sided
    t_p, p_p = scipy_stats.ttest_rel(para_means, orig_means)
    p_p = float(p_p / 2)
    d_s = float(
        (np.mean(sent_means) - np.mean(orig_means)) /
        (np.sqrt((np.var(sent_means) + np.var(orig_means)) / 2) + 1e-10)
    )
    d_p = float(
        (np.mean(para_means) - np.mean(orig_means)) /
        (np.sqrt((np.var(para_means) + np.var(orig_means)) / 2) + 1e-10)
    )
    print(f"Paired t (orig vs sent): t={t_s:.3f}, p={p_s:.4f} {'***' if p_s<0.01 else '*' if p_s<0.05 else ''}")
    print(f"Paired t (orig vs para): t={t_p:.3f}, p={p_p:.4f} {'***' if p_p<0.01 else '*' if p_p<0.05 else ''}")
    print(f"Cohen d (sentence): {d_s:.3f}, (paragraph): {d_p:.3f}")
else:
    t_s = p_s = t_p = p_p = d_s = d_p = 0
    print("Insufficient samples for paired t-test")

# ── 6. Per-story CV comparison ──
def per_story_cv(stories):
    cvs = []
    for story in stories:
        tan = extract_tan_theta([story])
        if tan is not None and len(tan) > 0:
            valid = np.isfinite(tan)
            t = tan[valid]
            if len(t) > 1:
                m = np.mean(t)
                s = np.std(t, ddof=1)
                if m > 0:
                    cvs.append(float(s / m))
    return np.array(cvs)

orig_cvs_arr = per_story_cv(variants["original"])
sent_cvs_arr = per_story_cv(variants["sentence"])
para_cvs_arr = per_story_cv(variants["paragraph"])
print(f"\nPer-story CV means: orig={np.mean(orig_cvs_arr):.4f}, sent={np.mean(sent_cvs_arr):.4f}, para={np.mean(para_cvs_arr):.4f}")

# ── 7. Sliding window test ──
conserv = TheoremConservation(window_size=8)
window_result = conserv.sliding_window_test(tan_orig) if tan_orig is not None else {}
print(f"\nSliding window:")
print(f"  local_cv_mean = {window_result.get('local_cv_mean', 'N/A')}")
print(f"  cv_ratio = {window_result.get('cv_ratio', 'N/A')}")
print(f"  pt_residual = {window_result.get('parallel_transport_residual', 'N/A')}")
print(f"  verdict = {window_result.get('verdict', 'N/A')}")

# ── 8. Verdict ──
gradient_ok = (
    not np.isnan(cv_orig) and not np.isnan(cv_sent) and not np.isnan(cv_para)
    and cv_orig < cv_sent < cv_para
)
per_story_gradient = (
    np.mean(orig_cvs_arr) < np.mean(sent_cvs_arr) < np.mean(para_cvs_arr)
    if len(orig_cvs_arr) > 0 and len(sent_cvs_arr) > 0 and len(para_cvs_arr) > 0
    else False
)

confidence = 0.0
if gradient_ok:
    confidence += 0.30
if per_story_gradient:
    confidence += 0.10
if p_s < 0.05:
    confidence += 0.25
if p_p < 0.05:
    confidence += 0.25
cv_ratio = window_result.get("cv_ratio", 999)
if cv_ratio < 0.8:
    confidence += 0.10

verdict = "SUPPORT" if confidence >= 0.70 else "WEAK" if confidence >= 0.40 else "INCONCLUSIVE"

print(f"\n{'='*60}")
print(f"VERDICT: {verdict} (confidence={confidence:.0%})")
print(f"  Gradient (batch): {gradient_ok}")
print(f"  Gradient (per-story): {per_story_gradient}")
print(f"  Paired p (sent): {p_s:.4f}")
print(f"  Paired p (para): {p_p:.4f}")
print(f"  Window CV ratio: {cv_ratio}")
print(f"{'='*60}")

# ── 9. Summary for diagnosis ──
print(f"\nDIAGNOSIS:")
print(f"  tan_orig n_valid: {np.isfinite(tan_orig).sum() if tan_orig is not None else 0}")
print(f"  tan_orig range: [{np.nanmin(tan_orig):.2f}, {np.nanmax(tan_orig):.2f}]")
print(f"  N_per_story (orig/sent/para): {n_orig}/{n_sent}/{n_para}")
