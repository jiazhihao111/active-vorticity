#!/usr/bin/env python3
"""exp10b per-story CV analysis"""
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJ = Path(__file__).parent
config = load_config(PROJ / "causal_gauge_field" / "config.yaml")
model = create_high_dim_model(config, base_dim=128, max_seq_len=256).to(DEVICE)
ckpt = torch.load(PROJ / "exp10_outputs/exp10b_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

gen_cfg = dict(config)
gen_cfg["data"] = dict(config["data"])
gen_cfg["data"]["enh_min_steps"] = 6
gen_cfg["data"]["enh_max_steps"] = 12
generator = EnhancedClosureGenerator(gen_cfg, seed=777)
(_, _), (_, _), (test_pos, test_neg) = generator.generate_dataset(num_stories=200)
positive_test = [s for s in test_pos if s.is_positive]

def shuffle_story_steps(story, block_size, seed=None):
    rng = np.random.RandomState(seed)
    steps = list(story.steps)
    n = len(steps)
    if n < block_size * 2:
        return story
    n_blocks = max(2, n // block_size)
    actual_block_size = n // n_blocks
    blocks = [steps[i:i+actual_block_size] for i in range(0,n,actual_block_size)]
    blocks = [b for b in blocks if len(b) > 0]
    if len(blocks) < 2:
        return story
    indices = list(range(len(blocks)))
    rng.shuffle(indices)
    shuffled_steps = []
    for idx in indices:
        shuffled_steps.extend(blocks[idx])
    return Story(story.story_id, shuffled_steps, story.personality, story.is_positive, story.violation_type, deepcopy(story.causal_graph))

tokenizer = NPNWTokenizer()
fs_analyzer = FrenetSerretAnalyzer(eps=1e-8)
MAX_LEN = 256

def extract_tan(story):
    try:
        steps_data = [{"state": s.state, "action": s.action, "causal_labels": s.causal_labels} for s in story.steps]
        token_ids = tokenizer.encode_story(steps_data)
        if len(token_ids) > MAX_LEN:
            token_ids = token_ids[:MAX_LEN]
        if len(token_ids) < 4:
            return None
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            _, hidden = model(input_ids)
        result = fs_analyzer.analyze(hidden.unsqueeze(0) if hidden.dim()==2 else hidden, compute_chern=False)
        return result.tan_theta.flatten().detach().cpu().numpy()
    except Exception:
        return None

N = min(20, len(positive_test))
print(f"Analyzing {N} stories per variant...")

results = []
for i in range(N):
    story = positive_test[i]
    n_steps = len(story.steps)
    orig_tan = extract_tan(story)
    sent_tan = extract_tan(shuffle_story_steps(story, 3, seed=42+i))
    para_tan = extract_tan(shuffle_story_steps(story, max(8, n_steps//4), seed=99+i))

    def cv_m(tan):
        if tan is None:
            return np.nan, np.nan
        valid = np.isfinite(tan)
        t = tan[valid]
        if len(t) < 2:
            return np.nan, np.nan
        m = np.mean(t)
        s = np.std(t, ddof=1)
        return s/m if m > 0 else np.nan, m

    ocv, om = cv_m(orig_tan)
    scv, sm = cv_m(sent_tan)
    pcv, pm = cv_m(para_tan)
    results.append((ocv, scv, pcv, om, sm, pm))

results = np.array(results)
orig_cv = results[:, 0]
sent_cv = results[:, 1]
para_cv = results[:, 2]
orig_mean = results[:, 3]
sent_mean = results[:, 4]
para_mean = results[:, 5]

valid = ~(np.isnan(orig_cv) | np.isnan(sent_cv) | np.isnan(para_cv))
orig_cv = orig_cv[valid]
sent_cv = sent_cv[valid]
para_cv = para_cv[valid]
orig_mean = orig_mean[valid]
sent_mean = sent_mean[valid]
para_mean = para_mean[valid]

print(f"Valid paired stories: {len(orig_cv)}")

# ── CV comparison ──
print(f"\n=== Per-story CV ===")
print(f"Mean CV: orig={np.mean(orig_cv):.4f}, sent={np.mean(sent_cv):.4f}, para={np.mean(para_cv):.4f}")

t_cv_s, p_cv_s = scipy_stats.ttest_rel(sent_cv, orig_cv)
p_cv_s = float(p_cv_s / 2)
t_cv_p, p_cv_p = scipy_stats.ttest_rel(para_cv, orig_cv)
p_cv_p = float(p_cv_p / 2)

print(f"Paired t CV (orig vs sent): t={t_cv_s:.3f}, p={p_cv_s:.4f}")
print(f"Paired t CV (orig vs para): t={t_cv_p:.3f}, p={p_cv_p:.4f}")

# Sign test
n_increase = int(np.sum(sent_cv > orig_cv))
frac_increase = np.mean(sent_cv > orig_cv)
try:
    binomial_p = scipy_stats.binomtest(n_increase, len(orig_cv), 0.5).pvalue
except AttributeError:
    binomial_p = scipy_stats.binom_test(n_increase, len(orig_cv), 0.5)
print(f"Fraction sent_CV > orig_CV: {frac_increase:.2%} ({n_increase}/{len(orig_cv)}, binomial p={binomial_p:.4f})")

n_para_increase = int(np.sum(para_cv > orig_cv))
frac_para_increase = np.mean(para_cv > orig_cv)
print(f"Fraction para_CV > orig_CV: {frac_para_increase:.2%} ({n_para_increase}/{len(orig_cv)})")

# ── Mean comparison ──
print(f"\n=== Per-story mean ===")
t_m_s, p_m_s = scipy_stats.ttest_rel(sent_mean, orig_mean)
p_m_s = float(p_m_s / 2)
print(f"Mean tanTheta: orig={np.mean(orig_mean):.4f}, sent={np.mean(sent_mean):.4f}")
print(f"Paired t mean (orig vs sent): t={t_m_s:.3f}, p={p_m_s:.4f}")

# Effect sizes
d_cv_s = float((np.mean(sent_cv)-np.mean(orig_cv))/(np.sqrt((np.var(sent_cv)+np.var(orig_cv))/2)+1e-10))
d_m_s = float((np.mean(sent_mean)-np.mean(orig_mean))/(np.sqrt((np.var(sent_mean)+np.var(orig_mean))/2)+1e-10))
print(f"\nCohen d (CV sent): {d_cv_s:.3f}")
print(f"Cohen d (mean sent): {d_m_s:.3f}")

# ── Per-story detail ──
print(f"\n=== Per-story detail ===")
for i in range(min(10, len(orig_cv))):
    delta_cv = sent_cv[i] - orig_cv[i]
    delta_mean = sent_mean[i] - orig_mean[i]
    arrow_cv = ">" if delta_cv > 0 else "<"
    arrow_mean = ">" if delta_mean > 0 else "<"
    print(f"  S{i}: CV orig={orig_cv[i]:.4f} {arrow_cv} sent={sent_cv[i]:.4f} (dCV={delta_cv:+.4f}) | mean={orig_mean[i]:.2f} {arrow_mean} {sent_mean[i]:.2f} (dMean={delta_mean:+.2f})")

# ── Wilcoxon signed-rank test (more robust) ──
try:
    w_cv, wp_cv = scipy_stats.wilcoxon(sent_cv, orig_cv, alternative='greater')
    print(f"\nWilcoxon signed-rank (CV, sent > orig): W={w_cv:.1f}, p={wp_cv:.4f}")
    w_m, wp_m = scipy_stats.wilcoxon(sent_mean, orig_mean, alternative='greater')
    print(f"Wilcoxon signed-rank (mean, sent > orig): W={w_m:.1f}, p={wp_m:.4f}")
except Exception as e:
    print(f"Wilcoxon failed: {e}")

# ── Summary verdict ──
print(f"\n{'='*50}")
print("SUMMARY")
signals = []
if p_cv_s < 0.05 or wp_cv < 0.10:
    signals.append(f"CV paired {'t' if p_cv_s<0.05 else 'Wilcoxon'} significant")
if frac_increase > 0.60:
    signals.append(f"CV increase in {frac_increase:.0%} of stories")
if p_m_s < 0.05:
    signals.append("Mean change significant")
if signals:
    print(f"Signals found: {len(signals)}")
    for s in signals:
        print(f"  + {s}")
else:
    print("No significant signals found")
print(f"{'='*50}")
