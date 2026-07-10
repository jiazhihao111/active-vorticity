import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()

pos_texts = [
    "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。",
    "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。",
    "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。",
    "厨师先准备好食材：鸡蛋、面粉和糖。他把面粉倒入碗中，加入鸡蛋和糖搅拌，然后放进烤箱。三十分钟后，蛋糕做好了。",
    "侦探仔细检查了犯罪现场。他发现窗户上有指纹，地毯上有泥脚印。顺着线索，他找到了嫌疑人藏身的旅馆。",
]

scrambled_texts = [
    "小明走进房间，宝箱突然出现了。他打开门，发现桌上有一把钥匙。钥匙用他打开了宝箱，然后密室走进了他。",
    "天刚亮，蔬菜就摘下了农夫。他先装进篮子，然后浇了水，最后满载而归。施肥把他放进烤箱，蛋糕走到田里。",
    "科学家在实验室里成功了。她调整了第一次，优化了第二次。实验反复失败，终于参数重试了。结果更好了实验。",
    "厨师把烤箱倒进面粉中。鸡蛋准备好了食材，糖搅拌了他。然后碗加入蛋糕，三十分钟后，面粉做好了厨师。",
    "侦探找到了旅馆，他检查了嫌疑人。窗户上有泥脚印，地毯上有指纹。犯罪现场顺着线索，他藏身在旅馆里发现。",
]

gamma = 0.01

def get_hidden(texts):
    results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1].squeeze(0).cpu().float()
            results.append(h)
    return results

def estimate_alpha(hidden_list, gamma):
    alphas = []
    for h in hidden_list:
        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        F_res = acc[:min_t] + gamma * v_for[:min_t]
        P_raw = (F_res * v_for[:min_t]).sum(dim=-1)
        P_active = (v_for[:min_t] * v_for[:min_t]).sum(dim=-1)
        if P_active.abs().mean() > 1e-10:
            alphas.append(P_raw.mean().item() / P_active.mean().item())
    return float(np.mean(alphas)) if alphas else None

def compute_Pc(hidden_list, alpha, gamma):
    Pc_list = []
    Praw_list = []
    ranks = []
    for h in hidden_list:
        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        F_c = acc[:min_t] + (gamma - alpha) * v_for[:min_t]
        P_c = (F_c * v_for[:min_t]).sum(dim=-1)
        P_raw = ((acc[:min_t] + gamma * v_for[:min_t]) * v_for[:min_t]).sum(dim=-1)
        Pc_list.append(P_c.mean().item())
        Praw_list.append(P_raw.mean().item())
        vel_np = vel.numpy()
        cov = np.cov(vel_np.T)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        total = eigvals.sum()
        if total > 1e-10:
            cum = np.cumsum(eigvals) / total
            rank = int(np.searchsorted(cum, 0.95) + 1)
            ranks.append(rank)
    return np.array(Pc_list), np.array(Praw_list), ranks

# === 随机token neg ===
print("=== Generating random token negatives ===")
np.random.seed(42)
random_hidden = []
with torch.no_grad():
    for _ in range(5):
        random_ids = torch.randint(100, tokenizer.vocab_size - 100, (1, 30)).to(device)
        outputs = model(input_ids=random_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1].squeeze(0).cpu().float()
        random_hidden.append(h)

# === 分析 ===
pos_h = get_hidden(pos_texts)
scr_h = get_hidden(scrambled_texts)

alpha_star = estimate_alpha(pos_h, gamma)
print("alpha* (from pos) = {:.4f}".format(alpha_star))

# pos vs scrambled
pos_Pc, pos_Praw, pos_ranks = compute_Pc(pos_h, alpha_star, gamma)
scr_Pc, scr_Praw, scr_ranks = compute_Pc(scr_h, alpha_star, gamma)
rnd_Pc, rnd_Praw, rnd_ranks = compute_Pc(random_hidden, alpha_star, gamma)

print("\n=== pos vs scrambled vs random ===")
print("pos  P_c: mean={:.4f}, |mean|={:.4f}, |Pc/Praw|={:.6f}".format(
    pos_Pc.mean(), abs(pos_Pc.mean()), abs(pos_Pc.mean())/(abs(pos_Praw.mean())+1e-10)))
print("scr  P_c: mean={:.4f}, |mean|={:.4f}, |Pc/Praw|={:.6f}".format(
    scr_Pc.mean(), abs(scr_Pc.mean()), abs(scr_Pc.mean())/(abs(scr_Praw.mean())+1e-10)))
print("rnd  P_c: mean={:.4f}, |mean|={:.4f}, |Pc/Praw|={:.6f}".format(
    rnd_Pc.mean(), abs(rnd_Pc.mean()), abs(rnd_Pc.mean())/(abs(rnd_Praw.mean())+1e-10)))

print("\nvel ranks: pos={}, scr={}, rnd={}".format(
    np.mean(pos_ranks), np.mean(scr_ranks), np.mean(rnd_ranks)))

# t-tests
t1, p1 = stats.ttest_ind(np.abs(pos_Pc), np.abs(scr_Pc), equal_var=False)
t2, p2 = stats.ttest_ind(np.abs(pos_Pc), np.abs(rnd_Pc), equal_var=False)
t3, p3 = stats.ttest_ind(np.abs(scr_Pc), np.abs(rnd_Pc), equal_var=False)
print("\n|Pc| t-tests:")
print("  pos vs scr: t={:.4f}, p={:.4f}".format(t1, p1))
print("  pos vs rnd: t={:.4f}, p={:.4f}".format(t2, p2))
print("  scr vs rnd: t={:.4f}, p={:.4f}".format(t3, p3))

# Per-trajectory details
print("\n=== Per-trajectory |P_c| ===")
for i in range(len(pos_Pc)):
    print("  traj {}: pos={:.2f}, scr={:.2f}, rnd={:.2f}".format(
        i, abs(pos_Pc[i]), abs(scr_Pc[i]), abs(rnd_Pc[i]) if i < len(rnd_Pc) else 0))

# Key insight: P_c/P_raw ratio
print("\n=== P_c/P_raw ratio (key metric) ===")
for i in range(len(pos_Pc)):
    pos_ratio = abs(pos_Pc[i]) / (abs(pos_Praw[i]) + 1e-10)
    scr_ratio = abs(scr_Pc[i]) / (abs(scr_Praw[i]) + 1e-10)
    rnd_ratio = abs(rnd_Pc[i]) / (abs(rnd_Praw[i]) + 1e-10) if i < len(rnd_Pc) else 0
    print("  traj {}: pos={:.6f}, scr={:.6f}, rnd={:.6f}".format(i, pos_ratio, scr_ratio, rnd_ratio))

del model
torch.cuda.empty_cache()