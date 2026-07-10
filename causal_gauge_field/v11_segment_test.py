import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()

text_causal = "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。"
text_halluc = "宝箱突然钥匙了他，然后密室走进了他。门后是一间打开了，里面放着突然。"
text_mixed = text_causal + text_halluc
text_pure = "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。"

gamma = 0.01
alpha = 1.50

def get_hidden(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1].squeeze(0).cpu().float()

def analyze(h, alpha, gamma, window=5):
    vel = h[1:] - h[:-1]
    acc = vel[1:] - vel[:-1]
    v_for = vel[1:]
    min_t = min(acc.size(0), v_for.size(0))
    F_c = acc[:min_t] + (gamma - alpha) * v_for[:min_t]
    F_total = acc[:min_t] + gamma * v_for[:min_t]
    P_c = (F_c * v_for[:min_t]).sum(dim=-1)
    P_raw = (F_total * v_for[:min_t]).sum(dim=-1)
    cum_Pc = torch.cumsum(P_c, dim=0)
    cum_Praw = torch.cumsum(P_raw, dim=0)
    cum_ratio = torch.abs(cum_Pc) / (torch.abs(cum_Praw) + 1e-10)
    win_Pc = torch.zeros_like(P_c)
    win_Praw = torch.zeros_like(P_raw)
    for i in range(len(P_c)):
        start = max(0, i - window + 1)
        win_Pc[i] = P_c[start:i+1].sum()
        win_Praw[i] = P_raw[start:i+1].sum()
    win_ratio = torch.abs(win_Pc) / (torch.abs(win_Praw) + 1e-10)
    return cum_ratio.numpy(), win_ratio.numpy(), P_c.numpy(), P_raw.numpy()

h_mixed = get_hidden(text_mixed)
h_pure = get_hidden(text_pure)

cum_m, win_m, pc_m, praw_m = analyze(h_mixed, alpha, gamma)
cum_p, win_p, pc_p, praw_p = analyze(h_pure, alpha, gamma)

tokens_mixed = tokenizer.encode(text_mixed)
tokens_causal = tokenizer.encode(text_causal)
tp = len(tokens_causal) - 2

print("=== Mixed text (causal -> scrambled) ===")
print("Causal part: ~{} tokens, Total: {} tokens".format(tp, len(tokens_mixed)))
print()
print("Step | cum_ratio | win_ratio | marker")
print("-" * 50)
for i in range(len(cum_m)):
    marker = " <-- TRANSITION" if abs(i - tp) < 2 else ""
    print("{:4d} | {:.6f}  | {:.6f}  |{}".format(i, cum_m[i], win_m[i], marker))

print()
print("=== Pure causal text ===")
print("Step | cum_ratio | win_ratio")
print("-" * 40)
for i in range(len(cum_p)):
    print("{:4d} | {:.6f}  | {:.6f}".format(i, cum_p[i], win_p[i]))

print()
print("=== Key comparison ===")
print("Mixed overall P_c/P_raw: {:.6f}".format(abs(pc_m.sum())/(abs(praw_m.sum())+1e-10)))
print("Pure  overall P_c/P_raw: {:.6f}".format(abs(pc_p.sum())/(abs(praw_p.sum())+1e-10)))
print("Mixed causal segment (0:{}) P_c/P_raw: {:.6f}".format(tp, abs(pc_m[:tp].sum())/(abs(praw_m[:tp].sum())+1e-10)))
print("Mixed halluc segment ({}:end) P_c/P_raw: {:.6f}".format(tp, abs(pc_m[tp:].sum())/(abs(praw_m[tp:].sum())+1e-10)))

del model
torch.cuda.empty_cache()