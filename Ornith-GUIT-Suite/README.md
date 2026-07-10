# Ornith-GUIT-Suite

**Ornith-1.0-9B 专属 GUIT 工程化套件** —— 将 [`GUIT-TRT v9.2`](..)（大语言模型隐空间非平衡态热力学活性涡流）理论工程化，为 `Ornith-1.0-9B`（Qwen3.5 架构，D=4096，混合注意力）的 Agentic-Coding / Self-Scaffolding 场景提供组件。

> 理论核心：LLM 隐状态轨迹是非平衡态活性系统 `m·a + γ·v = α*·v + F_c + ξ`，
> 合法轨迹约束力做功 `P_c = P_raw − P_active ≈ 0`；`P_c/P_raw`（**逐 token** 计算）是幻觉/动力学相变的核心指标。

---

## 四大组件

| 组件 | 职责 | 关键 API |
|------|------|----------|
| `ThermoPhysics` | 潜空间热力学功率计算（P_raw / P_active / P_c / P_c/P_raw） | `powers()`, `pc_ratio()`, `trajectory_metrics()`, `calibrate_alpha_star()` |
| `OrnithAutoCalibrator` | 代码因果脊线提取 + α* 标定 | `calibrate()`, `project()`, `reconstruct()`, `calibrate_from_swebench()` |
| `StreamingAffineCompressor` | 流式增量正交迭代脊线追踪（压缩/解压激活） | `forward()`, `compress_store()`, `restore()` |
| `OrnithGuard` | 测试代码热力学相变守卫（防崩溃） | `update_state_machine()`, `process_step()`, `register_hook()`, `close()` |
| `DynamicKVCacheEvictor` | 因果贡献度 KV 淘汰（CausalKV） | `update_scores()`, `get_eviction_mask()` |
| `PhysicsInformedLoop` | **PI-LOOP** 物理感知自优化循环（事中熔断 + 语义反思 + 元层进化） | `run()`, `generate_with_guard()`, `evaluate_with_physics()` |
| `OrnithLatentSimulator` | 忠实潜空间代理（无 GPU 时可测） | `generate_trajectory()`, `generate_test_code_stream()`, `generate_niah_context()` |
| `NESSDiagnostics` | 非平衡态定态诊断（σ/J/⟨v⟩/CV）+ 异常曲率 K_sub | `diagnose()`, `compare_regimes()` |
| `VorticityAnalyzer` | 活性涡流分析：速度雅可比反对称分解 + RMT KS 检验 | `analyze()`, `compare()` |
| `HallucinationDetector` | 滑动窗口 P_c/P_raw 零样本实时幻觉检测器 | `step()`, `reset()` |
| `AffineConstraintProjector` | 推理时硬约束投影回因果脊线子空间 | `project()`, `project_trajectory()`, `pc_before_after()` |

---

## 安装

```bash
cd Ornith-GUIT-Suite
pip install -e .          # 需要 torch >= 2.0, numpy
```

---

## 快速上手

### 1. 物理指标计算（逐 token）

```python
import torch
from ornith_guit import ThermoPhysics

eng = ThermoPhysics(alpha_star=1.41, gamma=0.01)
h_t, h_t1, h_t2 = (torch.randn(256) for _ in range(3))
pc_ratio, vel_norm = eng.pc_ratio(h_t, h_t1, h_t2)   # P_c/P_raw, ‖v‖
print(f"P_c/P_raw={pc_ratio:.4f}, vel_norm={vel_norm:.4f}")
```

### 2. 脊线提取与标定

```python
from ornith_guit import OrnithAutoCalibrator

cal = OrnithAutoCalibrator(hidden_dim=256, target_r=16)
report = cal.calibrate([hidden_states_list])   # 每条 [T, D]
print(report)   # 含 r(有效秩), 压缩率, alpha*
coords = cal.project(hidden_states)            # [T, D] -> [T, r]
recon  = cal.reconstruct(coords)                # [T, r] -> [T, D]
```

### 3. 流式压缩（应对流形漂移）

```python
from ornith_guit import StreamingAffineCompressor

comp = StreamingAffineCompressor(hidden_dim=256, ridge_dim=16, drift_threshold=0.05)
for h in decode_stream:                 # h: [B, 1, D]
    h_rec, info = comp(h)
    if info["basis_updated"]:
        print("脊线漂移，基底已更新")
```

### 4. 测试代码防崩溃守卫

```python
from ornith_guit import OrnithGuard, PhaseTransitionException

guard = OrnithGuard(alpha_star=1.41, pc_threshold=0.08)
for token, h in generation_loop:
    guard.update_state_machine(token)   # 检测 <test> ... </test>
    try:
        guard.process_step(h)           # 相变时抛 PhaseTransitionException
    except PhaseTransitionException as e:
        agent.backtrack_or_fix(e)       # 触发 Agent 回退/修正
```

### 5. 因果 KV 淘汰

```python
from ornith_guit import DynamicKVCacheEvictor

evictor = DynamicKVCacheEvictor(max_capacity=512, n_sink=4, n_recent=10)
for t, (h_t, h_t1, h_t2, attn) in enumerate(decode_stream):
    evictor.update_scores(h_t, h_t1, h_t2, attn_weights=attn)
mask = evictor.get_eviction_mask()      # True = 待淘汰
```

### 6. PI-LOOP 物理感知自优化循环

将 GUIT 微观物理熔断嵌入 LOOP 宏观语义反思，构成"事中毫秒级熔断 + 事后精准修正 + 越用越懂代码库"的三轨闭环。生成后端抽象为 `GenerationBackend`，可对接真实 Ornith，也可用 `SimulatedCodingBackend` 无 GPU 验证。

```python
from ornith_guit import PhysicsInformedLoop, SimulatedCodingBackend, OrnithAutoCalibrator
from ornith_guit.simulator import OrnithLatentSimulator

sim = OrnithLatentSimulator()
backend = SimulatedCodingBackend(sim, flaw_start=12, flaw_len=40, fix_after=2)
cal = OrnithAutoCalibrator(hidden_dim=sim.D, target_r=sim.r)
cal.calibrate([sim.generate_trajectory("pos", 40)[0] for _ in range(3)])

loop = PhysicsInformedLoop(
    backend, alpha_star=1.41, pc_threshold=0.15,   # 真实 Ornith 用 0.08
    max_iterations=5, calibrator=cal,              # 传入 calibrator 启用元层进化
    guard_kwargs={"consecutive_hits": 3},
)
code, trace = loop.run("为金融函数写 pytest 用例", max_new_tokens=128)
print(trace["physics_interventions"], trace["total_tokens_saved"], trace["converged"])
```

真实部署时自定义 `GenerationBackend.stream()`，逐 token 产出 `(token_text, hidden_state)` 即可（`hidden_state` 为最后一层 decode 隐状态 `[D]`）。

---

## 运行对比测试

```bash
cd Ornith-GUIT-Suite/benchmarks
python run_comparison.py
```

生成 `benchmarks/results.json`（9 组对比：压缩 T1、幻觉检测 T2、防崩溃 T3、KV 淘汰 T4、PI-LOOP T5、NESS 诊断 T6、活性涡流 RMT T7、仿射投影 T8、P_c 假象 T9）。
详细解读见 [`comparison_report.md`](comparison_report.md)。

| 测试 | 内容 | 核心结论（代理环境） |
|------|------|----------------------|
| T1 | 静态 SVD vs 流式脊线 | 流式平均误差降 60.8%，末步仅静态 12% |
| T2 | P_c/P_raw vs vel_norm vs 熵 | 因果梯度复现 pos<scr<rnd<halluc；熵探针失效 |
| T3 | OrnithGuard vs vel_norm vs 语法 | GUIT/vel_norm recall=1.0；语法检查召回 0 |
| T4 | CausalKV vs H2O vs Random | keep_30% 下 needle 召回 100% vs 50% |
| T5 | PI-LOOP 事中熔断 vs 事后 LOOP | 收敛所需 token 节省 57.3% |
| T6 | NESS 非平衡态诊断 | pos 判为 NESS 定态；异常曲率 K_sub 合法(0.70)≪缺陷(6.18) |
| T7 | 活性涡流 RMT 检验 | pos 结构化涡流拒绝 RMT 零假设；rnd 随机涡流自洽不拒绝 |
| T8 | 仿射硬约束投影 | off-ridge 缺陷块 P_c/P_raw 投影后降 75.2% |
| T9 | P_c 批量 vs 逐token 假象 | 批量与逐 token 均值不同；铁律要求逐 token 计算 |

---

## 真实模型 A/B 测试（Ornith-1.0-9B）

`benchmarks/run_real_ab.py` 把全套件**真正跑在 9B 真实模型上**（8-bit 加载，
逐 token 提取 4096 维隐状态），对比 A（无工具）/ B-default（默认阈值 0.08）/
B-calibrated（按真实轨迹重标定阈值）。完整报告见 [`real_ab_report.md`](real_ab_report.md)，
原始数据见 `benchmarks/real_ab_results.json`。

| 分支 | 阈值 | 物理熔断 | 输出 | 误熔断率 | 结论 |
|------|------|----------|------|----------|------|
| **A 无工具** | — | 无 | 完整 80 tok | — | 基线（无保护） |
| **B-default** | 0.08 | 4 次/条 | 被砍至 26–35 tok | **1.0** | ❌ 默认阈值误熔断正常代码 |
| **B-calibrated** | 0.309 | 0 次/条 | 完整 80 tok | **0.0** | ✅ 重标定后非侵入、收敛 |

**真实模型核心发现**：
- `α*` 标定 **1.429 ≈ 1.41**（迁移误差 1.35%）→ 活性驱动力系数可跨模型迁移。
- NESS 定态、活性涡流 RMT（拒绝零假设 **p=0.0**）在真实 9B 隐空间上稳健成立 → GUIT 理论获真实数据支持。
- **部署铁律补充**:`pc_threshold` 必须按目标模型 NESS-健康轨迹标定（本模型 p99.5≈0.309），禁止硬编码 0.08；真实 r(0.95)=145、压缩率 0.965，远不及仿真 r=25/99.4%；K_sub≈0.66（仿真≈0.01）应改作相对量。

> **完整报告**：仿真代理验证（T1–T9）与真实模型 A/B 测试已整合为 [`complete_report.md`](complete_report.md)（含执行摘要、交叉验证与偏差分析、部署铁律补全版）。

---

## 边界约束（GUIT 铁律）

- 仅适用于**推理阶段冻结权重**的 LLM；脊线 SVD 须在 bf16/fp32 下进行。
- `P_c/P_raw` **必须逐 token 计算**，禁止批量平均（真实 Ornith 实测批量平均≈0.84 假象；本代理符号结构方向可能相反，但铁律不变）。
- **`pc_threshold` 必须按目标模型标定**:默认 0.08 源自低维仿真/论文,在真实 Ornith-1.0-9B 上连贯生成 P_c/P_raw 均值≈0.09、约半数 token 越阈,硬编码会 100% 误熔断（见真实模型 A/B 测试）。
- 本仓含 `OrnithLatentSimulator` 忠实代理（无 GPU 可测）；**绝对数值以真实 Ornith 实测为准**。

---

## 目录结构

```
Ornith-GUIT-Suite/
├── ornith_guit/
│   ├── physics.py                  # ThermoPhysics, calibrate_alpha_star
│   ├── core/
│   │   ├── ornith_calibrator.py    # OrnithAutoCalibrator
│   │   └── streaming_compressor.py # StreamingAffineCompressor
│   ├── detection/
│   │   ├── ornith_guard.py         # OrnithGuard, PhaseTransitionException
│   │   └── detector.py             # HallucinationDetector (滑动窗口 P_c/P_raw)
│   ├── steering/
│   │   ├── causal_kv.py            # DynamicKVCacheEvictor (CausalKV)
│   │   └── affine_projector.py     # AffineConstraintProjector (硬约束投影)
│   ├── thermo/
│   │   ├── __init__.py             # 导出 NESS/Vorticity 分析器
│   │   ├── ness.py                 # NESSDiagnostics, ness_metrics, abnormal_curvature
│   │   └── vorticity.py            # VorticityAnalyzer, RMT KS 检验, 雅可比分解
│   ├── loop/
│   │   ├── __init__.py
│   │   └── pi_loop.py              # PhysicsInformedLoop (PI-LOOP) + 仿真后端
│   ├── simulator/
│   │   └── __init__.py             # OrnithLatentSimulator
│   └── integrations/
│       └── huggingface_ornith.py   # ThermoHookManager (真实模型部署)
├── benchmarks/
│   ├── run_comparison.py           # 9 组对比测试驱动 (T1-T9, 仿真代理)
│   ├── run_real_ab.py             # 真实模型 A/B 测试 (加载 9B, 8-bit)
│   ├── results.json                # 仿真代理测试结果
│   └── real_ab_results.json        # 真实模型 A/B 测试结果
├── tests/
│   └── test_suite.py               # 14 个冒烟/单元测试
├── comparison_report.md            # 详细对比测试报告 (仿真代理)
├── real_ab_report.md               # 真实模型 A/B 测试报告
├── complete_report.md              # 完整评测报告 (仿真+T1-T9 + 真实 A/B 整合)
├── README.md
└── pyproject.toml
```

---

## 测试

```bash
cd Ornith-GUIT-Suite
python -m pytest tests/ -q
```
