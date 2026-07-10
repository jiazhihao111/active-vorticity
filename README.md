# VORTEX · 大语言模型隐空间的非平衡态热力学活性涡流研究项目

> **涡流框架（VORTEX, Active Vorticity Theory of LLM Latent Spaces）**
> GUIT-TRT（大一统通用信息论-热力学重构）工程 · 非平衡态活性涡流版

本项目把大语言模型（LLM）的自回归生成重新刻画为**子黎曼流形上维持非平衡态定态（NESS）的活性涡流（Active Vorticity）**：合法生成是受仿射-非完整混合约束的高阶矩环流，幻觉则是系统从"高有效温度的活性涡流态"向"低温被动布朗态"的**动力学相变**。

---

## 1. 项目简介

| 项 | 内容 |
|----|------|
| 核心隐喻 | **涡**——健康生成=高有效温度的环流，幻觉=涡的破裂 |
| 理论内核 | 过阻尼活性 Langevin 方程 `m·a + γ·v = α*·v + F_c + ξ`；约束力做功 `P_c = F_c·v ≈ 0` 于合法轨迹 |
| 核心指标 | `P_c/P_raw`（**必须逐 token 计算**）——相对健康基线的抬升标志动力学相变 / 幻觉 |
| 理论版本 | GUIT-TRT v9.2（活性涡流版），论文历经 v5.0–v20.0 理论迭代与 v22 跨模型 A/B 验证 |
| 覆盖架构 | MiniCPM5-1B(1536D)、Qwen2.5-7B(3584D)、Ornith-1.0-9B(4096D，混合注意力) |

### 四大核心发现
1. **仿射-非完整混合约束原理**：因果约束流形近乎完美平坦（`R²=1.0`），25% 完整约束构成位置空间"仿射骨架"，75% 非完整约束构成速度空间"因果导流管"，升级为子黎曼流形。
2. **活性涡流（Active Vorticity）**：RMT 以 `p < 10⁻³⁸` 证明速度场雅可比反对称部分偏离 Wigner 半圆律，确证高阶矩相空间涡流；宏观细致平衡未破缺（`⟨v⟩≈0`），但概率流 `J > 0`、熵产生 `σ > 0`（NESS）。
3. **幻觉的动力学相变理论**：`P_c/P_raw` 相对健康基线抬升为相变标志；但跨模型 A/B 下该指标塌缩为 ≈0.10–0.15 近平坦、排序非单调，**须按目标模型现场标定**，非普适阈值梯度。
4. **因果维度的强涌现本质**：有效约束维度随上下文亚线性增长，且与任一注意力头正交，是残差流全网络协同涌现的"时变非完整分布"。

---

## 2. 论文

- **主论文（Markdown）**：[`2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.md`](2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.md)
- **主论文（PDF）**：[`2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.pdf`](2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.pdf)
- **方法论边界章节（§7）**：解释力 / 解决力 / 可证伪性（H1–H5）/ 局限性——基于 GUIT 七维诊断与边界铁律撰写。
- 相关理论文档：`2026.7.10_GUIT-TRT_v9.0_非平衡态活性涡流版.md`、`2026.7.9.21.13GUIT-TRT v8.0：非平衡态活性动力学版本.md` 等（根目录 `*.md`）。

---

## 3. 代码库导航

| 代码库 | 角色 | 版本 / 状态 | 安装 | 入口文档 |
|--------|------|------------|------|----------|
| **`causal_gauge_field/`** | 论文核心实证代码库（v5.0–v22.0 实验脚本与子模块：NPNW、RMT、NESS、仿射压缩、跨模型 A/B） | 研究原型（含 `v20_*`、`v21_ab`、`v22_*` 等） | 直接运行脚本 | 见 `causal_gauge_field/方案.md` 与各 `v*_report.json` |
| **`llm-thermodynamics/`** | 正式可安装的 `llm_thermo` 库：热力学引擎、NESS 评估、子黎曼几何、RMT 涡流、仿射压缩、量化守卫 | `v0.3.0`（MIT，56 测试通过） | `pip install -e ".[hf]"` | [`llm-thermodynamics/README.md`](llm-thermodynamics/README.md) |
| **`Ornith-GUIT-Suite/`** | Ornith-1.0-9B 专属工程化套件：脊线压缩、相变守卫、因果 KV 淘汰、PI-LOOP 自优化 | `v1.0.0`（MIT） | `pip install -e .` | [`Ornith-GUIT-Suite/README.md`](Ornith-GUIT-Suite/README.md) |

> 三者关系：`causal_gauge_field` 是论文结论的**产生地**（实验原型）；`llm-thermodynamics` 是将结论**固化为通用库**的正式交付；`Ornith-GUIT-Suite` 是将理论**工程化到 Agentic Coding 场景**的专属套件。

---

## 4. 快速开始

### 4.1 通用库 `llm_thermo`（推荐入门）

```bash
cd llm-thermodynamics
pip install -e ".[hf]"      # 含 HuggingFace 集成
```

```python
from llm_thermo import ThermodynamicEngine, get_preset

preset = get_preset("qwen2.5-7b")
engine = ThermodynamicEngine(alpha_star=preset.alpha_star)

# hidden_states: [T, D] 隐状态轨迹
result = engine.compute_per_token_ratio(hidden_states)
print(f"P_c/P_raw = {result['per_token_ratio']:.4f}")   # 合法文本 ≈ 0.10–0.15
```

实时幻觉检测、NESS 评估、子黎曼分析、RMT 涡流、仿射压缩、量化守卫等完整 API 见 [`llm-thermodynamics/README.md`](llm-thermodynamics/README.md)。

### 4.2 Ornith-1.0-9B 专属套件

```bash
cd Ornith-GUIT-Suite
pip install -e .
```

```python
import torch
from ornith_guit import ThermoPhysics
eng = ThermoPhysics(alpha_star=1.41, gamma=0.01)
pc_ratio, vel_norm = eng.pc_ratio(h_t, h_t1, h_t2)
print(f"P_c/P_raw={pc_ratio:.4f}, vel_norm={vel_norm:.4f}")
```

PI-LOOP 物理感知自优化、测试代码相变守卫、因果 KV 淘汰等见 [`Ornith-GUIT-Suite/README.md`](Ornith-GUIT-Suite/README.md)。

---

## 5. 边界与局限（GUIT 铁律）

所有工程落地必须遵守以下边界，详见论文 §7.4 与子库 README：

1. **精度边界**：核心微积分对高频噪声极敏感；`vel_norm` 等一/二阶导数指标在 INT4/INT8 下反转，**脊线提取须在 BF16/fp32**；`P_c` **必须逐 token 计算**（批量平均产生 ≈0.84 假象）。
2. **阈值标定铁律**：`P_c/P_raw` 排序非普适单调（MiniCPM5: scr>pos≈rnd；Qwen2.5-7B: rnd>pos>scr），检测阈值**必须按目标模型 NESS 健康轨迹 p99.5 现场标定**（如 Ornith≈0.309），**禁止硬编码**他模型经验值（会 100% 误熔断）。
3. **架构泛化边界**：`α*` 非宇宙常数，依赖归一化类型（RMSNorm/LayerNorm）与规模；SSM/Mamba 等非 Transformer 架构尚未验证。
4. **反身性边界**：仅适用于强因果任务（推理、事实问答、代码生成）；弱因果/强反身性领域（创意写作）的因果贡献度度量可能误删"灵感节点"。
5. **因果推断边界**：框架识别几何-热力学约束结构（相关性与不变性层面），**不等同** do-演算意义上的强干预性因果。
6. **固化度自评**：基于 GUIT 七维光谱，本理论处于 **0.70–0.78（弱反身性/统计稳定学科）** 区间——物理内核（`R²=1.0`、RMT、NESS）已强固化，但**跨模型普适性（复制信息维度）** 仍受文本集与精度影响，为主要短板。

---

## 6. 顶层目录结构

```
对称、破缺与约束场/
├── README.md                                # 本文件（项目总说明）
├── 2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.md   # 主论文
├── 2026.710.9.10大语言模型隐空间的非平衡态热力学活性涡流论文.pdf  # 主论文 PDF
├── causal_gauge_field/                      # 论文核心实证代码库（实验原型）
│   ├── npnw/ geometry/ dynamics/ theorems/ newton/ losses/ models/ experiments/ utils/
│   ├── v20_*  v21_ab*  v22_*               # 各版本实验脚本
│   └── 方案.md                               # 实验方案说明
├── llm-thermodynamics/                      # 正式库 llm_thermo (v0.3.0)
│   ├── llm_thermo/  examples/  tests/
│   ├── pyproject.toml  README.md
├── Ornith-GUIT-Suite/                      # Ornith-1.0-9B 专属工程化套件 (v1.0.0)
│   ├── ornith_guit/  benchmarks/  tests/
│   ├── pyproject.toml  README.md  complete_report.md
├── outputs/  exp9_outputs/  exp10_outputs/  logs/   # 实验产出与日志
└── （根目录其余 *.md 为 GUIT-TRT 理论迭代与验证报告）
```

---

## 7. 引用与许可

### 论文引用（BibTeX）

```bibtex
@article{guit_trt_vortex_2026,
  title={Non-Equilibrium Thermodynamics of LLM Latent Spaces: From Affine Constraints to Active Vorticity},
  author={Jia, Zhihao},
  year={2026},
  note={GUIT-TRT v9.2 Active-Vorticity Edition}
}
```

### 许可
- 代码库 `llm-thermodynamics` 与 `Ornith-GUIT-Suite`：MIT。
- 论文与理论文档：用于学术与工程目的，引用请注明出处。

---

> **免责声明（GUIT 边界铁律）**：本框架所有结论均为"在 X 条件下成立、在 Y 条件下失效"的限定宣称，不构成无边界普适理论；工程使用须严格遵守精度、阈值标定与架构边界。
