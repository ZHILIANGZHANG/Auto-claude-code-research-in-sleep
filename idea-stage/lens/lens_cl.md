# Lens 报告：Continual Learning（training-free / 少训练 / 低资源，预训练模型时代）

日期：2026-10-17。子代理：ARIS idea-creator Phase 1 + 1.5（CL 子方向）。

## 0. 方法与可信度说明（必读）

- 环境限制：arxiv.org / export.arxiv.org / alphaxiv / huggingface / semanticscholar 的直接 fetch 全部被代理 403 拦截；WebSearch 可用但有 200 次/turn 的共享预算，本次用掉 38 次后预算耗尽。
- 因此：**所有条目均来自搜索返回的摘要片段 + 会议页面（iclr.cc / icml.cc / neurips.cc / proceedings）**，未能逐篇读 intro / limitation。下文「自述 limitation」只在搜索片段中明确出现时才填写，否则标「未读到」。
- 所有 arXiv ID 与标题都是搜索结果中直接出现的；凡是我凭记忆补充但本次未核实的锚点论文，统一标注 `[未核实-记忆]`，请主代理在后续 turn 复核。**（2026-10-17 更新：已由核实子代理逐篇对照 arXiv abs 页复核，结果见 §0.5，标记已移除。）**
- 日期窗口：重点 2025-07 → 2026-10，但为锚点也纳入 2023-2025H1 的奠基工作。

## 0.5 核实记录（2026-10-17，abs 页面）

核实方式：逐篇 curl `arxiv.org/abs/<ID>`（标题/作者/日期/comments/摘要），关键数字另抓 `arxiv.org/html/<ID>` grep；非 arXiv 论文用 CrossRef、aclanthology、icml.cc / iclr.cc 虚拟页、openreview 搜索接口；少量 WebSearch。标记：✅ 一致 / ✏️ 修正 / ❌ 未找到或 ID 错误 / ⚠ 仍无法核实。**所有修正已同步改入 §1 与 §4。**

### 0.5.1 原「未核实-记忆 / 二手 / venue 未确认」条目
- InfLoRA 2404.00228 | ✅ 一致：Liang & Li，CVPR 2024，干扰消除子空间 LoRA。
- SimpleCIL/APER 2303.07338 | ✏️ 修正：venue → **IJCV**（v2 2024-08）；摘要还明确说「以 ImageNet 为基础的旧基准因与预训练数据重叠而不适用，提出 ImageNet-A / ObjectNet / OmniBenchmark / VTAB 四个新基准」——与 Idea 1 的「零泄漏协议」部分重叠。
- ZSCL 2303.06628 | ✅ 一致：ICCV 2023；特征蒸馏 + 训练中权重平均；MTIL 基准；MTIL 平均 +9.7%。
- RAIL 2406.18868 | ✏️ 修正：venue → **NeurIPS 2024**（v4 2024-12）；其余（递归岭回归 adapter、training-free 融合模块、X-TAIL、「absolutely preserves」零样本）✅。
- F-OAL | ✏️ 修正：ID = **arXiv 2403.15751**，NeurIPS 2024（CrossRef 10.52202/079017-1314）；冻结编码器 + 特征融合 + RLS 线性分类器，无反传。**注意：原文把 2403.15751 错标给了 AOCIL**，AOCIL 的真实 ID 未查到（❌ ID 错误，已改为「ID 待查」）。
- Spark memory 2607.22157 | ❌ ID 错误：2607.22157 实为 **「Learning on the Job: Continual Learning from Deployment Feedback for Frozen-Weights Agents」**（Tablan, Taylor, Bernhem，2026-07-24，无 venue）：冻结模型 + 外部记忆把每个 episode 蒸馏成自然语言规则；τ-bench banking 上 outcome 反馈 1.6×、corrections 2.6×。原「Spark memory」名称来源不明，条目已按真实内容改写。
- GAP-Prompt 2608.23782 | ✅ 一致：2026-08-24，无 venue 信息（abs 无 comments）；CUB-200 87.29%（joint 88.00%）。
- 2601.18699 v1 vs v2 | ✅ 矛盾属实，结论：**不要引用**。v1（2026-01-26）作者「Olaf Yunus Laitinen Imanov」，声称「109B–400B 模型」、r=0.87、15–23% 注意力头受损；v2（2026-06-13）作者名变为「Gustav Olaf Yunus Laitinen-Fredriksson Lundstrom-Imanov」，改为「20 个 2026 年中 SOTA 模型（含闭源 Claude Fable 5 / GPT-5.5 / Gemini 3.5）」+ LRCP 方法「缓解 94.2% 能力」。同一 ID 下作者署名、实验对象、贡献全部改变，闭源模型做权重轨迹分析亦不可能。

### 0.5.2 无 arXiv ID 条目
- Audit Before You Merge | ✏️ 修正：全名「Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition」（Aharon, Glazer, Aharon）；**不是 ICML 2026 主会，是 ICML 2026 workshop「Continual Adaptation at Scale: Towards Sustainable AI (CATS)」poster**（icml.cc/virtual/2026/74593；openreview `vgsNcwf81w`）。数字（GSM8K −10.5、HumanEval −18.3、TA λ=0.5 在基线 1 SE 内、Mistral-7B 4 个候选 2 个不过审）✅ 与 ICML 摘要一致。无 arXiv ID。
- Is Our Benchmark Enough? An Analysis of Continual Learning for MLLMs | ✏️ 修正：**ICML 2026 CATS workshop oral，非主会**（icml.cc/virtual/2026/74636；作者 Tran, Gowda, Dzaferagic, Ruffini，TCD + TU/e）；内容（RePro training-free/replay-free 原型路由匹配 MR-LoRA 路由器；共享专家无益；MLLM-CL 任务高度可分 + 单一固定顺序；建议重叠流形/多顺序/细粒度域偏移/forward transfer）✅。无 arXiv ID。
- CL-Bench | ✏️ 修正：ID = **arXiv 2606.05661**「Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments」（Asawa … Zaharia, Gonzalez；Berkeley + Snorkel + UW-Madison，2026-06-04）。NeurIPS 2026 仅来自 Snorkel 博客，neurips.cc 未核实（⚠）；icml.cc 上的条目是 CATS workshop invited talk。「6 域、朴素 ICL 胜专门记忆系统」✅；摘要明确**不评估参数化方法（如 TTT）**。
- PTA | ✏️ 修正：ID = **arXiv 2604.21360**（Huang, Zhang, Liu, Chao, Ji；v2 2026-05-13），ICML 2026 主会 poster ✅；65.64→69.38（10 跨域基准）✅，保留 92% CLIP 推理速度；TF（无反传）✅。
- CF-Router | ✏️ 修正：**ICLR 2026 workshop「Lifelong Agents: Learning, Aligning, Evolving (LLA)」poster，非主会**（iclr.cc/virtual/2026/10012549；openreview `7tWtYHTIDD`）。方法：用 MLLM 末层隐藏态做描述子，**正则化最小二乘分类器选择 expert LoRA**。无 arXiv ID。
- In-Place TTT | ✏️ 修正：ID = **arXiv 2604.06169**（Feng … Cai，ByteDance Seed + PKU），ICLR 2026 **Oral** ✅；把 MLP 末投影矩阵当 fast weights 在推理中更新，chunk-wise 更新。
- LoRA vs Full FT: An Illusion of Equivalence | ✏️ 修正：ID = **arXiv 2410.21228**（Shuttleworth, Andreas, Torralba, Sharma），NeurIPS 2025 ✅（CrossRef 10.52202/085713-5808）；摘要已含「intruder dimensions 因果导致遗忘；缩小其奇异值可恢复预训练分布；顺序微调中 intruder dims 累积、CL 表现更差」。
- Chronos | ✏️ 修正：ACL 2026 Findings 论文标题为 **「RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World」**（2026.findings-acl.546 页面 ✅）；Chronos 是其中提出的 time-aware retrieval baseline（Event Evolution Graph，无需训练）。该基准同时评估 continual finetuning / 知识编辑 / RAG，报告遗忘 + 时间不一致。无 arXiv ID。
- Slim Adaptation Modules | ✅ NeurIPS 2025 存在（neurips.cc/virtual/2025/123855，WebSearch）；PAM 2603.21170 为同作者改名版（CPAL 2026, PMLR v328）。

### 0.5.3 全部 closest work（§4）
- CLDyB 2503.04655 | ✅ 一致（v2 2025-05，无 venue）。摘要：预训练阶段「潜在数据污染」+ 静态基准饱和 → MDP + MCTS 动态任务序列。**不做泄漏量化**，Idea 1 核心未被覆盖。
- Data Leakage 2508.17416 | ✅ 一致（2025-08-24，无 venue）。html 正文：ImageNet test hard 1.54% / soft 1.95%（val 1.58%/1.78%）✅；LAION→基准：引用 OpenCLIP pHash 估计 1–5%，自测表中部分基准达数个百分点 ✅。不涉及 CL 方法。
- PAM 2603.21170 | ✅ 一致：CPAL 2026；正文「2–5× 可训参数、2–6× 总参数」✅（摘要写「up to ~5× / ~6×」）；冻结 ResNet + 稀疏任务层；自述方法学 gap ✅。少训练，非 TF。
- RL's Razor 2509.04259 | ✅ 一致（NeurIPS 2025 未在 abs 标注，⚠ venue 来自搜索）。摘要：遗忘由新任务上 fine-tuned vs base 的 KL 决定；**已在 LLM 与 robotic foundation models 上验证**（不只 LLM）。正文 §6：L1 / Fisher-L2 / 谱范数、稀疏度（bf16 伪影）、更新秩均不预测遗忘 ✅。
- MagMax 2407.06322 | ✅ 一致：ECCV 2024；顺序微调 + 最大幅度选择；摘要明说「权重平均与随机选择出人意料地强」✅。
- SFA 2501.05559 | ✏️ 修正：**v2（2026-10-14）改名「Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives」**（Kleiman, Dziugaite, Frankle, Kakade, Paul）。v2 摘要：SFA + replay-free Lookahead；Qwen3-14B adapter CL 上固定配置胜 replay 与 adaptive merging，训练时间 −36.3%；跨语言/视觉任务与复杂方法相当。正文比较 TA / TIES / WiSE-FT，EWC 在附录 G ✅。「胜 TA/TIES/L2/EWC」改为上述表述。
- Fly-CL 2510.16877 | ✅ 一致：ICLR 2026（v2 2026-03）；多重共线性 + 果蝇嗅觉回路稀疏投影；训练时间大幅下降。摘要只讲 ViT 分类类 CIL，不涉及 MLLM/路由。
- RanPAC 2307.02251 | ✏️ 修正：错误率下降 **20%–62%**（原文写 10–62%）；NeurIPS 2023 ✅；TF（冻结随机投影 + 原型去相关）✅。
- RanDumb 2402.08823 | ✅ 一致：NeurIPS 2024；摘要已含「预训练特征上只训线性分类器即超过多数 continual fine-tuning 与 prompt-tuning」。只有总体结论，无逐方法分解。
- TTABC 2606.14299 | ✏️ 修正：**NeurIPS 2026 Datasets & Benchmarks (E&D) Track**（v3 2026-10-12）；标题「What Drives Test-Time Adaptation for CLIP? A Controlled Empirical Study from an Update Perspective」。20+ 方法、三范式、「增益来自测试时证据 + 可靠代理而非重优化」「跨/当前样本证据 + 轻量原型更新」「无 silver bullet」✅；正文 Tab.4「norm-layer 适配专精 corruption」、label shift 用 StatA 类先验估计（46.89→53.51）✅。**只含 TTA 方法，无 CIL 方法。**
- BayesTTA 2507.08607 | ✅ 一致（2025-07，无 venue）：CT-TTA（渐变偏移）设定；增量估计类条件高斯混合 + GDA + norm 层自适应（非完全 TF）。
- UL-TTA 2511.09101 | ✅ 一致：单作者 7 页；完全 TF/无反传，仅调原型/先验/温度，闭式贝叶斯更新；+4.7 点、ECE −20–30%、200K 长流不崩。
- MS-TTA 2507.00462 | ✏️ 补充 venue：**IEEE TCSVT**（v2 2026-03）；单步 kNN Mean-Shift，TF ✅。
- WikiBigEdit 2503.05683 | ✏️ 补充 venue：**ICML 2025**；50 万+ QA ✅；正文「局部修改法在前 250 次更新内急剧下降；WISE <500 次可比 RAG，10K 内退回更新前水平」✅；「RAG 远胜编辑法（推理成本更高）；等推理成本下持续 LoRA-FT/LoRA-Merge 更好」✅。**已在 26K/步、最多 500K 更新的真实时间流上对比编辑 / RAG / LoRA。**
- SMF 2510.15103 | ✅ 一致：Lin … Oğuz（Meta FAIR），2025-10-16；NQ F1 全参 −89% / LoRA −71% / SMF −11% ✅；评估两个 QA 任务。「承认 RAG 是自然方案」在摘要中未出现（⚠ 来自正文/搜索片段）。
- Spurious Forgetting 2501.13453 | ✅ 一致：ICLR 2025；性能下降反映任务对齐丢失而非知识丢失；Freezing 底层策略。摘要未提 TRACE（⚠ 「TRACE 上反弹」为搜索片段细节）。
- GraphMemory 2610.02687 | ✏️ 修正：**NeurIPS 2026 workshop「TTCL」（Test-Time Continual Learning），非主会**（comments: "neurips workshop: TTCL"）；token ↓81–85% ✅；无参数更新 ✅。
- ContinualSkillBench 2608.03874 | ✅ 一致（v3 2026-10-15；PKU 来自作者单位，abs 未标 venue）：5 域×100 任务；「保留上下文与反馈、不维护技能库 ≈ 显式技能库」（2 模型×3 域）✅；弱模型（GPT-4o）积累多、复用少 ✅。
- SLAO 2512.23017 | ✅ 一致（ICLR 2026 由 iclr.cc poster 10008003 确认）：单 LoRA 正交初始化 + 顺序合并 + time-aware scaling，O(1) 内存，Llama 系列。
- EWC-LoRA 2602.17559 | ✅ 一致：ICLR 2026；对共享低秩更新做 EWC，存储/推理成本常数。
- AnaCP 2511.13880 | ⚠ venue 未核实（abs 无 comments，「NeurIPS 2025」来自搜索）；内容 ✅：解析分类器无法适配特征 → 无梯度的增量特征适配，达 joint training 上界。
- VILA 2602.13670 | ✏️ 补充 venue：**ICML 2026**（v3 2026-05）；「representation rigidity 为主要瓶颈」+ 双分支视觉-语言校准 ✅。
- 2511.21437 | ✏️ 修正：标题「A Systematic Study of In-the-Wild Model Merging for Large Language Models」（Hitit, Girrbach, Akata），**TMLR 2026-03**；6 种合并法（含子空间法）× 4 个开源 LLM × 每模型 12 个 ckpt × 16 基准；结论：**只有 Task Arithmetic 可靠带来增益**，干扰感知/子空间法基本不优于 base。原文「Llama-3、Qwen3、Model Stock」为未核实细节，已删。
- Continual-CLIP 2210.03114 | ✅ 一致：冻结 CLIP 零样本在 CIL/DIL/TAIL 五基准上胜多数 CL 方法。
- Audit / Is Our Benchmark Enough / CL-Bench / PTA / Chronos / LoRA illusion：见 0.5.2。

### 0.5.4 各 idea 的「closest work 摘要已覆盖核心做法」判定（novelty 风险）
- **Idea 3（闭式 MLLM 路由）—— 高风险**：CF-Router（ICLR 2026 LLA workshop）已经是「MLLM 末层隐藏态 + 正则化最小二乘闭式分类器选 expert LoRA」，与 Idea 3 的 (1)(2)(3) 几乎重合；Is Our Benchmark Enough 的 RePro 已是 training-free 原型路由。剩余差异只剩 Fly-CL 式稀疏随机投影 + 去相关，以及非 IID ability track。建议改为「闭式路由族（RePro / CF-Router / 去相关 Fly 头）的对照 + 困难流」而非「首次闭式路由」。
- **Idea 7（统一遗忘账本）—— 中高风险**：「RAG or Learning?」(ACL 2026 Findings) 已在同一时间戳基准上对比 continual finetuning / 知识编辑 / RAG，并报告遗忘 + 时间不一致；WikiBigEdit 已在同一流上对比编辑 / RAG / LoRA。未被覆盖的只剩「三类遗忘的分离定义」+「等 FLOPs 预算」+ ICL/SMF 线。必须以此为主卖点。
- **Idea 9（跨基准可分性）—— 中风险**：Is Our Benchmark Enough 已给出可分性诊断与「重叠流形/多顺序」建议（但只是 workshop 论文，正式版可能扩展到更多基准）。跨 4 基准 + 困难版仍是增量。
- **Idea 8（更新次数 scaling）—— 语言侧中风险**：WikiBigEdit 已在 1→500K 更新上画出编辑 / WISE / RAG / LoRA 退化曲线（250、500、10K 拐点）。语言侧只剩 SMF 稀疏更新这一条新曲线；视觉侧 1,000 任务闭式头 vs 单 LoRA 的曲线未被覆盖。
- **Idea 6（增益分解）—— 中低风险**：LoRA illusion (2410.21228) 已做「缩放 intruder 奇异值」的因果干预（对应步骤 b）；RanDumb 已给「线性头即超多数 prompt-tuning」总体结论。逐方法三分量分解仍未有人做。
- **Idea 2（KL-Razor for Vision）—— 低中风险**：RL's Razor 已在 robotic foundation models 上验证 KL（非纯 LLM），但未做视觉 CIL/CLIP、未做合并系数干预。
- **Idea 1（泄漏分层）—— 低风险，但需引用 APER**：APER 已因「数据重叠」提出新基准；CLDyB 只提污染；无人做泄漏分层。
- **Idea 5（审计合并基准）—— 低中风险**：Audit 是 workshop 论文、4 adapter；2511.21437 已做 6 方法 in-the-wild 基准（结论 TA 唯一可靠）但无审计。注意 Audit 本身是 workshop，主会版本可能正在扩展。
- **Idea 4（CTTA ∪ TF-CIL 同一流）—— 低风险**：TTABC 明确只含 TTA 方法；UL-TTA / BayesTTA / PTA 都是「原型 + 统计量」闭式更新，恰好支持统一形式，但无人把 RanPAC/RAIL 放进同一流。
- **Idea 10（预训练规模扫描）—— 低风险**：三篇 closest 都不扫描预训练规模。

### 0.5.5 仍无法核实
- AOCIL 真实 arXiv ID（原 2403.15751 属 F-OAL）。
- AnaCP、RL's Razor、Nested Learning/HOPE、GoD-MoE、CoIN、C-CLIP 的 venue（abs 无标注；本次未逐一查会议页）。
- CL-Bench 的 NeurIPS 2026 录用（仅 Snorkel 博客）。
- SMF「承认 RAG 是自然方案」、Spurious Forgetting「TRACE 上反弹」两句来自正文/片段，摘要未含。
- openreview 页面被浏览器验证拦截，Audit / CF-Router 正文未读，数字以 icml.cc 摘要为准。
- §1 其余 ~45 篇非 closest-work 条目（L2P/DualPrompt/CODA 等经典文献与 1.4–1.6 多数条目）本次未逐篇核实。

---

## 1. 论文清单（按子方向分组，共 ~70 篇）

格式：标题 | arXiv/venue | 日期 | 问题 | 方法一句话 | 关键数字 | training-free? | 自述 limitation

### 1.1 预训练 ViT 上的 prompt / adapter / LoRA CL（最饱和）

锚点：
- Learning to Prompt (L2P) | arXiv 2112.08654, CVPR 2022 | prompt pool + key-query 选择 | 非 TF（训 prompt）
- DualPrompt | arXiv 2204.04799, ECCV 2022 | G-prompt + E-prompt | 非 TF
- CODA-Prompt | arXiv 2211.13218, CVPR 2023 | 注意力加权的 prompt 分解，软匹配 | 非 TF；自述 average forgetting 略高于 DualPrompt
- Hierarchical Decomposition of Prompt-Based CL | arXiv 2310.07234 (NeurIPS 2023) | 复现 CODA-Prompt 官方代码数字与原文有明显差异；CODA 需要更小 lr + cosine decay | 诊断类
- One-stage Prompt-based CL (OS-Prompt) | arXiv 2402.16189 | L2P/DualPrompt 的 top-k 硬匹配比 CODA 的软匹配更稳 | 非 TF
- Prompt Customization | arXiv 2404.18060 | CODA-P 训练时间 ≈ DualPrompt 的 4× | 非 TF
- Additive Prompt Tuning | arXiv 2503.07979 (2025-03) | prompt pool 开销大 → 单一共享 prompt 集 | 非 TF
- GAP-Prompt: Gated Adaptive Prompting | arXiv 2608.23782 (2026-08-24) | 实例级门控选 prompt 注入层 + 动态知识融合 + 共享 prompt 蒸馏；CUB-200 87.29%（joint 88.00%） | 非 TF | abs 无 venue 信息
- SD-LoRA | arXiv 2501.13198, ICLR 2025 Oral | LoRA 幅度/方向解耦，rehearsal-free，推理无需选组件；声称收敛到各任务共享低损区 | 非 TF
- InfLoRA | arXiv 2404.00228, CVPR 2024 | 干扰消除子空间 LoRA（已核实） | 非 TF
- SAFE: Slow and Fast PEFT | arXiv 2411.02175 | 第一会话训 slow learner，之后 fast learner；slow learner 使最终准确率 +3.2 | 非 TF
- PLASTIC: Enhancing Plasticity for First-Session Adaptation | arXiv 2310.11482 v3 (2025-07), PMLR v330 (2026) | FSA 在异质任务序列上弱 → 测试时用无标签数据微调 LayerNorm | 半 TF（无标签 TTA） | 自述：FSA 在异质序列上塑性不足
- Revisiting Weight Regularization for Low-Rank CL (EWC-LoRA) | arXiv 2602.17559, ICLR 2026 | EWC 在 PEFT-CL 里被忽视；对单个共享低秩更新做 EWC，存储/推理成本不随任务数增长 | 非 TF
- Merge before Forget (SLAO) | arXiv 2512.23017, ICLR 2026 | 单 LoRA 持续合并：新任务从旧 LoRA 的正交基起步 + time-aware scaling；内存 O(1)；Llama 系列 | 非 TF（少训练）
- Pruned Adaptation Modules (PAM) | arXiv 2603.21170, CPAL 2026 (PMLR 328)；近似同文 Slim Adaptation Modules @ NeurIPS 2025 | **诊断/基线**：领域转向 FM-CIL 时没有和强轻量卷积基线对比；冻结 ResNet + 稀疏任务层，可训参数 ↓2-5×、总参数 ↓2-6×，仍超 SOTA FM-based CIL | 少训练 | 自述：方法学 gap——无法判断近期进展是真实还是基线弱
- 综述：Parameter-Efficient Continual Fine-Tuning: A Survey | arXiv 2504.13822 (2025-04)

### 1.2 完全 training-free / 解析式（analytic）/ 原型式 CL

- RanPAC | arXiv 2307.02251, NeurIPS 2023 | 冻结随机投影 + 非线性 + 原型去相关（Gram 矩阵逆）；7 个 CIL 基准相对以往 PTM 方法错误率 ↓20-62%（已核实，原文误写 10-62%），无 rehearsal | **TF**（仅闭式）
- RanDumb | arXiv 2402.08823, NeurIPS 2024 | 随机 RBF 特征 + 线性分类器即可超越在线 CL 的表示学习方法；用预训练特征时线性分类器超过多数 prompt-tuning | **TF** | 结论限定 low-exemplar / online 场景
- SimpleCIL / APER | arXiv 2303.07338, IJCV | SimpleCIL：冻结 PTM + 原型分类器不训练即胜 SOTA；APER：聚合 PTM 与适配模型嵌入；**并因 ImageNet 类基准与预训练数据重叠而提出 ImageNet-A / ObjectNet / OmniBenchmark / VTAB 新基准**（与 Idea 1 相关） | SimpleCIL 为 TF
- AOCIL | ID 待查（原文误标 2403.15751，该 ID 属 F-OAL），venue 未核实 | 冻结 ViT + RLS 更新的解析分类器，在线 exemplar-free | TF
- REAL | arXiv 2403.13522 | 自述解析 CL 的表示无效/利用不足 → 双流 base pretraining + 表示增强蒸馏 | 非 TF
- RAIL | arXiv 2406.18868, NeurIPS 2024 | CLIP 上递归岭回归 adapter + training-free 融合模块；提出 X-TAIL（跨域、无域提示、含未见域）；零样本能力「绝对保持」 | TF
- AnaCP: Analytic Contrastive Projection | arXiv 2511.13880（NeurIPS 2025 未在 abs 标注，待核实） | 自述 limitation：解析法「无法把 PTM 特征适配到下游，性能次优」→ 解析对比投影 | 近 TF
- LR-RGDA | arXiv 2602.00144 (2026-02) | ViT-CIL 的分类器重建多数靠迭代 SGD → 低秩因子化解析分类器（Woodbury） | TF
- VILA | arXiv 2602.13670 (2026-02), ICML 2026 | 解析 CIL 的瓶颈是「表示僵化 (representation rigidity)」+ 累积误差/特征不兼容 → 双分支 | 部分 TF
- Fly-CL | arXiv 2510.16877 (2025-10-19), ICLR 2026 | 预训练特征在相似度匹配阶段存在多重共线性；果蝇 PN→KC 式高维稀疏随机投影 + top-k + 流式岭分类；训练时间大幅下降、性能≥SOTA | **TF**（闭式）| 自述动机：RanPAC 类方法对低延迟场景太贵
- F-OAL (forward-only online analytic learning) | arXiv 2403.15751, NeurIPS 2024 | 冻结编码器 + 特征融合 + RLS 线性分类器，无反传 | TF
- Continual-CLIP | arXiv 2210.03114 | 冻结 CLIP 零样本就是强 CL 基线 | TF
- PTA: Prototype-Based TTA of VLMs | arXiv 2604.21360, ICML 2026 poster（主会） | 类原型按零样本置信加权积累测试样本知识，无 cache；CLIP 10 个跨域基准 65.64→69.38，保留 92% 推理速度 | TF

### 1.3 模型合并 / task arithmetic 用于 CL

- MagMax | arXiv 2407.06322, ECCV 2024 | 顺序微调减少符号冲突 + 最大幅度选择；**发现：普通权重平均/随机选择就很强，常胜复杂合并** | 少训练
- Sequential Fine-tuning with Averaging (SFA) | arXiv 2501.05559 (v1 2025-01；v2 2026-10-14 改名「Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives」) | 训练中周期性与早期 checkpoint 插值，免 replay；v2：Qwen3-14B adapter CL 上固定配置胜 replay 与 adaptive merging，训练时间 −36.3%；正文另比较 TA / TIES / WiSE-FT，EWC 在附录 | 少训练
- Scalable Strategies for CL with Replay | arXiv 2505.12512 (2025-05) | 比较并行合并(TA)、EMA、顺序合并用于长任务流；DARE-TIES 倾向保留预训练知识
- Layer-Aware Task Arithmetic (LATA) | arXiv 2502.20186 | 按层区分任务知识 vs 指令遵循知识加权 task vector；WikiText-2/GSM8K/HumanEval
- Closed-form LoRA merging for federated CL | arXiv 2410.17961, ICLR 2025 | 闭式合并使合并模型匹配各模块输出
- Aligned model merging for VLM CL (DeepMind) | arXiv 2506.03189 (2025-06) | 新任务参数与旧参数对齐合并，多任务顺序下遗忘减少
- Unraveling LoRA Interference | arXiv 2505.22934 | 正交子空间减少 LoRA 合并干扰
- Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition | ICML 2026 **CATS workshop** poster（非主会；无 arXiv ID） | **诊断**：把每次合并公共 LoRA 当作顺序任务（Task 0 = base 能力）；公共 hub 中大量 adapter base 版本不匹配或 Δ≈0，**会使测得的遗忘失真**；经审计的 4-adapter 集上朴素合并 GSM8K −10.5、HumanEval −18.3（Llama-3.1-8B-Instruct）；Mistral-7B 上 4 个候选中 2 个不过审 | TF（合并）
- A Systematic Study of In-the-Wild Model Merging for LLMs | arXiv 2511.21437, TMLR 2026-03 | 6 种合并法（含子空间法）× 4 个开源 LLM × 每模型 12 个 ckpt × 16 基准；**只有 Task Arithmetic 可靠带来增益**，干扰感知/子空间法基本不优于 base | TF（合并）
- MeMo: Memory as a Model | arXiv 2605.15156 (2026-05) | 把合并视为避免顺序微调分布干扰的手段
- LoRA vs Full FT: An Illusion of Equivalence | arXiv 2410.21228, NeurIPS 2025 | LoRA 产生「intruder dimensions」（新的高奇异值方向），全参微调不产生；因果干预（缩小其奇异值）证明它们导致遗忘；顺序微调中累积 | 对合并/遗忘的几何有含义

### 1.4 LLM 的持续学习（continual pretraining / instruction tuning / 知识编辑 / 轻量知识注入）

- Simple and Scalable Strategies to Continually Pre-train LLMs | arXiv 2403.08763 | re-warm + re-decay 增加适应也增加遗忘；replay 缓解；提出 infinite LR schedule
- Learning Dynamics in Continual Pre-Training | arXiv 2505.07796, ICML 2025 spotlight | CPT scaling law：分离分布偏移与 LR 退火效应，可预测任意步 loss，覆盖 peak LR / loss potential / replay ratio
- Spurious Forgetting in CL of LMs | arXiv 2501.13453 (ICLR 2025) | TRACE 上部分「遗忘」在后续任务后反弹——遗忘指标需谨慎
- Forget Forgetting: CL in a World of Abundant Memory | arXiv 2502.07274 | 在 TRACE 上研究 exemplar memory 充足时的策略
- TRACE | arXiv 2310.06762 | 8 域 LLM CL 基准，含 RCL（推理增强）
- GoD-MoE (Grow-on-Demand) | AAAI 2026 | attention 层插入 LoRA 专家，按任务动态激活小子集，胜 replay 基线
- Adaptation-aware probing | arXiv 2605.31025 (2026-05) | 持续微调中的任务专属参数发现；指出 replay / 任务 adapter 都有额外开销
- CLoRA | arXiv 2410.16801, ACL 2025 | LoRA 更新的零空间正交正则，限制输出漂移
- LoRA-Null | arXiv 2503.02659 | 从预训练知识激活的零空间构造 adapter
- RL's Razor | arXiv 2509.04259, NeurIPS 2025 | **同等新任务性能下 on-policy RL 比 SFT 遗忘少；遗忘由新任务上「与 base 的 KL」预测；权重变化幅度、稀疏度、梯度秩都不能稳定解释** | 诊断
- Continual Learning via Sparse Memory Finetuning | arXiv 2510.15103 (2025-10-16, Meta FAIR+Berkeley) | memory-layer 模型，仅更新相对预训练用量被新知识高度激活的 memory slot（TF-IDF 排名）；学新事实后 NQ F1：全参 −89%、LoRA −71%、本法 −11% | 少训练 | **自述：只评 factual QA；承认 RAG 是当下 factual learning 的自然方案**
- Nested Learning / HOPE | NeurIPS 2025 (Google) | 模型 = 多频率嵌套优化问题；Continuum Memory System；HOPE 是 Titans 改版 | 架构类 | 自述：camera-ready 被大幅压缩
- In-Place TTT | arXiv 2604.06169, ICLR 2026 Oral | 把已有 MLP 末投影矩阵当 fast weights 在推理中更新（next-token 对齐目标、chunk-wise）→ 走向 LLM 持续学习
- Test-Time Learning for LLMs (TLM) | arXiv 2505.20633, ICML 2025 | 用测试输入困惑度最小化适应；LoRA 比全参更能缓解遗忘
- Agentic TTT (aTTT) | arXiv 2607.03441 (2026-07) | 多轮 agent 的持续 TTT 会形成自训练回路放大漂移；token 级重加权
- Mechanistic Analysis of Catastrophic Forgetting (2601.18699) | 2026-01 | v1 vs v2 作者/机构/内容严重不一致（v2 含可疑的 1.5T 模型、94.2% 等数字）——**不建议引用**
- 知识编辑持续性：
  - WilKE | arXiv 2402.10987 | ROME/MEMIT 终身编辑「毒性累积 / 毒性闪现」
  - Rebuilding ROME (r-ROME) | arXiv 2403.07175 | 解决顺序编辑崩溃；自述下游退化在规模化时仍存在
  - Lifelong KE requires Better Regularization | arXiv 2502.01636, EMNLP 2025 Findings | 退化来自激活过优化 + 编辑矩阵范数增长 → 早停 + Frobenius 约束
  - WikiBigEdit | arXiv 2503.05683, ICML 2025 | 50 万+ 真实 Wikidata 编辑；局部编辑法（MEMIT/ROME）**250 次更新内趋近零**；专门编辑法不如简单持续 LoRA 微调；RAG 用推理时间换性能
  - Towards a Principled Evaluation of Knowledge Editors | arXiv 2507.05937 | string-match 评估产生假阳性；ICL/RAG 也是一种编辑
  - Selective Knowledge Suppression | arXiv 2604.19089 (2026-04) | 可扩展终身编辑
- 综述：2402.01364；2404.16789；2603.12658 (2026-03, CL in LLMs: Methods, Challenges, Opportunities)

### 1.5 记忆 / 检索式 CL、in-context CL、agent memory

- SelfAug | arXiv 2509.03934, EMNLP 2025 Findings | RAG 场景 SFT 的遗忘 = 分布偏移 → logit 自对齐
- 「RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World」(Chronos) | ACL 2026 Findings (2026.findings-acl.546) | 带时间戳真实事件基准，同一基准上评估 continual finetuning / 知识编辑 / RAG；vanilla RAG 和多种学习法都受「遗忘 + 时间不一致」困扰；提出无训练的 time-aware retrieval baseline Chronos（Event Evolution Graph） | TF（Chronos）
- Memento 2: Stateful Reflective Memory | arXiv 2512.22716 (2025-12) | 对 episodic memory 做反思即可持续适应，不改参数；给出收敛证明
- Decoupling Memory from Context (GraphMemory) | arXiv 2610.02687 (2026-10-06), NeurIPS 2026 **TTCL workshop**（非主会） | 共享上下文不断追加 → token 成本/窗口/退化；图记忆只检索相关子图；记忆构建 token ↓81-85% | TF
- Modular Memory is the Key to CL Agents | arXiv 2603.01761 | 立场文：ICL 快适应 + in-weight 慢更新
- ContinualSkillBench | arXiv 2608.03874 (PKU) | 5 域×100 链式子任务；**ICL 平均上与显式技能库维护相当**；弱模型会积累更碎片化的技能
- CL-Bench (Continual Learning Bench) | arXiv 2606.05661 (2026-06, Berkeley + Snorkel + UW-Madison)；NeurIPS 2026 仅 Snorkel 博客称，未核实 | 6 域真实有状态环境；**朴素 ICL 胜过专门记忆系统**；摘要明确不评估参数化方法（TTT 等）
- AGENTCL | arXiv 2606.02461 | 现有基准偏向长上下文检索/推理，提出更严格的 agent CL 评测
- Letta「Continual Learning in Token Space」(博客, 2025-12) | 非论文
- Learning on the Job: CL from Deployment Feedback for Frozen-Weights Agents | arXiv 2607.22157 (2026-07-24, 无 venue) | 冻结模型 + 外部记忆把每个 episode 蒸馏为自然语言规则；τ-bench banking 上 outcome 反馈 1.6×、corrections 2.6×；记忆可跨模型迁移 | TF（原文误称「Spark memory」）

### 1.6 MLLM / VLM 持续学习

- CoIN | NeurIPS 2024 D&B | 10 数据集 8 任务；**遗忘主因是指令对齐失败而非推理知识遗忘**
- Continual Instruction Tuning for LMMs | arXiv 2311.16206 | replay 与模型扩展有效；正则法只在联合指令微调过的模型上有效
- Improving MLLMs Using CL | arXiv 2410.19925, CoLLAs 2025 | 把 LLaVA 集成当作 CL：文本能力退化 ↓15%
- HiDe-LLaVA | arXiv 2503.12941 (2025-05) | 按层 CKA 相似性决定任务专属扩展 vs 通用融合
- MLLM-CL | arXiv 2506.05453 (2025-06) | 域 CL（遥感/医疗/自驾/科学/金融，IID）+ 能力 CL（OCR/数学/感知/GUI，非 IID）；MR-LoRA 参数隔离 + MLLM 路由
- MLLM-CTBench | arXiv 2508.08275 (2025-08) | 7 任务 6 域；**同时评分 CoT 轨迹**；GRPO（有 KL 控制）比 SFT 跨任务保持更稳，去掉 KL 加剧遗忘
- Routing-Based CL for MLLMs | arXiv 2511.01831 (2025-11, v3 2026-04, Amazon) | token 级路由专家；2B-8B 上匹配多任务上界且计算固定；跨模态知识迁移
- CF-Router | ICLR 2026 **LLA workshop** poster（非主会；无 arXiv ID） | MLLM 末层隐藏态做描述子 + 正则化最小二乘分类器选 expert LoRA（闭式路由） | 近 TF
- φ-DPO | arXiv 2602.22601 (2026-02) | 公平 DPO 用于 LMM CL；在 MLLM-CL 上 RS 85.58 / Finance 95.28 等
- **Is Our Benchmark Enough? An Analysis of CL for MLLMs** | ICML 2026 **CATS workshop** oral（非主会；无 arXiv ID；TCD + TU/e） | **诊断**：(1) training-free、replay-free 原型路由 RePro（冻结特征 + 任务原型）匹配 MR-LoRA 的 MLLM 路由器；(2) 共享专家无益；(3) MLLM-CL 任务在表示空间高度可分、单一固定任务顺序 → 基准奖励「孤立学习」而非持续迁移；建议重叠任务流形、多任务顺序、细粒度域偏移、forward-transfer 指标
- C-CLIP | ICLR 2025 | 多域 image-caption 的多模态 CL 基准，追踪零样本遗忘
- ZSCL | arXiv 2303.06628, ICCV 2023 | 特征蒸馏（无标签参考集）+ 训练中权重平均保护零样本；提出 MTIL 基准，平均 +9.7%（已核实）
- MG-CLIP: Mind the Gap | arXiv 2507.09118, ICCV 2025 | 模态 gap 大小追踪预训练知识保留程度；gap 保持 + gap 补偿，无 replay
- Open-domain VLM CL (SMU, 2025-08) | 流中含已见/未见域的新类
- Generative prompt learning (VAE replay) | arXiv 2407.15793
- Dual-phase CL with unlabeled test data | arXiv 2406.13653, TMLR 2026 | 教师-学生 + 稀疏更新，用无标签测试数据减少 VLM CIL 遗忘

### 1.7 评估问题 / 诊断

- CLDyB | arXiv 2503.04655 (2025-03) | **PTM 预训练可能污染 CL 基准**，静态基准饱和 → MDP + MCTS 动态生成任务序列
- Data Leakage in Visual Datasets | arXiv 2508.17416 (2025-08) | ImageNet test 内部泄漏 1.54%(hard)/1.95%(soft)；LAION 等预训练集对基准的泄漏可达数个百分点
- Hierarchical Decomposition (2310.07234)：CODA-Prompt 复现差异
- PAM / Slim Adaptation Modules：基线缺失
- Is Our Benchmark Enough (ICML 2026 CATS workshop)：MLLM-CL 可分性
- Audit Before You Merge (ICML 2026 CATS workshop)：adapter 来源污染遗忘测量
- Spurious Forgetting (2501.13453)：遗忘反弹
- RL's Razor：遗忘预测量
- WikiBigEdit：编辑法 vs LoRA vs RAG
- Rethinking 类：「Towards a Principled Evaluation of Knowledge Editors」(2507.05937)

### 1.8 与 TTA / online / test-time CL 的交叉

- TTABC / What Drives TTA for CLIP | arXiv 2606.14299 (2026-06, v3 2026-10), NeurIPS 2026 D&B Track | 20+ 方法统一基准；**增益主要来自测试时证据 + 可靠代理而非重优化；跨/当前样本证据 + 轻量原型更新即可有竞争力；无普适最优范式**（norm 层更新适合 corruption，stream-aware 适合 label shift）；在线累积参数更新带来效率与稳定性问题
- BayesTTA | arXiv 2507.08607 (2025-07) | CLIP 在渐变偏移（光照/季节）下退化；现有持续 TTA 假设突变剧烈偏移
- MS-TTA | arXiv 2507.00462, IEEE TCSVT | 单步 kNN Mean-Shift，TF
- UL-TTA | arXiv 2511.09101 (2025-11) | 冻结骨干，只调 logit 级参数（原型/先验/温度），TF、无反传，流式
- MINT | arXiv 2506.03190 | 记忆 prompt bank（可学习）
- BPRE | arXiv 2503.09394 | 原型-奖励共进化
- Just Shift It (TPS) | arXiv 2403.12952 | 测试时原型平移
- PLASTIC（见 1.1）：FSA + TTA
- Dual-phase（见 1.6）
- NeurIPS 2026 workshop「Towards Test-Time Continual Learning Agents」已成立——方向正在被制度化

---

## 2. Landscape map：饱和 vs 起步

| 子方向 | 状态 | 证据 |
|---|---|---|
| ViT 上 prompt-pool / LoRA-pool CIL（CIFAR-100、ImageNet-R、CUB） | **做烂**（饱和，且被诊断论文围攻） | L2P→DualPrompt→CODA→OS-Prompt→Additive→GAP-Prompt；SD-LoRA/InfLoRA/EWC-LoRA/SLAO；RanPAC/RanDumb/PAM 反复证明 TF 或轻量基线即可打平 |
| 解析式 / 随机投影 TF-CIL | **饱和中但仍有空间**：方法已多（RanPAC、Fly-CL、AnaCP、LR-RGDA、VILA、AOCIL），空间在「表示僵化」与「特征不适配」问题（AnaCP、VILA 自述） | |
| ViT-CIL 基准有效性（预训练泄漏） | **刚起步**：CLDyB、Data Leakage 只给了问题，没给「泄漏量 ↔ TF 方法优势」的因果证据 | |
| 模型合并用于 CL（视觉） | 中等饱和：MagMax、SFA、2505.12512 已把「平均就很强」说透 | |
| 模型合并用于 CL（LLM、公共 adapter） | **刚起步**：Audit Before You Merge 刚指出测量污染；2511.21437 刚做公开 ckpt 基准 | |
| LLM 顺序 LoRA/MoE CL（TRACE 类） | 饱和：O-LoRA/CLoRA/LoRA-Null/GoD-MoE/SLAO | |
| LLM 遗忘的「可预测量」 | **刚起步、高价值**：RL's Razor 只给 KL 一个量，且只在 RL vs SFT 对比中验证 | |
| memory-layer / sparse-slot 持续注入 | **刚起步**：SMF 只在 factual QA；需要 memory-layer 架构 | |
| 终身知识编辑 | 饱和于「ROME/MEMIT 崩溃」；WikiBigEdit 已宣判「不如 LoRA」 | |
| RAG/ICL vs 参数化 CL 的正面对比 | **起步且结论混乱**：CL-Bench、ContinualSkillBench 说朴素 ICL 赢；WikiBigEdit 说 RAG 换推理时间；Chronos 说 vanilla RAG 时间不一致；SMF 承认 RAG 是自然方案 | |
| MLLM 持续指令微调 | 方法饱和（HiDe-LLaVA、MR-LoRA、Routing、φ-DPO），**基准被诊断为无效**（Is Our Benchmark Enough） | |
| MLLM CL 的推理过程 / RL 视角 | 起步：MLLM-CTBench 一篇 | |
| continual TTA ↔ CL 交叉（CLIP） | 起步：TTABC 刚统一；BayesTTA 指出渐变偏移未被覆盖；无「test-time CIL」标准设定 | |
| agent memory 作为 CL | 热但混乱：多基准互相矛盾，缺统一的「遗忘」定义 | |

---

## 3. 五个 lens 的结构性 gap

### Lens A：Method-transfer（跨域方法迁移）

**A1. 「闭式/随机投影分类头」从 ViT-CIL 迁移到 MLLM 的持续指令微调路由与能力保持。**
依据：Fly-CL (2510.16877)、RanPAC (2307.02251) 证明冻结特征 + 闭式头在 ViT 上足够；Is Our Benchmark Enough (ICML 2026) 证明 training-free 原型路由 RePro 可匹配 MR-LoRA 的 MLLM 路由；Routing-Based CL (2511.01831) 仍用可训练 token 路由。缺口：没有人把 RanPAC/Fly-CL 式的去相关闭式头直接当作 MLLM 的 LoRA-expert 路由器或作为「能力 CL」里输出头，并在非 IID 的 ability track 上验证。

**A2. 「KL-to-base 作为遗忘预测量」（RL's Razor）迁移到视觉 PTM-CL 与模型合并。**
依据：RL's Razor (2509.04259) 发现 LLM 遗忘由新任务上相对 base 的 KL 预测，而权重变化幅度/稀疏度/梯度秩都不行；MagMax (2407.06322) 与 SFA (2501.05559) 用权重空间启发式（幅度、平均）选参数；Audit (ICML 2026) 用 Δ-norm 做审计。缺口：视觉 CL 与合并领域仍用权重空间量作为「干扰」代理，没有人检验 output-distribution KL 是否在 ViT/CLIP 上同样是更好的预测量，也没人用它作为合并系数的选择准则。

**A3. 「时间感知检索」(Chronos) 与「Spurious forgetting」思想迁移到 agent memory 基准的遗忘定义。**
依据：Chronos (ACL 2026 Findings) 指出 vanilla RAG 时间不一致；Spurious Forgetting (2501.13453) 指出 LLM CL 中遗忘会反弹；CL-Bench / ContinualSkillBench 对 ICL vs memory 结论不一致。缺口：agent memory CL 基准没有「遗忘 vs 暂时对齐丢失」的区分，也没有时间一致性维度。

### Lens B：Contradiction（文献内矛盾）

**B1. 「RAG/ICL 是无遗忘的 CL」 vs 「RAG 本身会遗忘/不一致」。**
CL-Bench (NeurIPS 2026) 与 ContinualSkillBench (2608.03874) 说朴素 ICL ≥ 专门 memory 系统；SMF (2510.15103) 承认 RAG 是自然方案；但 Chronos (ACL 2026) 说 vanilla RAG 有遗忘与时间不一致，GraphMemory (2610.02687) 说追加式 context 性能退化，SelfAug (2509.03934) 说 RAG 场景 SFT 仍遗忘，WikiBigEdit (2503.05683) 说 RAG 用推理时间换。矛盾核心：各家「遗忘」的操作定义不同（能力遗忘 vs 事实过期 vs 技能碎片化）。无一篇在同一任务流、同一模型、同一预算下用统一的遗忘定义对比 ICL / RAG / LoRA-CL / 编辑法。

**B2. 「KL 控制减少遗忘」 vs 「合并/平均减少遗忘」——两种解释是否同源？**
RL's Razor (2509.04259)、MLLM-CTBench (2508.08275, GRPO 的 KL 控制更稳)、EWC-LoRA (2602.17559, 参数空间正则) 、SFA (2501.05559, checkpoint 平均)、MagMax（平均即可）。RL's Razor 明确说权重空间量不预测遗忘，但 MagMax/SFA 的成功完全是权重空间操作。未被解释：权重平均之所以有效，是否恰恰因为它在 output KL 上最小？目前没有把两类方法放在同一 KL 坐标系下的实验。

**B3. 「解析/TF 方法够用」 vs 「解析方法特征僵化」。**
RanDumb (2402.08823)、PAM (2603.21170)、Fly-CL、RePro (ICML 2026) 说 TF/轻量即可；AnaCP (2511.13880)、VILA (2602.13670)、REAL (2403.13522) 自述解析法因无法适配特征而次优。两边都对，但没有人回答：在什么「预训练-下游域距离」或「泄漏比例」下 TF 开始落后？CLDyB (2503.04655) 与 Data Leakage (2508.17416) 提供了测这件事的工具却未结合。

### Lens C：Untested-assumption（未检验假设）

**C1. 假设：ViT-CIL 基准（CIFAR-100、ImageNet-R、CUB、Cars）上 TF 方法的胜利反映的是方法优劣，而非 ImageNet-21K/LAION 预训练对这些数据集的覆盖。**
依据：CLDyB 只提出「可能污染」；Data Leakage 只测了 ImageNet 自身与 LAION→基准的泄漏比例；PAM 指出基线缺失。无人做过「按泄漏/相似度分层后 TF 与可训练方法差距如何变化」的分析。

**C2. 假设：公共 LoRA 合并实验里测到的「遗忘」来自干扰。**
Audit Before You Merge 已指出大量 adapter base 版本不匹配或 Δ≈0——即过去用公共 adapter 做的合并-遗忘结论（含 2511.21437 一类的公开 ckpt 基准）可能部分是伪影。未检验：审计前后，TIES/DARE/TA 的相对排名是否改变。

**C3. 假设：MLLM CL 基准中「任务可分」是普遍现象还是 MLLM-CL 特有。**
Is Our Benchmark Enough 只分析了 MLLM-CL；CoIN、MLLM-CTBench、2311.16206 是否同样高度可分（从而同样被原型路由「解决」）未检验。若是，整条 MLLM-CL 方法线（HiDe-LLaVA、MR-LoRA、Routing、φ-DPO）的增益需要重估。

**C4. 假设：continual TTA 的「渐变偏移」与 CL 的「离散任务」是不同问题。**
BayesTTA 指出 CTTA 假设突变；TTABC 指出 stream-aware 设计适合 label shift。CL 社区的 class-incremental 本质是极端 label shift。无人在同一 CLIP 骨干、同一数据流上同时跑 CTTA 方法（UL-TTA、MS-TTA）和 TF-CIL 方法（RanPAC、RAIL），看它们是不是同一方法族的两种参数化。

### Lens D：Scaling-regime（规模/资源区间）

**D1. 「任务数 → 数百/数千」区间下 TF 闭式头 vs 单 LoRA 合并的内存/精度曲线缺失。**
SLAO、EWC-LoRA 强调 O(1) 内存；RanPAC/Fly-CL 的 Gram 矩阵在类数极大时需要正则（Fly-CL 自述多重共线性）；WikiBigEdit 显示编辑法 250 次内崩溃。无人把「更新次数」作为主轴在 10→10,000 的范围内画 TF vs 合并 vs 编辑的曲线（视觉与语言各一条）。

**D2. 小模型/单卡区间下的 LLM CL：SMF (memory layer)、Routing (2B-8B)、RL's Razor 都在 ≥1B 模型；TF 式 LLM CL（例如用冻结 LLM 的隐藏态 + 闭式头做分类型持续任务）在 0.5B-3B 的 Qwen/Llama 上几乎没有系统研究。而 TRACE 类基准多数任务是分类/短生成，恰好适合闭式头。**

**D3. 预训练数据规模 → TF 优势：Continual-CLIP (2210.03114) 与 PTA (ICML 2026) 暗示零样本能力越强 TF 越占优；Data Leakage 说明 LAION 规模决定泄漏。没有「同一架构、不同预训练数据量（CLIP 400M / LAION-2B / DataComp）下 TF-CL 与训练式 CL 差距」的扫描。**

### Lens E：Diagnostic（诊断类）

**E1. 「训练式 PTM-CL 方法到底学到了什么」：** 把 CODA-Prompt/SD-LoRA/InfLoRA 学到的更新投影到「冻结特征 + 闭式头」能解释的子空间，看残差贡献。依据：Hierarchical Decomposition (2310.07234) 已发现复现差异；PAM 发现基线缺失；RanDumb 说表示学习贡献小；LoRA intruder dimensions (NeurIPS 2025) 给出几何工具。无人做「可训练方法的增益分解 = 分类头效应 + 特征适配效应 + 调参效应」。

**E2. 「MLLM 持续指令微调的遗忘到底是什么」：** CoIN 说是指令对齐失败；MLLM-CTBench 说 CoT 轨迹揭示隐藏遗忘；Spurious Forgetting 说 LLM 的遗忘可逆；Is Our Benchmark Enough 说基准可分。缺一个把「对齐丢失 / 知识丢失 / 路由错误」三者定量拆开的诊断（例如用 RePro 式原型路由作为对照：若原型路由恢复了大部分性能，则遗忘主要是路由/对齐问题）。

**E3. 「权重空间量 vs 输出分布量」作为遗忘预测器的系统评估（跨视觉/语言/合并）**：RL's Razor 只在 RL-vs-SFT 中验证了 KL；Audit 用 Δ-norm；SFA/MagMax 用幅度。缺一个跨 8-10 种 CL 方法、2 模态的「预测器 vs 实测遗忘」相关性表。

---

## 4. 候选 idea（10 个）

### Idea 1：Leakage-Stratified CL — 预训练泄漏能解释多少 training-free 方法的优势？
- **Method**：(1) 对 CIFAR-100 / ImageNet-R / CUB / Cars / ObjectNet / 新采集的 2025-后数据，用 CLIP/DINO 特征对 ImageNet-21K 与 LAION 子集做近重复检索（复用 Data Leakage 2508.17416 的 hard/soft 定义），给每个类打「泄漏分」；(2) 在同一 ViT-B/16 (IN21K) 上跑 RanPAC、Fly-CL、SimpleCIL（TF）与 CODA-Prompt、SD-LoRA、EWC-LoRA（训练式）；(3) 按泄漏分分层报告 TF−trained 差距；(4) 构造「零泄漏」子集作为新协议。
- **Hypothesis**：TF 方法的优势随泄漏分单调增加；在零泄漏层训练式方法反超 ≥3 pts。
- **MVE**：ViT-B/16 IN21K；6 数据集；指标：分层 Last-Acc 差、与泄漏分的 Spearman；GPU：特征提取 + 6 方法×6 数据集 ≈ 30-40 单卡小时（A100/4090）。
- **Closest work**：CLDyB (2503.04655)：提出污染但未量化；Data Leakage (2508.17416)：量化泄漏但不涉及 CL 方法；PAM (2603.21170)：基线批评但不涉及数据；**APER (2303.07338, IJCV) 已因数据重叠提出 ImageNet-A/ObjectNet/OmniBenchmark/VTAB 新基准，须引用并区别**。区别：首次把泄漏量与方法类型交叉。
- **Novelty 7/10**；**Risk**：泄漏分与「域相似度」混淆，需要用 DINO 域距离做协变量。
- **Contribution type**：诊断 + 评测协议。
- **Reviewer objection**：「近重复检索阈值任意」；「泄漏≠记忆」。
- dedup_key：`cl_leakage_stratified_tf_vs_trained`

### Idea 2：KL-Razor for Vision — 输出分布 KL 是否也是 PTM-CL/合并的遗忘预测量？
- **Method**：(1) 在 ViT/CLIP 的 CIL（ImageNet-R、CUB）和 CLIP 的 MTIL 上跑 8 种方法（CODA、SD-LoRA、EWC-LoRA、SLAO、MagMax、SFA、TA、全微调）；(2) 每步记录：新任务数据上相对 base 的 KL、权重 Δ-norm、更新稀疏度、梯度秩、intruder-dimension 数；(3) 回归实测遗忘；(4) 把 KL 作为 MagMax/SFA 的合并系数选择准则做干预实验。
- **Hypothesis**：KL 在视觉上同样是最强预测量（r>0.8），权重量 r<0.5；用 KL 选合并系数比固定系数遗忘 ↓20%。
- **MVE**：ViT-B/16；ImageNet-R 10 任务 + MTIL；指标：预测器-遗忘相关性、干预后 AvgAcc/Forgetting；GPU：约 40-60 单卡小时。
- **Closest**：RL's Razor (2509.04259，LLM + robotic foundation models、RL vs SFT；正文已排除 L1/Fisher-L2/谱范数/稀疏度/秩)；MagMax (2407.06322，权重启发式)；LoRA intruder dims (2410.21228, NeurIPS 2025，已做缩放 intruder 奇异值的因果干预)。区别：视觉 CIL/CLIP 验证 + 作为合并准则的干预。
- **Novelty 7/10**；**Risk**：视觉分类的 KL 在 class-incremental 下因输出头变化难以定义，需用特征空间 KL 或 logit 子集。
- **Contribution**：诊断 + 方法准则。
- **Objection**：「KL 与性能是循环定义」——需用新任务 KL 预测旧任务遗忘来避免。
- dedup_key：`kl_forgetting_predictor_vision_merging`

### Idea 3：Closed-Form Ability Router — 用 RanPAC/Fly-CL 式闭式头替代 MLLM 路由器与输出头做 MLLM 持续能力学习
- **Method**：(1) 冻结 MLLM（Qwen2-VL-2B / InternVL-2B），取视觉+文本池化隐藏态；(2) 高维稀疏随机投影 + 去相关（Fly-CL）构造任务路由器，闭式更新；(3) 每任务只训 rank-8 LoRA，由路由器选择；(4) 在 MLLM-CL ability track（非 IID）和 CoIN 上评估，并与 RePro（原型路由）、MR-LoRA 比较。
- **Hypothesis**：闭式去相关路由在非 IID ability track 上比原型路由高 ≥3 pts，且无需 MLLM 推理做路由（延迟 ↓10×）。
- **MVE**：2B 模型；MLLM-CL ability + CoIN；指标：AvgAcc、BWT、路由准确率、延迟；GPU：LoRA 训练 ~20h + 评估 ~10h 单卡。
- **Closest**：**CF-Router (ICLR 2026 LLA workshop)：已用 MLLM 末层隐藏态 + 正则化最小二乘闭式分类器选 expert LoRA——与本 idea 的 (1)(2)(3) 几乎重合**；Is Our Benchmark Enough (ICML 2026 CATS workshop, RePro training-free 原型路由)；Fly-CL (2510.16877, 仅 ViT 分类)；Routing-Based CL (2511.01831, 训练路由)。区别只剩：去相关/稀疏随机投影路由 vs 岭回归路由，以及非 IID ability track + 多顺序。**不能再声称「首次闭式路由」**，应改为闭式路由族对照。
- **Novelty 4/10（核实后下调）**；**Risk**：CF-Router 已是闭式 LS 路由；Is Our Benchmark Enough 若已指出 ability track 也可分，增益可能小。
- **Contribution**：方法（轻量）。
- **Objection**：「只是换了路由器」；需展示在多任务顺序和重叠任务流形上仍成立。
- dedup_key：`closed_form_router_mllm_cl`

### Idea 4：One Stream, Two Communities — CTTA 与 TF-CIL 在同一 CLIP 数据流上的统一对照
- **Method**：(1) 构造 CLIP 数据流：ImageNet-C 渐变 corruption（BayesTTA 风格）× 类增量到达（CIL）× 标签偏移；(2) 跑 CTTA 侧 UL-TTA、MS-TTA、PTA、TPS 和 CL 侧 RanPAC、RAIL、Continual-CLIP；(3) 把两族都写成「原型 + 统计量更新」的统一公式，分析差异仅在更新规则；(4) 提出混合更新规则。
- **Hypothesis**：两族在混合流上各自崩在对方擅长的维度；统一的「原型 + 协方差 + 先验」闭式更新同时拿到两边 ≥90% 的最优值。
- **MVE**：CLIP ViT-B/16；ImageNet-C/R + CIFAR-100-C 流；指标：流式准确率、Forgetting、延迟；GPU：全部 TF，<15 单卡小时。
- **Closest**：TTABC (2606.14299, NeurIPS 2026 D&B, 只有 TTA 族，已确认不含 CIL 方法)；BayesTTA (2507.08607, 渐变偏移，GDA 增量估计)；UL-TTA (2511.09101, 闭式原型/先验更新)；PTA (2604.21360, 原型积累)；RAIL (2406.18868, NeurIPS 2024, X-TAIL)。区别：首次把 CTTA 与 CIL 方法放进同一流并给统一形式。
- **Novelty 7/10**；**Risk**：流设计被批「人造」。
- **Contribution**：诊断 + 统一框架 + 轻量方法。
- **Objection**：「两个社区问题定义不同，对比不公平」——需多种流配置。
- dedup_key：`ctta_cil_unified_clip_stream`

### Idea 5：Audited Merging Benchmark — 审计前后 TIES/DARE/TA 的持续合并排名是否改变？
- **Method**：(1) 复用 Audit Before You Merge 的两项检查（base 匹配、Δ-norm）扫描 HF 上 Llama-3.1-8B / Qwen2.5-7B 的 200+ 公共 LoRA；(2) 构造「审计通过」与「未审计」两个 8-adapter 序列；(3) 顺序合并（TA、TIES、DARE-TIES、SFA 式平均、SLAO 式正交合并）；(4) 测 GSM8K/HumanEval/MMLU/IFEval 的遗忘，比较两组排名。
- **Hypothesis**：未审计集里 ≥30% 的「低遗忘」来自 Δ≈0 伪影；审计后方法排名发生显著变化（Kendall τ<0.5）。
- **MVE**：7-8B 模型，仅推理；GPU：评估 5 方法×2 组×4 基准 ≈ 25 单卡小时（vLLM）。
- **Closest**：Audit Before You Merge (ICML 2026 CATS workshop，只做 4 adapter、仅比较 naive vs TA λ=0.5，未比较 TIES/DARE 排名)；2511.21437（TMLR 2026，6 方法 in-the-wild 合并基准，结论只有 TA 可靠，无审计）；MagMax（视觉）。注意 Audit 为 workshop 文，其主会扩展版可能正在做本 idea。
- **Novelty 5/10**；**Risk**：公共 adapter 质量过差导致无可用序列。
- **Contribution**：评测 + 诊断。
- **Objection**：「增量小」——需补充 adapter-hub 规模统计与自动审计工具。
- dedup_key：`audited_lora_continual_merging_ranking`

### Idea 6：Decomposing the Gain — 训练式 PTM-CL 方法的增益 = 分类头 + 特征适配 + 调参
- **Method**：(1) 对 CODA-Prompt、SD-LoRA、EWC-LoRA、InfLoRA 各取训练后模型；(2) 三步消融：a) 冻结其特征，换成 RanPAC 闭式头；b) 用其特征但把更新投影掉 intruder dimensions；c) 用原始冻结特征 + 其分类头；(3) 对每个方法报告三者贡献；(4) 用 Hierarchical Decomposition 的复现设置控制调参效应。
- **Hypothesis**：≥60% 的增益来自分类头 + 调参；特征适配贡献 <2 pts，且在高泄漏数据集上趋零。
- **MVE**：ViT-B/16；CIFAR-100、ImageNet-R、CUB；指标：各分量贡献；GPU：~30 单卡小时。
- **Closest**：RanDumb (2402.08823，只给「线性头即超多数 prompt-tuning」总体结论)；PAM/Slim (2603.21170 / NeurIPS 2025)；LoRA intruder dims (2410.21228，已做「缩放 intruder 奇异值」干预，即本 idea 步骤 b 的雏形)。区别：逐方法的三分量增益分解。
- **Novelty 6/10**；**Risk**：分解不唯一。
- **Contribution**：诊断。
- **Objection**：「Shapley 式分解更严谨」。
- dedup_key：`ptm_cl_gain_decomposition`

### Idea 7：Unified Forgetting Ledger — 同一任务流下 ICL / RAG / LoRA-CL / 编辑 / sparse-memory 的统一遗忘定义对比
- **Method**：(1) 用 WikiBigEdit 子集 + TRACE 的 2 个任务 + Chronos 式时间戳事件构造一条 2,000 更新的流；(2) 定义三种遗忘：能力遗忘（MMLU/GSM8K 掉分）、事实过期（旧答案仍被输出）、暂时对齐丢失（Spurious 式可恢复）；(3) 方法：ICL（全上下文）、RAG、持续 LoRA、WISE/AlphaEdit、SMF 式稀疏更新；(4) 固定推理+训练总 FLOPs 预算。
- **Hypothesis**：按三种遗忘拆开后，CL-Bench「ICL 赢」与 Chronos「RAG 不一致」不再矛盾：ICL 零能力遗忘但事实过期最高。
- **MVE**：Qwen2.5-1.5B/3B；指标：三类遗忘 + 等 FLOPs；GPU：LoRA/编辑训练 ~30h，推理 ~15h 单卡。
- **Closest**：WikiBigEdit (2503.05683, ICML 2025，编辑 vs LoRA vs RAG，但无能力/时间维度)；CL-Bench (2606.05661，明确不评参数化方法)；**「RAG or Learning?」/Chronos (ACL 2026 Findings)：已在同一时间戳基准上对比 continual finetuning / 编辑 / RAG 并报告遗忘 + 时间不一致——与本 idea 的 (1)(3) 大幅重合**。区别只剩：三类遗忘的分离定义、等 FLOPs 预算、加入 ICL 与 SMF 线。
- **Novelty 5/10（核实后下调）**；**Risk**：工作量大，需严格裁剪到 2 域；与「RAG or Learning?」的差异必须靠遗忘定义而非基准本身。
- **Contribution**：评测协议 + 诊断。
- **Objection**：「每条线都有更强的专门方法」。
- dedup_key：`unified_forgetting_ledger_icl_rag_lora_edit`

### Idea 8：Scaling the Update Count — 10→10,000 次更新下 TF 闭式头 vs 单 LoRA 合并 vs 编辑的退化曲线
- **Method**：(1) 视觉：ImageNet-21K 子集按 1 类/任务切成 1,000 任务，跑 RanPAC、Fly-CL（闭式）与 SLAO、EWC-LoRA（单 LoRA）；(2) 语言：WikiBigEdit 1→10,000 编辑，跑 LoRA-CL、SMF 式稀疏更新、AlphaEdit；(3) 记录精度、内存、Gram 条件数 / 编辑矩阵范数；(4) 拟合退化曲线并找拐点。
- **Hypothesis**：闭式头在类数 >2,000 时因 Gram 病态开始退化（需 Fly-CL 式稀疏化救回），而单 LoRA 合并在 ~200 任务后遗忘超越闭式头；两条曲线交点给出「何时该训练」的实用规则。
- **MVE**：ViT-B/16 + Qwen2.5-1.5B；GPU：视觉几乎 TF（~10h），语言 ~40h。
- **Closest**：WikiBigEdit（ICML 2025；已在 1→500K 更新上画出 MEMIT/WISE/RAG/LoRA 退化曲线：局部编辑 250 次内崩、WISE <500 次可比 RAG、10K 内退回——语言侧曲线已大部分存在）；SLAO（O(1) 内存，任务数 ≤ 几十）；Fly-CL（多重共线性）。区别：视觉侧 1,000 任务闭式头 vs 单 LoRA 曲线 + 语言侧加入 SMF 稀疏更新；两条曲线的交点规则。
- **Novelty 6/10**；**Risk**：1,000 任务视觉基准太人造。
- **Contribution**：scaling 诊断。
- **Objection**：「极端任务数无现实意义」——用 WikiBigEdit 的真实时间流回应。
- dedup_key：`update_count_scaling_tf_vs_lora_vs_edit`

### Idea 9：Is Separability the Whole Story? — 跨 4 个 MLLM-CL 基准的任务可分性与「原型路由上界」
- **Method**：(1) 对 MLLM-CL、CoIN、MLLM-CTBench、2311.16206 基准，用冻结 CLIP/MLLM 特征计算任务可分性（线性探针准确率、Fisher 比）；(2) 对每个基准跑 RePro 式原型路由 + per-task LoRA 作为「路由上界」；(3) 与各基准上报告的 SOTA 方法比较；(4) 用任务顺序随机化与任务混合（overlapping manifold）构造「困难版」。
- **Hypothesis**：4 个基准中 ≥3 个可分性 >95%，原型路由达到 SOTA 的 ≥95%；困难版上 SOTA 方法增益缩减一半以上。
- **MVE**：LLaVA-1.5-7B 或 Qwen2-VL-2B；GPU：特征提取 + LoRA ~30h 单卡。
- **Closest**：Is Our Benchmark Enough (ICML 2026 CATS workshop，仅 MLLM-CL；已提出重叠流形/多顺序/forward-transfer 建议，正式版可能扩展)；CoIN (NeurIPS 2024)；MLLM-CTBench (2508.08275)。区别：跨基准泛化 + 困难版构造。
- **Novelty 5/10**；**Risk**：与 ICML 2026 文章重叠度高，必须靠跨基准 + 困难版差异化。
- **Contribution**：诊断 + 新基准变体。
- **Objection**：「增量工作」。
- dedup_key：`mllm_cl_benchmark_separability_audit`

### Idea 10：Pretraining-Scale Sweep for TF-CL — 预训练数据量如何决定 TF 与训练式 CL 的差距
- **Method**：(1) 固定 ViT-B/16 架构，取 OpenCLIP 不同预训练集（CC12M / LAION-400M / LAION-2B / DataComp-1B）与 IN1K/IN21K 监督模型；(2) 对每个骨干跑 Continual-CLIP、RanPAC、RAIL（TF）与 CODA、SD-LoRA（训练）；(3) 在 ImageNet-R、CUB、Aircraft、EuroSAT、DTD（含远离自然图像的域）上报告 TF−trained 差距随预训练规模的曲线；(4) 关联 Data Leakage 的泄漏估计。
- **Hypothesis**：TF 优势随预训练规模单调增长，但在远域（EuroSAT、DTD、医学）上饱和甚至反转；差距可由「预训练集-下游域距离」解释。
- **MVE**：5 骨干 × 5 数据集 × 5 方法；GPU：~60 单卡小时（训练式方法占多数）。
- **Closest**：Continual-CLIP (2210.03114)；PTA (2604.21360, ICML 2026)；Data Leakage (2508.17416)。三者摘要均不扫描预训练规模。区别：首次做预训练规模主轴的 TF vs trained 扫描。
- **Novelty 6/10**；**Risk**：骨干间架构/训练细节差异混淆。
- **Contribution**：scaling 诊断。
- **Objection**：「骨干不可控」——用 OpenCLIP 同代系列缓解。
- dedup_key：`pretraining_scale_sweep_tf_cl_gap`

---

## 5. 未完成 / 建议后续核实
- 搜索预算在第 38 次后耗尽，以下未能核查：ACL 2026 CL 主会论文、CVPR 2026 CL 论文、OPCM/MINGLE 类持续合并、Parametric RAG (DyPRAG)、MemoryLLM/M+/Cartridges、ModalPrompt/LLaVA-c/SEFE/CL-MoE、CTTA 的 NeurIPS 2025 论文、CLIP-CL 的 ICLR 2026 新方法。
- 2601.18699 两个版本内容矛盾（已核实，见 §0.5），勿引用。
- InfLoRA、SimpleCIL/APER、ZSCL 已于 2026-10-17 核实（§0.5）。
- closest work 的关键数字已对照 abs/html 核实（§0.5）；§1 其余条目的数字仍来自摘要片段。
