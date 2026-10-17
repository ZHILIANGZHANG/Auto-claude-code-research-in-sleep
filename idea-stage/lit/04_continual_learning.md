# 04 · 免训练 / 低训练的大模型持续学习（LLM / MLLM / 预训练模型）——文献地图与候选 idea

> 生成日期：2026-10-17。检索方式：arXiv 列表检索（按公告时间倒序）+ arXiv API 摘要核验 + WebSearch 定向查新；精读了 2 篇 PDF（2610.09835、2610.08718）。
> 文中所有 arXiv id 都经 arXiv API 或 arXiv 检索页核验过。“月份”取自 id 前缀（首次提交月份），不是最新版本的日期。
> 没有 Codex/Gemini 交叉评审，也没有跑 pilot。下面的新颖性结论只是**同一模型的初步判断**，以 2–4 次定向检索为依据，不能当作最终裁决。

---

## 1. 领域地图（2025 → 2026-10）

### 1.1 模型合并用于 CL（continual / sequential merging）
- 典型工作：*Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable Continual Model Merging* (2501.09522, 2025-01)；*MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging* (2505.11883, 2025-05)；*Null-Space Filtering for Data-Free Continual Model Merging* (2509.21413, 2025-09)；*AIMMerging* (2509.17348, 2025-09)；*RECALL: Representation-aligned … Hierarchical Model Merging* (2510.20479, 2025-10)；*K-Merge: Online Continual Merging of Adapters for On-device LLMs* (2510.13537, 2025-10)；*Merge before Forget: A Single LoRA Continual Learning via Continual Merging* (2512.23017, 2025-12)；*Unlocking the Potential of Continual Model Merging: An ODE Perspective* (2605.19409, 2026-05)；*Tunable MAGMAX* (2605.20803, 2026-05)；*SAFE-Merge: Data-Free Continual Model Merging with General Knowledge Preservation* (2608.01184, 2026-08)；*When the Merge Coefficient Stops Mattering: Proximity Regularized Merging for Continual LoRA Adaptation* (2609.32332, 2026-09)；*ChainLoRA: Geometry-Preserving Task Vector Merging for CL in LLMs* (2610.00431, 2026-10)；*Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning* (2610.05872, 2026-10)。
- RL 模型合并：*Sparsity Curse: Understanding RLVR Model Parameter Space from Model Merging* (2606.18521, 2026-06)。结论是 RLVR 的更新稀疏、彼此近似正交，**反而更难合并**。
- 合并推理能力：*ReasonAny* (2601.05560, 2026-01)、*RAIN-Merging* (2602.22538, 2026-02)、*Reasoning Vectors* (2509.01363, 2025-09)。
- **饱和度：高。** 零数据 / 零梯度的顺序合并已有很多变体（零空间、ODE、掩码、SVD/Procrustes）。评测大多停在 CLIP-ViT 8–20 个任务或短 LLM 序列上。

### 1.2 解析式 / 闭式 CL（RLS、岭回归）
- 视觉侧：RanPAC (2307.02251)；*Advancing Analytic CIL through Vision-Language Calibration (VILA)* (2602.13670, 2026-02)；*Spectral-Aware Analytic CIL for Long-Tailed* (2607.22931, 2026-07)；*SPARCL: Spectral Partitioned Analytic CL* (2608.21307, 2026-08)。SPARCL 指出共享的 (R+λI)^{-1} 会造成“谱干扰”。
- 迁移到 LLM/MLLM 已经发生：*Analytic Subspace Routing (ASR)* (2503.13575, 2025-03) 用 RLS 做 LLM CL 的路由；*MAny: Merge Anything for MCIT* (2604.14016, 2026-04) 用 RLS 闭式合并 LoRA，免训练；*Analytic Task Scheduler* (2506.09623, 2025-06) 用于具身基础模型；*Modality-Decoupled Online Recursive Editing (M-ORE)* (2605.20273, 2026-05) 是用 Sherman-Morrison 递推的 MLLM 在线编辑；*Soft RLS for Lifelong LLM Editing (RLSEdit)* (2601.15686, 2026-01)。
- 闭式合并的源头：RegMean (2212.09849)、RegMean++ (2508.03121)。
- **饱和度：中。** “把 RLS 用到 LLM”这个迁移已经被 ASR、MAny、RLSEdit、M-ORE 占了。真正还空着的是**谱层面的诊断**，以及**生成式任务上的闭式更新**（后者可行性存疑）。

### 1.3 冻结骨干的 prompt / adapter / 路由式 CL；小预算 LoRA CL
- O-LoRA (2310.14152)；*Orthogonal LoRA in Lie Groups* (2509.06100, 2025-09)；*Gated Integration of LoRA* (2505.15424)；*JumpLoRA* (2604.16171, 2026-04)；*Latent-LoRA: Gradient-Free Routing* (2607.23837, 2026-07)，它的结论是冻结 embedding 层的池化特征加 GMM 就足以识别任务；*LAPS: Anchored Prompt Space* (2609.32499, 2026-09)；*Functional Subnetwork Localization (CiMEO)* (2610.21537, 2026-10)；*Subspace Tuning (SST)* (2610.22293, 2026-10)；*Muon-OGD* (2605.08949, 2026-05)；*Sparse Memory Finetuning* (2510.15103, 2025-10) 及其复现 (2605.03229, 2604.05248)。
- MLLM 持续指令微调（MCIT）：*MLLM-CL* (2506.05453)；*MCITlib* (2508.07307)；*HiDe-LLaVA* (2503.12941)；*MLLM-CTBench* (2508.08275)；*Routing-Based CL for MLLMs* (2511.01831)；*ProtoAda* (2606.02576)；*CRAM* (2606.02502)；*SAME* (2602.01990)；*SiGMA* (2607.20511)；*InduceKV* (2607.02010)；*CoRe-MoE* (2608.27867)；*ReCAP* (2609.37889)；*CART: Compatible Routing* (2610.22137, 2026-10)；*StrLoRA: Streaming CVIT* (2605.16353)；*CL-VISTA*（视频 LLM CL 基准，2604.00677）。
- **基准批评已经出现在 MLLM 侧**：*Is Our Benchmark Enough? An Analysis of CL for MLLMs* (2606.20961, 2026-06) 发现 MLLM-CL 的任务高度可分，免训练原型路由就能追平 MLLM 路由器；*Beyond Routing Saturation (FLEX)* (2608.01437, 2026-08) 发现文本指纹泄漏任务身份，路由接近饱和。
- **饱和度：MCIT 方法极度拥挤**，每月 5 篇以上的 MoE / LoRA 路由论文。**LLM 侧（TRACE、SuperNI-long）还没有对应的基准审计。**

### 1.4 知识编辑作为持续更新（lifelong editing）及其崩溃
- AlphaEdit (2410.02355)；WISE (2405.14768)；*UltraEdit* (2505.14679, 2025-05)；*MEMOIR* (2506.07899)；*Model Merging for Knowledge Editing* (2506.12384, 2025-06)；*Spectral Characterization … Sequential Editing Collapse (REVIVE)* (2601.11042, 2026-01)；*RLSEdit* (2601.15686)；*Reversible Lifelong Editing via Semantic Routing LoRA* (2603.11239)；*HiEdit* (2604.11214)；*HoReN* (2605.08143)；*More Edits, More Stable: Lifelong Normalization* (2605.11836, 2026-05)；*The Labyrinth and the Thread* (2605.26670, 2026-05) 证明了一次性编辑与顺序编辑的等价性，稳定性来自对累积约束的正确处理；*CRANE: Knowledge Editing for Reasoning MLLMs* (2606.09033)；*Sequential knowledge editing breaks a model's ability to tell good evidence from bad* (2609.29587, 2026-09)；*Generalizable Lifelong Editing via Preference Optimization* (2609.36748)；*EngramEdit* (2610.10533)；*FedAlphaEdit* (2610.11033)。
- 编辑 × 后续微调：*Can Fine-Tuning Erase Your Edits?* (2511.05852)、*On the Retention of Edited Knowledge in Fine-tuned LMs* (2507.14198)。这个方向已经做过了。
- **饱和度：高。** 编辑器设计已经红海化。剩下的空间主要是**机制诊断**（谱、协方差）和**“编辑造成的隐性能力损失”**。

### 1.5 检索 / 记忆式 CL；上下文内 CL；自我进化 agent（不改权重）
- *From RAG to Memory (HippoRAG 2)* (2502.14802)；*In-context CL Assisted by an External Continual Learner* (2412.15563)；*ICL can Perform CL Like Humans* (2509.22764)；*Understanding Generalization and Forgetting in In-Context CL* (2605.28705)；*MIITA* (2607.22556)；*Self-Consolidating LMs* (2605.07076)。
- Agent 记忆 / 经验：*Dynamic Cheatsheet* (2504.07952)；*Memento* (2508.16153)；*ReasoningBank* (2509.25140)；*ACE: Agentic Context Engineering* (2510.04618)；*Evo-Memory* (2511.20857)；*MemRL* (2601.03192)；*Live-Evo* (2602.02369)；*MemCollab*（跨模型记忆，2603.23234）；*Memory Transfer Learning* (2604.14004)；*CL-Bench* (2606.05661, 2026-06)，其结论是**朴素 ICL 胜过专门的记忆系统**；*Reliable Memory from Unstable Growth (RBC)* (2610.18427, 2026-10)，它发现同一流程写出的 playbook 可能比不用 playbook 低 69.6 分；*Beyond Downstream QA: Rethinking How We Evaluate Agent Memory* (2610.22209)；*Decoupling Memory from Context (GraphMemory)* (2610.02687)。
- **饱和度：极高。** 仅 2026-10 前两周，标题含 “agent memory” 的论文就有约 40 篇。跨模型迁移、投毒、评测失真都已有人做。

### 1.6 测试时适应 / 测试时训练
- *In-Place Test-Time Training* (2604.06169)；TTCS (2601.22628)；*Prequential Test-Time Learning for LLM Agents* (2609.35911)；*SelectTTRL* (2610.16104)；*Sharpen Before You Adapt* (2610.00903)；VLM 免训练 TTA（TDA 系列，如 2604.21360）。
- **饱和度：中高。** 主要处理单一分布。“流式 TTRL 的遗忘”研究较少，但离本方向主线较远。

### 1.7 后训练中的遗忘：SFT vs RL vs 自蒸馏（核心矛盾区）
- 主张“RL 遗忘更少”的工作：*RL's Razor* (2509.04259, ICLR'26)，以新任务上的 KL 为预测量；*Retaining by Doing* (2510.18874)；*RFT Naturally Mitigates Forgetting in Continual Post-Training* (2507.05386)；*Why RFT Enables MLLMs Preserve Prior Knowledge* (2506.23508)；*Mechanistic origins … why RL preserves circuits* (2605.28860)，指出 RL 适应更慢；*Learning Style, Forgetting Semantics* (2610.02437) 从理论上用风格漂移解释。
- 反面证据：*RL Forgets! Towards Continual Policy Optimization* (2607.04364)，在 MRCL 基准上 RL 照样遗忘，作者认为该看的是旧任务分布上的 KL；*Evolutionary Strategies lead to Catastrophic Forgetting* (2601.20861)，发现 GRPO 的 KL 与遗忘不相关；*Denser ≠ Better: Limits of On-Policy Self-Distillation* (2607.01763)，SDPO 遗忘比 GRPO 更多。
- 自蒸馏 / 中间路线：*SDFT: Self-Distillation Enables CL* (2601.19897)；*TMS* (2602.03073)；*On-Policy Replay (OPR)* (2605.29495)，在 TRACE 上 1% 回放就把 BWT 从 −13.9 拉到 −2.3；*SynthSelf* (2610.13113)；*iSDFT* (2609.24646)；*Finetuning with Sampling (MCMC)* (2610.02140)；*SFT-as-Context*（免训练，2610.11132）。
- 遗忘机制：*Spurious Forgetting* (2501.13453)；*Real-Time Detection of Spurious Forgetting* (2512.20634)；*When Forgetting is not Catastrophic* (2610.08718)，提出“公共方向平移”，并发现删掉每个权重更新的 top 奇异方向就能恢复旧事实（OLMo-2-1B）；*A Deafening Silence* (2610.09835)，发现遗忘集中在新语料中缺席 token 的输出 embedding 上，由单侧 softmax 梯度经 Adam 放大造成，事后改写这些行只能恢复不到 5%；*RL Fine-Tuning Heals OOD Forgetting in SFT* (2509.12235)，与奇异向量旋转有关；*Catastrophic Forgetting is Low-Rank* (2606.18024)；*Spectral Unforgetting (DG-Hard)*（事后谱修复，2605.20296）；*Rotation-Preserving SFT* (2605.10973)；*Geometry Conflict* (2605.09608)；*8-bit Quantization Improves CL* (2512.18934)；*Representation Drift: LoRA vs N-gram Memory* (2610.14921)；*Can a LM Learn Facts Continually in Its Weights?* (2607.11020)。
- **饱和度：极热，但结论互相矛盾。** 这是做“诊断 / 解决矛盾”类论文最好的切入点。

---

## 2. 结构性缺口（按 lens）

| Lens | 观察 | 缺口 |
|---|---|---|
| 矛盾 | RL's Razor 认为新任务 KL 决定遗忘；RL Forgets! 认为该看旧任务 KL；ES 论文发现 GRPO 的 KL 与遗忘无关；SDPO 遗忘多于 GRPO | 没有人把多个预测量放在多种范式上**正面对比**；也没有人用**机制**解释 “RL < SFT” 的差距，现有解释停在行为层或电路层 |
| 机制迁移 | 2610.09835：遗忘集中在缺席 token 的输出 embedding，原因是单侧梯度经 Adam 放大 | GRPO 用的是**零均值优势**，缺席 token 的梯度有正有负，可能相互抵消。这一点没人检验 |
| 机制迁移 | 2610.08718：公共平移可逆，事实可在权重空间去方向后恢复（1B 模型，事实任务） | 激活空间里“每层一个均值平移向量”的免训练修复还没在指令 CL、TRACE、MCIT、通用能力上测过 |
| 未检验假设 | MLLM 基准已被指出任务可分、指纹泄漏（2606.20961、2608.01437） | TRACE、SuperNI-long、“Standard CL 5 tasks”这类 **LLM 基准没有同等审计**；kNN-ICL 零更新基线和存储对齐的回放基线普遍缺席 |
| 隐性遗忘 | 编辑会破坏证据仲裁（2609.29587）；指令微调会造成 context-parametric inversion（2410.10796） | 持续 SFT、RL、合并、自蒸馏是否同样损伤“信检索还是信参数”的能力？MMLU 看不出来 |
| 机制迁移（解析 CL → 编辑） | SPARCL 发现共享逆协方差导致谱干扰；AlphaEdit 与 RLSEdit 也共享 (K0K0ᵀ+ΣKKᵀ)^{-1} | 编辑崩溃从**键协方差谱**的角度还没被分析。REVIVE 分析的是权重谱，不是键协方差谱 |
| 未探索的条件 | 大多数比较不对齐学习量 | “RL 遗忘少”可能只是“RL 学得少、走得近”。用免训练插值把 SFT 对齐到相同新任务增益后，差距还在吗？ |

---

## 3. 候选 idea（按优先级排序）

> 成本统一假设：1 张 24–80GB GPU，0.5–8B 开源模型，每个 idea 的最小验证 ≤ 2 GPU-day。

### Idea 1 ★ 零均值优势抵消缺席 token 漂移：用机制解释 “RL 遗忘少于 SFT”
- **假设**：SFT 与 RL 遗忘差距的一大部分来自输出 embedding。SFT 里，新语料缺席 token 收到的是单侧 softmax 梯度（+p_i），Adam 的二阶矩归一化把它放大成满幅更新。GRPO/RLOO 的组内优势是零均值的，同一缺席 token 在正、负优势样本上收到方向相反的梯度，大致抵消，因此没有系统性漂移。
- **机制 / 预测**：(a) SFT 下缺席 token 的 unembedding 行漂移量远大于 GRPO；(b) 只给 SFT 的 output projection 调高 Adam ε（或冻结 lm_head）能关闭相当一部分 SFT–RL 遗忘差距；(c) 反向操作：只保留正优势样本的 GRPO（即 RAFT / 拒绝采样 SFT）应当重新出现漂移和遗忘，而只用负样本的 NSR 应当更少。
- **最小实验**：OLMo-2-1B 或 Llama-3.1-8B（选**非 tied embedding** 的模型。Qwen2.5 小模型绑定了 embedding，要单独讨论）。沿用 RL's Razor 的设置，在 Countdown / MATH 子集 / SciQ 上，在**新任务精度相同**的条件下比较 SFT、GRPO、正样本 GRPO、SFT+ε_out、SFT+冻结 lm_head。遗忘探针用 MMLU、IFEval、HellaSwag、旧任务。同时记录每个 token 行的 Δ‖u_i‖ 与 token 计数的关系。约 1–2 GPU-day。
- **新颖性**：2610.09835 只做了持续预训练和指令 SFT，PDF 里没有 RL 或 GRPO；RL's Razor 用的是行为层的 KL；2605.28860 停在电路层；2610.02437 给的是风格漂移理论。“零均值优势导致梯度抵消”这一机制以及正样本反事实，我没检索到。
- **最接近工作**：2610.09835、2509.04259、2510.18874、2605.28860、2610.02437、2607.04364。
- **风险**：差距可能主要在网络主体而不在输出头，那么 (b) 只能关闭很小一部分差距。即便如此，这也是一个有信息量的负结果。tied embedding 会混淆结论。GRPO 中 p_i 极小的 token 的梯度方差很大，抵消可能不完全。
- **判定：NOVEL-LIKELY**（检索 3 次：WebSearch 两次 “freeze lm_head SFT RL forgetting gap”，以及 PDF grep）。

### Idea 2 ★ 每层一个均值平移向量：免训练撤销“伪遗忘”（Steer-Back）
- **假设**：持续指令微调或 MCIT 后，相当一部分旧任务和通用能力的下降属于公共平移造成的“访问丢失”（2610.08718），不是真正的知识擦除。用无标签通用文本估计每层残差流的均值平移 μ_ℓ = E[h_ft − h_base]，推理时减去 α·μ_ℓ（或把它沿 top-1 方向投影掉），就能恢复旧能力，同时基本保住新任务。
- **机制**：从激活空间实现 2610.08718 的权重空间“去 top 方向”。它不需要旧任务数据，也不碰权重，可以按层、按 token 位置（只作用在开头几个 token，对应 2512.20634 说的浅对齐）做门控。
- **最小实验**：Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct 在 TRACE 的 8 个任务上做顺序 LoRA，基线是 SeqFT、O-LoRA、OPR 1% 回放、WiSE-FT、DG-Hard (2605.20296)。评测 TRACE 的 OP/BWT 和通用能力（MMLU、GSM8K、IFEval、BBH）。扩展：LLaVA-1.5 在 CoIN 上。约 1 GPU-day，大部分时间花在训练 SeqFT 检查点上。
- **新颖性**：2610.08718 只在 1B 模型、事实任务、权重空间做过；2608.24988 研究的是“微调会不会撤销 steering”，方向相反；2605.20296 是权重谱阈值，不是激活平移；2501.13453 用的是冻结底层这种训练时方法。“用 base 与 FT 的激活差做修复向量”我没检索到。
- **最接近工作**：2610.08718、2501.13453、2512.20634、2605.20296、2608.24988。
- **风险**：TRACE 上的遗忘可能大多是真擦除，向量只能恢复格式和指令遵循。2610.08718 也指出公共平移在部分条件下会被训练自行撤回。如果恢复幅度不大于 WiSE-FT，故事就弱了。
- **判定：CROWDED-BUT-DELTA，接近 NOVEL**。可验证的 delta 一句话：在真实 CL 基准上，零权重修改、零旧数据的激活平移修复，与 WiSE-FT、DG-Hard 正面对比。

### Idea 3 “你的持续更新模型不再听检索”：CL 的隐性遗忘
- **假设**：顺序 SFT、合并、自蒸馏、RL 都会改变模型在参数知识与上下文证据冲突时的仲裁。MMLU 不变，ConflictBank、ClashEval、反事实 RAG 上的上下文依赖和选择性预测（AURC）却在退化，而且退化幅度因范式而异，SFT 应当大于 RL。
- **免训练修复**：把 base 模型冲突仲裁头（2609.25602 已定位这些头）的权重合并回去，或用 Idea 2 的平移向量修复。
- **最小实验**：复用 Idea 1、2 的检查点，额外评测知识冲突集与 AURC，几乎不增加成本，约 0.5 GPU-day。
- **新颖性**：2410.10796 只研究单次指令微调；2609.29587 只研究编辑；2609.25602 只比较 base 与 instruct。**跨 CL 范式、多任务顺序设置**下的比较没找到。
- **最接近工作**：2410.10796、2609.29587、2609.25602、2512.10150（安全遗忘）。
- **风险**：可能被看成“又一个评测维度”，需要靠修复方法或范式差异来撑起贡献。
- **判定：CROWDED-BUT-DELTA。**

### Idea 4 遗忘预测量锦标赛，外加“只用 prompt 的旧任务 KL”选择合并系数
- **假设**：把新任务 KL（RL's Razor）、旧任务 prompt 上的 KL（RL Forgets!）、权重距离、主奇异向量旋转（2509.12235）、缺席 token 输出漂移（2610.09835）、公共平移幅度（2610.08718）放在 SFT、GRPO、SDFT、ES、合并、编辑上正面对比，其中“旧任务 prompt KL”跨范式最稳。旧任务 prompt KL 只需要旧任务的**无标签 prompt**，所以可以直接拿来**免训练地逐层选择合并或插值系数**。
- **最小实验**：Qwen2.5-1.5B 或 OLMo-2-1B 跑 3 个任务序列 × 6 种范式 × 多个超参，收集约 100 个检查点，用 Spearman 相关和留一范式外推来评估预测量。随后在 TRACE 的 LoRA 合并上做系数选择，基线是 AdaMerging 和 ODE-M。约 2 GPU-day。
- **新颖性**：RL's Razor 只在 SFT 与 RL 之间做过变量消融；ES 论文给出了 KL 的反例；跨 6 种范式的正面对比我没找到。
- **最接近工作**：2509.04259、2607.04364、2601.20861、2605.19409（用校准集的 ODE-M）。
- **风险**：分析类论文的门槛高。基于 prompt-KL 的系数选择与 AdaMerging 的熵最小化思路接近。
- **判定：CROWDED-BUT-DELTA。**

### Idea 5 RL's Razor 是不是只因为步子小？增益对齐下的免训练插值对照
- **假设**：先把 SFT 检查点向 base 做插值（WiSE-FT，必要时外推），使其**新任务增益**与 RL 相同。此时 SFT 的遗忘与 RL 持平，甚至更低。如果成立，“RL 遗忘少”主要反映的是 RL 学得少、离 base 近，也就是 KL 小。如果不成立，就说明 RL 的解在“方向”上本质不同，可以进一步与 Idea 1 的机制联系起来。
- **最小实验**：与 Idea 1 共用设置。对 SFT 的 α 扫 0 到 1.2，画出 (新任务精度, 遗忘) 帕累托前沿，与 GRPO 的训练轨迹对比。约 0.5 GPU-day。
- **新颖性**：TMS (2602.03073) 用历史检查点课程改善 SFT 的前沿；grafting (2610.05872) 缩放更新，但对照对象是 OPSD；“插值后的 SFT 前沿 vs RL 前沿”的直接对比没检索到。
- **风险**：结果可能显而易见，审稿人会说插值就是 KL 控制。建议**作为 Idea 1 的一个对照实验**并入，不单独成文。
- **判定：CROWDED-BUT-DELTA。**

### Idea 6 LLM 持续学习基准审计（TRACE / SuperNI-long / Standard-5）
- **假设**：(i) 冻结 embedding 的原型或 kNN 分类器就能近乎 100% 识别任务身份，于是“CL”退化成路由加每任务专家；(ii) 零更新的 kNN-ICL（从任务流存 k 条示例）在存储对齐时能追平或超过 O-LoRA 系列；(iii) 存储字节对齐的 1% 回放（参考 OPR）优于大多数无回放方法。据此构造一个“去指纹”的 TRACE 变体。
- **最小实验**：Llama-3.1-8B-Instruct / Qwen3-8B 上跑任务 ID 探针、kNN-ICL、oracle 路由，基本不需要训练。约 0.5–1 GPU-day。
- **新颖性**：MLLM 侧已经有 2606.20961 和 2608.01437；Latent-LoRA (2607.23837) 把可分性当成特征来用，没有当成问题；CL-Bench (2606.05661) 发现 ICL 胜过记忆系统，但测的是 agent 环境。LLM 侧的系统审计没找到。
- **最接近工作**：2606.20961、2608.01437、2607.23837、2605.29495、2606.05661。
- **风险**：属于“把 MLLM 侧批评搬到 LLM 侧”，delta 偏薄。适合作为 workshop 论文，或其他 idea 的附带贡献。
- **判定：CROWDED-BUT-DELTA。**

### Idea 7 键协方差谱干扰：从解析 CL 视角解释顺序编辑崩溃（SPARCL → AlphaEdit / RLSEdit）
- **假设**：MEMIT、AlphaEdit、RLSEdit 的更新共享 (C0+ΣK_iK_iᵀ+λI)^{-1}。编辑键如果集中在少数主特征方向上，就会稀释这些方向的谱，扰动未编辑知识与早期编辑。按谱分块的正则（对旧主方向加大 λ），或者分块逆，可以延缓崩溃。
- **最小实验**：Llama-3-8B 或 Qwen2.5-7B，在 ZsRE / CounterFact 上做 1k–20k 次顺序编辑。追踪协方差特征谱的变化与通用能力的相关性，再在 AlphaEdit、RLSEdit 上加谱分块。约 1–2 GPU-day。
- **新颖性**：REVIVE 分析的是权重谱；Lifelong Normalization 分析的是值梯度归一化；Labyrinth 讨论的是约束累积。“键协方差谱稀释”这个角度检索未见。
- **最接近工作**：2608.21307、2601.15686、2601.11042、2605.11836、2605.26670。
- **风险**：编辑领域极度拥挤，增益可能只有个位数，最后变成又一个编辑器。
- **判定：CROWDED-BUT-DELTA（delta 薄）。**

### Idea 8 （降级）把指令任务当编辑单元：闭式任务写入做持续指令微调
- 设想用 MEMIT/RLS 在少数 MLP 层上闭式写入每个任务的 (key, value)，value 取自每任务 LoRA 专家的激活。问题在于这本质上就是 RegMean (2212.09849) 的递推版本，而 MAny 的 LPM (2604.14016) 已经在 MCIT 上用 RLS 闭式合并了 LoRA，ASR (2503.13575) 也有类似做法。
- **判定：CROWDED，接近 ALREADY-DONE。** 不推荐。

### 已排除（检索中发现已有人做）
- 事后重置缺席 token 的 unembedding 行：**ALREADY-DONE，负结果**。2610.09835 原文写明“post-hoc editing of the drifted rows recovers under 5%”。
- 免训练撤销奇异向量旋转以恢复 OOD 能力：**ALREADY-DONE**，见 2605.20296 (Spectral Unforgetting / DG-Hard)、2509.12235 及其后续、2605.10973。
- 编辑能否扛过后续微调：**ALREADY-DONE**，见 2511.05852、2507.14198。
- Agent 记忆跨骨干迁移：**ALREADY-DONE**，见 2603.23234 (MemCollab)、2605.30712 (ExpGraph)、2601.03192 (MemRL)、2606.07711。
- 领域 SFT 后用合并恢复推理能力：**CROWDED**，见 2601.05560、2602.22538、2609.25618 (NB-LoRA)、2610.05872。

---

## 4. 建议的组合与下一步
- **主线论文**：Idea 1 + Idea 5，可以共用一套检查点。题目方向是 “Why RL forgets less: zero-mean advantages cancel absent-token drift”，即从机制上解释后训练遗忘。
- **轻量配套**：Idea 2 + Idea 3，复用上面的检查点，加一个免训练修复方法和一个隐性遗忘评测。
- **先做的 pilot（<4 GPU 小时）**：用 OLMo-2-1B 在一个任务上对比 SFT 与 GRPO，记录逐 token 行的 unembedding 漂移与 token 计数，看漂移的符号结构是否与假设一致。如果 GRPO 的缺席 token 行漂移并不显著更小，就放弃 Idea 1。
- **待交叉模型复核**：Idea 1、2 的新颖性。重点再查 ICLR'27 投稿，以及 2610 下旬的 arXiv。
