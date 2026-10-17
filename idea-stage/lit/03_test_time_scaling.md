# 03 — LLM Test-Time Scaling（免训练 / 极低训练）：文献地图、结构性空白与候选 idea

> 生成日期：2026-10-17。检索方式：arXiv 列表检索（按公布日期倒序，带引号短语，约 50 组查询）+ arXiv 摘要页核对 + WebSearch。
> 下文所有 arXiv id 都已在 arXiv 摘要页或搜索结果页核对过标题（“月份”取 v1 提交月）。
> **局限说明**：2026 年本方向论文量非常大（仅 `"test-time scaling"` 一个短语，近 5 个月就有 200 多篇）。本调研覆盖面广，但做不到穷尽。对每个 idea，我只读了最接近工作的摘要，只有 2609.32035 读了全文 grep，所以新颖性判断是 **初筛**，不是定论。
> 关于 “TTS = text-to-speech”：语音合成侧的推理时扩展很少。相关的只有 Llasa（2502.04128，2025-02，同时扩 train-time 和 inference-time compute，使用 verifier 选择）和 MagpieTTS-LF（2606.18485，2026-06，推理时长语音生成，不需要训练）。它们和本调研关系不大，下文不再展开。

---

## 1. 领域地图（子方向 × 代表工作 × 饱和度）

### 1.1 并行采样 + 选择/投票（verifier-free 内部信号）—— **非常拥挤，且出现了正面矛盾**
- Deep Think with Confidence (DeepConf) — 2508.15260 (2025-08)：用局部 group confidence 过滤和加权 trace，并支持在线早停。
- Scalable Best-of-N Selection via Self-Certainty — 2502.18581 (2025-02)；Confidence Improves Self-Consistency (CISC) — 2502.06233 (2025-02)。
- Trace Length is a Simple Uncertainty Signal in Reasoning Models — 2510.10409 (2025-10)：仅凭 trace 长度就是很强的置信信号。
- Reliable Chain-of-Thought via Prefix Consistency — 2605.07654 (2026-05)：截断后重新生成，按答案复现率加权投票。
- LLMs as a Jury: Cross-Model Consensus Can Outperform PRMs — 2607.10139 (2026-07)：机制在于跨模型的**错误去相关**。
- Compare to Correct: Pairwise-Guided Logit Adjustment (PGLA) — 2610.18583 (2026-10)；Consilience for Verifier-Free TTS — 2608.09898 (2026-08)；Intrinsic Selection and Particle Resampling — 2606.08850 (2026-06)。
- **反方证据**：
  - Reasoning Concentrates Errors, and Self-Consistency Never Notices — 2609.32035 (2026-09)：横跨 280 个“方法×数据×模型”组合，做 Holm 校正后，**没有一个**置信加权投票方法优于普通多数投票。推理模式会让错误答案集中到同一个干扰项上。answer log-prob 的符号在 think 和 no-think 模式下相反。长度是很强的混杂因素。
  - Calibration, Not Answer Selection — 2609.33290 (2026-09)：hidden-state probe 的校准很好，但用来选答案时只和多数投票打平。
  - When Self-Consistency Backfires (small LLMs, GPQA) — 2608.11403 (2026-08)。
  - When More Sampling Hurts: Modal/Correlation Ceiling — 2606.28661 (2026-06)。
- 判断：**“再提出一种置信分数”基本已无空间**。真正的瓶颈是 *selection/identifiability*，而不是 generation（2608.18931、2606.28661、2609.13257 都得出同样的结论）。

### 1.2 并行思考的在线控制（中间答案 probe、早停、剪枝）—— **拥挤**
- Parallel-Probe (2D probing) — 2602.03845 (2026-02)；ParaTempo (temporal confidence) — 2608.16425 (2026-08)；MARS risk-controlled stopping — 2606.12935 (2026-06)。
- Funnel of Thoughts（按 hesitation 词汇剪枝）— 2608.15065 (2026-08)；Fork-Think with Confidence — 2606.31484 (2026-06)；Collaborative Parallel Thinking — 2605.27030 (2026-05)。
- Overscaling Curse of Parallel Thinking (LanBo) — 2601.21619 (2026-01)；Interpretable Adaptive Sampling — 2608.03961 (2026-08)。

### 1.3 单链高效推理 / 早退 / 防 overthinking（免训练部分）—— **非常拥挤**
- DEER: Dynamic Early Exit — 2504.15895 (2025-04)；NoWait — 2506.08343 (2025-06)；Budget Guidance — 2506.13752 (2025-06)；Don't Overthink it (short-m@k) — 2505.17813 (2025-05)。
- TRACE temporal aggregation — 2604.17304 (2026-04)；PUMA semantic-preserving exit — 2605.17672 (2026-05)；Detection-Extraction Gap / BAEE — 2604.06613 (2026-04)。
- `</think>` Doesn't Stop Reasoning: Spurious CoT Termination — 2609.03633 (2026-09)；Pareto Frontiers of Reasoning Termination — 2610.13570 (2026-10)。
- 激活转向方向：Halt Vector — 2608.28859 (2026-08)、Reflection Steering — 2608.25542 (2026-08)、Activation Steering for CoT Compression — 2507.04742 (2025-07)。
- 分析类：Mirage of TTS — 2506.04210 (2025-06)；Inverse Scaling in Test-Time Compute — 2507.14417 (2025-07)；Probing the Trajectories of Reasoning Traces — 2601.23163 (2026-01)。

### 1.4 混合思考模式 / 思考预算路由 —— **拥挤**
- DART: Draft-Agreement Routing — 2606.23181 (2026-06)：用两个 no-think 草稿是否一致来决定是否思考、思考多少。
- Increasing the Thinking Budget is Not All You Need — 2512.19585 (2025-12)；Zero-Step Thinking — 2510.19176 (2025-10)。

### 1.5 树搜索 / PRM / 顺序精炼 —— **中度拥挤，理论化趋势明显**
- Provable TTS for Beam Search — 2609.38672 (2026-09)；Limits of PRM-Guided Tree Search — 2510.20272 (2025-10)；Adaptive Parallel MCTS — 2604.00510 (2026-04)；Stochastic Backtracking — 2605.25143 (2026-05)。
- Hill Sampling — 2609.25510 (2026-09)；Recursive Self-Aggregation (RSA) — 2509.26626 (2025-09)；Refining Over Resampling — 2608.05643 (2026-08)；Better Exploration in Sequential TTS — 2609.39632 (2026-09)。

### 1.6 系统、服务与成本视角 —— **中度**
- Kinetics: Rethinking TTS Laws（memory-bound 视角）— 2506.05333 (2025-06)；Are We Scaling the Right Thing? — 2509.19645 (2025-09)。
- SpecScale: Speculative Search in LLM Serving — 2609.39334 (2026-09)；Speculative-decoding benchmark for TTS — 2509.04474 (2025-09)；Test-Time Speculation（长输出下 acceptance length 衰减到约 1）— 2605.09329 (2026-05)。
- Sample Count Is Not Enough (energy) — 2609.19499 (2026-09)。
- 量化：BitCal-TTS — 2605.05561 (2026-05)；Spend Bytes on Breadth (KV precision-count) — 2610.05685 (2026-10)；Quantized reasoning models think longer — 2606.00206 (2026-06)。

### 1.7 多样性来源 —— **有空隙**
- Test-Time Augmentation（匹配算力下比较输入侧与输出侧多样性）— 2608.09351 (2026-08)。
- Architectural Sampling（复用 decoder 层带来的计算多样性，**只在 VLM 上做过**）— 2610.01687 (2026-10)。
- Hard or Just Unreached?（残差流扰动作为多样性来源，用于难度诊断）— 2606.19636 (2026-06)。
- Strategy vs. Phrasing diversity — 2606.29985 (2026-06)。

### 1.8 多模态 / agent / diffusion LLM 的 TTS —— **各自都在快速变拥挤**
- VLM：On TTS for VLMs — 2606.28864 (2026-06)；AVIS — 2606.11576 (2026-06)；Decoding-format audit — 2608.01207 (2026-08)；MM-TTS survey — 2606.08231 (2026-06)。
- Agent：CATTS for WebAgents — 2602.12276 (2026-02)；BrowseConf — 2510.23458 (2025-10)；Beyond Confidence for search agents — 2608.24024 (2026-08)；Routing-guided TTS for SWE agents — 2608.22191 (2026-08)；Agentic Aggregation — 2604.11753 (2026-04)；Adaptive thinking budgets for multi-turn — 2604.05164 (2026-04)。
- dLLM：UnMaskFork — 2602.04344 (2026-02)；Prism (hierarchical search + self-verif) — 2602.01842 (2026-02)；Reward-guided stitching — 2602.22871 (2026-02)；Why deterministic PRM guidance underperforms — 2609.35472 (2026-09)；Prefix-Denoising Consistency — 2608.25311 (2026-08)。WebSearch 另外找到 SIGMA / frequency-aware remasking（NeurIPS'26 poster，未查到 id）。

### 1.9 Scaling law / 评测可信度 —— **中度，偏理论**
- Tail Competition in BoN — 2610.19362 (2026-10)；Cheap to Draw, Expensive to Trust — 2609.40190 (2026-09)；How Much Can LMs Gain from TTC (SELF-POT) — 2610.01110 (2026-10)。
- TTS in the Wild (open-ended)：瓶颈在 exploitation — 2608.18931 (2026-08)；TTS in reasoning LLMs: reproducibility — 2608.04001 (2026-08)。
- 经典锚点：Snell et al. — 2408.03314 (2024-08)；s1 — 2501.19393 (2025-01)；Does RL Really Incentivize Reasoning — 2504.13837 (2025-04)。

---

## 2. 结构性空白（按 lens 整理）

1. **矛盾 lens（最强的信号）**：DeepConf/CISC/self-certainty 报告了显著收益，而 2609.32035 和 2609.33290 认为置信加权≈多数投票。2609.32035 自己承认只覆盖 *weighting*，**没有覆盖 *filtering*（DeepConf 的 top-η 过滤）**；它也没有给出“如何利用 error concentration”的方法。
2. **2609.32035 留下的未解问题**：(a) 推理模式下 answer log-prob 的符号会翻转，原因不明；(b) 推理让错误集中，但**没人测试 think 与 no-think 混合投票能否去相关**；(c) 错误集中到的那个干扰项，是否就是模型“直觉”(no-think) 给出的答案？这一点没人问过。
3. **方法迁移 lens**：Architectural Sampling 只在 VLM 选择题/短答案上验证过，长 CoT 文本推理是否受益没人知道。Jury 的跨模型去相关可以对应到“单模型内的异构视角”，这一方向也还没有系统做。
4. **免费信号 lens**：服务端早就在跑 speculative decoding。draft/target 的接受轨迹是零额外算力的“跨模型一致性”信号。目前只被用于 backdoor 检测（SpecGuard 2609.11799）和 token 级修正（EASD 2512.23765），**没人用它做 trace 选择或早停**。
5. **未检验假设**：probe 类方法（Parallel-Probe、ParaTempo、MARS、TRACE）都要额外 forward。模型在思考文本里本来就会说 “so the answer is X… wait”，这些**文本中的自述中间答案**是零成本的。另外，所有新信号都应该过一遍 2609.32035 / 2510.10409 提出的“长度基线 + 长度残差化”检验，多数论文没有做。
6. **未覆盖的 regime**：同一 base 模型在 SFT 蒸馏和 RL 两种后训练下，TTS 信号质量（错误集中度、置信 AUROC、pass@k−maj@k gap）的因果比较。Devic 2510.10409 只看了长度，2609.32035 只做了 think 开关。

---

## 3. 候选 idea（按优先级排序）

> 统一实验平台：单卡 H100/A100-80G，vLLM，Qwen3-8B（think/no-think 同权重）、DeepSeek-R1-Distill-Qwen-7B，可选 Qwen3-4B/1.7B 做规模趋势。
> 基准：AIME24/25/26、HMMT25、GPQA-Diamond、MATH500-hard、LiveCodeBench（可选）。
> 统计协议沿用 2609.32035 的建议：按题目做 cluster bootstrap，多重比较校正，**必须报告 length-only 基线**。

### Idea 1 — 直觉陷阱：Prior-Discounted Voting（推理错误是否集中在 no-think 直觉答案上？）
- **假设**：think 模式下“集中的错误”有很大一部分恰好等于同一模型 no-think 模式的众数答案，即推理没能摆脱直觉锚定。如果成立，可以把 no-think 先验作为负证据或条件证据，修正 think 投票。这也可能解释 2609.32035 观察到的 log-prob 符号翻转：模型在锚定答案上自信度过高。
- **机制**：先采 M=16–32 个 no-think 样本（每个约 300 token，成本不到 think 的 5%），得到先验分布 p₀(a)。think 投票分数改为 `s(a)=log c_think(a) − λ·g(p₀(a), margin)`：只有当 think 票分歧且领先答案恰好是直觉答案时才折扣。λ 在 AIME24 上调，在 AIME25/26/HMMT/GPQA 上测试。
- **最小实验**（约 0.5–1 GPU-day）：
  1. 诊断：在 think-maj 错误的题上，统计 P(think 错误答案 = no-think 众数)，并与随机错误答案的基线比较。
  2. 方法：比较 maj@64、DeepConf、长度过滤、PGLA 和 prior-discount。
- **新颖性论证**：DART 把 no-think 一致性当作“容易”的信号，用来跳过思考，方向相反。Pre-commitment 工作（2607.16451）只做单题的行为和激活分析。2609.32035 报告了 error concentration，但没有和 no-think 答案做匹配。
- **最接近工作**：2609.32035、2606.23181、2607.16451、2604.06613、2502.06233。
- **风险**：在简单题上直觉往往是对的，折扣会伤害准确率，必须做条件触发。效应可能只在 GPQA/选择题上出现。no-think 模式存在“推理泄漏”，两种模式不完全分离。
- **初筛判定**：**NOVEL-LIKELY**（查了 3 次：arXiv 上 anchoring/no-think prior 查询零命中；WebSearch 只返回人类认知文献；DART 摘要确认方向相反）。

### Idea 2 — SpecSignal：把 speculative decoding 的接受轨迹当作免费 verifier-free 信号
- **假设**：同族小 draft（Qwen3-0.6B 或 1.7B）对 target trace 的逐段期望接受率 `Σ min(1, q/p)` 含有正确性信息。错误 trace 里会出现 draft 难以预测的“跳步或幻觉段”（也可能方向相反：正确 trace 含有非显然的洞见）。无论方向如何，它都可以零成本地用于 trace 选择、剪枝和早停，而且不需要暴露 logprob API。
- **机制**：离线用 teacher forcing 计算 draft 在 target trace 上的 token 概率（只需一次 0.6B 前向，成本可忽略），按位置归一化。2605.09329 显示接受长度会随生成位置衰减，所以必须先去除位置趋势。提取的特征包括 trace 均值、最差窗口、尾部窗口。
- **最小实验**（约 0.5 GPU-day）：在 Qwen3-8B 的 64 条 AIME/HMMT/GPQA trace 上计算 AUROC（题内）。与 self-certainty、DeepConf-tail、长度做对比，并报告**长度残差化之后的 AUROC**。最后接入投票或过滤，看能否超过 maj@64。
- **新颖性论证**：SpecGuard 用 per-query 接受率检测 backdoor；EASD/ConfSpec 在 token 级改解码或门控 verification。没有找到“接受率→推理正确性→TTS 选择”的工作。
- **最接近工作**：2609.11799、2512.23765、2602.18447、2605.09329、2607.10139（跨模型一致性思想）。
- **风险**：接受率可能只反映流畅度或可预测性（WebSearch 汇总的证据倾向这一点），被长度和难度混杂后，残差可能接近 0。即使是负结果，也可以写成诊断结论。
- **初筛判定**：**NOVEL-LIKELY**（3 次检索：arXiv `"acceptance rate" reasoning correctness` 零相关；`"acceptance length" speculative signal` 只返回加速类工作；WebSearch 只命中 SpecGuard/EASD/SFAD 这些相邻工作）。

### Idea 3 — Intra-Model Jury：以错误去相关为目标的单模型异构采样
- **假设**：同一模型的不同“视角”产生的错误重合率不同，这些视角包括温度、think/no-think、复用层的 architectural sampling、题面改写、语言切换和残差扰动。在匹配 token 预算下，按 pairwise wrong-answer collision 选出混合比例，可以把 maj@N 推近 Jury（2607.10139）的效果，而且只需要一个模型。
- **机制**：
  1. 用 2609.32035 的 collision 指标，为每种来源估计错误重合率。
  2. 部署时没有标签，用 Dawid–Skene 或一致性结构做无监督估计（也可在一个 dev 集上做一次校准）。
  3. 求解小规模的预算分配问题。
- **最小实验**（约 1–1.5 GPU-day）：Qwen3-8B，六种来源各 16 个样本，在 AIME/HMMT/GPQA 上测 collision 矩阵、匹配算力的 maj 准确率和 pass@k。
- **新颖性论证**：TTA（2608.09351）只比较了输入侧和输出侧，没有以错误重合为目标，也没有覆盖 think/no-think 和计算路径。Jury 依赖多个模型。Architectural sampling 只做过 VLM。
- **最接近工作**：2607.10139、2609.32035、2608.09351、2610.01687、2606.19636、2609.19499。
- **风险**：低准确率的来源虽然去相关，但会带来更多错误票，净收益可能为 0（Condorcet 条件）。
- **初筛判定**：**CROWDED-BUT-DELTA**（WebSearch 没有找到“think 与 no-think 混合投票去相关”的工作；但多样性来源对比已有 2608.09351 和 2609.19499）。

### Idea 4 — 文本自述的中间答案：零成本 commitment 特征（对比 probe 类方法）
- **假设**：从 thinking 文本中正则抽取的中间答案序列，比如“答案翻转次数”、“最终答案首次出现的相对位置”、“最终答案被重复确认的次数”，能达到 probe 类信号（Parallel-Probe、ParaTempo、prefix consistency）的大部分效果，而且额外 forward 为 0。
- **最小实验**（约 0.3 GPU-day，可复用 Idea 1/2 的 trace）：对每个特征计算题内 AUROC、长度残差化 AUROC 和投票增益，并与 prefix consistency（2605.07654）在匹配 token 下比较。
- **新颖性论证**：现有工作都需要额外 probe 或续写（2504.20708 subthought 续写、2605.07654 重新生成、2602.03845 2D probe）。
- **风险**：2609.32035 发现 hedging 计数在长度残差化后跌到 0.474，低于随机。翻转次数很可能同样是“长度的伪装”，这是核心风险，需要在第 1 天就验证。
- **初筛判定**：**CROWDED-BUT-DELTA**（3 次检索命中 Parallel-Probe、ParaTempo、MARS、Prefix Consistency、2504.20708，但都不是零成本的文本抽取）。

### Idea 5 — Filtering vs. Weighting：调停 DeepConf 与 “Self-Consistency Never Notices” 的矛盾
- **假设**：DeepConf 的收益主要来自 *过滤*，即删掉截断、循环、无答案的 trace，相当于去除 no-answer 和超长 trace，而不是来自置信度本身对正确性的区分。一个“删掉触顶或重复 trace + 长度过滤”的平凡基线，就能复现它的大部分收益。
- **最小实验**（约 0.5 GPU-day）：复现 DeepConf-low/high（η=10%/90%），加入平凡过滤基线、长度过滤、随机过滤；按题目做 cluster bootstrap 和 Holm 校正；在 Qwen3-8B、R1-Distill-7B 上比较。
- **新颖性论证**：2609.32035 明确把 filtering 列为 threat 并且没有覆盖；FoT（2608.15065）的 hesitation 剪枝从侧面支持“删病态 trace 有效”的说法。
- **最接近工作**：2508.15260、2609.32035、2510.10409、2608.15065。
- **风险**：属于纯分析贡献，需要写得非常扎实才能中稿；如果 DeepConf 收益确实存在，结论会变得平淡。
- **初筛判定**：**CROWDED-BUT-DELTA**。

### Idea 6 — 后训练 regime 如何改变 TTS 信号（同 base 的 SFT 蒸馏 vs RL）
- **假设**：在同一 base 上，RL（相对 SFT 蒸馏）会提高错误集中度（collision↑），并让 answer-confidence 由“正向预测正确”变为“预测错误”。这会使置信投票失效，并扩大 pass@k−maj@k gap。
- **最小实验**（约 1 GPU-day）：
  - 公开 checkpoint 链 1：Qwen2.5-Math-7B → R1-Distill-Qwen-7B → 在其上做 RL 的公开模型（如 Skywork-OR1-7B、AceReason-Nemotron-7B）。
  - 公开 checkpoint 链 2：Qwen2.5-Math-7B → SimpleRL-Zoo 或 Oat-Zero（zero-RL）。
  - 指标：collision 率、置信 AUROC、各投票法增益。
- **最接近工作**：2510.10409、2609.32035、2504.13837、2609.33290、2604.16830（On-Policy Distillation calibration）。
- **风险**：不同 checkpoint 链的训练数据不同，难以干净地归因；只能做相关分析。
- **初筛判定**：**CROWDED-BUT-DELTA**。

### Idea 7 — Architectural Sampling → 长 CoT 文本推理的迁移检验
- **假设**：在 Qwen3-8B 上复用早期层块（不训练），可以在 AIME/GPQA 上带来比温度采样更高的 pass@k 和更低的 collision。但长 CoT 中误差会累积，可能需要“只在前 k 个 token 或问题编码阶段”使用改动后的路径。
- **最小实验**（约 0.5 GPU-day）：九种层复用配置 × 每种 8 个样本，与 72 个温度样本在匹配算力下比较 pass@k 和 maj@k。
- **最接近工作**：2610.01687（只做 VLM）、2507.07996（Skip a Layer or Loop it，按输入自适应深度，不是为了多样性）、2606.19636（残差扰动）。
- **风险**：长生成中模型可能直接崩坏；可以并入 Idea 3 作为其中一种来源。
- **初筛判定**：**CROWDED-BUT-DELTA**（WebSearch 和 arXiv 都没有找到针对文本 LRM 的版本，但这是显而易见的迁移，随时可能被人抢先）。

### Idea 8 — 用 TTS 曲线形状做污染探针（低优先级）
- **假设**：在被记忆的题上，maj@k 很早饱和、pass@k 曲线平坦，而且在数值扰动版本上答案依然“粘”在原值。在难度匹配的新题（AIME26 对比 AIME24）上，曲线形状不同。
- **最接近工作**：2501.11790（unseen random-variable questions）、2607.21433、2508.13654。
- **风险**：难度和曲线形状强混杂，与 TTS 主线的关系较远。
- **初筛判定**：不确定，倾向 **NOVEL-LIKELY 但价值存疑**。

### 已排除（ALREADY-DONE / 过度拥挤）
- 基于中间答案 probe 的跨分支早停和剪枝：Parallel-Probe 2602.03845、ParaTempo 2608.16425、MARS 2606.12935、TRACE 2604.17304。
- 用 no-think 草稿一致性路由思考预算：DART 2606.23181。
- 量化 × TTS：BitCal-TTS 2605.05561、BreadthKV 2610.05685。
- forced-`</think>` 与自由续写的差距：BAEE 2604.06613、2609.03633。
- dLLM 解码顺序多样性：UnMaskFork 2602.04344、SIGMA（NeurIPS'26）。
- 开放式生成的 selection：TTS in the Wild 2608.18931、Consilience 2608.09898。
- 置信与长度混杂的纯诊断：2609.32035、2510.10409 已经做过。

---

## 4. 建议的执行顺序
1. 先跑一批共享 trace：Qwen3-8B 的 think×64 + no-think×32，覆盖 AIME24/25/26、HMMT25、GPQA-D，约 0.5–1 GPU-day。在这批数据上可以同时检验 Idea 1、4、5，几乎零增量。
2. 在同一批 trace 上加 0.6B draft 的 teacher-forcing 计算，检验 Idea 2（不到 0.2 GPU-day）。
3. 如果 1 和 2 中有一个正向信号，再扩展到 R1-Distill-7B 和 Qwen3-4B；在此基础上补充 Idea 3/7 的多样性来源实验。
4. 所有结论都要过三项检验：长度残差化、cluster bootstrap、多重比较校正。这是 2026 年审稿人的默认要求（参考 2609.32035 和 2609.40190）。
