# 05 — 其他低资源 / Training-free LLM·MLLM 方向：文献地图 + 候选 Idea

> 日期：2026-10-17。检索方式：arXiv HTML search（按 announced date 倒序）+ WebSearch + 逐篇拉 abs 页核验标题。下文所有 arXiv id 均已通过 arxiv.org/abs 页面或检索结果列表核验真实存在（未核验的条目不列出）。
> 范围：激活引导/表征工程、training-free 幻觉缓解、内部状态不确定性、模型合并、长上下文、机制可解释性驱动干预、LLM-as-judge/污染检测、推理期安全、Agent/RAG 压缩、dLLM 与统一多模态模型。不覆盖纯效率 / test-time scaling / 持续学习（其他 agent 负责）。
> 方法：参考 skills/research-lit、idea-creator、novelty-check 的流程，但没有 Codex/Gemini 外部评审、没有 GPU 试点。所有 novelty 判定都是"2-3 次定向检索 + 读摘要"级别，**置信度中等**，正式立项前应再读最接近论文的全文。

---

## 0. 总体判断（TL;DR）

1. **最拥挤**：LVLM 物体幻觉的 training-free 解码/注意力干预（2026 年每月 10+ 篇，几乎所有"对比解码 + 注意力重加权 + 头级干预"组合都已被做过）；LLM-as-judge 偏差分析；探针式幻觉检测（"均值偏移即可"已被证明，见 2608.28930）。
2. **快速变挤**：dLLM（扩散语言模型）上的 training-free 方法。2026 下半年已有幻觉检测（DynHD/TRE/DeMTS）、激活引导（TRACESTEER）、安全（SADDER、safety neuron 迁移）、重掩码策略（CoDR）等。只剩"机制继承/迁移"和"视觉多模态 dLLM"相对空。
3. **相对开放的结构性空白**：
   - **"谱系迁移"**：在父模型（instruct / AR）上算好的探针、引导向量、头，能否零成本复用到子模型（R1-distill 推理模型、long-to-short 合并模型、AR→dLLM 转换模型）？权重层面已有（2610.08108 将 AR post-training delta 复用到 dLLM），激活/监控层面系统研究缺失。
   - **合并的安全副作用**：long-to-short 合并已被广泛用于效率，但其对拒答/越狱鲁棒性的影响几乎没人测。
   - **对旧 training-free 技巧在 2026 年强模型上的复现审计**：大部分 VCD/OPERA 类方法的证据仍来自 LLaVA-1.5。
   - **长 CoT 中的干预衰减**：引导向量在数千 token 推理中的"半衰期"和被 reflection 重置的现象，只有零散证据（2602.06941、2605.26772）。

---

## 1. 分子领域文献地图

### 1.1 激活引导 / 表征工程（steering, SAE, refusal direction）

代表作（2025-2026）：
- Understanding (Un)Reliability of Steering Vectors in Language Models — 2505.22637（2025-05）：引导向量可靠性的基础负面结果。
- Steering LLM Reasoning Through Bias-Only Adaptation — 2505.18706（2025-05）；Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors — 2509.06608（2025-09）：RL 增益可压进每层一个向量。
- Base Models Know How to Reason, Thinking Models Learn When — 2510.07364（2025-10）：base 模型 + 引导向量恢复大部分推理模型差距。
- Endogenous Resistance to Activation Steering in Language Models — 2602.06941（2026-02）：模型在生成中途"抵抗"引导（"wait, that's not right"）。
- Steering Vector Fields for Context-Aware Inference-Time Control — 2602.01654（2026-02/09）。
- Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal — 2605.26772（2026-05）：在 LRM 中 CoT 会重建拒答信号，单方向引导失效。
- Reflection Steering: Disentangling Reflection from Reasoning in Activation Space — 2608.25542（2026-08）。
- GAPS: Dimension-Level Gates for Conditional Activation Steering — 2609.01878（2026-09）；Programming Refusal with Conditional Activation Steering — 2409.05907（早期基线）。
- LLM Layers Immediately Correct Each Other — 2609.07876（2026-09）：相邻层互相抵消，挑战"特征持久存在"假设。
- Disentangling Steering Vectors — 2609.07037（2026-09）；The Sharp Directions Are Against You: Curvature Analysis of Activation Steering — 2610.16084（2026-10）。
- Steering Under Compression: Dose-Response… in Quantized LLMs — 2609.06473（2026-09）：引导 × 量化已被系统研究。
- On the Steering Dimensionality of Refusal in Language Models — 2610.04245（2026-10）；Fast Multi-dimensional Refusal Subspaces via RFM-AGOP — 2607.02396（2026-07）。
- One Token Can Be Enough: Bridging Prompting and Activation Steering with Prefix Steering — 2610.04967（2026-10）。
- Activation Steering for Masked Diffusion Language Models — 2512.24143（2025-12）；Steering Without Breaking (discrete dLLM) — 2605.10971（2026-05）。
- 多模态：Textual Steering Vectors Can Improve Visual Understanding in MLLMs — 2505.14071（2025-05）；Harnessing Textual Refusal Directions for Multimodal Safety — 2606.31876（2026-06）；Do Unified Multimodal Models Think in One Space? Cross-Branch Steering — 2607.26411（2026-07）。

饱和 vs 开放：
- 饱和：拒答方向（单/多维、旋转、门控）、条件引导、SAE 特征选择、引导 × 量化、跨分支（统一模型）引导、RL→向量。
- 开放：引导在**长 CoT 推理中的时间动态**（衰减、被反思重置）只有间接证据；引导向量**跨谱系复用**（instruct→distill、AR→dLLM）缺少系统性正面研究（2512.24143 仅给出原生 MDLM→AR 的负结果）。

### 1.2 Training-free 幻觉缓解（MLLM / VLM）

代表作：
- KVSmooth — 2602.04268（2026-02）；Look Carefully: Adaptive Visual Reinforcements — 2602.24041（2026-02）。
- Decoding by Perturbation (Dynamic Textual Perturbation) — 2604.12424（2026-04）。
- Cognitive Pivot Points and Visual Anchoring (MLRM 幻觉) — 2604.10219（2026-04）；Mitigating Visual Forgetting via Take-along Visual Conditioning — 2503.13360（2025-03）。
- Vocabulary Hijacking in LVLMs — 2605.10622（2026-05）；CHASD — 2605.23344；Region-Aware Attention Recalibration — 2605.24957；Adversarial Orthogonal Disentanglement — 2605.25377（2026-05）。
- See Only When Needed: Context-Aware Attention Intervention — 2606.29847（2026-06）；Spectral Query-Key Product Weight Steering — 2606.20419（2026-06）。
- Targeting the Attention Heads Behind Object Hallucination in LLaVA — 2608.24966（2026-08）；Same Attention, Different Truths: Logit-Lens over Visual Attention — 2608.07302（2026-08）。
- MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads (HEAL) — 2609.09206（2026-09）。
- Lost in Contrast: Instruction-Preserving Decoding — 2610.16722（2026-10）：指出 VCD 会吞掉合理的 abstention。
- Attention Sinks in VLMs Are Borrowed, Not Seen — 2610.15855（2026-10）。
- 关系幻觉：RECAP — 2609.26093；Relation-aware Visual Enhancement — 2605.11808；When Relations Break — 2605.05045。
- 扩散 VLM：Reliability Challenges in Diffusion Vision-Language Models — 2609.01318（2026-09）；Before the Token Commits (DynaHall) — 2609.34772（2026-09）：dVLM 的视觉幻觉在答案 token 揭示前就已"定型"；ST-Veto — 2607.17884；DAVET — 2608.01821。

饱和 vs 开放：
- 极度饱和：LLaVA-1.5 上 POPE/CHAIR 的对比解码、注意力重加权、头级干预、sink 处理。任何"新组合"都很难过审。
- 开放：(i) **对强 2026 模型（Qwen2.5/3-VL、InternVL3.x）的系统复现审计**——web 检索未找到；(ii) **扩散 VLM** 上利用"早期定型"现象的干预——只有 benchmark（DynaHall）和效率工作（DAVET），缺少专门针对幻觉的早期窗口干预。

### 1.3 内部状态的不确定性 / 校准 / 幻觉检测探针

代表作：
- Reasoning Models Know When They're Right: Probing Hidden States — 2504.05419（2025-04）。
- Efficient Hallucination Detection: Adaptive Bayesian Semantic Entropy — 2603.22812（2026-03）。
- Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing — 2604.13068（2026-04）。
- HalluSAE — 2604.16430；The First Token Knows: Single-Decode Confidence — 2605.05166（2026-05）。
- Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs — 2606.02628（2026-06）。
- From Signals to Transfer: A Factorised Study of Probe-Based UE — 2606.27679（2026-06/10）。
- The Hallucination Signal Is a Mean Shift: Why Simple Probes Suffice — 2608.28930（2026-08）。
- Gradients for Interventions and Activations for Detection — 2609.32355（2026-09）：解释了检测/纠正不对称。
- Effectiveness of Hidden-State Probes for Localizing Reasoning Failures — 2610.22046（2026-10）；RL-ARC — 2610.11352；Test-time Calibration Learning — 2610.02695。
- dLLM：DynHD — 2603.16459；TRE: Training-Free Hallucination Detection for dLLMs — 2607.22661；DeMTS — 2608.14632；A Heckler in the Hidden State — 2609.36783；Bayesian Entropy-based Reordering for Calibrated dLLMs — 2610.05125。
- 多模态：VL-Uncertainty — 2411.11919；PDCR: Perception-Decomposed Confidence Reward — 2605.13467（训练式）；Estimating Uncertainty of Omnimodal LLMs — 2610.16933。

饱和 vs 开放：
- 饱和：同一模型内训练探针；dLLM 去噪轨迹检测（4+ 篇）；语义熵加速。
- 开放：**探针的跨模型（谱系）迁移**——2606.27679 研究的是跨数据分布迁移，不是跨模型；2608.28930 的"单一均值偏移"发现暗示该方向可能在后训练中保持，这正是零成本迁移的理论依据，但无人验证。**多模态推理模型中感知 vs 推理不确定性的 training-free 分解**也较空（PDCR 是训练 reward）。

### 1.4 模型合并（非 CL）与权重插值

代表作：
- Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging — 2503.20641（2025-03）。
- Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging — 2505.05464（2025-05）；Training-Free Reasoning and Reflection in MLLMs — 2505.16151（2025-05）。
- Weight Ensembling Improves Reasoning in Language Models — 2504.10478。
- Revisiting Model Interpolation for Efficient Reasoning — 2510.10977（2025-10）。
- Extrapolative Weight Averaging Reveals Correctness-Efficiency Frontiers in Code RL — 2605.28751（2026-05）；ThinkSwitch — 2606.01080。
- Rethinking Heterogeneous LLM Merging: A Weighted Model Averaging Perspective — 2607.18026（2026-07）。
- Asymmetric Collapse in Model Merging: When Refusal Over-writes Recognition — 2607.27240（2026-07，1B 安全微调合并）。
- When Skills Meet Safety: Jailbreak Robustness of Skill-Merged LLMs — 2608.08542（2026-08）。
- Enhancing Diffusion Language Models with Autoregressive Post-Training Weights — 2610.08108（2026-10）：AR post-training delta 直接加到 dLLM 上有效。
- SLEM: Spectrum-Aware Latent Expert Merging — 2610.16484。

饱和 vs 开放：
- 饱和：long-to-short 插值本身、视觉+推理合并、AR delta→dLLM。
- 开放：**long-to-short 合并对推理模型安全性的影响**（web 检索未命中直接研究；现有安全合并研究集中在 skill/safety 微调合并）；合并后**监控器（探针/拒答方向）是否仍有效**。

### 1.5 Training-free 长上下文 / 上下文工程

代表作：
- Attention Reveals More Than Tokens: Training-Free Long-Context Reasoning with Attention-guided Retrieval — 2503.09819（2025-03）。
- LongLLaDA — 2506.14429（2025-06）：dLLM 的 training-free 长度外推。
- Attention Sinks in Diffusion Language Models — 2510.15731（2025-10）。
- MrRoPE — 2601.22181（2026-01）；DySCO: Dynamic Attention-Scaling Decoding — 2602.22175（2026-02）；VERA: Visual Evidence Retrieval Heads — 2602.10146（2026-02）。
- Reasoning Shift: How Context Silently Shortens LLM Reasoning — 2604.01161（2026-04）。
- LongGuard: Training-Free Mitigation of Long-Context Failure in Safety Guardrails — 2608.27580（2026-08）。
- Mechanics of Long-Context Hybrid Models Part 1.1 — 2610.10114（2026-10）。

饱和 vs 开放：位置插值/NTK 类已极成熟；检索头用于 KV 压缩被效率方向覆盖。开放点：**推理模型长 CoT 期间对 prompt/检索文档的注意力衰减**的文本版（视觉版 TVC 已有）；dLLM 的检索头是否从 AR 父模型继承（未找到研究）。

### 1.6 机制可解释性驱动的廉价干预（MLLM / 推理模型）

- Attention in Space: Functional Roles of VLM Heads for Spatial Reasoning — 2603.20662；Gaze Heads — 2606.14703；CoRe Heads — 2606.05843。
- Reasoning Theater: Disentangling Model Beliefs from CoT — 2603.05488（2026-03）：答案在 CoT 很早就可被探针解码。
- Induction in Both Directions: ICL in Masked Diffusion LMs — 2607.15893（2026-07）。
- A Comparative Analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs — 2603.07475（2026-03/08）：Dream-7B 保留 AR 式动态。
- Understanding Unified VLMs: From Attention Probing to Guidance Tuning — 2610.15227（2026-10）。
- COMPASS: Finding Where Reasoning Lives in Language Models — 2610.07469。

饱和 vs 开放：VLM 头的功能分类已很多；"AR→dLLM 转换后机制保留多少"只有表征相似度分析（2603.07475）和安全神经元迁移（2608.07430），**可复用于正向用途（检测、长上下文、引导）的系统地图缺失**。

### 1.7 LLM-as-judge 与评测方法学 / 污染检测

- Assessing Judging Bias in Large Reasoning Models — 2504.09946；Explicit Reasoning Makes Better Judges — 2509.13332。
- Opening LLM Judges: Recovering Preference Signals Beyond the Final Verdict — 2609.32407；The First Token Is Not the Verdict — 2610.00054；How Much Do LLM-as-a-Judge Design Choices Matter? — 2610.05094；A CaLLM Judge is a Better Judge — 2610.18108。
- 污染：On The Fragility of Benchmark Contamination Detection in Reasoning Models — 2510.02386；The Illusion of Reasoning: Zero-CoT Truncation — 2605.21856；When Is Benchmark Contamination Detectable? — 2608.07914；Excess Separability: Residual-Stream Probing for Contamination — 2608.12652；WildTrace — 2610.16152；多模态：Both Text and Images Leaked! — 2411.03823。

饱和 vs 开放：judge 偏差（位置、自偏好、格式、推理长度）全面饱和；污染检测"理论上限 + 探针脆弱性"刚被系统指出（2608.07914、2608.12652），white-box 推理模型污染信号仍有缝隙但风险高。

### 1.8 推理期安全（越狱防御、推理模型安全、内部状态守卫）

- Jailbreak Antidote — 2410.02298；Jailbreaking Leaves a Trace — 2602.11495；LLM Safety From Within — 2604.18519；Agent Safety From Within — 2609.33039。
- Mitigating Adaptive Attacks against Reasoning Models with Activation Consistency Training — 2605.28467；Refusal Reads Only a Slice of What the Model Knows — 2609.14759。
- dLLM 安全：The Devil behind the mask (DIJA) — 2507.11097；Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits — 2608.07430（AR 初始化的 dLLM 继承父模型 safety neuron）；SADDER — 2610.17636。
- ReFrame: Test-Time Safety Alignment in MLLMs — 2608.21100；Mitigating LLM Over-Refusal via Dynamic Semantic Routing — 2609.25049。

饱和 vs 开放：内部状态越狱检测已很多；开放点集中在**合并/蒸馏等廉价后处理对安全的副作用**。

### 1.9 Agent / RAG：记忆压缩、提示压缩、工具使用效率

- ReFold: Training-Free Reversible Inter-Turn Context Folding — 2610.07863；Skip the Compressor: Coding Agents Fix More Issues When They Pick Whole Files — 2610.21073（压缩器可能有害）；Adapting Context Compression for Long-Horizon Agents with Counterfactual Continuations — 2609.36526；Will My Assistant Remember My Allergy? — 2609.05767；ToolChoiceConfusion — 2606.06284；ASA: Representation Engineering for Tool-Calling Agents — 2602.04935；Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG — 2511.09803。

饱和 vs 开放：工程化方法多、评测分散；学术上 novelty 难以证明，本报告不重点推荐。

### 1.10 dLLM 与统一多模态模型的 training-free 改进

- dLLM：CoDR: Training-Free Confidence-Drift Remasking — 2610.08833；One Block Ahead — 2610.19510；Where to Place the Query? Positional Bias in ICL for dLLMs — 2606.19349；Follow the Latent Roadmap (anchor tokens) — 2606.16847；TRACESTEER — 2610.18173；Noise Out, Bias In — 2610.05894。
- 统一模型：Flash-Unified — 2603.15271；UniRect-CoT — 2604.13540；Steering Visual Generation in UMMs with Understanding Supervision — 2605.05781；Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models — 2607.11886；UniCache — 2609.32831；SegBanana — 2609.34235。

饱和 vs 开放：dLLM 解码顺序/重掩码极度拥挤（且属效率）；统一模型"理解指导生成"已做过，反向（生成辅助理解幻觉）较少但证据不足。

（以上共列出 100+ 篇真实论文，均来自 arXiv 检索结果或 abs 页核验。）

---

## 2. 结构性空白（按 lens）

| Lens | 观察 | 空白 |
|---|---|---|
| Method transfer | 探针/引导向量都在"同一模型内"训练与使用 | 父模型→子模型（distill / merge / AR→dLLM）零成本迁移 |
| Contradiction | 2608.28930 称幻觉信号是单一均值偏移；2604.13068 称沿该方向引导无法纠正；2609.32355 用梯度解释 | 若信号是均值偏移，那么后训练很可能只"平移"它——可用无标签重中心化迁移 |
| Untested assumption | long-to-short 合并被当作"免费午餐"，只报告准确率/长度 | 安全/拒答/越狱鲁棒性随 α 的变化未测 |
| Unexplored regime | 引导向量多在短生成中评估 | 数千 token CoT 下的效应衰减与反思重置 |
| Diagnostic nobody asked | 2024 年的 VCD/OPERA 等在 LLaVA-1.5 上有效 | 在 2026 强 MLLM 上还有效吗？是否损害通用能力与 abstention？ |
| New-architecture | dVLM 幻觉在答案揭示前已定型（2609.34772） | 只在早期去噪窗口施加视觉对比，代价/效果 |

---

## 3. 候选 Idea（按推荐度排序）

资源假设：单卡 24-80GB，开源 7-8B 模型（Qwen2.5-7B-Instruct、DeepSeek-R1-Distill-Qwen-7B、Dream-v0-Instruct-7B、LLaDA-8B、LLaDA-V、Qwen2.5-VL-7B、InternVL3-8B），每个想法验证 ≤ 2-3 GPU-day。

### Idea 1 —— 监控器的谱系继承：父模型探针/方向零成本迁移到推理蒸馏与合并子代
- **假设**：在 Qwen2.5-7B-Instruct 上训练的幻觉/正确性线性探针和拒答方向，在 R1-Distill-Qwen-7B、long-to-short 合并模型、AWQ 量化版上，只需**无标签的均值/协方差重中心化**（用约 200 条无标签 prompt）就能恢复 ≥90% 的同模型 AUROC。
- **机制**：2608.28930 显示信号以单一均值偏移为主；后训练主要是低秩、近正交更新（参见 2610.08108 的观察），因此该方向大概率被保留，只发生平移和缩放。
- **最小实验**：TriviaQA/SciQ/MATH 正确性标签；在父模型逐层训练 LR 探针 → 直接用于 / 重中心化后用于 4 个子代；对比在子代上从头训练的探针（上限）和 first-token entropy（2605.05166，下限）。同时测拒答方向在子代上的消融/添加效果。约 1 GPU-day。
- **Novelty 论据**：现有探针迁移研究针对数据分布（2606.27679），不针对模型谱系；2509.06608 只测 RL 训练出的引导向量在"近似模型"间迁移；2602.09434 把拒答向量当作指纹，而不是当作可复用的监控器。
- **最接近工作**：2608.28930、2606.27679、2504.05419、2509.06608、2602.09434。
- **风险**：长 CoT 子代的"答案位置"与父模型不同（需对齐到 `</think>` 后的位置）；结果可能是"部分迁移"，叙事需要包含"什么时候失效"（如 RL 越多，迁移越差）。
- **判定**：**NOVEL-LIKELY**（3 次检索未命中直接工作）。

### Idea 2 —— Long-to-short 合并的隐藏安全代价与拒答方向保持合并
- **假设**：在 instruct ↔ reasoning 之间做插值时，拒答率/越狱 ASR 随 α **非单调**变化，中间 α 最差。原因是 CoT 被缩短，而拒答方向又与两端都不对齐（参见 2605.26772：CoT 本身承载拒答）。
- **机制 / 修复**：逐层计算两端模型拒答方向的余弦相似度，在低相似层回退到更安全端的权重；或者在合并后 training-free 地把拒答方向投影回去。
- **最小实验**：Qwen2.5-7B-Instruct × R1-Distill-Qwen-7B（同 base）用 Task Arithmetic/TIES/SLERP，α 取 11 个点；指标为 HarmBench/StrongREJECT/XSTest 过度拒答、WildJailbreak ASR、MATH500/GSM8K 准确率与长度。约 1.5 GPU-day（vLLM 推理）。
- **Novelty 论据**：L2S 合并文献（2503.20641、2510.10977）只报告准确率与长度；安全合并文献（2607.27240、2608.08542）研究的是 skill/安全微调模型的合并，不涉及推理↔非推理插值。
- **最接近工作**：2503.20641、2510.10977、2607.27240、2608.08542、2605.26772。
- **风险**：效应可能是单调的（缺少"非单调"卖点，但"安全随效率下降"本身仍然可报告）；需要用 LLM judge 判定有害性（可用 Llama-Guard-3-8B 本地运行）。
- **判定**：**NOVEL-LIKELY**。

### Idea 3 —— 2026 年强 MLLM 上 training-free 幻觉缓解的复现审计
- **假设**：VCD、M3ID、OPERA、PAI、ICD、DoLa-style、头级干预等方法在 Qwen2.5-VL-7B / InternVL3-8B / Qwen3-VL-8B 上的 POPE/CHAIR/AMBER 增益缩小到 ≤1 个点，并且会损害 MMBench/MMMU 和 abstention（参见 2610.16722）；增益与基线模型的"语言先验强度"负相关。
- **最小实验**：统一代码库，跑 8 种方法 × 4 个模型 × 6 个 benchmark（含通用能力与 abstention 的 decision utility），并在相同 compute 下对照"更好的 prompt / 更大分辨率"基线。约 3 GPU-day。
- **Novelty 论据**：web 检索未找到此类复现研究；新方法论文常把 Qwen2.5-VL 作为 backbone，但很少同时复现旧方法。
- **最接近工作**：2610.16722、2605.25377、2509.21997、VCD（2311.16922）。
- **风险**：属于"负结果 / benchmark"类论文，更适合 TMLR 或 D&B 赛道；需要非常干净的实现。
- **判定**：**NOVEL-LIKELY**（作为审计论文）；方法层面不新。

### Idea 4 —— AR→dLLM 转换中的机制继承地图（探针、引导向量、检索头）
- **假设**：Dream-7B（由 Qwen2.5-7B 初始化）继承父模型的检索头、真值/幻觉方向和情感/拒答引导向量，在 dLLM 中只需按去噪步做缩放即可直接复用；原生训练的 LLaDA 则不能复用。
- **最小实验**：(a) 在 Qwen2.5-7B 上做 NIAH 检索头打分 → 在 Dream 上消融同一组头，比较 NIAH 掉点与随机头的差异；(b) 父模型的 truthfulness 探针用于 Dream 的 masked 位置隐藏态；(c) 父模型引导向量按去噪步注入。约 1.5 GPU-day。
- **Novelty 论据**：2608.07430 只证明 safety neuron 被继承，并用于攻击；2610.08108 是权重 delta 的复用；2603.07475 只做表征相似度分析；2512.24143 给出原生 MDLM→AR 的负迁移。"正向用途的激活级复用"缺少系统研究。
- **最接近工作**：2608.07430、2610.08108、2603.07475、2512.24143、2605.10971、2607.15893。
- **风险**：2608.07430 已占据"继承"这个叙事，新意需落在"头 + 探针 + 引导"的统一地图和实际收益上（例如零成本 dLLM 幻觉检测，与 DynHD/TRE 对比）。
- **判定**：**CROWDED-BUT-DELTA**。可与 Idea 1 合并成一篇"Lineage-transferable monitors"论文。

### Idea 5 —— 扩散 VLM 的"承诺窗口"视觉对比引导
- **假设**：既然 dVLM 的幻觉在早期去噪步就已定型（2609.34772），那么只在前 k% 步施加图像条件的 CFG/VCD（有图 vs 无图/扰动图），就能拿到全步 VCD ≥90% 的收益，而额外代价降到约 k%。
- **最小实验**：LLaDA-V / Dream-VL / MMaDA，在 POPE、AMBER、DynaHall 上扫描 k∈{5,10,20,50,100}%，同时测 MMBench 是否受损；比较按 token 置信度触发与按固定步触发。约 1 GPU-day。
- **Novelty 论据**：DAVET（2608.01821）按去噪步分配视觉证据，但目标是效率；ST-Veto（2607.17884）用 image-attention 否决 token；TRACESTEER（2610.18173）是纯文本 dLLM 的引导。web 检索未找到 dVLM 上按步调度的视觉对比解码。
- **最接近工作**：2609.34772、2608.01821、2607.17884、2610.18173、2610.15227。
- **风险**：DynaHall 原文可能已附带类似 baseline（只读了摘要，需读全文）；dVLM 生态小，审稿人关注度有限。
- **判定**：**CROWDED-BUT-DELTA**。

### Idea 6 —— 长 CoT 中的引导半衰期与反思触发的再注入
- **假设**：固定强度的引导效应随生成长度衰减；在 "Wait/Alternatively" 等反思 token 后会出现阶跃式重置（与 2602.06941 的 ESR 一致）。只在反思起始 token 处再注入，可以用更低的平均强度达到同等行为控制，并减少能力损失。
- **最小实验**：R1-Distill-Qwen-7B，取 3 个行为方向（拒答、简洁性、语言切换），逐 token 测量方向投影和行为探针随位置的变化；对比恒定注入、衰减补偿注入与事件触发注入的行为 / MATH 准确率权衡。约 1 GPU-day。
- **Novelty 论据**：2602.06941 在非推理模型中观察到抵抗现象，2605.26772 只研究拒答且 CoT 固定，2609.07876 研究的是层间纠正而非时间维度。"时间维度的半衰期曲线 + 事件触发"尚未见到。
- **最接近工作**：2602.06941、2605.26772、2608.25542、2609.07876、2604.02113。
- **风险**：贡献偏分析性，"事件触发"可能被视为 CAST（2409.05907）的变体。
- **判定**：**CROWDED-BUT-DELTA**。

### Idea 7 —— 多模态推理模型的感知/推理不确定性 training-free 分解 + 选择性回看
- **假设**：对同一 CoT 前缀做"有图 / 空白图"两次前向，二者 next-token 分布的 KL 衡量感知依赖；把这个量与熵组合，可以区分感知不确定和推理不确定。只在感知不确定步重新注入视觉 token（类 TVC），即可提升 MathVista/MathVerse。
- **最小实验**：Qwen2.5-VL-7B-based 推理模型（如 VL-Rethinker-7B / R1-OneVision-7B），在 MathVerse vision-only 子集上测 AUROC 和选择性回看的准确率 / 额外 FLOPs。约 1 GPU-day。
- **最接近工作**：PDCR 2605.13467（训练式 reward）、2604.10219、2503.13360、2411.11919、2609.05708。
- **风险**：空白图对比与 VCD/M3ID 思路相近，审稿人可能认为是旧技巧换个位置。
- **判定**：**CROWDED-BUT-DELTA**。

### Idea 8 —— 文本长 CoT 推理模型在 RAG 中的"上下文注意力衰减"与 training-free 重锚定
- **假设**：随着 CoT 增长，模型分配给检索文档的注意力质量单调下降，晚期推理步更容易违背文档（上下文不忠实）；在反思 token 处对文档 span 做注意力温度提升（DySCO 式），可以恢复忠实度。
- **最小实验**：R1-Distill-Qwen-7B，在 HotpotQA-distractor / FaithEval / ConFiQA 上测注意力质量与忠实度随步数的曲线，再做干预。约 1 GPU-day。
- **最接近工作**：2503.09819、2602.22175、2503.13360、2604.01161。
- **风险**：效应可能主要来自"答案已经确定"（2603.05488 reasoning theater），而不是注意力衰减。
- **判定**：**CROWDED-BUT-DELTA**。

### Idea 9 —— 推理模型的白盒污染指纹：思考前答案可解码性 + 同构扰动对照
- **假设**：被污染的题目在 `<think>` 起始位置，探针就能以高置信度解码出最终答案；同一题的数值同构扰动版本则不能。二者之间的"预承诺差"是一种比 Zero-CoT 截断更稳健的污染信号。
- **最小实验**：在 Qwen2.5-7B 上故意用 GSM8K/MATH 的一半测试题做 LoRA 污染（可控 ground truth），再做 RL-style 后训练（参照 2510.02386 的协议），比较 ZCP、Min-K% 与本方法。约 2 GPU-day。
- **最接近工作**：2605.21856、2603.05488、2608.12652、2510.02386、2608.07914。
- **风险**：**高**。2608.12652 指出探针式污染检测方差大于零分布；2608.07914 给出可检测性上限。
- **判定**：**CROWDED-BUT-DELTA（高风险）**。

### 已检查但判定为已做 / 不推荐（避免重复）
- RL 增益 = 引导向量 / base 模型 + 向量复现推理模型：**ALREADY-DONE**（2505.18706、2509.06608、2510.07364）。
- 统一多模态模型中理解分支向生成分支的跨分支引导：**ALREADY-DONE**（2607.26411）。
- AR post-training 权重 delta 迁移到 dLLM：**ALREADY-DONE**（2610.08108）。
- 引导 × 量化交互：**ALREADY-DONE**（2609.06473）。
- dLLM 去噪轨迹幻觉检测：**ALREADY-DONE / 拥挤**（2603.16459、2607.22661、2608.14632）。
- 推理 judge 的偏差分析：**拥挤**（2504.09946、2509.13332 及 2026-10 的多篇）。
- dVLM 视觉 token 剪枝：**拥挤**（2601.20168、2511.15098、2511.12428，且属效率方向）。

---

## 4. 推荐组合

- **首选**：Idea 1 + Idea 4 合并为一篇"Lineage-Transferable Internal Monitors"（instruct→distill/merge/quant/AR→dLLM），并以 Idea 2 作为安全应用章节（合并后拒答方向漂移 + 修复）。三者共享同一套代码（探针 + 方向提取 + vLLM 评测），总计约 4 GPU-day。
- **备选（快速出结果）**：Idea 3 审计（工程量大、风险低）；Idea 5（dVLM，算力最省）。
- **注意**：所有 novelty 判定只基于摘要级检索；Idea 2、Idea 5 立项前必须精读 2503.20641 / 2510.10977 的安全部分（如果有）和 DynaHall（2609.34772）全文。
