# 阅读进度

这个文件只记录**读过的**和**正在读的**论文。还没读的论文在 `to_read.md`，下载链接在 `to_read_links.md`。每读完一节（或换一篇）就更新这里，写法见 `skills/paper-close-reading/SKILL.md` §4。

两种“读过”分开记：
- **精读**：一节一节带读，有分节结论、挑刺和问题追踪。
- **通读**：读完全文，写了 NOTES.md 条目（一句话 + 有意思的发现 + 我的看法）。

最后更新：2026-10-18

---

## 一、正在读

### 1. On Test-Time Scaling for Vision-Language Models（ECCV'26；VUB / imec）[2606.28864](https://arxiv.org/abs/2606.28864)
- **方式**：精读
- **读到**：§1 Intro（L41）、§2 Related Work（L164）已读完。**下一节：§3 Setup（L193）**
- **原文与章节行号**：重新下载用 `curl -sL https://arxiv.org/pdf/2606.28864 -o paper.pdf && pdftotext -layout paper.pdf paper.txt`，输出共 1875 行。§3 L193 · §4 L289 · §5 Attention Dynamics L615 · §6 L715 · §7 L786
- **目前的判断**：
  - 只测了 instruct 模型加 prompt 诱导的 TTS，没测 thinking 模型。
  - “小模型收益最大”可能反映的是模型代际，不是方法本身。
  - “初始短窗口之后 attention 衰减”目前看可能只有 attention 相关性证据，要到 §5 核实是否有因果证据。
  - 相关工作里，VLM 专用的 TTS 方法（CCoT、DCoT、DDCoT）全是“先转写、再推理”。
- **带着走的问题**：

| Q | 问题 | 状态 |
|---|---|---|
| Q1 | contribution drop 是怎么测的 | 未回答 |
| Q2 | 短窗口是固定长度，还是随推理长度变化 | 未回答 |
| Q3 | 结论能否推广到 thinking 模型 | 未回答 |
| Q4 | 和“丢失焦点”（#43、#94）是不是同一机制 | 未回答 |
| Q5 | §6 有没有测“不给图、只给推理链”是否足够 | 未回答 |

### 2. Look Light, Think Heavy（ACL'26；中科院自动化所）[2606.22565](https://arxiv.org/abs/2606.22565)
- **方式**：精读。之前已通读，见 NOTES #94。
- **读到**：Abstract、Intro、Setup、§3 的 RQ1/RQ2 已读完。**下一节：§4 视觉/文本子探针**；之后是反思曲线、mosaic 遮挡、attention 分析。
- **目前的判断**：视觉 CoT 的收益集中在数学和逻辑类任务上。你的观点是：这些收益不是“视觉推理”，而是从图里拿到信息之后做的文本/逻辑推理。由此引出了“图像 token 是隐式 caption”的三个命题：
  - A（表征）：基本不成立，probe 能把信息恢复出来；
  - B（读出）：成立，因为监督只有文本；
  - C（访问）：图像只读一次，编码与问题无关。

---

## 二、精读完

### Are VLMs Seeing or Just Saying?（ICML'26 Oral；NOTES #43）[2605.15864](https://arxiv.org/abs/2605.15864)
- 全文和附录都已读完，包括附录 H 的 role × 内容交叉实验。
- **关键结论**：
  - 只加 structure token 几乎没效果，只加内容也几乎没效果；只有“turn 边界 + 有意义的 user 指令”才能把 attention 恢复回来。
  - 所以起作用的是“带来源归属的指令”（source-attributed instruction），和 #81、#82 是一条线。
  - 表 15 关键数字（Probe NL / 高 PPL 有意义 / 仅 system token / 多轮）：8B-T 36.6 / 43.3 / 33.3 / 67.5；235B-T 34.1 / 47.8 / 37.9 / 85.4。
- **缺口**：
  1. 长度和冲突证据混在一起，需要加一组无冲突的 filler 作对照；
  2. role × 祈使句的交叉：把 user 指令放进 assistant turn，把自述放进 user turn；
  3. 均匀放大 2 倍 attention 只恢复一半（36.6→54.8，多轮是 67.5），原因不明；
  4. 跨模态能不能推广。

---

## 三、通读完（NOTES.md #1–#100）

| # | 论文 | 会议 | arXiv |
|---|---|---|---|
| 1 | Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | NeurIPS'25 Oral | [2504.13837](https://arxiv.org/abs/2504.13837) |
| 2 | When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following | NeurIPS'25 Spotlight | [2505.11423](https://arxiv.org/abs/2505.11423) |
| 3 | Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs | NeurIPS'25 Spotlight | [2501.18585](https://arxiv.org/abs/2501.18585) |
| 4 | Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization | EMPO | [2504.05812](https://arxiv.org/abs/2504.05812) |
| 5 | Wait, Wait, Wait... Why Do Reasoning Models Loop? | ICML'26 Spotlight | [2512.12895](https://arxiv.org/abs/2512.12895) |
| 6 | Base Models Know How to Reason, Thinking Models Learn When | ICML'26 | [2510.07364](https://arxiv.org/abs/2510.07364) |
| 7 | Do We Need Adam? Surprisingly Strong and Sparse RL with SGD in LLMs | ICML'26 Spotlight | [2602.07729](https://arxiv.org/abs/2602.07729) |
| 8 | On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning LMs | ICML'26 Spotlight | [2512.07783](https://arxiv.org/abs/2512.07783) |
| 9 | RAGEN-2: Reasoning Collapse in Agentic RL | ICML'26 Spotlight | [2604.06268](https://arxiv.org/abs/2604.06268) |
| 10 | Quantized Reasoning Models Think They Need to Think Longer, but They Do Not | NeurIPS'26 Oral | [2606.00206](https://arxiv.org/abs/2606.00206) |
| 11 | The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness | NeurIPS'25 Spotlight | [2505.14617](https://arxiv.org/abs/2505.14617) |
| 12 | The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes | ICML'26 Spotlight | [2602.15515](https://arxiv.org/abs/2602.15515) |
| 13 | Effective Reasoning Chains Reduce Intrinsic Dimensionality | ICML'26 | [2602.09276](https://arxiv.org/abs/2602.09276) |
| 14 | The Tell-Tale Norm: ℓ2 Magnitude as a Signal for Reasoning Dynamics | ICML'26 Spotlight | [2606.06188](https://arxiv.org/abs/2606.06188) |
| 15 | Stop When Further Reasoning Won't Help: Attention-State Adaptive Generation (ASAG) | ICML'26 Spotlight | [2606.15070](https://arxiv.org/abs/2606.15070) |
| 16 | When to Trust the Cheap Check: Weak and Strong Verification for Reasoning | ICML'26 Spotlight | [2602.17633](https://arxiv.org/abs/2602.17633) |
| 17 | Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability | SOAR | [2601.18778](https://arxiv.org/abs/2601.18778) |
| 18 | Checklists Are Better Than Reward Models For Aligning Language Models | RLCF | [2507.18624](https://arxiv.org/abs/2507.18624) |
| 19 | KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction | NeurIPS'25 Oral | [2505.23416](https://arxiv.org/abs/2505.23416) |
| 20 | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference | NeurIPS'25 Oral | [2506.09501](https://arxiv.org/abs/2506.09501) |
| 21 | FFN Fusion: Rethinking Sequential Computation in Large Language Models | NeurIPS'25 Spotlight | [2503.18908](https://arxiv.org/abs/2503.18908) |
| 22 | TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup | NeurIPS'25 Spotlight | [2502.07864](https://arxiv.org/abs/2502.07864) |
| 23 | MoBA: Mixture of Block Attention for Long-Context LLMs | NeurIPS'25 Spotlight | [2502.13189](https://arxiv.org/abs/2502.13189) |
| 24 | SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training | NeurIPS'25 Spotlight | [2505.11594](https://arxiv.org/abs/2505.11594) |
| 25 | Hogwild! Inference: Parallel LLM Generation via Concurrent Attention | NeurIPS'25 Spotlight | [2504.06261](https://arxiv.org/abs/2504.06261) |
| 26 | SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications | NeurIPS'25 | [2411.04975](https://arxiv.org/abs/2411.04975) |
| 27 | Skip a Layer or Loop It? Learning Program-of-Layers in LLMs | ICML'26 | [2606.06574](https://arxiv.org/abs/2606.06574) |
| 28 | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models | ICML'26 | [2604.21106](https://arxiv.org/abs/2604.21106) |
| 29 | Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation | NeurIPS'25 Spotlight | [2506.09991](https://arxiv.org/abs/2506.09991) |
| 30 | ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models | ICML'26 | [2512.07843](https://arxiv.org/abs/2512.07843) |
| 31 | Predictable Scale Part II, Farseer: A Refined Scaling Law in LLMs | NeurIPS'25 Spotlight | [2506.10972](https://arxiv.org/abs/2506.10972) |
| 32 | EvoLM: In Search of Lost Language Model Training Dynamics | NeurIPS'25 Oral | [2506.16029](https://arxiv.org/abs/2506.16029) |
| 33 | The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition | NeurIPS'25 | [2505.17863](https://arxiv.org/abs/2505.17863) |
| 34 | Procedural Pretraining: Warming Up Language Models with Abstract Data | ICML'26 | [2601.21725](https://arxiv.org/abs/2601.21725) |
| 35 | TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior | ICML'26 | [2512.20757](https://arxiv.org/abs/2512.20757) |
| 36 | How much do language models memorize? | NeurIPS'25 Oral？ | [2505.24832](https://arxiv.org/abs/2505.24832) |
| 37 | Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment | ICML'26 | [2601.10160](https://arxiv.org/abs/2601.10160) |
| 38 | Efficient Numeracy in Language Models through Single-token Number Encodings | ICML'26 | [2510.06824](https://arxiv.org/abs/2510.06824) |
| 39 | The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models | ICML'26 | [2601.15165](https://arxiv.org/abs/2601.15165) |
| 40 | WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference | 腾讯微信 AI 技术报告 | [2512.22737](https://arxiv.org/abs/2512.22737) |
| 41 | Caption This, Reason That: VLMs Caught in the Middle | NeurIPS'25 | [2505.21538](https://arxiv.org/abs/2505.21538) |
| 42 | VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs | NeurIPS'25 | [2507.13361](https://arxiv.org/abs/2507.13361) |
| 43 | Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination | ICML'26 | [2605.15864](https://arxiv.org/abs/2605.15864) |
| 44 | Rewarding Perception for Multimodal Reasoning (MoCA) | ICML'26 | [2605.14054](https://arxiv.org/abs/2605.14054) |
| 45 | Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in Transformers | ICML'26 | [2601.20796](https://arxiv.org/abs/2601.20796) |
| 46 | Head Pursuit: Probing Attention Specialization in Multimodal Transformers | NeurIPS'25 Spotlight | [2510.21518](https://arxiv.org/abs/2510.21518) |
| 47 | Vision Transformers Don't Need Trained Registers | NeurIPS'25 Spotlight | [2506.08010](https://arxiv.org/abs/2506.08010) |
| 48 | Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought | NeurIPS'26 | [2602.23959](https://arxiv.org/abs/2602.23959) |
| 49 | To Think or Not To Think: A Study of Thinking in Rule-Based Visual Reinforcement Fine-Tuning | NeurIPS'25 | [2503.16188](https://arxiv.org/abs/2503.16188) |
| 50 | Real-Time Visual Attribution Streaming in Thinking Models (VStream) | ICML'26 | — |
| 51 | Lost in Transmission: When and Why LLMs Fail to Reason Globally | NeurIPS'25 Spotlight | [2505.08140](https://arxiv.org/abs/2505.08140) |
| 52 | AbsenceBench: Language Models Can't Tell What's Missing | NeurIPS'25 D&B | [2506.11440](https://arxiv.org/abs/2506.11440) |
| 53 | Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) | NeurIPS'25 D&B Oral/Best Paper | [2510.22954](https://arxiv.org/abs/2510.22954) |
| 54 | What are you sinking? A geometric approach on attention sink | NeurIPS'25 | [2508.02546](https://arxiv.org/abs/2508.02546) |
| 55 | Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations | NeurIPS'25 | [2506.19004](https://arxiv.org/abs/2506.19004) |
| 56 | The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models | Anthropic Fellows / Oxford 等，2026 | [2601.10387](https://arxiv.org/abs/2601.10387) |
| 57 | Equivalence of Context and Parameter Updates in Modern Transformer Blocks | ICML'26 | [2511.17864](https://arxiv.org/abs/2511.17864) |
| 58 | Evaluation Awareness in Language Models Has Limited Effect on Behaviour | NeurIPS'26 投稿 / 2026 | [2605.05835](https://arxiv.org/abs/2605.05835) |
| 59 | Poisoning Attacks on LLMs Require a Near-Constant Number of Poison Samples | UK AISI / Anthropic / Turing / Oxford / ETH，2025 | [2510.07192](https://arxiv.org/abs/2510.07192) |
| 60 | Do LLMs Signal When They're Right? Evidence from Neuron Agreement | ICLR'26 投稿 / 复旦 / 哈工大 | [2510.26277](https://arxiv.org/abs/2510.26277) |
| 61 | Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? | NeurIPS'25 Spotlight | [2508.17536](https://arxiv.org/abs/2508.17536) |
| 62 | Why Do Multi-Agent LLM Systems Fail? | NeurIPS'25 D&B | [2503.13657](https://arxiv.org/abs/2503.13657) |
| 63 | Measuring Agents in Production (MAP) | ICML'26 | [2512.04123](https://arxiv.org/abs/2512.04123) |
| 64 | ECHO: Terminal Agents Learn World Models for Free | Microsoft Research AI Frontiers，2026 | [2605.24517](https://arxiv.org/abs/2605.24517) |
| 65 | Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation | NeurIPS'25 D&B Spotlight | [2508.13144](https://arxiv.org/abs/2508.13144) |
| 66 | Inference-Time Reward Hacking in Large Language Models | NeurIPS'25 | [2506.19248](https://arxiv.org/abs/2506.19248) |
| 67 | What Makes a Reward Model a Good Teacher? An Optimization Perspective | NeurIPS'25 Spotlight | [2503.15477](https://arxiv.org/abs/2503.15477) |
| 68 | Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning | ICML'26 | [2603.03818](https://arxiv.org/abs/2603.03818) |
| 69 | Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates | ICML'26 | [2601.18510](https://arxiv.org/abs/2601.18510) |
| 70 | Conditional Memory via Scalable Lookup: A New Axis of Sparsity | ACL'26 | — |
| 71 | Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening | EMNLP'25 | — |
| 72 | Exploring Attention Attractors in Large Language Models | ACL'26 | — |
| 73 | Multimodal Language Models See Better When They Look Shallower | EMNLP'25 | — |
| 74 | Massive SFT Experiments Reveal How Data, Layer, and Training Factors Shape Alignment | EMNLP'25 | — |
| 75 | When Long Helps Short: How Context Length in SFT Affects LLM Behavior | EMNLP'25 | — |
| 76 | The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination | ACL'26 | — |
| 77 | Language of Thought Shapes Output Diversity in LLMs | ACL'26 | — |
| 78 | To Think or Not to Think: The Hidden Cost of Meta-Training with Excessive CoT Examples | ACL'26 | — |
| 79 | Why Do Some Inputs Break Low-Bit LLM Quantization? | EMNLP'25 | — |
| 80 | The Illusion of Specialization: Domain-Invariant "Standing Committee" in MoE | ACL'26 | — |
| 81 | Language Models Struggle to Use Representations Learned In-Context | ACL'26 | — |
| 82 | Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers | EMNLP'25 | — |
| 83 | LLMs Cannot Spot Math Errors, Even When Allowed to Peek into the Solution | EMNLP'25 short | — |
| 84 | When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges | ACL'26 | — |
| 85 | Do RAG Systems Really Suffer From Positional Bias? | EMNLP'25 | — |
| 86 | When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in VLMs | ACL'26 | — |
| 87 | How Do LVLMs See Text in Image? Unveiling the Distinctive Role of OCR Heads | EMNLP'25 | — |
| 88 | Why Multimodal In-Context Learning Lags Behind? Unveiling the Inner Mechanisms | ACL'26 | — |
| 89 | Defenses Against Prompt Attacks Learn Surface Heuristics | ACL'26 | — |
| 90 | Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy | ACL'26 | — |
| 91 | Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets | ACL'26 | — |
| 92 | Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries | EMNLP'25 | — |
| 93 | AgentOCR: Reimagining Agent History via Optical Self-Compression | ACL'26 | — |
| 94 | Look Light, Think Heavy: What Multimodal Chain-of-Thought Reasoning Can and Cannot Do | ACL'26 | — |
| 95 | Reasoning Gets Harder for LLMs Inside A Dialogue | ACL'26 | — |
| 96 | Steer Like the LLM: Activation Steering that Mimics Prompting | ICML'26 spotlight | — |
| 97 | Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models | NeurIPS'25 | — |
| 98 | What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces? | ICML'26 | — |
| 99 | Latent Collaboration in Multi-Agent Systems | ICML'26 | — |
| 100 | Language Model Circuits Are Sparse in the Neuron Basis | ICML'26 | — |
