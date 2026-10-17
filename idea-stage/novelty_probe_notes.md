# Novelty probe（主代理，2026-10-17，WebSearch + arxiv.org/abs 核实）

## Idea 1 Why Compression Makes Models Talk Longer
- 2608.04771 ReCo (2026-08-05, work in progress)：明确写出"smaller cache leads the model to generate more tokens"，R-KV 25% 压缩在 MATH-500 上 R1-Distill-Qwen-7B 长度 +38.8%、R1-Distill-Llama-8B +79.0%；修复是 process-reward 估计器驱动的三组件（训练式）。
- 2510.01290 ThinKV（html 核实）："Unlike quantization, eviction does not cause an increase in generation length"；纯量化可使 R1-Llama-8B 长度膨胀最高 5.x×。
- 2510.08525 RLKV v3 (2026-05)：R-KV "often results in longer reasoning steps and poor performance"；并指出激进压缩下"长度减少"多因只解出简单题。
- → 结论：现象已被发表，但 **ThinKV（驱逐不拉长）与 ReCo/RLKV（驱逐拉长）直接矛盾**，机制无人解释，training-free 修复无人做。Idea 1 要重定位为"解决矛盾 + 机制 + TF 修复"，ReCo 作为训练式 baseline。Novelty 7→6。

## Idea 2 Steering Safety-Tax Audit + Refusal-Orthogonal Fix
- 2608.08383 (COLM 2026)："Safety Cost of Steering Vectors Is Separable and Reducible"——primal-dual 找单一方向消融，恢复安全。
- 2607.19806 OPIUM (ICML'26 MI workshop)：training-free 表示匹配消除 steering 外部性。
- 2605.01167：二阶矩加权的约束优化减 collateral damage。
- 2608.11227 Forecasting Side Effects：67 行为交叉矩阵；cos-sim 启发式与 refusal 效应无正相关。
- 2603.24543 Safety Pitfalls 本身已测过正交投影：|ΔASR| 只减 15–25%。
- → **修复部分被完全覆盖**，且我们提议的正交化已被证明只部分有效。Idea 2 作为方法 idea 淘汰；可保留的只剩"跨模态/跨应用（TTS、MLLM 擦除、CoT 压缩）安全税审计"，价值有限。移入 backlog 并降级。

## Idea 3 ConfMap
- 2609.32470（2026-09）：LRM 的 verbalized confidence prior 集中于少数高值并贯穿 RL（讨论的是 verbalized confidence，不是 token-logprob）。
- EACL'26 Findings "Think Just Enough"：置信控制在 RL 训练模型上可能更伤精度。
- PiCSAR 2508.21787、CGES 2511.02603。
- → 无人画"难度 × 模型 × 置信定义"相图；RLVR 轴有新证据支持。保留，Novelty 6，加 2609.32470 为相关工作。

## Idea 4 PruneDraft
- 2508.16201 SpecVLM (EMNLP'25 main)：training-free SD for Vid-LLM，两阶段剪枝 **draft 模型** 的视频 token（verifier 注意力引导 + 空间均匀），剪 90%，LLaVA-OV-72B 2.68×、Qwen2.5-VL-32B 2.11×。用的是独立小 Vid-LLM 作 draft。
- SpecPrune-VLA（openreview）：VLA 的 self-speculative pruning。
- → "剪枝后的自己当 drafter"（无独立 draft 模型）仍是 delta，但核心 insight（draft 对视频 token 剪枝不敏感）已被 SpecVLM 发表。Novelty 7→5。重定位：draft-model-free + 图像/短视频 + MMSpec 基准，必须与 SpecVLM 同台。

## Idea 5 DPE-for-Video
- 2409.12963 INTP-Video-LLMs (2024)：training-free 视频 token 重排 + LLM 上下文扩展，完全无训练地增加帧数。
- 2412.12735 GIRAFFE (2024)：M-RoPE++（需指令微调），指出直接外推不涨。
- 2602.14027 FLEX (2026)：视频**生成**的频率感知 RoPE 调制（低频插值高频外推）。
- → "逐维 / 频率感知地处理视频 M-RoPE"在理解侧仍无人做，但"training-free 让 Video-LLM 看更多帧"已有先例。Novelty 7→5–6。必须以 INTP-Video-LLMs 与 FLEX 为对照，卖点是"逐维有效长度估计 + 与 KV 压缩正交"。

## Idea 6 Reflect-or-Not
- 2506.11077 CyclicReflex (ICLR'26)：反思 token 过多/过少都降性能，三角波调度 logits（training-free）。
- 2608.25542 Reflection Steering (2026-08)：激活空间解耦反思与推理。
- 2602.12113 ARLCP (ICLR'26)：正确答案反思 token 更少；RL 惩罚。
- 2506.23840 "thinking trap"。
- → "难度 × 中间答案正确性"的 2×2 因果分解仍未见；但方向拥挤。Novelty 7→5–6。

## Idea 7 Leakage-Stratified CL
- 一次搜索无命中 2026 工作。保留，Novelty 7。

## Idea 8 KL-Razor for Vision
- 无视觉/合并系数命中；2605.26097 (2026) 说 replay 数据上约束 KL 比 NTP 更防遗忘（LLM）。保留，Novelty 7。

## Idea 9 One Stream, Two Communities
- RLAP-CLIP (ICLR'26) 原型 CIL；ARC 2405.14318 (ICLR'25) 是"CL 的 test-time training"（分类层单步梯度），应列为相关工作。无统一框架命中。保留，Novelty 6–7。

## Idea 10 Trajectory-Consensus Probe
- 2604.16529 (2026-04)：rollout 结构化摘要 + Recursive Tournament Voting（事后选择）。
- 2608.03222 FailFast (2026-08)：单轨迹早停（训练 0.6B 监视器）。
- tree-forking SWE-bench Pro（未核实 ID）。
- → 在线"跨轨迹共识早停"仍未见，但 2604.16529 的 rollout 摘要正是我们需要的"状态摘要"表示。Novelty 7→6，risk HIGH 不变。

## Idea 11 ReconVis
- 2608.24063 VisCache (2026-08)：training-free 视觉 KV 剪枝（关键帧过滤 + PruneKV），prompt-aware。
- StreamMem 2508.15717 query-agnostic（流式视频）。
- → "同图多问 prefix 复用"设置仍无人做。保留，Novelty 6。

## Idea 12 AR Speech-LM Steering + Attention-CD
- (a) 2609.33810 (SLT 2026, 2026-09-27)：**正是** AR TTS 的激活 steering 控语速，含 CosyVoice2，Seed-TTS-Eval。2603.05977 (SLT 2026) 口音中和 steering。EmoShift 2601.22873、CoCoEmo 2602.03420 情感 steering。
- (b) 2608.00722 ECCD (2026-08)：training-free 对比解码治 LM-TTS 幻觉，CosyVoice2/3、Llasa、GLM-TTS，WER −55.6%；并与 attention-constrained inference 比较。
- → **两条子线都已被覆盖**。淘汰 Idea 12。语音 TTS 的 TF 控制/去幻觉在 2026 下半年已不是空白（上一轮报告的"起步"判断过时）。
