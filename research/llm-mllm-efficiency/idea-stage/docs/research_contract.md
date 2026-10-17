# Research Contract: 压缩吃掉冗余 — CoT 长度压缩对投机解码与 KV 驱逐的隐性代价（C1+C5）

> 当前选定 idea 的工作契约（W1 → W1.5 交接用）。session 恢复时读这个文件，不用读完整的 IDEA_REPORT.md。

## Selected Idea
- **Description**: 长度惩罚 RL（ThinkPrune、L1、AdaptThink、LC-R1）缩短了推理 trace。我们检验它删掉的是不是恰好是投机解码（SD）和随机 KV 驱逐赖以生效的冗余文本，从而导致 (a) SD 接受长度 τ 下降，"token 节省"高估 wall-clock 节省；(b) 随机驱逐（Random Attention 2609.03430）在压缩后的 trace 上不再与打分驱逐打平。
- **Source**: `idea-stage/IDEA_REPORT.md`，Idea 1（C1 + C5 合并）
- **Selection rationale**: triage 第 1（N8/I9/F9）；新颖性 7/10 PROCEED；独立评审 PROCEED（5→6/10）；只用公开 checkpoint 做推理；第一个闸门实验约 1.5 GPU-h；正反结果都能发表。

## Core Claims（每条都是方法的验收标准，以实验结果为准）
1. 在相同解码位置、固定查找窗口下，长度惩罚 RL 会降低 n-gram 和（token 匹配重蒸馏后的）EAGLE-3 的期望接受长度。
2. 在 7B、batch 1 / 32 / 128 下，实测"有效压缩比"（wall-clock 比 / token 比）在开启 SD 时差于关闭 SD 时。必须同时考虑 KV 读取代价降低带来的反向效应（Kinetics）。
3. 在相同随机基线精度下，压缩模型上"随机 vs R-KV / SnapKV"的驱逐差距扩大，且在 ≥6 个模型点上与 EAGLE / drafter τ 的损失相关（不用 n-gram τ，避免循环论证）。

### C5 的预注册预测（全文核对 Random Attention 2609.03430v2 之后加入）
RA 只测了 Qwen3-4B/14B/32B 和 Phi-4-reasoning，并在附录 H 中写明"text-level redundancy is inferred, not measured"。shared draw 对照（所有头共用随机位置）只在 Qwen3-4B MATH500 上做过：0.871 vs 逐头 0.874（K=1024）。
- **P1**：冗长模型上 shared draw ≈ 逐头随机；压缩模型上 shared draw 显著低于逐头随机（跨头冗余从"非必需"变成"必需"）。
- **P2**：压缩模型上 R-KV 相对 Random Attention 的差距转为正值，且与"只说一次的事实比例"相关（RA Table 3：只说一次的口令 R-KV 0.836 vs RA 0.000）。
- **P3**：直接测量文本冗余，即后来被用到的中间值的复述次数。
- **附加测量**：驱逐导致的长度反弹（RA Table 7：Qwen3-4B MATH500 full 5.3K → RA 7.1K token）在压缩模型上是否更大。
- **预算口径**：沿用 RA 的设定（K 约为典型 trace 的 4× 压缩、保护整个 prompt），压缩模型按各自的典型长度设定相对 K，同时报告绝对 K。
- **代码**：SalesforceAIResearch/Random-Attention（VaSE HF 引擎，以及 TriAttention 的 vLLM 插件）。

## Method Summary
选同一父模型的冗长 → 压缩配对：R1-Distill-Qwen-1.5B → `…-thinkprune-iter2k`；DeepScaleR-1.5B → L1-1.5B-Max；R1-Distill-Qwen-7B → LC-R1-7B / AdaptThink-7B（按 think / no-think 分层）。在 MATH-500 / AIME 上以 T=0.6 采样 trace，一次前向拿到 target 的 log-prob，离线计算期望接受长度：prompt-lookup / suffix 和 R1-Distill-1.5B drafter，按绝对位置分桶 + 固定 2K 窗口 + 不匹配三种口径。L1 在同一权重下扫多个目标长度，作为模型内对照。

drafter 方面：按 token 数匹配，在各模型自己的 trace 上训练 EAGLE-3 head，做 d_v / d_c × m_v / m_c 的交叉评测，把"分布偏移"和"可预测 token 本身被删"两个原因分开。在 vLLM 中实测 wall-clock。驱逐方面：用 Random Attention 官方代码（逐头随机、保护 prompt），与 R-KV、SnapKV 对比，相对预算 + 绝对预算都做。

## Experiment Design
- **Datasets**: MATH-500（500）、AIME24/25（60），每题 4 个样本
- **Baselines**: SD 不开 / n-gram / 小 drafter / EAGLE-3；驱逐用 full KV / 逐头随机 / R-KV / SnapKV
- **Metrics**: 期望接受长度 τ（三种口径）、wall-clock / solved problem、有效压缩比、驱逐差距（在相同随机基线精度下）、n-gram 复述率
- **Controls**: both-correct 定义为 ≥3/4 次答对 + 不扰动的重采样对照；bootstrap 置信区间；预注册"每 token 税"与"删掉高 τ 段"两种结论
- **Compute budget**: 约 300 GPU-h（P0–P6，见 IDEA_REPORT 的 External Critical Review）；P0 闸门约 1.5 GPU-h

## Gate（P0，判别实验）
- **Kill**：两个 1.5B 模型对在所有匹配桶里的 τ 下降都 <10%，且不匹配口径下的下降在匹配后消失，且 L1 的长度扫描是平的。
- **Promote**：匹配位置 + 固定窗口下都下降 ≥15%，且同一权重下 τ 随 L1 目标长度变短而下降。

## Current Results
（还没有。环境无 GPU，pilot 跳过。）

## Key Decisions
- 主指标不是不匹配口径的 n-gram τ（有历史长度偏差），而是位置匹配 + 固定窗口的 τ。
- wall-clock 和 EAGLE 的结论放在 7B 上做；1.5B 只用来跑闸门。
- C5 并入本文作为第二维度，不单独成文。
- 已知风险：位置混杂；"填充 token 是低熵的"这一假设可能不成立；结果可能被评价为"只是测量"；Random Attention 作者的后续工作可能抢先。

## Status
- [x] Idea selected
- [x] 关键先例全文核对（Random Attention：确认没有测过压缩模型）
- [ ] P0 判别实验
- [ ] C5 闸门：一个冗长 / 压缩模型对上，逐头随机 vs shared draw vs R-KV（约 3 GPU-h）
- [ ] Baseline reproduced（Random Attention 代码，SD 基线）
- [ ] 7B trace + 离线 τ
- [ ] EAGLE-3 token 匹配重蒸馏
- [ ] wall-clock + 驱逐
- [ ] Paper draft
