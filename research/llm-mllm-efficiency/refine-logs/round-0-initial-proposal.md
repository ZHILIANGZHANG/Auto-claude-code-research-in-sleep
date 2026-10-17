# Research Proposal (Round 0): 压缩吃掉了冗余 — CoT 长度压缩与解码加速的隐性冲突

## Problem Anchor
（原样引用 `PROBLEM_ANCHOR.md`）
- **Bottom-line problem**: Efficient-reasoning work reports *token reduction* as its efficiency gain, and decode accelerators (speculative decoding, KV-cache eviction) report their own gains on *uncompressed* reasoners. In deployment they are stacked. We do not know whether these gains compose, because all three may draw on the same resource — the redundancy (restatement) inside long chains of thought.
- **Must-solve bottleneck**: There is no controlled measurement of how length-penalty RL changes (i) the draftability of the remaining trace and (ii) the robustness of signal-free KV eviction. The leading explanation for why random eviction works on reasoning (Random Attention 2609.03430: "the model restates what it still needs") is, in its authors' words, *inferred, not measured*.
- **Non-goals**: proposing a new length-compression RL method, a new drafter architecture, or a new eviction policy; long-input workloads; MLLMs; training any model larger than a draft model.
- **Constraints**: ≤ 8 × A100-80GB / RTX 4090-class; open checkpoints only; main experiments ≤ ~2 GPU-weeks; inference-dominant; target NeurIPS/ICML/ICLR (realistic: main borderline or *ACL Findings / MLSys).
- **Success condition**: On ≥ 3 verbose→compressed parent/child pairs at 2 scales, state with CIs (a) how much of the token saving survives as wall-clock saving once SD is on, decomposed into intrinsic draftability loss vs drafter mismatch vs attention-cost savings; and (b) whether signal-free eviction stays free on compressed traces — with a *directly measured* restatement statistic that predicts both — or a clean refutation of the shared-substrate story.

## Technical Gap
1. **当前评测的断点**：长度压缩论文（ThinkPrune、L1、AdaptThink、LC-R1、DLER、Laser）报的都是 token 数或准确率；投机解码和 KV 驱逐论文测的都是没压缩过的模型（Random Attention 用 Qwen3 / Phi-4-reasoning，R-KV 用 R1-Distill-Llama-8B / Qwen-14B，SD 论文用标准对话或推理模型）。三者叠加后的真实收益没人测过。
2. **为什么朴素做法不够**：
   - 直接比较"压缩前 / 后"的 vLLM 接受率会混入三个伪影：n-gram drafter 天然随历史长度变好；EAGLE 类方法的接受率随输出位置下降（LongSpec：超过约 2K token 后急剧下降）；vLLM 自带的接受长度统计在 EOS 处多计（issue #56101），对短输出偏差更大。
   - 只看 wall-clock 又混入"上下文变短、注意力本身变便宜"的反向效应（Kinetics）。
3. **最小充分干预**：不发明新方法，只造一套**测量仪器**，把冗余直接测出来，并把每一项伪影单独隔离。
4. **frontier 原语**：本研究的对象本身是 RL 训练出的推理模型。我们**不引入**任何新的 frontier 组件，也不需要。
5. **核心主张**：长度惩罚 RL 删掉的是"复述"，而复述正是投机解码和无信号驱逐共同依赖的底层资源。所以两类看似正交的加速在压缩模型上是**次乘法**（sub-multiplicative）的，并且一个直接测得的复述统计量能同时预测两边的损失。
6. **需要的证据**：在位置匹配下内在可投机性下降；随机驱逐的跨头冗余从"非必需"变成"必需"；复述统计量对两者都有预测力；wall-clock 分解。

## Method Thesis
**长度压缩删掉的是推理 trace 里的复述，而复述是投机解码和无信号 KV 驱逐共同依赖的资源，所以这三种效率技术并不能像各自论文暗示的那样相乘叠加。我们把复述直接测出来，并用它解释两边的损失。**

## Contribution Focus
- **主贡献（一个）**：一组受控测量，加上一个机制解释，用来回答"长度压缩是否吃掉了 SD 和随机驱逐的冗余"。测量仪器有两件：
  - **位置匹配的期望接受长度（EAL）**：包括无需 drafter 的 oracle-argmax 版本；
  - **复述统计量 O(K)**：被后续复用的"事实"中，上一次被提及发生在 K 个 token 之前的比例。它正是 Random Attention §5.3 所说的"只说一次、很久以后才用"的情形。
- **辅助贡献（至多一个）**：一个报告规范，"有效压缩比" ρ = wall-clock 比 / token 比，在 batch 1 和 serving batch 下分别给出，并分解为三项。
- **明确拒绝的复杂度**：不提出新的 drafter、新的驱逐策略或"压缩感知 RL"修复方法（结论里只作为 future work 提一句）；不训练 EAGLE-3（列为 nice-to-have）；不扩展到 MLLM 和长输入。

## Proposed Method（测量协议）

### Complexity Budget
- **冻结复用**：所有被测模型（公开 checkpoint）；vLLM（n-gram、suffix、draft-model 投机）；Random Attention 官方代码（HF harness + vLLM 插件）；R-KV 官方代码。
- **新写的东西**（都是脚本，不是模型）：
  1. 离线 EAL 计算器（teacher-forcing，用 target 的概率计算）；
  2. 事实抽取与复述统计；
  3. RA harness 的 Qwen2 前向移植，以及 shared-draw 变体（约十几行改动）；
  4. vLLM 计时脚本，接受长度自己从提交的 token 计算。
- **无可训练组件**（可选：一个 token 匹配 KD 的小 drafter，属于 nice-to-have）。

### System Overview（模型矩阵）
| 尺度 | 父模型（冗长） | 压缩子模型（公开 checkpoint） | 报告的长度变化 |
|---|---|---|---|
| 1.5B | DeepSeek-R1-Distill-Qwen-1.5B | `nvidia/DLER-R1-1.5B-Research`（−77%）、`Shiyu-Lab/DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k`（−65%）、`zx10086/LCR1_1.5B`（−52%）、`THU-KEG/AdaptThink-1.5B-delta0.05`（−53%，模式切换型，单独分层） | |
| 7B | DeepSeek-R1-Distill-Qwen-7B | `nvidia/DLER-R1-7B-Research`（−69%）、`zx10086/LCR1_7B`（−46%）、`hkust-nlp/Laser-D-L4096-7B`（−64%）、`THU-KEG/AdaptThink-7B-delta0.05`（−40%，模式切换型） | |
| 同权重剂量对照（附录） | DeepScaleR-1.5B → `l3lab/L1-Qwen-1.5B-Max`，用 prompt 预算 512 / 1K / 2K / 4K | 同一权重下改变长度 | |

所有 checkpoint ID 来自检索摘录，下载前需用 `huggingface-cli` 复核。

### Core Mechanism（三件测量仪器）

**(1) 位置匹配的期望接受长度 EAL**
- 在每个模型自己的采样 trace 上（T=0.6, top-p 0.95，与 RA 一致）离线计算，**不依赖 vLLM 的计数器**。
- **trace-coupled 精确估计（自查修正，原先的连乘公式需要草稿分支上的 target 条件概率，teacher forcing 给不出来）**：
  - 设 drafter 是确定性的，即草稿块只由前缀决定（n-gram 查找、贪心小 drafter、target argmax 都满足）。
  - 无损投机采样中，草稿 token d 以概率 p(d) 被接受，拒绝时从 p(·|≠d) 重采样。这与"从 p 采样 y，接受当且仅当 y=d"的 coupling 分布完全相同。
  - 因此，位置 t 的接受长度 = 草稿块 d_{1..k} 与实际 trace y_{t..t+k−1} 的**前缀匹配长度**。它在分布意义上精确、无偏。
  - 只有在前缀匹配时才需要下一位置的条件分布，而这恰好就是 teacher forcing 提供的。所以每个 drafter 只需在 trace 上做**一次前向**，n-gram 不需要前向。
  - 对每个位置 t 都发起一次草稿（k=4，同时报告 k=8），在位置桶内平均。
  - 另外报告逐位置的**首 token 接受率** α_t = 1[d_1 = y_t]，以及 i.i.d. 近似 EAL = (1 − α^{k+1}) / (1 − α)（Leviathan et al.），便于和文献对照。
- 三种 drafter：
  - **oracle-argmax**：草稿 = target 自己的 argmax。它给出"任何 drafter 能达到的内在上界"，与 drafter 是否偏移无关；
  - **n-gram / suffix**：在 prompt + 最近 W=2048 token 的窗口里查找（固定窗口），同时报告不加窗口的版本；
  - **小 drafter**：对 7B target 用 1.5B drafter，组成 2×2 = {父 1.5B, 同方法子 1.5B} × {父 7B, 同方法子 7B}，覆盖 DLER、LC-R1、AdaptThink 三种方法。无需训练就能把"drafter 偏移"与"内在可投机性"分开。
- **全部按绝对解码位置分桶**（0–1K, 1–2K, 2–4K, 4–8K）报告，主比较只在同一个桶内进行。

**(2) 复述统计量 O(K) 与 R**
- **事实抽取**：从 trace 中抽取数值字面量、规范化后的算式和赋值形式（如 `x = 3`、`AB = 5`），由 CPU 正则完成；数学任务上覆盖率较好，这一点列为局限。
- **复用事件**：一个事实在位置 u 被再次使用（出现在后续计算或最终答案中）。
- **O(K)** = 复用事件中，上一次提及距离超过 K token 的比例。按 RA 的机制，这正是随机驱逐会丢失、只有内容感知打分能救回来的情形。
- **R** = 每个被复用事实的平均提及次数。
- **注意**：O(K) 和 R 只用来预测 drafter EAL 和驱逐差距，**不**用来预测 n-gram EAL，避免评审指出的循环论证。

**(3) 驱逐协议（完全沿用 RA）**
- **策略**：full、逐头随机（RA）、shared-draw 随机（所有头共用随机位置）、R-KV、SnapKV + prompt 保护、recency + prompt。
- **预算**：主实验用每个模型自身典型 trace 长度的 4× 压缩（RA 口径）；在 MATH500 上扫 2× 和 8×；同时报告绝对 K ∈ {1024, 2048}。
- **指标**：准确率；**相同随机基线精度下**的"打分 − 随机"差距；驱逐引起的长度反弹（RA Table 7 显示驱逐会把输出拉长）。

### Training Plan
无。可选的 nice-to-have：用 token 匹配的 target trace 对 1.5B drafter 做 KD（每个约 5–10 GPU-h），或用 SpecForge 训练 EAGLE-3 head（每个约 48 H200-h ≈ 100 A100-h）。

### Failure Modes and Diagnostics
| 失败模式 | 诊断 / 应对 |
|---|---|
| 位置匹配后效应消失 | 这本身就是结论（"表观的可投机性税是长度伪影"），按预注册转为方法论 note |
| 效应全部来自 drafter 偏移 | 由 2×2 drafter 对照与 oracle-argmax 判定；如果 oracle-argmax 不降而 drafter 降，结论是"重训 drafter 即可" |
| AdaptThink / Thinkless 的模式切换混入 | 按 think / no-think 分层；不进入汇总回归 |
| both-correct 筛选导致向均值回归 | 定义为 ≥3/4 次答对；另加一组不做扰动的重采样对照 |
| 事实抽取覆盖率低 | 报告抽取覆盖率；在 50 条 trace 上人工标注，验证抽取的精度和召回 |
| RA harness 不支持 Qwen2 | Qwen2 与 Qwen3 的注意力只差 qk-norm，移植约 1 天；退路是在 RA 的 vLLM 插件上只跑随机类策略 |

### Novelty and Elegance Argument
没有新模块。新颖性在**发现**和**仪器**：
- 第一次在同父模型的压缩前后对照上测量 SD 可投机性和驱逐鲁棒性；
- 第一次直接测量 RA 所说的"复述"，并检验它是否对两类加速都有预测力。

最近的工作及区别：
- Random Attention：只有冗长模型，冗余是推断出来的；
- DREAM-R：长度奖励加在 drafter 上，而且是步级投机；
- ReCo：方法论文，把长度与 KV 联合优化；
- SD Performance-or-Illusion：标准模型。

## Claim-Driven Validation Sketch

### Claim 1（主）：压缩吃掉了两类加速共同依赖的冗余
- **实验**：在 8 个子模型 + 2 个父模型上测量：
  - 位置匹配的 oracle-argmax EAL 和 drafter EAL；
  - 驱逐中的 shared-draw 与逐头随机差距、R-KV 与随机差距；
  - O(K)、R。
- **基线 / 对照**：父模型；位置匹配；2×2 drafter；重采样对照。
- **决定性指标**：
  - 同一位置桶内 EAL 的相对变化；
  - 压缩模型上 shared-draw 的掉分（P1）；
  - R-KV 相对随机的差距（P2）；
  - O(K) 对驱逐差距和 drafter EAL 的跨模型 Spearman ρ（≥10 个模型点）。
- **预期方向**：EAL↓、shared-draw 掉分↑、R-KV 差距转正、O(K)↑；ρ 显著。

### Claim 2（辅）：token 节省高估了 SD 下的 wall-clock 节省
- **实验**：vLLM 下 {无 SD, n-gram, suffix, 小 drafter} × batch {1, 32, 128}，每题 wall-clock，计算 ρ = (wall_v / wall_c) / (tok_v / tok_c)。
- **分解**：每 token 代价随上下文增长（实测拟合）× 接受长度。
- **预期**：开 SD 时 ρ < 1；不开 SD 时 ρ 可能 > 1（反向效应），两者都报告。

## Experiment Handoff Inputs
- 必须证明：Claim 1 中位置匹配 EAL 下降（或可信的不下降）；驱逐的 P1、P2；O(K) 的预测力；Claim 2 的实测 ρ。
- 必须做的消融：位置 / 窗口匹配 vs 不匹配；2×2 drafter；shared-draw vs 逐头随机；重采样对照；AdaptThink 分层。
- 关键数据集：MATH500（×4 样本）、AIME24+25（×8 样本）。GPQA-D 放附录（检验数值抽取之外的泛化）。
- 最高风险假设：效应量。即位置匹配后 EAL 的下降是否 ≥15%，压缩模型上 shared-draw 是否明显掉分。

## Compute & Timeline Estimate（粗估，A100-80GB）
- trace 生成（vLLM，10 个模型）：约 8 GPU-h
- 离线 EAL / drafter 前向：约 5 GPU-h
- vLLM 计时：约 25 GPU-h
- 驱逐网格（HF + RA 的 vLLM 插件）：约 80 GPU-h
- 重采样对照：约 10 GPU-h
- **must-run 合计约 130 GPU-h**，留 25% 余量约 160 GPU-h（约 1 GPU-周）
- nice-to-have：EAGLE-3 head × 2 约 200 GPU-h；KD drafter 约 20 GPU-h；GPQA / LiveCodeBench 扩展约 30 GPU-h
- **工程量**：约 2–3 人周（EAL 计算器 2 天、事实抽取 2 天、RA 移植 + shared-draw 2–3 天、计时脚本 2 天、分析 3–5 天）
- **日历时间**：8 卡下约 3–4 周可以出完整结果；闸门 G0 约 4 GPU-h，1 周内可完成
