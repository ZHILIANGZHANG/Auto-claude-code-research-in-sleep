# Round 2 Refinement

## Problem Anchor
（与 `PROBLEM_ANCHOR.md` 逐字一致，未改动）
- **Bottom-line problem**: Efficient-reasoning work reports *token reduction* as its efficiency gain, and decode accelerators (speculative decoding, KV-cache eviction) report their own gains on *uncompressed* reasoners. In deployment they are stacked. We do not know whether these gains compose, because all three may draw on the same resource — the redundancy (restatement) inside long chains of thought.
- **Must-solve bottleneck**: There is no controlled measurement of how length-penalty RL changes (i) the draftability of the remaining trace and (ii) the robustness of signal-free KV eviction. The leading explanation for why random eviction works on reasoning (Random Attention 2609.03430: "the model restates what it still needs") is, in its authors' words, *inferred, not measured*.
- **Non-goals**: proposing a new length-compression RL method, a new drafter architecture, or a new eviction policy; long-input workloads; MLLMs; training any model larger than a draft model.
- **Constraints**: ≤ 8 × A100-80GB / RTX 4090-class; open checkpoints only; main experiments ≤ ~2 GPU-weeks; inference-dominant; target NeurIPS/ICML/ICLR (realistic: main borderline or *ACL Findings / MLSys).
- **Success condition**: On ≥ 3 verbose→compressed parent/child pairs at 2 scales, state with CIs (a) how much of the token saving survives as wall-clock saving once SD is on, decomposed into intrinsic draftability loss vs drafter mismatch vs attention-cost savings; and (b) whether signal-free eviction stays free on compressed traces — with a *directly measured* restatement statistic that predicts both — or a clean refutation of the shared-substrate story.

## Anchor Check
全部是定义层面的修正。Claim 2 改写为 S_c/S_v < 1，与锚点中"能否叠加"一句对齐得更紧，没有偏题。

## Simplicity Check
- **删掉**不可识别的逐头解析并集公式，改用一个统计量 O_K（单次抽样下整条提及的联合存活）。
- 两个回归**共用**同一个 O_K。
- **删掉**未经验证的 QwQ-32B EAGLE 方案。
- 唯一的新增是**数据点**，没有新增机制：
  - 父模型的 16× 预算点；
  - Laser-D-7B 升入主网格，使 7B 满足 3 对。

## Changes Made
1. **(N1/N2)** O_K 改为 Π_m (1 − s_K(Δ)^{n_tok(m)})，即每次提及的所有值 token 在单次抽样下同时存活。删掉逐头并集公式。P1 改为**经验**预测。shared-draw 与 RA 附录 D 一致：所有层、所有 KV 头共用同一组随机位置（需在代码中确认）。
2. **(N3)** 以 harness 实测记录的生存曲线为主。解析式采用 RA 附录 D 自己的公式：每轮存活率 (K − ℓ_p) / (K + r − ℓ_p)（prompt 占预算），轮数 n = ⌊u/r⌋ − ⌊m/r⌋（如果代码有保护 buffer，再减 1）。每个预算点使用该点的 K。
3. **(N4)** G0 改为以 **A = τ − 1**（每步接受的 token 数）为准，阈值重新标定，并补上中间区间的处理分支。
4. **(N5)** 父模型增加 16× 预算点；子模型如果在 8× 时仍 ≥95%，再追加 16×。删失情况按预注册报告为下界。E 按题目做 bootstrap 给 CI。"full 精度"一律取 vLLM 结果，并在 100 题子集上验证与 HF full-KV 一致。
5. **(N6)** 内存组合公式重新标注（见 D4）。
6. **(N7)** 机制检验：模型内合并各预算点，加预算固定效应；协变量改为该模型**自己的** full-KV 逐题通过率，并减去重采样翻转率；两个回归共用同一个 O_K；τ 一侧只用 drafter 和 G。
7. **(N8)** Claim 2 写成 S_c/S_v = ρ_SD / ρ_noSD < 1。ρ_SD < 1 作为单独的、更强的部署结论。b > 1 时的分解是模型，在 b ∈ {1, 32, 128} 三处做端到端验证。
8. τ 汇总：每一步归入它**起始**位置所在的桶，按比值之和汇总（N_total / steps_total）。
9. EAGLE-3：删掉 QwQ 方案；可选为 7B 父模型训练 1 个 head（约 100 A100-h），用于测部署格"父 head → 子模型"。

---

# Revised Proposal (Round 2)

## Method Thesis
长度惩罚 RL 删掉的是推理 trace 里的复述，而复述是投机解码和无信号 KV 驱逐共同依赖的资源。所以三者的收益不能相乘叠加：交互因子 S_c/S_v < 1、E_c/E_v < 1。并且直接测得的复述统计量 O_K，在模型内部同时预测这两类损失。

## Contribution Focus
- **主贡献**：受控测量加机制解释。三件仪器：
  - τ（更新步进的 tokens/step，位置匹配）；
  - G（target 贪心可预测性）；
  - O_K（复述统计量，按提及的联合存活计算）。
- **辅贡献**：组合报告规范，包括 S_c/S_v、E_c/E_v，以及 ρ 的对数可加三项分解。
- **明确拒绝**：新 drafter、新驱逐策略、压缩感知 RL；必做实验中不训练 EAGLE；不做 MLLM、不做长输入。

## 定义

**(D1) τ：更新步进的 tokens/step**
- trace 在 T=0.6、top-p 0.95 下采样，最大长度 32K。
- 确定性 drafter D，块长 k=4（同时报告 k=8）。
- 从 t=0 开始：L = min(k, |草稿|, 草稿与 y_{t:} 的前缀匹配长度)，然后 t ← t + L + 1。
- 每一步归入它起始位置所在的桶（0–1K / 1–2K / 2–4K / 4–8K）。
- τ = Σ tokens / Σ steps，按比值之和汇总；k+1 个起始偏移取平均。
- **A = τ − 1** 即每步接受的 token 数，闸门以 A 为准。
- drafter 有三种：
  - n-gram / suffix：查找范围 prompt + 最近 2048 token，同时报告不加窗口的版本；
  - 1.5B 贪心 drafter，用于 7B target：主格为父 drafter，子 drafter 格为次要；
  - target 自身贪心，即 G。
- CI 按题目做 bootstrap。

**(D2) G** = 以 target argmax 为草稿时的 τ，并报告逐桶平均熵。G 衡量的是 target 侧的可预测性，不是任何 drafter 的上界。

**(D3) SD 交互因子**
- wall(m, D, b) = Σ_steps c_m(ctx, b, k)。c 来自实测代价网格，父子模型同架构，每个尺寸只测一次。
- ρ_SD = (wall_v / wall_c) / (T_v / T_c) = (τ_c / τ_v) · (c̄_v / c̄_c)
- τ_c / τ_v = (G_c / G_v) × [(τ_c / G_c) / (τ_v / G_v)]，即"target 侧"乘"drafter 残差"。
- ρ_noSD = c̄_v / c̄_c
- **S_c / S_v = ρ_SD / ρ_noSD**，这是组合论点的核心数字。
- 在 b=1 时这是精确恒等式；b ∈ {32, 128} 时是模型，用端到端子集验证（误差 ≤ 10%）。
- 另给 per-solved-problem 版本。

**(D4) 驱逐可达压缩倍数 E 与内存组合**
- E(m, π) = 在 MATH500 上保持 ≥ 95% full 精度的最大相对压缩倍数（log 插值）。
- 网格：父模型 2/4/8/16×；子模型 2/4/8×，如果 8× 时仍 ≥ 95% 再加 16×。删失时报告下界。
- E 按题目做 bootstrap 给 CI。
- 每条序列的峰值 KV ≈ 典型长度 / E。
- 总节省（相对父模型 full KV）= (T_v / T_c) · E_c = [(T_v / T_c) · E_v] · (E_c / E_v)。
- **交互因子 E_c / E_v**。
- **P1（经验预测）**：E_c(shared) / E_c(per-head) < E_v(shared) / E_v(per-head)（后者 ≈ 1，复现 RA）。
- **P2**：E_c(R-KV) / E_c(random) > E_v(R-KV) / E_v(random)。

**(D5) O_K：复述统计量**
- **事实**：非平凡数值（排除 0–10 的单个整数和常见常数）、规范化算式、赋值绑定。记录每次提及的值 token 数 n_tok（Qwen 把每个数字切成单独 token）。
- **复用事件** u，先前提及为 m_1…m_n。
- **单条目生存率** s_K(Δ)：以实测记录的曲线为主；解析式为 ((K − ℓ_p) / (K + r − ℓ_p))^{⌊u/r⌋ − ⌊m/r⌋}，在 recency buffer 内为 1。
- **O_K** = mean_u Π_m (1 − s_K(u − m)^{n_tok(m)})，即"所有先前提及都没有完整存活"的概率。
- 每个预算点使用该点的 K。
- 抽取质量：50 条 trace 上人工标注复用**事件**的精确率和召回率。

**(D6) 机制检验（模型内为主）**
- 每个模型内、逐 (题, 预算)：
  - logit P(随机驱逐答错 | full 答对) ~ O_K + log 长度 + 预算固定效应 + 该模型自身的 full 通过率；
  - 翻转率扣除重采样对照的基线翻转率。
- 同一个 O_K（4× 预算点）也用于逐 trace 回归：drafter τ / G ~ O_K + log 长度。
- 跨模型（8 个模型点）只作描述，并报告对压缩率的偏相关。
- 已知局限：O_K 与 τ 的联系在构造上弱于它与驱逐的联系。

**(D7) 预注册控制**
- 循环 / 截断过滤：一个 50-token 片段重复 ≥ 3 次，或触达 32K，即标记，所有指标报告含 / 不含两版。
- 位置桶只比较两个模型都覆盖到的题，并报告 N。
- both-correct 定义为 ≥ 3/4 次答对，并加重采样对照。
- AdaptThink 按 think / no-think 分层。

## 模型矩阵（主网格满足 2 个尺度 × 3 对）
| 尺度 | 父 | 子（主） | 附录 |
|---|---|---|---|
| 1.5B | R1-Distill-Qwen-1.5B | DLER-R1-1.5B、ThinkPrune-iter2k、LCR1_1.5B | AdaptThink-1.5B（分层） |
| 7B | R1-Distill-Qwen-7B | DLER-R1-7B、LCR1_7B、Laser-D-L4096-7B | AdaptThink-7B（分层） |
| 同权重剂量 | DeepScaleR-1.5B | L1-1.5B-Max，prompt 预算 {512, 1K, 2K, 4K} | |

## Claims
- **Claim 1（主）**：压缩吃掉了 SD 和随机驱逐共同依赖的冗余。
  - 证据：位置匹配下 A 和 G 下降；E_c/E_v < 1；P1、P2 成立；模型内 O_K 显著。
  - 反证路径：A 不降且 E 不降，则结论为"三者可以相乘叠加"。
- **Claim 2（辅）**：S_c / S_v = ρ_SD / ρ_noSD < 1，并给出三项分解。ρ_SD < 1 单独报告，作为更强的部署结论。

## 闸门 G0（约 6 GPU-h，第 1 周）
- **G0-SD**：R1-1.5B → ThinkPrune-iter2k，以及 DeepScaleR → L1-Max；MATH500 中 200 题 × 4 样本；离线计算 n-gram 和 G 的 A，按位置桶、加窗口；L1 扫 3 个 prompt 预算。
  - **Kill**：两个对在所有匹配桶里 A 的相对下降都 < 10%，**并且** L1 的长度扫描是平的。
  - **Promote**：两个对的 A 在匹配位置 + 固定窗口下都下降 ≥ 25%，**并且**同一权重下 A 随 L1 预算变短单调下降。
  - **中间区间**：
    - 只有不匹配口径下的下降显著 → 预注册转入"删段组合效应"分支：SD 一侧只主张"压缩删掉了高 τ 段"；
    - 只有一个对达标 → 先用 7B 的 DLER 对和 LC-R1 对做离线复核（约 2 GPU-h），再决定。
- **G0-Evict**：R1-1.5B vs DLER-1.5B，MATH500 × 2 样本，4× 相对预算，逐头随机 / shared-draw / R-KV。
  - **Promote**：子模型上 shared-draw 比逐头低 ≥ 3 分（CI 不含 0），同时父模型上差 ≤ 1 分；**或者**子模型上 R-KV − 随机 ≥ +3 分，同时父模型上 ≤ +1 分。
  - **Kill**：子模型上两个差距都 < 1.5 分，**并且** O_K(c) / O_K(v) ∈ [0.9, 1.1]（bootstrap CI 覆盖）。

## Compute（A100-80GB）
| 项目 | GPU-h |
|---|---|
| trace 生成（11 个模型，MATH500 × 4 + AIME × 8，32K 上限） | ~10 |
| 重采样对照 | ~10 |
| 离线 τ / G / drafter 前向 | ~5 |
| 代价网格 + 端到端验证（b = 1 / 32 / 128） | ~18 |
| 驱逐主网格（8 个模型 × {逐头, shared, R-KV}；MATH500 × 2，2/4/8×，父模型加 16×；AIME × 8 只跑 4×） | ~135 |
| HF 与 vLLM 的 full-KV 一致性检查（100 题子集） | ~10 |
| L1 剂量扫描 | ~10 |
| **must-run 合计** | **~198 → 加 30% 余量 ≈ 260（约 1.5 GPU-周，低于上限 336）** |
| nice-to-have | 7B 父模型 EAGLE-3 head（约 100）+ 评测（约 5）；KD drafter（约 20）；SnapKV / recency 附录（约 13）；绝对 K（约 20）；GPQA-D（约 30） |

- **工程量**：约 2–3 人周。
- **日历时间**：8 卡下，第 1 周出 G0，第 3–5 周出主结果。

## Named Risks
1. **效应量**：可能只是 10–20% 的修正，被评价为"只是测量"。
2. **熵锐化**：可能抬高 G，掩盖复述的损失。
3. **主 drafter 不是 EAGLE-3 / MTP**：部署结论打折。可选的父模型 EAGLE head 能部分缓解。
4. **vLLM draft-model 的词表行数不一致**（151,936 vs 152,064）：离线 τ 不受影响。
5. **RA harness 的 Qwen2 移植**（QKV bias）；shared-draw 是否跨层共享需在代码中确认。
6. **E 的置信区间宽**：可能有 ±10–20%。
7. **O_K–τ 的联系弱于 O_K–驱逐**（构造使然）。
8. **抢先风险**：Random Attention 作者的后续工作。
9. **只覆盖数值类事实**：GPQA 只放附录。
