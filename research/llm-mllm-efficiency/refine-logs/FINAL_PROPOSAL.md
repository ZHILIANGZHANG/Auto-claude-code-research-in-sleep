# Final Proposal: 压缩吃掉了冗余 — CoT 长度压缩与解码加速的隐性冲突

> research-refine 第 3 轮后定稿（2026-10-17）。评审分数 6.4 → 7.1 → 7.8（REVISE，已收敛）。评审建议停止打磨、直接跑 G0。
> 评审是同家族的 Claude 子代理，**不是** ARIS 要求的跨模型评审。
> 完整历史见 `REVIEW_SUMMARY.md` 和 `REFINEMENT_REPORT.md`。

## Problem Anchor
- **Bottom-line problem**: Efficient-reasoning work reports *token reduction* as its efficiency gain, and decode accelerators (speculative decoding, KV-cache eviction) report their own gains on *uncompressed* reasoners. In deployment they are stacked. We do not know whether these gains compose, because all three may draw on the same resource — the redundancy (restatement) inside long chains of thought.
- **Must-solve bottleneck**: There is no controlled measurement of how length-penalty RL changes (i) the draftability of the remaining trace and (ii) the robustness of signal-free KV eviction. The leading explanation for why random eviction works on reasoning (Random Attention 2609.03430: "the model restates what it still needs") is, in its authors' words, *inferred, not measured*.
- **Non-goals**: proposing a new length-compression RL method, a new drafter architecture, or a new eviction policy; long-input workloads; MLLMs; training any model larger than a draft model.
- **Constraints**: ≤ 8 × A100-80GB / RTX 4090-class; open checkpoints only; main experiments ≤ ~2 GPU-weeks; inference-dominant; target NeurIPS/ICML/ICLR (realistic: main borderline or *ACL Findings / MLSys).
- **Success condition**: On ≥ 3 verbose→compressed parent/child pairs at 2 scales, state with CIs (a) how much of the token saving survives as wall-clock saving once SD is on, decomposed into intrinsic draftability loss vs drafter mismatch vs attention-cost savings; and (b) whether signal-free eviction stays free on compressed traces — with a *directly measured* restatement statistic that predicts both — or a clean refutation of the shared-substrate story.

## Technical Gap
- **没人测过压缩前后的对照。** 长度压缩论文（ThinkPrune、L1、AdaptThink、LC-R1、DLER、Laser）只报 token 数和准确率。SD 和驱逐论文只测没压缩过的模型。三者叠加的效果没有人做过受控测量。
- **朴素比较有三个伪影：**
  - n-gram drafter 的接受长度天然随历史长度变好；
  - EAGLE 类方法的接受率随输出位置下降；
  - vLLM 的接受长度计数有已知问题。
- **wall-clock 还混入反向效应**：上下文变短，注意力本身就变便宜（Kinetics）。
- **我们的做法**：不发明新方法，只造测量仪器，把这些伪影逐一隔离，并把 RA 所说的"复述"直接测出来。

## Method Thesis
长度惩罚 RL 删掉的是推理 trace 里的复述，而复述是投机解码和无信号 KV 驱逐共同依赖的资源。因此三者的收益不能相乘叠加：交互因子 **S_c/S_v < 1** 且 **E_c/E_v < 1**。并且一个直接测得的复述统计量 **O_K** 在模型内部同时预测这两类损失。

## Contribution Focus
- **主贡献**：受控测量加机制解释。三件仪器：
  - τ / A：更新步进的位置匹配 tokens/step；
  - G：target 贪心可预测性；
  - O_K：复述统计量，按提及的联合存活计算。
- **辅贡献**：组合报告规范，包括 S_c/S_v、E_c/E_v，以及 ρ 的对数可加三项分解。
- **明确拒绝**：新 drafter、新驱逐策略、压缩感知 RL（结论里只提一句 future work）；MLLM；长输入。
- **frontier 原语**：没有新的 frontier 组件。研究对象本身就是 RL 训练出的推理模型。生产级 drafter（EAGLE-3）以"父模型 head 复用到子模型"的形式纳入，属于标准 draft model，不是新架构。

## 定义（测量协议）

**(D1) τ 与 A：更新步进的 tokens/step**
- **trace**：T=0.6、top-p 0.95、最大 32K。
- **步进规则**：确定性 drafter，块长 k=4（同时报告 k=8）。从 t=0 开始，L = min(k, |草稿|, 草稿与 y_{t:} 的前缀匹配长度)，然后 t ← t + L + 1。
- **为什么精确**：在 coupling 下，拒绝时的修正 token 或全部接受后的 bonus token 都等于 y_{t+L}，所以步数与真实投机解码在该样本上的步数完全一致。
- **汇总**：每一步归入其起始位置所在的桶（0–1K / 1–2K / 2–4K / 4–8K）；τ = Σ tokens / Σ steps；k+1 个起始偏移取平均。
- **A = τ − 1**，即每步接受的 token 数。
- **drafter 三种**：
  - n-gram / suffix：查找范围 prompt + 最近 2048 token，同时报告不加窗口的版本；
  - 1.5B 贪心 drafter（用于 7B target）：父 drafter 为主格，子 drafter 为次格；
  - target 自身贪心（G）。
- **EAGLE-3**：在 G0 判定 promote 之后，加入一个用 7B 父模型训练的 head，用它测父模型和每个 7B 子模型。EAGLE 草稿依赖隐状态，所以在 trace 上 teacher forcing 跑 target 和 head 一次即可得到确定性草稿。
- **CI**：按题目做 bootstrap。

**(D2) G** = 以 target argmax 为草稿时的 τ，同时报告逐桶熵。它衡量 target 侧的可预测性，不是任何 drafter 的上界。

**(D3) SD 交互因子**
- wall = Σ_steps c_m(ctx, b, k)，c 来自实测代价网格（父子同架构，每个尺寸测一次）。
- ρ_SD = (τ_c / τ_v) · (c̄_v / c̄_c)
- τ_c / τ_v = (G_c / G_v) × drafter 残差
- ρ_noSD = c̄_v / c̄_c
- **S_c / S_v = ρ_SD / ρ_noSD**。b=1 时精确；b ∈ {32, 128} 时是模型，用端到端子集验证（误差 ≤ 10%）。
- 无损 SD 不改变准确率，所以 per-solved 归一化在 S_c / S_v 中抵消。ρ 本身另报一个 per-solved 版本。

**(D4) E 与内存**
- E(m, π) = MATH500 上保持 ≥ 95% full 精度的最大相对压缩倍数。
  - 先对"保持率 vs log 压缩倍数"做 isotonic 拟合，再取第一个穿越点。
  - 网格：父模型 2/4/8/16×；子模型 2/4/8×，如果 8× 时仍 ≥ 95% 再加 16×。
  - 删失时报告下界。若 E_v 删失而 E_c 有限，E_c/E_v 是假设方向上的单侧上界。
  - CI：按题目做 bootstrap。
- **full 精度**：取 vLLM 结果。在 100 题 × 4 样本上与 HF full-KV 做等价性检验（TOST，|Δ| ≤ 1.5 分）；不通过则改用 HF full-KV。
- **峰值 KV 的总节省**（相对父模型 full KV）= (T_v / T_c) · E_c = [(T_v / T_c) · E_v] · (E_c / E_v)。时间积分的 KV 另行报告（∝ T_v²/2 vs T_c²/E_c）。
- **驱逐导致的长度反弹**作为一个报告量。

**(D5) O_K：复述统计量**
- **事实**：非平凡数值、规范化算式、赋值绑定，并记录每次提及的值 token 数 n_tok。
- **单条目存活率 s_K(Δ)**：以 harness 实测记录的曲线为主；解析校验式为 ((K − ℓ_p) / (K + r − ℓ_p))^{⌊u/r⌋ − ⌊m/r⌋}，若有保护 buffer 再减一轮。
- **O_K(题 i, K)** = 该模型 full-KV 样本上，复用事件 u 的 mean_u Π_m (1 − s_K(u − m)^{n_tok(m)}) 的均值。它是一个**题目级**预测量，在驱逐之前就能算出来。
- **抽取质量**：50 条 trace 上人工标注复用事件的精确率和召回率。

**(D6) 机制检验（主检验在模型内）**
- **主回归**：一个跨模型分层的逐样本 logistic 混合模型，以 **shared-draw 这一组**为主臂（它的存活模型与 O_K 完全一致）：
  - correct_{i,s} ~ policy × O_K(i, K) + log 长度 + 预算 FE + 模型 FE + 通过率_i + (1 | 题目)
  - policy 取 {重采样对照, shared-draw}，通过率来自独立的重采样样本。
  - **机制主张** = policy × O_K 交互项为负。
  - 逐头随机和 R-KV 作为次要臂。报告 O_K 与 log 长度之间的 VIF。
- **τ 一侧**：逐 trace 回归 drafter τ / G ~ O_K(4×) + log 长度 + 模型 FE。不使用 n-gram，避免循环论证。

**(D7) P1：存活模型判别（预注册，不作为 Claim 1 的证据）**
- 比较 E(shared) / E(per-head) 在父子模型之间的变化：
  - 子模型上 shared < per-head → 跨头并集确实有用，O_K 对逐头一组是保守的（损失上界）；
  - 两者相同 → 单次抽样的 O_K 就是正确的统计量。
- shared-draw 必须在**所有层、所有 KV 头**之间共用位置，需在 RA 代码中确认，否则自己实现。

**(D8) 预注册控制**
- 循环 / 截断过滤：一个 50-token 片段重复 ≥ 3 次，或触达 32K，即标记，所有指标报告含 / 不含两版。
- 位置桶只比较两个模型都覆盖到的题，并报告 N。
- both-correct 定义为 ≥ 3/4 次答对，并加重采样对照。
- AdaptThink 按 think / no-think 分层。

## 模型矩阵
| 尺度 | 父 | 子（主网格） | 附录 |
|---|---|---|---|
| 1.5B | R1-Distill-Qwen-1.5B | `nvidia/DLER-R1-1.5B-Research`、`Shiyu-Lab/DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k`、`zx10086/LCR1_1.5B` | `THU-KEG/AdaptThink-1.5B-delta0.05` |
| 7B | R1-Distill-Qwen-7B | `nvidia/DLER-R1-7B-Research`、`zx10086/LCR1_7B`、`hkust-nlp/Laser-D-L4096-7B` | `THU-KEG/AdaptThink-7B-delta0.05` |
| 同权重剂量 | DeepScaleR-1.5B-Preview | `l3lab/L1-Qwen-1.5B-Max`，prompt 预算 {512, 1K, 2K, 4K} | |

ID 来自检索摘录，下载前需复核。DLER 采用 NSCLv1 许可，仅限研究用途。

## Claims
- **Claim 1（主）**：压缩吃掉了 SD 和随机驱逐共同依赖的冗余。
  - 证据：
    - 位置匹配下 A 和 G 下降；
    - E_c / E_v < 1（shared 和逐头两组都看）；
    - P2：子模型上 R-KV 相对随机的优势上升；
    - 主回归中 policy × O_K 交互项显著为负。
  - 反证路径：A 不降且 E 不降，则结论为"三者可以相乘叠加"，同样成文。
- **Claim 2（辅）**：S_c / S_v = ρ_SD / ρ_noSD < 1，并给出三项分解。ρ_SD < 1 单独报告，作为更强的部署结论。

## 闸门 G0（约 6 GPU-h，第 1 周）
- **G0-SD**：R1-1.5B → ThinkPrune-iter2k，以及 DeepScaleR → L1-Max；MATH500 中 200 题 × 4 样本；离线计算。
  - **判定指标**：n-gram（窗口 2048）的 A，按覆盖受限的匹配桶做 token 加权汇总。G 同时报告；"每个桶方向一致"作为次要判据。
  - **Kill**：两个对的下降都 < 10%，**并且** L1 的长度扫描是平的。
  - **Promote**：两个对都下降 ≥ 25%，**并且** A 随 L1 预算变短单调下降 → 开始训练 EAGLE-3 head，进入完整网格。
  - **两个对都在 10–25%**：先用 7B 的 DLER 对和 LC-R1 对做离线复核（约 2 GPU-h），再按"实测的中等每 token 税"继续 SD 一侧的主张。
  - **只有不匹配口径下降显著**：转入"删段组合效应"分支。
  - **只有一个对达标**：先做 7B 离线复核，再决定。
- **G0-Evict**：R1-1.5B vs DLER-1.5B，MATH500 × 2 样本，4× 相对预算，逐头随机 / shared-draw / R-KV。
  - **Promote**：子模型上 E 或准确率的相对父模型差距 ≥ 3 分（CI 不含 0）；**或者**子模型上 R-KV − 随机 ≥ +3 分，同时父模型上 ≤ +1 分。
  - **Kill**：子模型上两个差距都 < 1.5 分，**并且** O_K(c) / O_K(v) ∈ [0.9, 1.1]。

## Compute（A100-80GB）
| 项目 | GPU-h |
|---|---|
| M0：RA harness 移植后复现 RA 的 Qwen3-4B MATH500 结果；walk-τ 与 vLLM 实测接受率对照 | ~4 |
| trace 生成（11 个模型） | ~10 |
| 重采样对照（只做 MATH500） | ~5 |
| 离线 τ / G / drafter / EAGLE 前向 | ~5 |
| 代价网格 + 端到端验证（b = 1 / 32 / 128） | ~18 |
| 驱逐主网格（8 个模型 × 3 臂；MATH500 × 2，2/4/8×，父模型加 16×；AIME × 4，4×） | ~117 |
| full-KV 等价性检验（100 题 × 4 样本） | ~15 |
| L1 剂量扫描 | ~10 |
| EAGLE-3 head 训练（7B 父模型，SpecForge，G0 promote 后才做） | ~105 |
| **合计** | **~289；加 15% 余量 ≈ 332（≤ 336）** |
| 不含 EAGLE | ~184；加 30% 余量 ≈ 240 |

- **工程量**：约 2–3 人周。
- **日历时间**：8 卡下，第 1 周出 G0，第 4–5 周出主结果。

## Named Risks
1. **效应量**：10–25% 是最可能的区间，存在被评价为"只是测量"的风险。G0 的中间分支专门处理这种情况。
2. **熵锐化**：可能抬高 G，掩盖复述的损失。
3. **EAGLE-3 只有父模型 head**：如果 G0 没有 promote、没训练 head，部署结论写作"仅对 n-gram / draft-model 类 SD 成立"。
4. **模型年代**：R1-Distill 时代的 checkpoint 可能被 2026 年的审稿人认为过时。Qwen3 系列没有公开的压缩子模型，只能如实说明。
5. **vLLM draft-model 的词表行数不一致**：离线 τ 不受影响。
6. **RA harness 的 Qwen2 移植**（QKV bias）；shared-draw 是否跨层共享需确认。
7. **E 的置信区间宽**（可能 ±10–20%）。
8. **O_K–τ 的联系弱于 O_K–驱逐**（构造使然）。
9. **抢先风险**：Random Attention 作者的后续工作。
10. **只覆盖数值类事实**：GPQA 只放附录。
