# Round 1 Refinement

## Problem Anchor
（与 `PROBLEM_ANCHOR.md` 逐字一致，未改动）
- **Bottom-line problem**: Efficient-reasoning work reports *token reduction* as its efficiency gain, and decode accelerators (speculative decoding, KV-cache eviction) report their own gains on *uncompressed* reasoners. In deployment they are stacked. We do not know whether these gains compose, because all three may draw on the same resource — the redundancy (restatement) inside long chains of thought.
- **Must-solve bottleneck**: There is no controlled measurement of how length-penalty RL changes (i) the draftability of the remaining trace and (ii) the robustness of signal-free KV eviction. The leading explanation for why random eviction works on reasoning (Random Attention 2609.03430: "the model restates what it still needs") is, in its authors' words, *inferred, not measured*.
- **Non-goals**: proposing a new length-compression RL method, a new drafter architecture, or a new eviction policy; long-input workloads; MLLMs; training any model larger than a draft model.
- **Constraints**: ≤ 8 × A100-80GB / RTX 4090-class; open checkpoints only; main experiments ≤ ~2 GPU-weeks; inference-dominant; target NeurIPS/ICML/ICLR (realistic: main borderline or *ACL Findings / MLSys).
- **Success condition**: On ≥ 3 verbose→compressed parent/child pairs at 2 scales, state with CIs (a) how much of the token saving survives as wall-clock saving once SD is on, decomposed into intrinsic draftability loss vs drafter mismatch vs attention-cost savings; and (b) whether signal-free eviction stays free on compressed traces — with a *directly measured* restatement statistic that predicts both — or a clean refutation of the shared-substrate story.

## Anchor Check
所有修改都是**测量定义层面**的修正，问题本身没有变。把驱逐收益改写为"可达压缩倍数"，是让锚点里"这些收益能否叠加"这句话有了统一的计量单位，不是换题。

## Simplicity Check
- 把 O(K) 和 R **合并**为一个统计量 O_K。
- 驱逐策略从 6 个**砍到** 4 个主策略。
- **删掉**全量 batch-1 计时，改为"代价网格 × walk-τ"的恒等分解。
- 2×2 drafter 的子模型格**降级**为次要。
- 没有新增任何模块或机制。

## Changes Made
1. **主指标**：逐位置算术平均 EAL → **更新步进（renewal walk）的 tokens/step**。这修正了 CRITICAL 偏差。
2. **EAL 定义**：EAL := E[L] + 1，与 Leviathan 和 vLLM 口径一致。逐位置 α_t 只保留为诊断指标。
3. **oracle-argmax** 改名为 **target 贪心可预测性 G**，删掉"上界"的说法，并报告逐桶熵（熵降低方向上的效应与假设相反，正文明说）。
4. **O(K)** → 生存核统计量 **O_K**，按各模型自身的相对预算计算，并给出逐头和 shared-draw 两个版本，使 P1 成为解析预测。
5. **2×2 drafter**：主部署数字改为 d_v→m_c vs d_v→m_v；子模型格降为次要；"重训 drafter 即可"这一结论只在跑了 KD drafter 时才主张。
6. **统一计量单位**：S（SD 加速比）和 E（驱逐可达压缩倍数）。论点写成两个交互因子 S_c/S_v 和 E_c/E_v 都 < 1。
7. **机制检验**：主检验改为**模型内、逐题**、控制长度；跨模型相关只作描述，并给出控制压缩率的偏相关。
8. **新增控制**：循环和截断 trace 的预注册过滤；位置桶只比较两边都覆盖到的题，并报告 N。
9. **ρ 写成精确恒等式**，以对数可加的三项分解给出，并增加 per-solved-problem 版本。
10. **算力重估并裁剪网格**，闸门 G0 写成明确条文。

---

# Revised Proposal: 压缩吃掉了冗余 — CoT 长度压缩与解码加速的隐性冲突

## Method Thesis
长度惩罚 RL 删掉的是推理 trace 里的复述，而复述正是投机解码和无信号 KV 驱逐共同依赖的资源。所以压缩、SD、驱逐三者的收益不能相乘叠加：两个交互因子 S_c/S_v 和 E_c/E_v 都小于 1，并且一个直接测得的复述统计量 O_K 在模型内部能同时预测两边的损失。

## Contribution Focus
- **主贡献**：受控测量加机制解释。它由三件测量仪器组成，每一件都对应一个已知伪影的修正：
  1. 更新步进的位置匹配 tokens/step；
  2. target 贪心可预测性 G，与 drafter 无关；
  3. 生存核复述统计量 O_K。
- **辅助贡献**：效率报告的组合规范，包括交互因子 S_c/S_v、E_c/E_v，以及 ρ 的三项分解。
- **明确拒绝**：新 drafter、新驱逐策略、"压缩感知 RL"修复方法（只在讨论中提一句）；训练 EAGLE-3 不列为 must-run；不做 MLLM、不做长输入。

## 记号与定义

**模型对**：父模型 v，子模型 c（同父）。trace 在 T=0.6、top-p=0.95 下采样，最大长度 32K。

**(D1) 更新步进的 tokens/step τ**（主指标）
- 对确定性 drafter D 和块长 k：从 t=0 开始，令 L = min(k, |草稿|, 草稿与 y_{t:} 的前缀匹配长度)，然后 t ← t + L + 1，计一步。
- 在 coupling 下，拒绝时的修正 token 或全部接受后的 bonus token 都等于 y_{t+L}，所以步数与真实 SD 在该样本上的步数完全一致。
- 定义 τ = N_tokens / N_steps，按位置桶统计；对 k+1 个起始偏移取平均。
- 无提案的步计 1 token。drafter 包括：
  - n-gram / suffix：查找范围限定为 prompt + 最近 W=2048 token，同时报告不加窗口的版本；
  - 小 drafter（7B target 用 1.5B drafter），贪心，在 trace 上 teacher forcing 一次；
  - target 自身贪心，即 G。
- CI：按题目做 bootstrap。

**(D2) target 贪心可预测性 G**
- 定义为以 target argmax 为草稿时的 τ（上面第三种 drafter）。
- 同时报告逐桶平均熵 H̄。
- 解释：G 是"target 侧"的可预测性，不是任何 drafter 的上界。

**(D3) 交互因子 S**
- S(m, D, b) = 吞吐(有 SD) / 吞吐(无 SD)，在 batch b 下。
- 组合公式：wall(m) = Σ_steps c_m(ctx, b, k)；c 来自**实测代价网格**。父子模型同架构，所以每个尺寸只需测一次。
- 恒等式：ρ_SD = (wall_v / wall_c) / (T_v / T_c) = (τ_c / τ_v) · (c̄_v / c̄_c)
  - τ_c / τ_v = (G_c / G_v) · [(τ_c / G_c) / (τ_v / G_v)]，即"target 侧 × drafter 残差"；
  - c̄_v / c̄_c 是注意力代价项（Kinetics 反向效应）。
  - 三项对数可加。
- 无 SD 时 ρ_noSD = c̄_v / c̄_c，可能 > 1，同样报告。
- 另给 per-solved-problem 版本：wall 除以答对的题数。

**(D4) 驱逐可达压缩倍数 E**
- E(m, π) = 策略 π 在 MATH500 上保持 ≥ 95% full 精度时，可达到的最大相对压缩倍数。
- 在 2× / 4× / 8× 三点上，用 log 压缩倍数做线性插值。
- 交互因子 E_c / E_v。内存侧的组合关系：KV 占用 ∝ T / E，所以压缩 + 驱逐的总节省 = (T_v / T_c) · (E_c / E_v)。
- **P1**：E_c(shared) / E_c(per-head) < E_v(shared) / E_v(per-head) ≈ 1。
- **P2**：E_c(R-KV) / E_c(random) > E_v(R-KV) / E_v(random)。

**(D5) 生存核复述统计量 O_K**
- **事实**：非平凡数值字面量（排除 0–10 的单个整数和常见常数）、规范化算式、赋值绑定（如 `x = 3`、`AB = 5`）。
- **复用事件** u：事实在后续计算或最终答案中出现，且之前有提及 m_1 … m_n。
- **随机策略的单条目生存函数**：RA 每 r=64 步驱逐一次，每个候选每轮以 K/(K+r) 的概率存活，所以 s_K(Δ) ≈ (K/(K+r))^{⌊(Δ−r)/r⌋}；Δ ≤ r 时（recency buffer）s=1。RA 附录 D 给出 K=1024 时 0.94^n。我们同时从 harness 记录经验曲线做校验。
- **逐头版本**：一次提及在 H 个 KV 头上至少有一份副本存活的概率为 1 − (1 − s)^H。shared-draw 版本就是 s 本身。
- **O_K** = 复用事件上"所有先前提及都没有存活"的概率的均值 = mean_u Π_m (1 − s̃_K(u − m))。
- 按各模型**自身的相对 K** 计算。
- 抽取质量：在 50 条 trace 上人工标注**复用事件**的精确率和召回率。

**(D6) 机制检验（主检验在模型内）**
- 每个模型内、逐题，做以下两个回归，都控制 log trace 长度和题目难度（父模型的通过率）：
  - 随机驱逐相对 full 的失败指示 ~ O_K(题)；
  - 逐 trace 的 drafter τ ~ 复述统计量。
- 跨模型（≥ 7–8 点）只作描述，报告对压缩率的偏相关。

**(D7) 预注册控制**
- **循环过滤**：一个 50-token 片段重复 ≥ 3 次，或触达 32K 上限，即标记该 trace。所有指标都报告"含 / 不含"标记 trace 两个版本。
- **位置桶覆盖**：只比较两个模型都覆盖到该桶的题，并报告 N。
- **both-correct**：定义为 ≥ 3/4 次答对；另加一组不做扰动的重采样对照。
- **AdaptThink**：按 think / no-think 分层，不进入汇总回归。

## 模型矩阵
| 尺度 | 父 | 子（主） | 子（附录） |
|---|---|---|---|
| 1.5B | R1-Distill-Qwen-1.5B | DLER-R1-1.5B（−77%）、ThinkPrune-iter2k（−65%）、LCR1_1.5B（−52%） | AdaptThink-1.5B（分层） |
| 7B | R1-Distill-Qwen-7B | DLER-R1-7B（−69%）、LCR1_7B（−46%） | Laser-D-L4096-7B、AdaptThink-7B（分层） |
| 同权重剂量 | DeepScaleR-1.5B | L1-1.5B-Max，prompt 预算 {512, 1K, 2K, 4K} | — |

## Claim-Driven Validation Sketch
- **Claim 1（主）**：压缩吃掉了 SD 和随机驱逐共同依赖的冗余。
  - 证据：在位置匹配、固定窗口下 τ 和 G 下降；E_c / E_v < 1；P1、P2 成立；模型内 O_K 有预测力。
  - 反证路径：如果位置匹配后 τ 不降，同时 E 不降，那么结论就是"三者可以相乘叠加"，同样成文。
- **Claim 2（辅）**：token 节省高估了 SD 下的 wall-clock 节省。
  - 证据：开 SD 时 ρ_SD < 1，并给出三项分解；代价网格的预测在小子集上与端到端实测吻合（误差 ≤ 10%）。

## 闸门 G0（约 6 GPU-h，第一周）
- **G0-SD**：两个 1.5B 对（R1-1.5B → ThinkPrune-iter2k；DeepScaleR → L1-Max），MATH500 中 200 题 × 4 样本；离线 n-gram 与 G 的 walk-τ，分位置桶并加窗口；L1 扫 3 个 prompt 预算。
  - **Kill**：两个对在所有匹配桶里的下降都 < 10%，**并且** L1 的长度扫描是平的。
  - **Promote**：两个对的下降都 ≥ 15%，**并且**同一权重下 τ 随 L1 预算变短而下降。
- **G0-Evict**：R1-1.5B vs DLER-1.5B，MATH500 × 2 样本，4× 相对预算，比较逐头随机、shared-draw、R-KV。
  - **Promote**：子模型上 shared-draw 比逐头随机低 ≥ 3 分（CI 不含 0），同时父模型上差 ≤ 1 分（复现 RA）；**或者**子模型上 R-KV − 随机 ≥ +3 分，同时父模型上 ≤ +1 分。
  - **Kill**：子模型上两个差距都 < 1.5 分，**并且** O_K(c) ≈ O_K(v)。

## Compute（A100-80GB，重估）
| 项目 | GPU-h | 依据 |
|---|---|---|
| trace 生成（10 个模型，MATH500 × 4 + AIME × 8，32K 上限） | ~10 | 约 9700 万 token；1.5B 约 6K tok/s，7B 约 2.5K tok/s；长尾 ×1.5 |
| 重采样对照（再做一遍 full 生成） | ~10 | 同上 |
| 离线 τ / G / drafter 前向 | ~5 | prefill 吞吐 15–50K tok/s；只存 argmax 和 p(y) |
| 代价网格（2 个尺寸 × 5 种 ctx × 3 种 batch × 3 种配置）+ 端到端验证子集（100 题 × 4 个模型 × 3 种配置 × batch {1, 32}） | ~15 | 父子模型共享架构，所以网格每个尺寸只测一次 |
| 驱逐主网格（7 个模型 × {逐头, shared, R-KV}；MATH500 × 2 扫 2/4/8×；AIME × 8 只跑 4×；HF harness） | ~96 | 1.5B 约 2K tok/s、7B 约 600 tok/s（R-KV 慢 30%），驱逐导致的长度反弹 ×1.3 |
| **must-run 合计** | **~136 → 加 30% 余量 ≈ 175** | 低于锚点上限 ~336 GPU-h |
| nice-to-have | SnapKV / recency 附录（1.5B 对，约 13）；绝对 K（约 20）；KD drafter（约 20）；QwQ-32B ThinkPrune + 公开 EAGLE head（未验证，约 40）；训练 EAGLE-3 × 2（约 200） | |

- **工程量**：约 2–3 人周。
- **日历时间**：8 卡下，G0 在第 1 周出结果，主结果 3–4 周。

## Named Risks（只记录，不加机制）
1. **效应量**：可能只是 10–20% 的修正，评价为"只是测量"。
2. **熵锐化**：长度惩罚 RL 可能降低熵、提高 G，掩盖复述的损失。如实报告。
3. **EAGLE-3 缺失**：主 drafter 不是生产中用的 EAGLE-3 / MTP，部署结论因此打折。
4. **vLLM draft-model 词表**：1.5B 的 embedding 有 151,936 行，7B 有 152,064 行，vLLM 可能拒绝组合，待核实。离线 τ 不受影响；计时可以改用对齐 vocab 的手工 harness。
5. **RA harness 移植**：Qwen2 有 QKV bias，Qwen3 有 qk-norm，需要移植，约 1–2 天。
6. **抢先风险**：Random Attention 作者的后续工作。G0 必须在 1 周内完成。
7. **事实抽取只覆盖数值类事实**：GPQA 等非数值任务只做附录。
