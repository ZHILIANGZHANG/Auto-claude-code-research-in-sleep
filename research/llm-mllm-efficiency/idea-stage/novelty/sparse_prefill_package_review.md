# 外部研究包评估：sparse-prefill-scheduling-2026-10-18.zip

- 日期：2026-10-17（本会话）。评估者为 Claude，与该包内部评审同家族，因此**不是独立的跨家族评审**。
- 包状态（自述）：文献整理与研究设计已完成。GPU 实验、调度器实现、代价模型训练都还没开始；新颖性 NOT_CONFIRMED；评审结论 provisional。
- 本次实际做的核查：
  - 读了 FINAL_PROPOSAL、EXPERIMENT_PLAN、SCIENCE_REVIEW、REVIEW_RESPONSE、REVISION_CHECK、BASELINE_CAPABILITIES、CLAIM_EVIDENCE；
  - 对包里附带的 FlashPrefill V2 源码快照（commit 75b58f2）逐行核对了评审引用的位置；
  - 跑了 CPU 测试：11 项全部通过。这只能说明 CPU 参考实现的规则对，不证明任何 GPU 行为。

## 已核实的代码事实
1. `schedule_policy.py:398–406`：`rem_chunk_tokens -= mixed_with_decode_tokens`。也就是说，和 decode 混合时，chunk 预算会被 decode 的 token 数挤掉。
2. `add_chunked_req`（schedule_policy.py:599–610）还会按 `rem_total_tokens` 截短 q，旁边有一条注释："必须把 chunked_req 加入列表，否则内存泄漏"。
3. `scheduler.py:2370–2402`：continuation 优先处理，之后仍会遍历 waiting queue，可能继续加入新的 prefill。
4. 引擎自带的 `enable_dynamic_chunking` 只在 `pp_size > 1` 时启用，基于 profiling 的长度预测，单卡下是关闭的。调度器里看不到任何基于稀疏度的逻辑（`hisparse` 是另一个机制）。
5. `prefill.py:338–345`：密度统计通过 `.item()` 加 `print` 输出，有同步开销。正式使用需要改成异步计数。
6. README 写明运行要求是 **NVIDIA Hopper（H20 / H100, sm_90a）**。

结论：包里评审提出的 R1（实际 chunk 大小并不固定）在代码上属实。

## 判断
- **研究问题本身**：问题界定得好，自我约束很强（信息增量 C1 和系统收益 C2 分开验证，有停止规则，有强基线清单）。作为研究设计，质量高于一般水平。
- **能不能做**：E1（同形状、不同内容下的残差诊断）便宜而且关键，值得做。整个系统项目暂时不应承诺投入。
- **粗略的 FLOP 估算**（Qwen3-4B：36 层，32 个 Q head × 128，8 个 KV head，MLP 中间维 9728）：
  - 每 token 每层的线性部分约 202 MFLOP，注意力部分约 16384·p FLOP，二者之比 ≈ p / 12.3K；
  - 也就是说，前缀 p 为 16K / 32K / 64K 时，稠密注意力分别是线性部分的约 1.3 / 2.7 / 5.3 倍；
  - 以 p=32K 为例：密度从 10% 升到 30%，整步计算量大约增加 40%（没有计入 decode 部分、selector 开销和 kernel 效率）；
  - 所以只要相同长度下密度随内容有 ±10 个百分点的变化，就足以影响整步时间。E1 有不错的通过概率。
- **最大风险不在 E1，而在 E2，即 M2 能否胜过 M0-H**：
  - 设计中每步最多只有一个 prefill 请求。这种情况下，同一请求上一个 chunk 的实测时延残差，本身就几乎概括了这个请求的密度偏差；
  - 因此，稀疏遥测相对于廉价的时延反馈，可能只带来很小的额外信息。它可能的优势在于：按层的块数，能对"前缀变长后密度如何变化"做机制性外推；
  - 我估计 M2 达到预设的"恢复 M1 收益的一半以上"的概率不到一半。如果 M0-H 胜出，贡献就退化为"基于反馈修正代价的 chunk 准入"，新颖性很弱。
- **控制杠杆偏弱**：只能对一个 continuation、在固定 chunk 大小下做准入或暂缓。只有当预测值恰好落在预算边界两侧时，动作才会不同，goodput 提升可能达不到 10%。在 G2 的对比中，带反馈的动态切块方法（如 SLOWeave 类）可能吃掉大部分收益。
- **硬件是硬约束**：必须有 Hopper。如果只有 A100 或 4090，只能换后端（MInference、XAttention 的 Triton 实现），包里整套 SGLang + FlashPrefill V2 的代码依据都要重做。
- **工程风险集中在固定边界 adapter 和暂停/恢复的生命周期管理**，算力本身不是瓶颈：包里估算 E0–E2 需 8–16 GPU-h，E3/E4 需 40–120 GPU-h。我估计全部做完的人力约 2–3 人月。

## 建议（相对于包内原有计划）
1. **把 E1 提到 adapter 之前做**。E1 只需要在固定形状下直接驱动 model runner 测整步时延，不需要改调度器。这样可以在投入数周工程之前就判断方向是否成立。
2. **用同一批 E1 数据离线先算一遍 M0-H 和 M2**：比较"同请求上一 chunk 的残差"与"块数统计"对下一 chunk 时延的预测力，零额外算力。这是核心贡献的最早淘汰测试。
3. 如果 E1 显示内容方差很大，而 M0-H 又已经足够，可以考虑转向"把稀疏度本身作为闭环控制变量"（调度器决定每个请求用多稀疏，受精度预算约束）。这一方向不在原设计内，新颖性更高，但需要做精度评测。
4. 开工前确认手上有 Hopper 机器。
