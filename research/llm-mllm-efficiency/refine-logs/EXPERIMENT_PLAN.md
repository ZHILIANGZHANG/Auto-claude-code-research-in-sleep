# Experiment Plan

**Problem**: 长度压缩、投机解码（SD）、KV 驱逐三者的收益能否叠加（见 `PROBLEM_ANCHOR.md`）
**Method Thesis**: 长度惩罚 RL 删掉的是复述，而复述是 SD 和无信号驱逐共同依赖的资源，所以 S_c/S_v < 1、E_c/E_v < 1；复述统计量 O_K 在模型内同时预测这两类损失。
**Date**: 2026-10-17
**Source**: `FINAL_PROPOSAL.md`（定义 D1–D8、模型矩阵、闸门）

## Planning Gate
- **最终论点**：见上。
- **主贡献**：受控测量 + 机制解释，仪器为 τ/A、G、O_K。
- **拒绝的复杂度**：新 drafter、新驱逐策略、压缩感知 RL、MLLM、长输入。
- **仍需验证的评审关切**：
  - 效应量；
  - 位置 / 历史长度伪影；
  - drafter 偏移 vs 内在可投机性；
  - 熵锐化；
  - 向均值回归；
  - P1 判别；
  - E 删失；
  - b > 1 时分解的有效性。
- **frontier 原语**：没有。EAGLE-3 用来让部署结论更真实，不构成 frontier 主张，所以不设"frontier 必要性"实验块。

## Claim Map
| Claim | 为什么重要 | 最低可信证据 | 实验块 |
|---|---|---|---|
| **C1a（主，SD 侧）** 压缩降低剩余 trace 的可投机性 | 每篇高效推理论文都用 token 数当头条指标 | 两个尺度、≥5 个子模型上，位置匹配 + 窗口的 n-gram A 和 G 下降，CI 不含 0；L1 同权重剂量扫描单调 | B1, B4 |
| **C1b（主，驱逐侧）** 压缩让无信号驱逐不再免费，且 O_K 能预测 | 直接检验 RA 自己说"未测量"的机制 | E_c/E_v < 1（shared 和逐头两组）；P2 成立；分层回归中 policy × O_K 交互项为负且显著 | B2, B4, B5 |
| **C2（辅）** token 节省高估了 SD 下的 wall-clock 节省 | 部署中的叠加收益 | 在 b ∈ {1, 32, 128} 下 S_c/S_v = ρ_SD/ρ_noSD < 1，并给出三项分解；代价网格的预测与端到端实测误差 ≤ 10%；EAGLE-3 父模型 head 上同样成立（若训练） | B3 |
| **要排除的反面解释** | | | |
| 只是历史长度变短 | | 匹配位置 + 固定窗口下效应仍在 | B4 |
| 只是 drafter 偏移 | | G 下降；父 drafter → 子模型 vs 父 drafter → 父模型 | B4 |
| 只是难度或选择偏差 | | ≥3/4 both-correct + 重采样对照 + 自身通过率作协变量 | B2, B4 |
| 只是循环 / 截断 | | 含 / 不含标记 trace 两版结论一致 | B4 |
| 只是模式切换 | | AdaptThink 分层，不进入汇总 | B4 |

## Paper Storyline
- **正文必须证明**：
  - 图 1：交互因子 S_c/S_v 和 E_c/E_v 的总览；
  - 表 1：C1a，即逐对的 A 和 G；
  - 表 2：C1b，即 E、P2 和回归结果；
  - 图 2：C2，即 ρ 的三项分解；
  - 一张反面解释的排除表。
- **附录支持**：SnapKV / recency、绝对 K、GPQA-D、AdaptThink 分层、KD drafter、时间积分 KV、P1 判别细节、事实抽取的标注质量。
- **明确砍掉**：MLLM；长输入；Qwen3.8-27B 压缩对（太贵）；QwQ-32B（没有可靠的 EAGLE head）；任何新方法。

## Experiment Blocks

### B1：SD 侧主结果（C1a）
- **Claim**：压缩降低剩余 trace 的位置匹配可投机性。
- **为什么存在**：这是锚点问题 (a) 的核心证据。
- **数据**：MATH500（500 题 × 4 样本）；AIME24+25（60 题 × 8 样本）。T=0.6、top-p 0.95、最大 32K。
- **比较对象**：2 个父模型 vs 6 个主子模型；L1-Max 在 4 个 prompt 预算下。
- **指标**：
  - 主指标：n-gram（窗口 2048）的 A = τ − 1，按位置桶、只比较覆盖到的题；
  - 次要：G、逐桶熵、不匹配口径的 A、suffix A、k=8。
- **设置**：用 vLLM 生成 trace，离线做更新步进（D1），按题目 bootstrap 4000 次。
- **成功标准**：≥ 2/3 的父子对上 A 的相对下降 ≥ 15%（CI 不含 0），两个尺度方向一致；L1 剂量扫描单调。
- **失败解读**：位置匹配后下降 < 10% → "表观的可投机性税是长度伪影"，并支持"可以相乘叠加"。这两条都作为正面结论写。
- **目标位置**：表 1、图 1 左。
- **优先级**：MUST-RUN

### B2：驱逐侧主结果（C1b）
- **Claim**：压缩让无信号驱逐不再免费，且 O_K 能预测失败。
- **数据**：MATH500（× 2 样本），父模型扫 2/4/8/16×、子模型扫 2/4/8×（若 8× 时仍 ≥ 95% 再加 16×）；AIME24+25（× 4 样本）只跑 4×。
- **比较对象**：8 个主模型 × {full（vLLM）, shared-draw 随机（主臂）, 逐头随机（RA）, R-KV}，再加重采样对照。
- **指标**：
  - E（isotonic 拟合后取第一个穿越点，bootstrap CI）、E_c/E_v；
  - P2：E(R-KV)/E(random) 在父子模型之间的变化；
  - 分层混合 logistic 回归中 policy × O_K 交互项（D6），报告 VIF；
  - 驱逐导致的长度反弹。
- **设置**：RA 的 HF harness，移植 Qwen2（QKV bias）；shared-draw 改为跨层共享；保护 prompt；r=64。
- **成功标准**：≥ 2/3 的对上 E_c/E_v < 1（CI 不含 1）；交互项 < 0（p < 0.01）；P2 方向成立。
- **失败解读**：E_c ≈ E_v 且交互项为 0 → 复述不是长度惩罚 RL 删掉的那种冗余，RA 的机制解释不能推广到压缩模型。这是一个可以发表的反例。
- **目标位置**：表 2、图 1 右。
- **优先级**：MUST-RUN

### B3：wall-clock 组合（C2）
- **Claim**：S_c/S_v < 1；token 节省高估 SD 下的 wall-clock 节省。
- **数据**：代价网格，即 2 个尺寸 × ctx {1K, 2K, 4K, 8K, 16K} × b {1, 32, 128} × {无 SD, n-gram k=4, draft-model k=4, EAGLE-3 k=4}。端到端验证子集：100 道 MATH500 + 30 道 AIME，模型取 1.5B 和 7B 的 DLER 对。
- **比较对象**：同一负载下的父 / 子模型。
- **指标**：S_c/S_v、ρ_SD、ρ_noSD、三项分解（G 比、drafter 残差、c̄ 比）、per-solved ρ。
- **设置**：vLLM，接受长度自己从提交的 token 计算；固定版本；单卡独占。
- **成功标准**：预测与实测误差 ≤ 10%；三个 batch 下 S_c/S_v < 1。
- **失败解读**：S_c/S_v ≈ 1 → 在 wall-clock 意义上可以相乘叠加；ρ_noSD > 1 的反向效应占主导时也照实写。
- **目标位置**：图 2。
- **优先级**：MUST-RUN；EAGLE 部分在 G0 promote 后才做。

### B4：反面解释的排除，以及 P1 判别
- **Claim**：效应不是伪影。
- **内容**：
  - 匹配 vs 不匹配、窗口 vs 不加窗口的 A；
  - drafter 偏移分解：父 1.5B drafter → 父 / 子 7B，子 1.5B drafter → 子 7B（次要格）；
  - EAGLE 父 head → 父 / 子（若训练）；
  - 熵；
  - 含 / 不含循环标记 trace；
  - 重采样对照；
  - AdaptThink 分层；
  - P1：shared vs 逐头在父子模型之间的变化（存活模型判别）。
- **成功标准**：每一项反面解释都单独排除，或如实量化。
- **目标位置**：表 3（排除表），P1 放附录。
- **优先级**：MUST-RUN，P1 本身作为附录。

### B5：简洁性检验，O_K 是否足够
- **Claim**：一个统计量就足以解释驱逐失败，不需要更复杂的预测量。
- **比较对象**（都用于预测驱逐失败）：
  - 只用 log 长度；
  - 只用模型的压缩率；
  - n-gram 复述率；
  - O_K；
  - O_K 加上前三者。
- **指标**：交叉验证 AUC 或 ΔlogLik；O_K 的偏效应。
- **成功标准**：O_K 明显优于前三者，而"全部加上"相对 O_K 提升很小。
- **失败解读**：如果长度本身就够了，机制解释降级为"更短 = 更脆弱"，在正文中如实写。
- **优先级**：MUST-RUN（复用 B2 的数据，零额外算力）。

### 附录 / nice-to-have
SnapKV + prompt 和 recency + prompt（1.5B 对，4×）；绝对 K ∈ {1024, 2048}（2 个对）；GPQA-D；KD drafter（用 token 匹配的 target trace 训练 1.5B drafter）；时间积分 KV。

## Run Order and Milestones
| Milestone | 目标 | 运行 | 决策闸门 | 成本 (GPU-h) | 风险 |
|---|---|---|---|---|---|
| **M0 Sanity**（第 1 周前半） | 工具链正确 | 下载并复核 checkpoint ID；RA harness 移植 Qwen2，并复现 RA 的 Qwen3-4B MATH500 结果（K=1024：逐头 0.874、shared 0.871，±1.5 分以内）；walk-τ 估计与 vLLM n-gram 实测接受率在 50 题上对照（误差 ≤ 3%）；事实抽取器人工标注 50 条 | 复现失败或误差超限 → 先修工具，不进入 G0 | ~4 | RA 代码对 Qwen2 的支持；vLLM 版本差异 |
| **M1 G0**（第 1 周） | 判生死 | G0-SD 和 G0-Evict（见 FINAL_PROPOSAL） | Kill / Promote / 中间分支按预注册执行；Promote 后开始训练 EAGLE-3 | ~6 | 效应落在 10–25% → 走中间分支 |
| **M2 Traces + 离线 SD**（第 2 周） | B1 全部、B4 的 SD 部分 | 11 个模型的 trace、重采样对照、离线 A / G / drafter | B1 成功标准；若失败，转入"可相乘叠加"的叙事 | ~20 | AIME 的长尾；循环 |
| **M3 驱逐网格**（第 2–4 周） | B2、B5 | 8 个模型 × 3 臂 × 预算网格；full-KV 等价性检验 | B2 成功标准 | ~132 | HF harness 吞吐低于估计 → AIME 降到 × 2 |
| **M4 计时 + EAGLE**（第 3–4 周） | B3 | 代价网格；端到端验证；EAGLE-3 训练与评测 | 预测误差 ≤ 10% | ~18 + 105 | SpecForge 不支持 R1-Distill-Qwen-7B → 降级为 Named Risk 3 |
| **M5 分析 + 附录**（第 4–5 周） | 图表、附录 | 回归、排除表、P1、L1 剂量扫描、附录实验 | 进入 `/auto-review-loop` | ~10 + 附录 | — |

## Compute and Data Budget
- **总 GPU-h**：must-run 约 289（含 EAGLE），加 15% 余量约 332，在锚点上限 336 以内。不含 EAGLE 约 184，加 30% 余量约 240。
- **G0 的约 6 GPU-h 不额外计入**：G0-SD 的 trace 是 R020 的子集，G0-Evict 是 R030 中 4× 预算点的子集，后续直接复用。所以里程碑表中 M1 的 6 已包含在 M2 和 M3 里。
- **首周投入**：M0 + G0 约 10 GPU-h。
- **数据准备**：MATH500、AIME24 / 25（MathArena）、GPQA-D（附录）。EAGLE 训练数据为 7B 父模型自己的 trace，约 2 亿 token，生成成本已计入 105 GPU-h。
- **人工评估**：事实抽取器的复用事件标注，50 条 trace，约 1 人日。
- **最大瓶颈**：驱逐网格。HF harness 不分页、吞吐低，占总预算的 40% 以上。

## Risks and Mitigations
- **效应量落在中间** → G0 已预注册中间分支；论文叙事改为"实测的中等每 token 税 + 驱逐侧结论"。
- **熵锐化导致反向偏差** → 报告 G 和熵，作为"部分抵消"的机制如实写。
- **RA 移植失败** → 退回到 RA 的 vLLM 插件，只跑随机类策略；R-KV 改用其官方 HF 代码。
- **HF 与 vLLM 的 full 精度不一致** → 统一用 HF full-KV（成本 +20–30 GPU-h，在余量内）。
- **EAGLE 训练失败或超预算** → 降级为 Named Risk 3，部署结论限定为 n-gram / draft-model 类 SD。
- **抢先** → M0 + M1 在第 1 周内完成；结果一旦成立，优先写 arXiv 短版。

## Final Checklist
- [x] 正文表格都有对应实验块（表 1–3、图 1–2）
- [x] 新颖性被隔离（B4 的反面解释排除）
- [x] 简洁性得到辩护（B5：一个 O_K 足够）
- [x] 明确不主张 frontier 贡献（EAGLE 只用于部署真实性）
- [x] must-run 与 nice-to-have 已分开
