# Idea Discovery Report — 免训练 / 少训练、低资源的 LLM & MLLM 研究方向

**Direction**: LLM/MLLM efficiency、LLM test-time scaling (TTS)、continual learning (CL)，以及其他免训练/少训练、不吃资源的方向
**Date**: 2026-10-18
**Pipeline**: research-lit → idea-creator（5 个方向并行 lens fan-out）→ novelty-check（3 组深度查新）→ ~~research-review~~ → ~~research-refine-pipeline~~

> ⚠️ **本次运行的限制（请先读）**
> - **没有跨模型评审**：本环境没有 Codex / Gemini MCP，`/novelty-check` 的 Phase C 与 `/research-review` 都未执行，所有新颖性判断均出自同一模型家族（Claude），属于“自审”。按 ARIS 的 evidence gate，本报告状态为 **BLOCKED: novelty-check / research-review cross-model receipt missing**。
> - **没有 GPU pilot**：容器无 GPU，所有 idea 只有“纸面验证”，下面给出的是 pilot 设计 + kill criterion，而不是实验结果。
> - **检索覆盖**：arXiv API + arXiv HTML 检索 + WebSearch；Semantic Scholar 全程 429 未贡献；ICLR 2027 投稿尚未被索引。累计引用约 500+ 篇 arXiv 论文（2025 → 2026-10），深度查新阶段引用的 id 已用 `tools/verify_papers.py` 校验存在；共精读约 25 篇全文，其余基于摘要。
> - 这些领域每周新增 10+ 篇，**查新结论有效期大约 1–2 个月**。

详细笔记（每个方向的文献地图、全部候选 idea、“已被做过”清单）：

| 文件 | 内容 |
|---|---|
| `lit/01_llm_efficiency.md` | 文本 LLM 推理效率（KV、稀疏注意力、投机解码、量化/剪枝、MoE、dLLM） |
| `lit/02_mllm_efficiency.md` | MLLM / Video / VLA / omni 效率 |
| `lit/03_test_time_scaling.md` | Test-time scaling（并行采样、自适应算力、免训练过度思考压缩、verifier-free 选择） |
| `lit/04_continual_learning.md` | 免训练/少训练持续学习（合并、解析解 CL、知识编辑、记忆、SFT vs RL 遗忘） |
| `lit/05_other_training_free.md` | steering、幻觉、不确定性、合并、长上下文、可解释性、安全、dLLM |
| `lit/06_novelty_A.md` / `07_novelty_B.md` / `08_novelty_C.md` | 三组深度查新（精读最接近论文全文） |

---

## Executive Summary

1. **“换一个打分函数”类工作在几乎所有免训练子领域都已饱和**：KV 驱逐（2026-10 前两周就 20+ 篇）、单图视觉 token 剪枝（一个月 50+ 篇）、置信度加权投票、VLM 对比解码、拒答方向 steering、agent memory。在这些方向继续卷新 score 很难发。
2. **真正还有空间的是四类“结构性”机会**：(a) 新场景/新架构带来的新冗余（推理模型多轮对话、zoom 类视觉 agent、MoE、混合线性注意力、扩散 LLM）；(b) **端到端成本核算**——很多效率方法省了 prefill 却让推理链变长（已被多篇证实），以“整条回答的总成本”重评旧方法；(c) **严格统计的审计/反思**——2609.32035、2607.25196 一类论文正在推翻“置信度投票”“对比解码”的收益，2026 新模型上的复现审计是低成本高引用的方向；(d) **后训练谱系（lineage）**——同一基座派生出的 instruct / distill / merge / 量化 / 扩散后代之间，探针、方向、安全性如何迁移与退化。
3. **推荐优先做的 2 个 idea（PROCEED）**：**ReThink-Reuse**（推理模型多轮对话中被模板删掉 thinking 后的 KV 复用）和 **GeoDedup**（zoom 类视觉 agent 的跨轮几何去重，pilot 用 Mini-o3）。另有 6 个“谨慎继续”的 idea 都能在 ≤1 GPU-day 内用 kill criterion 快速判死活，其中 TTS 的两个 idea 可共享一套采样轨迹。

---

## Literature Landscape

### 1. 文本 LLM 推理效率
- **饱和**：长上下文 KV 驱逐/量化/低秩；推理模型 KV 压缩（R-KV 2505.24133、ThinKV 2510.01290、BreadthKV 2610.05685 等）；稀疏 prefill/decode 注意力；自投机/层跳过/检索式 drafter；权重量化（旋转类）；dLLM 缓存与并行解码。
- **矛盾**：Random Attention 2609.03430 显示“保住 prompt + 随机驱逐”即可追平最强基线，与 2512.12008 的“heavy-hitter 占优”冲突，且只在短 prompt + 长 CoT 上验证过。
- **开放**：推理模型多轮对话中模板删除 thinking 导致前缀缓存失效；MoE 专家剪枝/动态 top-k 在 thinking 模式下的 token 膨胀；多种压缩叠加时的“复合膨胀”；KV 压缩下多轮 system prompt 遵循的衰减。

### 2. MLLM 效率
- **极度饱和**：单图 token 剪枝/合并、视频压缩（SimpleCluster 2609.35394 显示简单聚类即可追平复杂方法）、流式视频 KV、多模态投机解码、omni 音视频压缩（~12 篇/4 个月）、VLA 剪枝（~25 篇）。
- **评测坑已有人做**：OCR 证据覆盖（2608.00077）、真实加速（2608.03649）、置信度（2604.12035）、安全（2610.09703）；剪枝让思考型 MLLM 输出变长已被 VisionPulse 2605.31457 / ViMoD 2610.12060 / DSTP 2604.12358 证实。
- **开放**：多轮 zoom/“thinking with images” agent 的跨轮冗余；新架构（Qwen3-VL DeepStack、混合线性注意力 V-CoLA 2610.11251 / DeltaS 2609.27470 刚出现）；统一理解+生成模型。
- **新发现**：2608.06270 审计显示 DeepEyes 的放大工具“结构性失活”（换随机裁剪仅差 ≤1.6 分），Mini-o3 / Qwen3-VL 才真正依赖裁剪。

### 3. Test-time scaling
- **饱和且被质疑**：内部信号/置信度选择（DeepConf 2508.15260、self-certainty 2502.18581 等）——2609.32035 在 280 种组合上发现经 Holm 校正后没有一种加权投票稳定胜过多数投票，且 think 模式错误会集中到同一干扰项、轨迹长度是强混杂。
- **拥挤**：并行思考的在线早停/剪枝（Parallel-Probe 2602.03845 等）、免训练 early exit / 过度思考压缩（DEER 2504.15895、NoWait 2506.08343 等）、thinking budget 路由（DART 2606.23181）、量化 × TTS。
- **开放**：多样性来源（输入扰动、层复用、think/no-think 混合）的系统比较；零成本的跨模型信号（小 draft 模型对大模型轨迹的打分）；过滤 vs 加权的严格对照。
- 语音合成（text-to-speech）方向几乎没有免训练效率工作（Llasa 2502.04128、MagpieTTS-LF 2606.18485），本次未展开。

### 4. 持续学习（免训练/少训练）
- **拥挤**：持续/顺序模型合并、多模态持续指令微调（MoE/LoRA 路由）、终身知识编辑、agent memory / 自进化 agent（CL-Bench 2606.05661 发现朴素 ICL 胜过专用记忆系统）。
- **中等**：解析解/闭式 CL 已迁移到 LLM（MAny 2604.14016、RLSEdit 2601.15686）。
- **最热且结论互相矛盾**：SFT vs RL 遗忘——RL's Razor 2509.04259、RL Forgets! 2607.04364、2601.20861、2607.01763 结论不一；机制线索：2610.09835（遗忘集中在新数据中未出现 token 的输出嵌入行，被 Adam 放大，但只研究了 SFT）、2610.08718（大量遗忘是可逆的共享偏移）。
- **开放**：遗忘机制的低成本小模型实验（OLMo-2-1B）；免训练的推理时“撤销偏移”；LLM CL 基准（TRACE/SuperNI）的泄漏审计（多模态侧已有 2606.20961、FLEX 2608.01437）。

### 5. 其他免训练方向
- **饱和**：拒答方向/条件 steering、SAE 特征 steering；VLM 幻觉的对比解码与注意力重加权（每月 10+ 篇，且 2607.25196 证明 VCD/ICD/SID 在 Qwen2.5-VL/InternVL3 上不如贪心）；内部探针做不确定性；LLM-as-judge 偏差；内部状态越狱检测。
- **开放**：长 CoT 中 steering 效果的衰减；**谱系迁移**（探针/方向在 distill、merge、量化、AR→扩散后代上的迁移；base→instruct/MLLM 已被 2606.15821、2504.02904 做过）；reasoning↔instruct 合并的安全代价；扩散 VLM 的早期去噪窗口干预（DynaHall 2609.34772 已部分占位）；长 CoT RAG 中对上下文注意力的衰减。

---

## Ranked Ideas

评分为单模型自评（0–10），仅供排序参考；所有 idea 均为免训练或极少训练，pilot 均可在单张 24–80GB GPU 上 ≤1 GPU-day 完成。

### 🏆 Idea 1: ReThink-Reuse — 推理模型多轮对话中删除 thinking 后的 KV 复用 — RECOMMENDED
- **方向**：LLM 推理效率 · **查新**：PROCEED（6/10）
- **问题仍然存在**：Qwen3 / 3.5 / 3.6 默认模板仍删除上一轮 thinking（Qwen3.6 的 `preserve_thinking` 默认 false），DeepSeek V4 有 `drop_thinking`；vLLM issue #39321 明确指出答案 KV 在缓存中却必须重算，现有修复（SGLang PR #23315、vLLM PR #39806）只是驱逐 thinking 块；llama.cpp `--cache-reuse` 是通用删除+平移，无推理模型质量研究。
- **方法**：删除 thinking 的 KV，RoPE 重新旋转对齐位置，可选只重算偏差最大的 top-r% token；并研究“看过 thinking 的答案 KV”是否反而对下一轮有益。
- **最接近**：KVEraser 2606.17034（擦除场景下删除+平移失败，因残留信息——在本场景中这可能是收益）、Models Take Notes at Prefill 2606.17107、CacheBlend 2405.16444、RoPERelay 2610.21126。
- **Delta**：首次测量“模板强制删 thinking 后复用答案 KV”的质量 vs TTFT，并检验 thinking 残留对下一轮的作用。
- **Pilot**（~0.5 GPU-day）：Qwen3-8B，~200 条三轮数学链 + 100 条 BFCL 多轮；对比 exact re-prefill / preserve_thinking / 删除+平移 / 重算 top 5/10/20%。
- **Kill**：重算 20% 仍比 re-prefill 低 >2 分且无残留收益，或真实历史长度下 TTFT 加速 <1.5×。
- **风险**：可能被视为工程；新模型逐渐保留交错 thinking，问题可能只存在于部分模型族。

### 🏆 Idea 2: GeoDedup — zoom 类视觉 agent 的跨轮几何去重 — RECOMMENDED（先过 token 数门槛）
- **方向**：MLLM 效率 · **查新**：PROCEED（6.5/10）
- **方法**：裁剪图返回后，按已知 bbox 把全局图（以及被替代的更粗裁剪）中被覆盖的视觉 token 合并/删除，释放上下文。
- **最接近**：PRCR 2606.26631（其 limitations 明说裁剪被重新编码时失效，正是 zoom agent 场景）、AwaRes 2603.16932、2512.03794（均需训练）。
- **关键调整**：pilot 换成 **Mini-o3**（DeepEyes 工具结构性失活且全局图主导，可省仅 1–10%）。Mini-o3 每图 ~2.5k token、平均 6.3–6.5 轮、32K 上下文是硬约束（降低像素换更多轮可把 VisualProbe-Hard 从 36.1 提到 48.0），去重省出的上下文可直接换成更多轮 → 更高准确率。
- **Pilot**：先测 V* / VisualProbe-Hard 上的 token 构成（~0.2 GPU-day），再做去重。
- **Kill**：可去重 token 占轨迹视觉 token <20%，或同轮数上限下准确率下降 >1.5 分。
- **评测**：用“整条轨迹总成本”（prefill + decode FLOPs、上限命中率）核算。

### Idea 3: TTS 共享轨迹包 — SpecSignal + Prior-Discounted Voting — BACKUP
- **方向**：Test-time scaling · **查新**：两者均 PROCEED WITH CAUTION
- **共享 pilot**：Qwen3-8B think×32 + no-think×32，GPQA-D / MMLU-Pro(500) / AIME24/25 / HMMT，~0.5 GPU-day，再加 Qwen3-0.6B/1.7B teacher-forced 打分 ~0.2 GPU-day。所有结论必须过长度残差化、题目级 bootstrap、Holm 校正。
- **3a SpecSignal**：小 draft 模型对大模型采样轨迹的期望接受率作为零成本选择/早停信号。最接近 **Cross-Model Perplexity 2603.25450**（其 limitations 说弱 verifier 情形未测、预期退化；仅贪心短输出）。Delta：更弱的同族 draft、长采样 thinking 轨迹、题内排序、接受率形式。**Kill**：pooled AUROC <0.55 或比同 draft 的 CMP 提升 <0.02。
- **3b Prior-Discounted Voting**：think 票分裂时，对与 no-think 众数相同的答案降权。最接近 **JointThinking 2508.03363**（方向相反：一致即接受）。⚠️ 其 Table 8 数据与本前提相反（think 错误与 no-think 重合率低于与另一 think 样本的重合率，AIME/AMC 上几乎不发生），只可能在有限选项集（GPQA-D、MMLU-Pro）上成立。**Kill**：think 多数错误时等于 no-think 众数的比率不比选项匹配基线高 ≥10pp。可能的零 GPU 第一步：2609.32035 发布的 Qwen3-8B/32B 双模式逐样本数据（未确认可下载）。

### Idea 4: 谱系监测 + 合并安全 — 一篇论文两部分 — BACKUP
- **方向**：其他免训练 · **查新**：两者均 PROCEED WITH CAUTION（5/10）
- **4a 谱系迁移监测器**：在父模型上拟合的幻觉探针/拒答方向，用无标注均值/协方差重新居中后迁移到重度后训练的后代（R1-distill、long-to-short merge、AR→扩散 Dream-7B）。base→instruct/MLLM 已被 **2606.15821（ICML'26）**、**2504.02904** 做过，需重新定位。注意 R1-Distill-Qwen-7B 源自 Qwen2.5-Math-7B、Dream-7B 源自 Qwen2.5-7B base，父模型要选对；评测须采用 2609.30275 的噪声底 null（INT4 会旋转拒答方向）。**Kill**：重居中 AUROC 在 R1-distill 和 Dream 上都 <0.75×原生，或原始迁移已 ≥95%。
- **4b long-to-short 合并的安全代价**：2503.20641 / 2510.10977 全文确认无安全评测；但 **2604.04120（COLM'26）** 已在每种方法单一系数上测过 HarmBench，2608.08542 已有投影修复 SubSafe-Merge（技能合并场景）。Delta：instruct↔thinking 插值的完整 α 曲线 + 自适应攻击（PAIR）+ 首 token 拒答读出 + 胜过 SubSafe-Merge 的修复；按 2609.23587 校正“尝试但失败”。测试平台改为 Qwen3-4B-Instruct-2507 × Thinking-2507。**Kill**：自适应攻击成功率随 α 单调，且偏离端点连线 ≤±5 分。

### Idea 5: Steer-Back — 推理时减去层均值偏移以撤销伪遗忘 — BACKUP
- **方向**：持续学习 · **查新**：PROCEED WITH CAUTION
- **最接近**：2502.11019（ICLR'25，推理时减去 function vector 降低遗忘）、2406.12227（TRACE 后加 instruction vector），二者均只作分析未作方法基准；2610.08718 选择了权重空间而非激活空间。
- **Delta**：任务无关、只用通用文本估计的 base−finetuned 均值偏移，无需旧任务数据和 head 搜索，对比 WiSE-FT / DG-Hard / function-vector 减法。
- **Pilot**（~0.5 GPU-day）：Qwen2.5-7B-Instruct，在 3 个 TRACE 任务上顺序 LoRA，用 1k C4 估计偏移，扫 α。**Kill**：在新任务损失 ≤2 分下恢复 <25% 通用能力损失，或被 WiSE-FT / FV 减法支配。

### Idea 6: MoE 动态专家截断在 thinking 模式下的 token 膨胀 + 分阶段专家预算 — BACKUP
- **查新**：PROCEED WITH CAUTION（4.5/10）。RAZOR 2609.30465 已测静态剪枝（关闭 thinking）下的长度膨胀（REAP 50% 剪枝中位长度 2.43k→3.94k）；2610.20826 提示分阶段路由统计不可靠。
- **Pilot**（~1 GPU-day）：GPT-OSS-20B、Qwen3-30B-A3B-Thinking，k=100/67/50/33%，AIME + MATH500-200。**Kill**：准确率保持 ≥97% 时膨胀 ≤5%，或分阶段调度不比均匀截断多省 10% 成本或多 2 分。

### Idea 7: 互补视图集成 — BACKUP
- **查新**：PROCEED WITH CAUTION（4.5/10）。最接近 **Dropout Decoding 2412.06474**（K=3 随机视觉 mask 投票，非互补、非等算力）、ParVTS 2511.18875。等算力只在短答任务成立。
- **Kill**：2×25% 在通用基准上不比单次 50% 高 ≥1 分，或在 POPE 上不能以更低成本胜过 Dropout Decoding。

### Idea 8: 遗忘机制——on-policy 性是否决定“缺席 token”嵌入漂移（重构版）— EXPLORATORY
- **原假设已被反驳**：2510.18874 Table 1 中无 baseline 的 0/1 REINFORCE（梯度权重全非负，无可抵消）遗忘仍接近 GRPO，作者明确排除优势估计是原因；零均值项最多解释 ~18% 差距。
- **重构问题**：是 on-policy 性（而非优势符号）决定缺席 token 漂移吗？2610.09835 没有任何 RL 对照。
- **Pilot**（<0.5 GPU-day）：OLMo-2-1B-Instruct（非绑定嵌入），六组：SFT / Self-SFT / Iterative-SFT / REINFORCE-0/1 / GRPO / 输出层加大 Adam ε 的 SFT。**Kill**：Self-SFT 与 GRPO 行漂移差 <1.5× 而遗忘差 >3×，或 ε 调整弥合 <20% 差距。

### 其余保留项（低优先级）
- **幻觉方法审计（收窄版）**：对比解码那半已被 2607.25196 做完；只剩 PAI / OPERA / DoLa 式 / head 干预在 2026 模型上的审计（4/10）。
- **扩散 VLM 早期窗口视觉引导**：DynaHall 2609.34772 已在 pre-commitment 窗口用 PGS 干预，且报告 VCD/DoLa 在扩散 VLM 上增加假阳性；剩余 delta 是步窗口化的 CFG 引导 + 解释 VCD 为何有害（4/10）。**先决条件**：全步引导在 LLaDA-V 上 POPE-adv F1 至少 +1。
- 调研阶段的其他候选（Random Attention 在长输入/代码上的失效、多种压缩叠加的复合膨胀、steering 在长 CoT 中的半衰期、长 CoT RAG 的上下文再锚定、LLM CL 基准泄漏审计等）见 `lit/01–05`，均为 CROWDED-BUT-DELTA，未做深度查新。

---

## Novelty Verification

深度查新共 13 个 idea（3 组并行，精读约 25 篇最接近论文全文）：

| Idea | 判定 | 最接近工作 | 查新中新发现的关键事实 |
|---|---|---|---|
| ReThink-Reuse | PROCEED | 2606.17034, 2606.17107 | vLLM #39321 确认问题；主流模板仍删 thinking |
| GeoDedup | PROCEED | 2606.26631 | 2608.06270：DeepEyes 工具失活 → 改用 Mini-o3 |
| SpecSignal | CAUTION | 2603.25450 (CMP) | 调研阶段漏掉的 CMP |
| Prior-Discounted Voting | CAUTION | 2508.03363 (JointThinking) | 其数据与前提相反，仅限有限选项集 |
| 谱系迁移监测器 | CAUTION | 2606.15821, 2504.02904 | base→instruct 已做；父模型谱系需纠正 |
| 合并安全代价 | CAUTION | 2604.04120, 2608.08542 | 单系数 HarmBench 已测 |
| Steer-Back | CAUTION | 2502.11019, 2406.12227 | 推理时向量减法已作为分析出现 |
| MoE token 膨胀 | CAUTION | 2609.30465 (RAZOR) | 静态剪枝长度膨胀已测 |
| 互补视图集成 | CAUTION | 2412.06474 | Dropout Decoding 已做随机 mask 集成 |
| 零均值优势抵消漂移 | 原框架被反驳 → 重构 | 2510.18874 | 0/1 REINFORCE 反例 |
| 幻觉方法审计 | CAUTION（收窄） | 2607.25196 | 对比解码部分已完成 |
| 扩散 VLM 早期窗口 | CAUTION | 2609.34772 | DynaHall 已占早期窗口 |
| 剪枝让思考 MLLM 想更久 | **ABANDON** | 2605.31457 | VisionPulse Tab.1/3 已证明 +11%~+108% |

---

## External Critical Review

**未执行**。本环境无 Codex / Gemini / manual-review MCP，无法获得非 Claude 家族的评审 receipt。建议在有 Codex MCP 的环境中对 Idea 1、2 运行：

```
/research-review "ReThink-Reuse ..." — composed: idea-stage/IDEA_REPORT.md
/novelty-check "GeoDedup ..."
```

并重点让外部评审核对：2606.15821 的跨族附录（Idea 4a）、2604.04120 附录 A（Idea 4b）、RoPERelay 2610.21126 是否测过多轮（Idea 1）。

---

## Eliminated Ideas

| Idea | 淘汰阶段 | 原因（具名论文） |
|---|---|---|
| 剪枝让思考型 MLLM 想更久（解决 ViMoD vs VisionPulse 矛盾） | Phase 3 | 矛盾不存在：VisionPulse 2605.31457 Tab.1/3 已证；ViMoD 2610.12060、DSTP 2604.12358 也已报告。保留其“整条回答总成本”评测方式用于 Idea 2/7 |
| GRPO 零均值优势抵消缺席 token 漂移（原框架） | Phase 3 | 2510.18874 Table 1 反例；已重构为 Idea 8 |
| VLM 对比解码在 2026 模型上的审计 | Phase 3 | 2607.25196 已完成 |
| probe 式早停/剪枝、DART 式 budget 路由、量化×TTS、dLLM unmask 顺序多样性、单独的长度混杂诊断 | Phase 2 | 2602.03845、2606.23181、2605.05561/2610.05685、2602.04344、2609.32035/2510.10409 |
| 兄弟样本作 draft、驱逐导致 CoT 变长、自生成 CoT 量化校准、多 head 覆盖驱逐、跨轮复用稀疏 prefill 模式、KV 压缩换采样数 | Phase 2 | 2503.05330、2512.12008/2512.07993、2608.01078/2504.04823、2606.29563/2609.06663、2507.13681、2506.05345 |
| 低置信度时全 token 重跑、解码中途丢视觉 token、MoE 路由感知剪枝、多图 token 预算 | Phase 2 | 2604.12035、2405.05803/2411.19628、2511.17885、2610.18858 |
| RL 收益作 steering 向量、统一模型跨分支 steering、AR 权重差加到 dLLM、steering×量化、dLLM 轨迹幻觉检测 | Phase 2 | 2510.07364、2607.26411、2610.08108、2609.06473、2607.22661 |
| 事后重置输出嵌入行、免训练撤销奇异向量旋转、编辑在后续微调中的存活、agent memory 跨骨干迁移 | Phase 2 | 2610.09835、2605.20296、2511.05852、2603.23234 |

---

## Next Steps

建议顺序（总计约 3–4 GPU-days 可把前 5 个 idea 全部判死活）：

- [ ] **Idea 2 门槛测量**（0.2 GPU-day）：Mini-o3 在 V* / VisualProbe-Hard 上的 token 构成 → 决定是否继续
- [ ] **Idea 1 pilot**（0.5 GPU-day）：Qwen3-8B 多轮复用四臂对比
- [ ] **Idea 3 共享轨迹包**（0.7 GPU-day）：一次采样同时检验 SpecSignal 与 Prior-Discounted Voting；先尝试 2609.32035 公开数据做零 GPU 检验
- [ ] **Idea 4 pilot**（1 GPU-day）：谱系探针迁移 + α 曲线安全扫描
- [ ] **Idea 5 / 8 pilot**（各 ≤0.5 GPU-day）：小模型遗忘实验
- [ ] 在有 Codex MCP 的环境补跑 `/research-review` 与 `/novelty-check` Phase C，获得跨模型 receipt
- [ ] pilot 通过后：`/research-refine-pipeline` → `/experiment-bridge` → `/auto-review-loop`
