# Research Idea Report：Training-free / 少训练 / 低资源的 LLM-MLLM 研究空间

**Direction**：LLM/MLLM 推理效率、Test-Time Scaling、Continual Learning，以及其他不依赖（大规模）训练、单卡或 API 即可完成的研究方向——哪里还有空间、方向是什么、潜在 idea 有哪些。
**Generated**：2026-10-17
**Pipeline**：ARIS `/research-lit`（Phase 1 landscape）→ `/idea-creator` Phase 1.5 lens fan-out（4 个子方向 × 5 lens：method-transfer / contradiction / untested-assumption / scaling-regime / diagnostic）→ Phase 3 机械合并去重 → Phase 6 报告。
**Ideas evaluated**：42 generated → 38 survived dedup → 0 piloted（本环境无 GPU）→ 12 recommended + 26 backlog

> **可信度与流程说明（必读）**
> - 本环境的 egress 代理拦截了 `export.arxiv.org` / `arxiv.org` / `huggingface.co` / `openreview.net` / `semanticscholar.org`（CONNECT 403），`tools/arxiv_fetch.py` 不可用；按 research-lit 的 Policy D2 降级为 **WebSearch-only**（4 路共约 160 次查询）。所有 arXiv ID 来自搜索结果，**未读 abs/HTML 正文**，论文的关键数字与 limitation 来自摘要片段。四份 lens 报告里带 `⚠` / `[片段]` / `[未核实-记忆]` 标记的条目需在可访问环境二次核实。
> - idea-creator 的 Phase 4 **跨模型 jury（Codex MCP）在本环境不可用**，排名由执行方（Claude）自评，按 `shared-references/acceptance-gate.md` 记为 `review_independence: same-family`, `acceptance_status: provisional`。下一步应跑 `/novelty-check` 与 `/research-review — reviewer: codex`。
> - 用户写的 "llm tts" 有两种解读：**test-time scaling** 与 **语音 text-to-speech**，两者均已覆盖（§2.2 与 §2.4.8）。
> - 四份 lens 原始报告（论文卡片 + 完整 gap 论证）在 `idea-stage/lens/`。

---

## 0. 一页结论（TL;DR）

**已经做烂、不建议再做"又一个方法"的子方向（红海）**

| 子方向 | 证据 |
|---|---|
| 单图 LLaVA 上的 training-free 视觉 token 剪枝 | FastV→SparseVLM→VisionZip→HoloV→D²Pruner→TOPS 每月数篇；CVPR'26 [2512.07580](https://arxiv.org/abs/2512.07580) 与 [2502.11501](https://arxiv.org/abs/2502.11501) 证明 ≥20 层后不如随机剪枝 |
| prefill 阶段 KV 驱逐 + 层/头预算分配；KV 驱逐 × 量化联合 | SnapKV/PyramidKV/LAVa/LU-KV/MixedDimKV；2026 上半年 RDKV/MoE-nD/HqeKV/CAPKV/ThinKV 密集涌现 |
| "又一个旋转"式 PTQ | SpinQuant→SingleQuant→OptRot→ReSpinQuant |
| 置信/熵驱动的自适应 self-consistency 与单轨迹 early-exit | 2025-08 至 2026-06 ≥15 篇用同一组信号（logprob、答案熵、试答置信、注意力、反思 token）：DeepConf、CGES、SeerSC、Parallel-Probe、DEER、CGRS、ARS、RPDI-EE、CoDE-Stop、ASAG… |
| ViT 上 prompt-pool / LoRA-pool 的 class-incremental | L2P→CODA→GAP-Prompt、SD-LoRA/EWC-LoRA/SLAO；被 RanPAC/RanDumb/PAM/Fly-CL 等 TF 基线反复打平 |
| MLLM 持续指令微调方法线 | HiDe-LLaVA/MR-LoRA/Routing/φ-DPO；ICML'26 "Is Our Benchmark Enough" 证明 TF 原型路由即可匹配 |
| Activation steering 方法本体、文本 DoLa 系对比解码、prompt 优化 | COLD-Steer/ODESteer/GER-steer 在小基准比 1–6 点；LayerCake 撤回；GEPA 已工程化 |

**仍有空间的六大主题（跨方向归纳，详见 §3）**

1. **"token ≠ 延迟" 与压缩/早停的真实代价**：KV 压缩让模型"话变多"、early-exit 让 token 降但 wall-clock 升，几乎所有方法论文只报 token/s。诊断 + TF 修复均空白。
2. **信号的失效相图**：置信类 TTS 选择器、注意力类剪枝、steering 向量都在"某些难度/模型/深度"失效，且顶会结论互相矛盾（DeepConf vs CGES vs Consilience；attention 剪枝 vs 随机）。没人画统一相图。
3. **基准有效性审计**：预训练泄漏 ↔ TF-CL 优势；MLLM-CL 任务可分性；公共 LoRA 合并的 Δ≈0 伪影；VLM-TTS 增益在匹配解码格式后消失。LLM 侧对应审计缺位。
4. **方法向新 regime 的迁移**：agent/工具调用/多轮、多图交错、长视频、端侧 ≤3B、MoE/MLA、W4A4 量化模型上几乎没有 TF 压缩/TTS/CL 评测。
5. **跨社区统一**：CTTA vs TF-CIL；权重插值 vs 激活 steering；RAG/ICL vs 参数化 CL；KL-to-base vs 权重空间量作为遗忘预测量。两边各有一半工具，没人放进同一坐标系。
6. **语音 LLM-TTS 的 AR 语义 token LM 侧**：training-free 控制只碰了 flow-matching（EmoSteer）或 EOS logit（TED-TTS）；LM 侧 steering、对比解码、自投机空白。

**推荐优先执行**（全部单卡 ≤60 GPU-h 或 ≤$100 API，见 §4）：
Idea 1 *Why Compression Makes Models Talk Longer* → Idea 2 *Steering Safety-Tax Audit + Refusal-Orthogonal Fix* → Idea 3 *ConfMap* → Idea 4 *PruneDraft* → Idea 5 *DPE-for-Video*。

---

## 1. 方法与来源

| 项目 | 内容 |
|---|---|
| Lens 子代理 | ① LLM/MLLM 推理效率（60 篇） ② Test-Time Scaling（55 篇） ③ Continual Learning（70 篇） ④ 其他方向广度扫描（64 篇） |
| 时间窗 | 重点 2025-07 → 2026-10；锚点回溯到 2022–2025H1 |
| 覆盖会议 | NeurIPS'25/26、ICLR'26、ICML'26、ACL'26、CVPR'26、AAAI'26、MLSys'25/26、COLM'26、Interspeech'25、EACL'26 |
| 来源 | WebSearch（arXiv abs 摘要片段、会议 proceedings 页、OpenReview/aclanthology 片段） |
| 去重规则 | 按 `dedup_key` 的假设语义做机械聚类；仅合并"同一假设"的候选，不以强弱淘汰（Phase 3 规则） |
| 去重结果 | 42 → 38：`tts_early_exit_latency_audit_batchsync` 与 `kv-compression-output-length-inflation` 归入同一主题但假设不同，保留两条；`early-exit-x-kv-eviction` 并入前者作为子实验；`mllm_cl_benchmark_separability_audit` 与 `closed_form_router_mllm_cl` 合并为一条两阶段 idea；`verbalized_difficulty_control_knob` 与 `tts_budget_allocation_memory_knn` 合并为"TF 预算/强度分配"一条；`interpolation_three_stage_scaling_law` 并入 `merge_steer_equivalence` 作为扩展实验 |

---

## 2. Landscape Summary

### 2.1 LLM / MLLM 推理效率（training-free）

**现状**。KV cache 侧已从"驱逐打分"走到"驱逐 × 量化 × 合并"的组合红海（KVzip [2505.23416](https://arxiv.org/abs/2505.23416)、Expected Attention [2510.00636](https://arxiv.org/abs/2510.00636)、GRKV [2605.31105](https://arxiv.org/abs/2605.31105)、SelKV、RDKV…）。真正新的是三条**诊断线**：(a) 压缩拉长输出、端到端延迟反升（MLSys'25 [2503.24000](https://arxiv.org/abs/2503.24000)、ThinKV [2510.01290](https://arxiv.org/abs/2510.01290)）；(b) 压缩的"物理"：约 90% 处出现幻觉悬崖（[2603.01426](https://arxiv.org/abs/2603.01426)），固定预算在生产中次优（Sparse Frontier [2504.17768](https://arxiv.org/abs/2504.17768)）；(c) 多指令提示下压缩会"选择性遗忘"指令（Pitfalls, ACL'26 [2510.00231](https://arxiv.org/abs/2510.00231)）。推理模型解码期 KV（R-KV/BeaconKV [2609.04971](https://arxiv.org/abs/2609.04971)/Moment-KV）揭示"重要性回流"（Thought Revisiting Tokens），推翻了 prefill 驱逐的"重要性持续"假设。

视觉 token 剪枝在单图上饱和且被根本性质疑，但**长视频/流式**（FLoC、InfiniPot-V [2506.15745](https://arxiv.org/abs/2506.15745)、ForestPrune）、**多模态 KV**（MadaKV、FlashCache [2511.16786](https://arxiv.org/abs/2511.16786)、RetentiveKV 的"视觉重要性迟发"）、**剪枝 × PTQ 交互**（QAPruner [2604.02816](https://arxiv.org/abs/2604.02816)）刚起步。投机解码侧，MMSpec [2603.14989](https://arxiv.org/abs/2603.14989) 证明文本 SD 在 VLM 上失效（Lookahead 仅 1.07×），SparseSpec-L [2607.27735](https://arxiv.org/abs/2607.27735) 刚证明"稀疏 KV 的自己"可当 drafter。早退方面 [2603.23701](https://arxiv.org/abs/2603.23701) 指出新一代模型层冗余更低、收益逐代递减。

**蓝海**：VLM 的 TF 投机解码；agent/工具调用/多轮压缩；端侧 ≤3B；驱逐打分在 W4A4 旋转量化模型上的迁移；多图交错；MLA/MoE 上的 KV 压缩；"为什么失败"的诊断。

### 2.2 Test-Time Scaling（training-free）

**现状**。并行采样 + 自适应 N + 置信选择（DeepConf [2508.15260](https://arxiv.org/abs/2508.15260)、CGES [2511.02603](https://arxiv.org/abs/2511.02603)、SeerSC、Parallel-Probe [2602.03845](https://arxiv.org/abs/2602.03845)、Consilience [2608.09898](https://arxiv.org/abs/2608.09898)）与单轨迹 early-exit / 反思压制（DEER [2504.15895](https://arxiv.org/abs/2504.15895)、CGRS、RPDI-EE、CoDE-Stop、ASAG [2606.15070](https://arxiv.org/abs/2606.15070)）两条线都已饱和。三个未解矛盾：(a) 置信选择在难题上有效（DeepConf AIME25 99.9%）还是崩溃（CGES 承认 GPQA 无信息；Consilience 称"灾难性失效"）；(b) verifier-free TTS 是否次优（理论说差距随预算扩大；实证 [2607.09438](https://arxiv.org/abs/2607.09438) 说 PRM beam 比 SC 低 0.39pp 贵 8×，[2608.01207](https://arxiv.org/abs/2608.01207) 说多数投票打败自验证）；(c) token 减少 ≠ 延迟减少（SpecExit/[2605.17672](https://arxiv.org/abs/2605.17672) 测出 DEER 端到端 0.52–0.71×）。

agent 侧 General AgentBench [2602.18998](https://arxiv.org/abs/2602.18998) 明确指出序列扩展撞"上下文天花板"、并行扩展撞"验证缺口"，两者都无实际增益；CATTS [2602.12276](https://arxiv.org/abs/2602.12276) 只做步级投票。training-free latent 推理只有 Soft Thinking 一篇且自认 OOD，**不建议投入**。轻量 TTT/TTA 中真正 TF 的只有记忆/规则类（Dynamic Cheatsheet [2504.07952](https://arxiv.org/abs/2504.07952)、TF-TTCL）。

**蓝海**：置信信号的难度 × 模型（RLVR 程度）相图；early-exit 的延迟诚实性与批友好设计；轨迹级共识迁移到 agent；极低预算（N≤4）区间的策略对比；"决策点"oracle-gap 评测协议。

### 2.3 Continual Learning（training-free / 少训练）

**现状**。ViT-CIL 上 TF 解析式方法（RanPAC [2307.02251](https://arxiv.org/abs/2307.02251)、Fly-CL ICLR'26 [2510.16877](https://arxiv.org/abs/2510.16877)、AnaCP、LR-RGDA）持续打平训练式方法，但 AnaCP/VILA 自述"表示僵化"；没人回答**在什么域距离或泄漏比例下 TF 开始落后**（CLDyB [2503.04655](https://arxiv.org/abs/2503.04655) 与 Data Leakage [2508.17416](https://arxiv.org/abs/2508.17416) 各给了一半工具）。LLM 侧最重要的新结论是 RL's Razor [2509.04259](https://arxiv.org/abs/2509.04259)：遗忘由新任务上相对 base 的 output-KL 预测，权重量不行——而视觉合并线（MagMax、SFA、Audit Before You Merge ICML'26）全靠权重空间启发式，两者未在同一坐标系比较。

记忆/检索式 CL 结论混乱：CL-Bench、ContinualSkillBench [2608.03874](https://arxiv.org/abs/2608.03874) 说朴素 ICL 赢；Chronos (ACL'26)、GraphMemory [2610.02687](https://arxiv.org/abs/2610.02687)、WikiBigEdit [2503.05683](https://arxiv.org/abs/2503.05683) 说 RAG 有过期/退化/推理成本——根源是各家"遗忘"定义不同。MLLM-CL 方法线被 ICML'26 诊断为基准任务高度可分。CTTA 与 CIL 两个社区（TTABC [2606.14299](https://arxiv.org/abs/2606.14299)、UL-TTA、PTA vs RanPAC、RAIL）在做同一件事的两种参数化，从未同台。

**蓝海**：泄漏分层协议；KL 作为跨模态遗忘预测量与合并准则；CTTA+CIL 统一；统一遗忘定义的等预算对比；更新次数 10→10,000 的退化曲线；审计后的合并排名。

### 2.4 其他 training-free / 低资源方向

| 方向 | 饱和度 | 要点 |
|---|---|---|
| 合并·Thinking/Instruct 插值可调推理 | 活跃→趋饱和 | Thinking Spectrum [2509.22034](https://arxiv.org/abs/2509.22034)、Revisiting Interpolation ACL'26 [2510.10977](https://arxiv.org/abs/2510.10977) 一致："简单插值即可调 token-精度曲线"；缺机制与规模律 |
| 合并·异构/跨 tokenizer/理论 | 起步 | Darwin Family、AutoMerge、OMP 移植 [2506.06607](https://arxiv.org/abs/2506.06607)；综述列"theoretical gaps" |
| Steering 可靠性 / 安全副作用 | 活跃（反面研究刚起步） | Braun [2505.22637](https://arxiv.org/abs/2505.22637)、Rogue Scalpel ICLR'26 [2509.22067](https://arxiv.org/abs/2509.22067)、Safety Pitfalls [2603.24543](https://arxiv.org/abs/2603.24543)：任意方向都抬高有害顺从率；无缓解方案，非安全用途 steering 从不报安全税 |
| CoT 长度 steering | 活跃 | ASC [2507.04742](https://arxiv.org/abs/2507.04742)、Activation Control、STU-PID；与权重插值的关系未研究 |
| 解码期对齐 | 活跃 | AAD ICML'26 [2509.26169](https://arxiv.org/abs/2509.26169)；仍需 DPO 对或奖励模型 |
| TF 长上下文扩展 | 活跃 | DPE [2504.18857](https://arxiv.org/abs/2504.18857)→MrRoPE ICLR'26 oral [2601.22181](https://arxiv.org/abs/2601.22181)："逐维/混合进制"新范式；从未用于视频 MLLM |
| Many-shot 压缩 / 函数向量 | 活跃 | MemCom [2510.16092](https://arxiv.org/abs/2510.16092)（8×，需训练压缩器）；FV "可 steer 不可解码" ICML'26；二者未交汇 |
| TF 路由 / 级联 | 活跃 | ZeroRouter AAAI'26、VDAR [2607.18098](https://arxiv.org/abs/2607.18098)；难度估计是瓶颈 |
| Agent TF 效率 | 起步（混乱） | 系统/博客多，学术化低；Tran & Kiela [2604.02460](https://arxiv.org/abs/2604.02460) 指出算力匹配后单 agent 不输多 agent |
| MLLM 高分辨率 zoom | 活跃→趋饱和 | HiDe ICML'26 [2510.00054](https://arxiv.org/abs/2510.00054) 改写归因为"背景干扰" |
| 视频 KV-as-memory | 活跃 | FlexMem CVPR'26 [2603.29252](https://arxiv.org/abs/2603.29252)、StreamMem [2508.15717](https://arxiv.org/abs/2508.15717)；位置编码从未被当变量 |
| 多图交错 TF | 起步 | PulseFocus [2603.04676](https://arxiv.org/abs/2603.04676) 单作者 ongoing |
| **LLM-TTS（语音）TF 加速/控制** | 起步 | EmoSteer [2508.03543](https://arxiv.org/abs/2508.03543) 只在 flow-matching 侧；TED-TTS ACL'26 [2601.03170](https://arxiv.org/abs/2601.03170) 只调 EOS logit；PCG [2511.13732](https://arxiv.org/abs/2511.13732) 语音化 SD 接受准则（仍需 draft）；AR 语义 token LM 侧 steering/对比解码/自投机空白 |
| TF 遗忘、tokenizer 迁移、dLLM 加速 | 起步 / 起步 / 活跃（Fast-dLLM 已成基线） | — |

---

## 3. 跨方向结构性 gap（去重后的六大主题）

每条列出 lens、依据论文、为什么是 gap。

### G1. "token ≠ 延迟"：压缩与早停的真实代价（contradiction + diagnostic）
- **效率侧**：KV 压缩/量化让生成变长、端到端延迟反升（[2503.24000](https://arxiv.org/abs/2503.24000)、ThinKV），方法论文只报 token/s；机制（结论锚点被驱逐？重复自检？）与 TF 修复空白。
- **TTS 侧**：DEER 报 token −19~80%，SpecExit/[2605.17672](https://arxiv.org/abs/2605.17672) 测出端到端 0.52–0.71×；SeerSC 指出序列早停破坏并行；几乎所有 early-exit/选择器论文不报 wall-clock。
- **交叉**：早停改变长度分布 → 移动 KV 驱逐最优预算，两者从未联合评测。
- **共同缺口**：一个"延迟诚实"的评测协议 + 零额外前向的批友好设计。

### G2. 信号的失效相图（contradiction + untested-assumption）
- **置信类 TTS 选择器**：DeepConf 99.9% vs CGES "GPQA 无信息" vs Consilience "难题灾难性失效"；三者数据集/模型/置信定义不同，无人按难度桶 × 模型 × 置信定义画相图。置信早停的有效性很可能依赖 RLVR 后训练带来的校准（TTSV [2512.04748](https://arxiv.org/abs/2512.04748) Qwen +17 分、LLaMA 0 分；[2601.06423](https://arxiv.org/abs/2601.06423) SC 对 GPT-5.2 与 Claude 方向相反）。
- **注意力类剪枝**：attention 剪枝"领先基线"（HoloV/TOPS/ERASE）vs "≥20 层后不如随机"（[2512.07580](https://arxiv.org/abs/2512.07580)）；"信息地平线"是否随分辨率/帧数/模型代际前移未测。
- **Steering 向量**：Braun 发现 cos 一致性与可分性预测可 steer 性；steering 在 MoE、≤1.7B、长上下文扩展后是否漂移均无数据。
- **反思 token**：NoWait/CGRS（压制派）vs s1/α1（插入派）vs underthinking（切换有害），缺"难度 × 中间答案正确性"的 2×2 分解。

### G3. 基准有效性审计（diagnostic + untested-assumption）
- **CL**：ViT-CIL 基准上 TF 方法的胜利可能反映 ImageNet-21K/LAION 对 CIFAR/ImageNet-R/CUB 的覆盖（CLDyB + Data Leakage 各给一半工具）；MLLM-CL 任务高度可分（ICML'26），CoIN/MLLM-CTBench 是否同样可分未检验；公共 LoRA 合并的遗忘测量被 Δ≈0 伪影污染（Audit Before You Merge）。
- **TTS**：VLM 侧 [2608.01207](https://arxiv.org/abs/2608.01207) 证明一类"一致性选择"的增益在匹配解码格式与预算后消失；LLM 侧 DeepConf/Consilience/RPDI 没有同类受控审计。
- **效率**：视频 token 压缩从未与"均匀帧采样 + 随机空间"强基线公平对比。
- **合并**：成败无事前预测量，只能事后评测。

### G4. 方法向新 regime 的迁移（scaling-regime + method-transfer）
- **agent / 工具调用 / 多轮**：Pitfalls (ACL'26) 证明多指令会被选择性遗忘；工具 schema + 多轮观测的 TF 压缩无人做，BFCL/τ-bench 未被用于压缩评测；agent 上并行 TTS 卡在验证缺口（General AgentBench），轨迹级共识未见；真实 agent trace 的跨上下文重叠度无实测（KV 共享类方法的前提）。
- **多模态**：VLM 的 TF 投机解码（MMSpec 证明文本法失效 + SparseSpec-L 的"稀疏自己当 drafter"+ 视觉 token 深层可丢 = 天然 drafter，无人组合）；query-agnostic 重建式打分（KVzip）未迁移到视觉 KV；逐维 RoPE 分析（DPE/MrRoPE）未用于视频；多图交错剪枝空白。
- **模型/硬件 regime**：端侧 ≤3B 的 TF 压缩评测几乎无；KV 驱逐打分在 W4A4 旋转量化模型上的有效性未验证（QAPruner 只证明了视觉剪枝 × PTQ 有交互）；MLA/MoE 上 KV 压缩与专家跳过的联合空白；插值三阶段只在 4B/30B-A3B 观察。
- **CL regime**：更新次数 10→10,000 下 TF 闭式头 vs 单 LoRA 合并 vs 编辑的曲线；0.5–3B LLM 上冻结 + 闭式头的 CL；预训练数据量 → TF 优势的扫描。

### G5. 跨社区统一（contradiction + method-transfer）
- **CTTA vs TF-CIL**：UL-TTA/MS-TTA/PTA 与 RanPAC/RAIL 都是"原型 + 统计量更新"，CIL 本质是极端 label shift，从未在同一 CLIP 数据流对比。
- **权重插值 vs 激活 steering**：Revisiting Interpolation（权重 α）与 ASC（激活 λ）都在调 CoT 长度，是否同一条轴未知；steering 领域的几何预测量（cos 一致性）可迁移为合并的逐层 α 选择。
- **RAG/ICL vs 参数化 CL**：各家"遗忘"操作定义不同（能力遗忘 / 事实过期 / 暂时对齐丢失 [2501.13453](https://arxiv.org/abs/2501.13453)），缺等预算统一对比。
- **KL-to-base vs 权重空间量**：RL's Razor 说 output-KL 预测遗忘；MagMax/SFA 的成功全是权重平均——权重平均是否恰因在 KL 上最小？未验证。
- **难度估计**：路由（VDAR）、steering 强度（STU-PID）、zoom（LazyMCoT）、TTS 预算（SeerSC）各自估难度，无统一 TF 旋钮。

### G6. 语音 LLM-TTS 的 AR LM 侧（method-transfer）
- CosyVoice2 等的 LM 阶段与文本 LLM 同构，但 TF 控制只碰了 flow-matching（EmoSteer）与 EOS logit（TED-TTS）；CoT 长度 steering → 语速/停顿/强调、注意力级对比解码 → 跳字/重复、层跳/早退式自投机 → 替代需训练的 draft（SSD/PCG），均空白；也缺"控制强度 × SIM × WER × 延迟"的联合评测。

---

## 4. Recommended Ideas（ranked）

排序依据：gap 的证据强度 × 结果无论正负都有价值 × 单卡/API 可完成 × 与 2026 顶会已发表工作的区分度。Novelty 为同家族自评（provisional）。

### Idea 1：Why Compression Makes Models Talk Longer —— KV 压缩导致生成膨胀的归因与 TF 修复
- **Method（我们实际做什么）**：(1) 在 R1-Distill-Qwen-7B / Qwen3-8B 上跑 SnapKV、R-KV、KIVI-2bit、Expected Attention，逐样本记录生成长度、重复 n-gram 率、"答案出现后继续生成的 token 数"；(2) 用 token 级归因（删哪些 KV 导致膨胀）判定是"结论/计划锚点被驱逐"还是"自检循环"；(3) TF 修复：给答案候选/结论标记及邻域 KV 加保护位，并用重复熵触发 budget forcing 提前收尾；(4) 以端到端延迟（不是 token/s）汇报，并附早停 × 驱逐的联合 Pareto 面（原 Idea "Early-Stop × KV-Evict" 作为子实验）。
- **Hypothesis**：长度膨胀主要源于早期锚点被驱逐引发的重复自检；保护 <2% 的锚点 KV 可消除 >70% 的膨胀。
- **Minimum experiment**：AIME24/25、MATH500、LiveCodeBench 子集、LongBench 生成类；指标 准确率 / 平均长度 / 端到端延迟 / 峰值显存；单卡 A100-80G 约 40–60 GPU-h。
- **Expected outcome**：成功 = 归因定位到可解释的锚点类型 + 修复后延迟回到或低于无压缩；失败 = 膨胀与被驱逐 token 类型无关（则转为"压缩改变停止行为"的负结果诊断，仍可发）。
- **Novelty 7/10** — closest：[2503.24000](https://arxiv.org/abs/2503.24000)（发现现象未解释）、ThinKV [2510.01290](https://arxiv.org/abs/2510.01290)（量化拉长未归因）、BeaconKV（重访 token，目标是精度）。
- **Feasibility**：公开权重 + KVPress 库即可；无训练。
- **Risk**：MEDIUM。**Contribution type**：diagnostic + method。
- **Pilot result**：SKIPPED（本环境无 GPU）。
- **Reviewer's likely objection**：锚点保护可能等价于"保留首尾 token"技巧；现象是否在非推理模型上存在。
- **Why we should do this**：命中 G1 的核心矛盾，是整个压缩子方向都需要却没人做的事；正负结果都有价值。
- dedup_key：`kv-compression-output-length-inflation-anchor-protection`

### Idea 2：Steering Safety-Tax Audit + Refusal-Orthogonal Fix —— 非安全用途 steering 的安全税与 TF 修复
- **Method**：(1) 复现 Rogue Scalpel 协议（HarmBench/AdvBench 有害顺从率）；(2) 对五类非安全用途 steering（ASC CoT 压缩、COLD-Steer 行为、DPS 个性化、指令函数向量、MLLMEraser 式擦除）逐一测安全税；(3) 用对比法估计 refusal 子空间（1–3 维），把任意 steering 向量投影到其正交补再施加；(4) 报告"目标效果保留率 vs 安全税消除率"。
- **Hypothesis**：安全税主要来自 steering 向量在 refusal 方向上的非零投影；正交化可消除 >80% 安全税而目标效果损失 <10%。
- **Minimum experiment**：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct；HarmBench 子集 + 各 steering 原任务集；单卡约 15 GPU-h。
- **Expected outcome**：成功 = 正交化显著降税且保效果；失败 = 随机方向也有税（Rogue Scalpel 暗示非方向性因素），则结论为"steering 的安全税不可由线性修复消除"，仍是重要负结果。
- **Novelty 7/10** — closest：Rogue Scalpel [2509.22067](https://arxiv.org/abs/2509.22067)、Safety Pitfalls [2603.24543](https://arxiv.org/abs/2603.24543)（只诊断不修）。
- **Feasibility**：最轻量的一条，纯推理。
- **Risk**：MEDIUM。**Contribution type**：diagnostic + defense。
- **Pilot result**：SKIPPED。
- **Objection**：正交化会削弱 steering 本身；refusal 子空间可能非线性。
- **Why**：steering 正被当作"免训练万能旋钮"推广（CoT 压缩、个性化、擦除），而三篇 2025–26 论文证明它有安全税且无人修——时间窗口正好。
- dedup_key：`steering_safety_tax_refusal_orthogonal`

### Idea 3：ConfMap —— 置信选择的"难度 × 模型 × 置信定义"失效相图
- **Method**：(1) 固定三种置信定义（DeepConf bottom-10% group conf、CGES 答案后验、Consilience 时间曲线）；(2) MATH-500 / AIME24-25 / GPQA-Diamond / LiveCodeBench 按 pass@1 分 5 个难度桶；(3) 每桶算置信-正确 AUROC、oracle 选择 vs 置信选择 vs 多数投票差距；(4) 加入 RLVR 轴：同一基座的 base / SFT / RLVR 检查点（原 Idea "RLVR-Calibration Dependence" 并入）；(5) 产出一条简单切换规则。
- **Hypothesis**：置信选择的增益集中在 pass@1 ∈ [0.2, 0.7] 的中等难度桶；最难桶 AUROC → 0.5 并被多数投票反超；RLVR 更重的模型相图整体右移。
- **Minimum experiment**：Qwen3-4B/8B、R1-Distill-7B、GPT-OSS-20B；N=32，约 80K 条轨迹，vLLM 单卡 40–60 GPU-h。
- **Expected outcome**：一张可操作相图 + 对三篇顶会矛盾的统一解释。
- **Novelty 6/10** — closest：DeepConf、CGES（Thm 3.5 承认失效但未画相图）、Consilience、TTSV（观察到 Qwen/LLaMA 差异未解释）。
- **Feasibility**：纯采样 + 离线分析，可用 API。
- **Risk**：LOW。**Contribution type**：diagnostic + rule。
- **Pilot result**：SKIPPED。
- **Objection**："benchmark 分析不是方法"；难度桶由 pass@1 定义与置信相关。
- **Why**：子方向已饱和到审稿人对新信号疲劳，但"什么时候信号失效"没人回答；低风险、高引用潜力。
- dedup_key：`tts_confidence_difficulty_phase_map`

### Idea 4：PruneDraft —— 用"剪掉视觉 token 的自己"做 VLM 的无训练无损自投机
- **Method**：(1) 草拟阶段用深层视觉 token 剪枝（保留 10–20%）+ 浅层早退的同一模型生成 k 个 token；(2) 验证阶段用全 KV 并行校验（标准 SD，无损）；(3) ViSkip 式视觉相关性门控 + SparseSpec-L 式熵控制器自适应 k；(4) 在 MMSpec 六类任务上报 wall-time speedup 与 p99。
- **Hypothesis**：视觉 token 占 VLM KV >80% 且深层信息"消失"，剪枝版自草拟接受率 >0.8，整体 1.6–2.2×，优于 Lookahead (1.07×)。
- **Minimum experiment**：Qwen2.5-VL-7B、LLaVA-1.5-7B；MMSpec + Video-MME 短答；单卡 48–80G 约 30 GPU-h。
- **Expected outcome**：成功 = 接受率与加速比达标；失败 = 草拟仍需全层使上限受限（则缩为"视觉剪枝作为 drafter 的接受率分析"）。
- **Novelty 7/10** — closest：SparseSpec-L [2607.27735](https://arxiv.org/abs/2607.27735)（文本）、MMSpec/ViSkip（门控不草拟）、LayerSkip（需训练）。
- **Risk**：MEDIUM。**Contribution type**：method。
- **Pilot result**：SKIPPED。
- **Objection**：与 EAGLE 类训练 drafter 相比速度可能不占优。
- **Why**：三条已发表结论（MMSpec 失效、SparseSpec-L 可行、深层视觉 token 可丢）拼起来就是它，窗口期短。
- dedup_key：`visual-token-pruned-self-speculative-decoding-vlm`

### Idea 5：DPE-for-Video —— 视频 MLLM 的逐维位置重映射（不驱逐也能看更多帧）
- **Method**：(1) 在 Qwen2.5-VL-7B / LLaVA-OneVision-7B 上，用 DPE 的注意力贡献度量分别对 M-RoPE 的时间/空间维度估计"有效长度"；(2) 只重映射关键时间维度，把可用帧从 ~64 扩到 ≥512，不做 KV 驱逐；(3) 与 FlexMem/StreamMem 比较并测"重映射 + 轻度压缩"的叠加；(4) 对比视频与文本的关键维度分布。
- **Hypothesis**：视频时间维度"训练不足"的维度比例高于文本，DPE 式重映射收益更大且与 KV 压缩正交。
- **Minimum experiment**：Video-MME、LongVideoBench、MLVU；单卡 3090/4090 约 30 GPU-h。
- **Novelty 7/10** — closest：DPE [2504.18857](https://arxiv.org/abs/2504.18857)、MrRoPE（文本）；FlexMem、StreamMem（压缩不改位置）。
- **Risk**：MEDIUM（M-RoPE 多维结构使估计不直接；基准对帧数饱和）。**Contribution type**：method-transfer + analysis。
- **Pilot result**：SKIPPED。
- **Objection**：收益可能被更简单的帧采样抵消——必须与均匀/自适应采样基线比。
- **Why**：视频记忆工作全把位置当常量，这是 G4 中最干净的一次"A 域成熟方法 → B 域未试"。
- dedup_key：`dimensionwise_rope_video_mllm`

### Idea 6：Reflect-or-Not —— 反思 token 何时有用的 2×2 因果分解
- **Method**：(1) 在每个 "Wait" 处强制补全得到中间答案；(2) 标注该反思为有益/有害/冗余；(3) 按"难度 × 中间答案正确性"做 2×2 统计；(4) 得到条件压制规则（仅在"中间答案已对且熵低"时压制），与 NoWait/CGRS/α1 比。
- **Hypothesis**：反思只在"中间答案错 + 中难题"象限净有益；条件压制同时超过 NoWait 的节省与 α1 的精度。
- **Minimum experiment**：R1-Distill-7B、Qwen3-8B；MATH-500/AIME/GPQA；单卡约 20 GPU-h。
- **Novelty 7/10** — closest：NoWait [2506.08343](https://arxiv.org/abs/2506.08343)、CGRS、α1、underthinking [2501.18585](https://arxiv.org/abs/2501.18585)。
- **Risk**：MEDIUM（CGRS 已部分做"熵高才允许反思"）。**Contribution type**：analysis + method。
- **Objection**："CGRS 的增量"——要靠 2×2 发现本身撑起。
- dedup_key：`tts_reflection_token_value_decomposition`

### Idea 7：Leakage-Stratified CL —— 预训练泄漏能解释多少 training-free 方法的优势？
- **Method**：(1) 用 CLIP/DINO 特征对 ImageNet-21K 与 LAION 子集做近重复检索（复用 Data Leakage 的 hard/soft 定义），给 CIFAR-100 / ImageNet-R / CUB / Cars / ObjectNet / 2025 后新数据的每个类打"泄漏分"；(2) 同一 ViT-B/16 上跑 RanPAC、Fly-CL、SimpleCIL（TF）与 CODA-Prompt、SD-LoRA、EWC-LoRA（训练式）；(3) 按泄漏分分层报告 TF−trained 差距；(4) 构造"零泄漏"子集作为新协议。
- **Hypothesis**：TF 优势随泄漏分单调增加；零泄漏层训练式方法反超 ≥3 pts。
- **Minimum experiment**：约 30–40 单卡小时。
- **Novelty 7/10** — closest：CLDyB [2503.04655](https://arxiv.org/abs/2503.04655)、Data Leakage [2508.17416](https://arxiv.org/abs/2508.17416)、PAM [2603.21170](https://arxiv.org/abs/2603.21170)——各做一半。
- **Risk**：MEDIUM（泄漏分与域相似度混淆，需 DINO 域距离做协变量）。**Contribution type**：diagnostic + protocol。
- **Objection**：近重复阈值任意；泄漏 ≠ 记忆。
- dedup_key：`cl_leakage_stratified_tf_vs_trained`

### Idea 8：KL-Razor for Vision & Merging —— output-KL 是否也是 PTM-CL / 模型合并的遗忘预测量
- **Method**：(1) ViT/CLIP 的 CIL + MTIL 上跑 8 种方法（CODA、SD-LoRA、EWC-LoRA、SLAO、MagMax、SFA、TA、全微调）；(2) 每步记录新任务 KL-to-base、Δ-norm、稀疏度、梯度秩、intruder-dimension 数；(3) 回归实测遗忘；(4) 把 KL 作为 MagMax/SFA 的合并系数准则做干预。
- **Hypothesis**：KL 在视觉上同样是最强预测量（r>0.8，权重量 r<0.5）；KL 选系数比固定系数遗忘 ↓20%。
- **Minimum experiment**：ViT-B/16；ImageNet-R 10 任务 + MTIL；约 40–60 单卡小时。
- **Novelty 7/10** — closest：RL's Razor [2509.04259](https://arxiv.org/abs/2509.04259)（仅 LLM、RL vs SFT）、MagMax、LoRA intruder dims (NeurIPS'25)。
- **Risk**：MEDIUM（class-incremental 下输出头变化使 KL 定义需用特征空间/logit 子集）。**Contribution type**：diagnostic + criterion。
- **Objection**："KL 与性能循环定义"——用新任务 KL 预测旧任务遗忘规避。
- dedup_key：`kl_forgetting_predictor_vision_merging`

### Idea 9：One Stream, Two Communities —— CTTA 与 TF-CIL 在同一 CLIP 数据流上的统一
- **Method**：(1) 构造 CLIP 数据流：ImageNet-C 渐变 corruption × 类增量到达 × 标签偏移；(2) 跑 CTTA 侧 UL-TTA、MS-TTA、PTA、TPS 与 CL 侧 RanPAC、RAIL、Continual-CLIP；(3) 把两族都写成"原型 + 统计量更新"统一式，差异仅在更新规则；(4) 提出混合更新规则。
- **Hypothesis**：两族各自崩在对方擅长的维度；统一"原型 + 协方差 + 先验"闭式更新同时拿到两边 ≥90% 的最优值。
- **Minimum experiment**：CLIP ViT-B/16；全 TF，<15 单卡小时。
- **Novelty 7/10** — closest：TTABC [2606.14299](https://arxiv.org/abs/2606.14299)（只有 TTA 族）、BayesTTA、RAIL。
- **Risk**：LOW。**Contribution type**：diagnostic + unified framework。
- **Objection**："两个社区问题定义不同对比不公平"——需多种流配置。
- dedup_key：`ctta_cil_unified_clip_stream`

### Idea 10：Trajectory-Consensus Probe —— 把 Parallel-Probe 的跨分支共识搬到多步 agent
- **Method**：(1) 并行 K 条 agent 轨迹；(2) 每 T 步对各轨迹"状态摘要"（页面/文件/工具调用序列哈希 + 自述子目标）聚类；(3) 共识簇占比超阈值即停其余分支、把算力转给共识簇；(4) 与 CATTS（步级）、均匀 best-of-K、AggAgent 比。
- **Hypothesis**：轨迹级共识早停能恢复 agent 上并行扩展的增益（General AgentBench 说目前没有），且比均匀 K 省 ≥40% token。
- **Minimum experiment**：WebArena-Lite 或 SWE-bench Verified 50 题；Qwen3-32B 或 API；K=8；约 $100 API 或 40 GPU-h。
- **Novelty 7/10** — closest：Parallel-Probe [2602.03845](https://arxiv.org/abs/2602.03845)、CATTS [2602.12276](https://arxiv.org/abs/2602.12276)、AggAgent [2604.11753](https://arxiv.org/abs/2604.11753)。
- **Risk**：HIGH（状态摘要难对齐；agent 评测噪声大）。**Contribution type**：method-transfer。
- **Objection**："agent 轨迹不可比较，共识定义模糊"。
- dedup_key：`tts_agent_trajectory_consensus_probe`

### Idea 11：ReconVis —— 一次压缩多问复用的 query-agnostic 视觉 KV 驱逐
- **Method**：(1) 图像/视频 prefill 后，用 MLLM 自身以"描述这张图"为 pseudo-query 做一次重建前向，取重建注意力作为视觉 KV 的 query-agnostic 重要性（KVzip 迁移）；(2) 低分视觉 KV 做 GRKV 式岭回归合并而非丢弃，保留"迟发重要性"通道；(3) 压缩 cache 作为 prefix 被同图多问复用；(4) 对比 query-aware 剪枝在多问设置的退化。
- **Hypothesis**：同图多问下，20% 视觉 KV 预算时比 VisionZip/SparseVLM 高 ≥5 点并允许 prefix 命中。
- **Minimum experiment**：Qwen2.5-VL-7B、LLaVA-OneVision-7B；MMBench/SEED 同图多问重组、VisDial、Video-MME 多问子集；约 30 GPU-h。
- **Novelty 7/10** — closest：KVzip [2505.23416](https://arxiv.org/abs/2505.23416)、RetentiveKV (ACL'26)、FlashCache。
- **Risk**：MEDIUM。**Contribution type**：method。
- **Objection**：重建前向多一次 prefill；多问基准是重组而非原生。
- dedup_key：`query-agnostic-visual-kv-reconstruction-multiquery`

### Idea 12：AR Speech-LM Prosody Steering + Attention-Contrastive Decoding —— LLM-TTS 的 LM 侧 training-free 控制与去幻觉
- **Method**（两条可拆分的子线）：(a) 在 CosyVoice2-0.5B 的 LM 隐状态上按 ASC 法抽"快→慢""平→强调"向量，按 token 段施加（复用 TED-TTS 的 segment 思路），评 WER/SIM/时长误差/UTMOS；(b) 定位文本-语音对齐注意力头，按 ACD/LayerCake 思路构造"受损"前向做对比解码治跳字/重复，在 Seed-TTS-eval hard 集测 WER 插入/删除率。
- **Hypothesis**：语义 token LM 内存在与文本"CoT 长度方向"同构的"时长方向"；跳字/重复对应对齐头注意力崩塌，注意力对比可在不伤 SIM 下把 hard 集 WER 降 15–30%。
- **Minimum experiment**：CosyVoice2-0.5B（开源）；Seed-TTS-eval zh/en + LibriTTS；每条子线单卡 10–12 GPU-h。
- **Novelty 7/10** — closest：EmoSteer [2508.03543](https://arxiv.org/abs/2508.03543)（flow-matching 侧）、TED-TTS [2601.03170](https://arxiv.org/abs/2601.03170)（logit/embedding）、ACD (ICML'26, VLM)、Selective CFG [2509.19668](https://arxiv.org/abs/2509.19668)。
- **Risk**：MEDIUM（时长可能主要编码在声学/flow 阶段；跳字可能来自 tokenizer）。**Contribution type**：method-transfer。
- **Objection**：与 CosyVoice 指令模式 / CFG / 重复惩罚相比优势何在——需对比并展示互补。
- dedup_key：`ar_speech_lm_prosody_steering` + `tts_attention_contrastive_decoding_hallucination`

---

## 5. Backlog：其余候选（按主题，均通过去重，未进 top-12）

| # | Idea | 主题 | Novelty | Risk | GPU-h | 一句话 |
|---|---|---|---|---|---|---|
| B1 | Latency-Honest Early Exit（批同步、零额外前向的退出 + 延迟审计） | G1 | 6 | LOW | 30 | 复现 DEER/RPDI/CoDE-Stop/ASAG 报四项延迟指标，设计批友好变体；Idea 1 的 TTS 侧孪生 |
| B2 | Oracle-Gap 评测协议（"决策点"作 early-exit 上界） | G1/G3 | 5 | LOW | 25 | 用 [2604.22266](https://arxiv.org/abs/2604.22266) 的强制补全定义 oracle 退出点，报 premature rate / overshoot |
| B3 | Matched-Control Audit for LLM TTS Selectors | G3 | 5 | MED | 40 | 把 [2608.01207](https://arxiv.org/abs/2608.01207) 的受控审计搬到 DeepConf/Consilience/RPDI/ST-BoN |
| B4 | Low-Budget Regime Study（N≤4、≤8K token 下哪种 TF 策略最优） | G4 | 5 | LOW | 30 | NoThinking×N vs 单条长 thinking+early exit vs 短 CoT+自纠正，compute-matched |
| B5 | Cross-Trajectory Hidden-State Convergence 作全局停止信号 | G1 | 6 | MED | 20 | ST-BoN 扩展：隐状态收敛先于答案收敛，零生成成本 |
| B6 | TF 预算/强度统一旋钮（Budget-from-Memory kNN + 口头化难度画像） | G5 | 5–6 | MED | 20 | 近邻记忆给 N 先验；VDAR 式难度画像控制插值 α / steering 强度 / zoom |
| B7 | Rotation-Aware KV Eviction（驱逐打分在 W4A4 旋转量化模型上的失效与修复） | G4 | 6 | LOW | 25 | 首次系统评测权重旋转量化 × KV 驱逐交互 |
| B8 | ToolKV（工具调用 agent 的压缩基准 + 段感知预算） | G4 | 6 | LOW | 30 | BFCL/τ-bench 下 schema 段冻结 + observation 段激进压缩 |
| B9 | CliffGuard（每请求自适应压缩率，控制在幻觉悬崖之下） | G1/G4 | 6 | MED | 40 | 用 Global Eviction Ratio / 头一致性熵做在线代理 |
| B10 | MoE-KV（MLA/MoE 上 KV 驱逐 × 专家跳过联合） | G4 | 6 | MED | 40 | Qwen3-30B-A3B / DeepSeek-V2-Lite；驱逐 token 与跳过专家是否同一批 |
| B11 | Depth-Switch Pruning（按"信息地平线"切换注意力剪枝/均匀剪枝） | G2 | 5 | LOW | 20 | 工作坊级；检验地平线随分辨率/帧数前移 |
| B12 | Video Token Pruning Random-Baseline Audit | G3 | 5 | LOW | 25 | 视频侧的 [2512.07580](https://arxiv.org/abs/2512.07580)；负结果风险 |
| B13 | Unified Forgetting Ledger（ICL/RAG/LoRA-CL/编辑/sparse-memory 的三维遗忘 + 等预算） | G5 | 7 | HIGH | 45 | 价值高但工作量大，需裁剪到 2 域 |
| B14 | Audited LoRA Merging Benchmark（审计前后 TIES/DARE/TA 排名） | G3 | 5 | MED | 25 | 扩展 Audit Before You Merge 到合并算法比较 |
| B15 | Decomposing the Gain（训练式 PTM-CL 增益 = 分类头 + 特征适配 + 调参） | G3 | 6 | MED | 30 | 逐方法消融；分解不唯一 |
| B16 | Update-Count Scaling（10→10,000 次更新下 TF 闭式头 vs 单 LoRA 合并 vs 编辑） | G4 | 6 | MED | 50 | 视觉几乎 TF，语言 ~40h |
| B17 | MLLM-CL 可分性跨基准审计 + 闭式去相关路由（两阶段） | G3/G5 | 5–6 | MED | 30 | 先证 CoIN/MLLM-CTBench 是否同样可分，再用 Fly-CL 式路由替代 MR-LoRA |
| B18 | Pretraining-Scale Sweep（预训练数据量 → TF-CL 优势） | G4 | 6 | MED | 60 | 骨干不可控是主要风险 |
| B19 | Merge-vs-Steer 等价性（插值 α 与 steering λ 是否同一条轴；含规模/MoE 外推） | G5 | 6 | MED | 25–40 | 若 cos 关系平凡则只是经验表 |
| B20 | Merge Predictability（steering 几何指标 → 逐层 α 选择） | G3/G5 | 6 | MED | 20 | 零搜索开销是卖点 |
| B21 | Head-Aware Many-Shot Compression（函数向量替代均匀逐层压缩） | G5 | 6 | MED | 12 | FV 在生成式任务上弱 |
| B22 | Agent-Trace KV-Overlap Measurement（KV 共享收益上界） | G4 | 5 | LOW | 0–5 | 纯 CPU 分析，analysis paper |
| B23 | Steering under Context Extension（向量在 DPE/MrRoPE 扩展后的漂移与重校准） | G2 | 6 | LOW | 15 | 无漂移亦可作 finding |
| B24 | Cross-Tokenizer Training-free Merge（OMP 移植 + 异构块合并） | G4 | 6 | HIGH | 15 | 合适模型对太少 |
| B25 | Early-Stop × KV-Evict 联合 Pareto | G1 | 4 | LOW | 20 | 已并入 Idea 1 作子实验 |
| B26 | 插值三阶段规模律 | G4 | 5 | LOW | 15–40 | 已并入 B19 |

---

## 6. Eliminated / 不建议投入

| 方向或 idea | 原因 |
|---|---|
| Training-free latent / soft reasoning | 唯一 TF 方案 Soft Thinking [2505.15778](https://arxiv.org/abs/2505.15778) 自认 concept token OOD、长链崩溃，根本解需要训练 |
| 又一个 KV 驱逐打分 / 又一个视觉 token 剪枝打分 / 又一个旋转 PTQ | 红海；且 [2512.07580](https://arxiv.org/abs/2512.07580) 已证深层不如随机 |
| 又一个置信/熵早停信号 | 2025-08 至 2026-06 ≥15 篇；审稿人疲劳 |
| 又一个 ViT prompt-pool CIL 方法 / 又一个 MLLM 持续指令微调 LoRA 路由 | 被 TF 基线打平；基准被诊断无效 |
| 终身知识编辑（ROME/MEMIT 系）的新正则 | WikiBigEdit 已宣判"250 次内趋零、不如持续 LoRA" |
| Activation steering 的新方向估计法（在 CAA/BiPO 小基准比点数） | 饱和；且可靠性/安全税问题未解，应先做 Idea 2 |
| 文本 DoLa 系对比解码变体 | LayerCake 撤回；视觉侧更活跃 |
| 多 agent 协调的 TF 调度 | Tran & Kiela 显示算力匹配后单 agent 不输；先做 B22 测前提 |

---

## 7. Suggested Execution Order

1. **Idea 2（Steering 安全税）或 Idea 3（ConfMap）先行**：两者最轻（15 / 40 GPU-h 或 API）、风险最低、正负结果都能发，且都对应"顶会结论互相矛盾"的 gap。
2. **Idea 1（压缩拉长输出）**：效率方向的旗舰；若归因成功可直接带出 TF 修复方法。
3. **Idea 4（PruneDraft）与 Idea 5（DPE-for-Video）**：多模态方向两条方法迁移，窗口期短，建议 1–2 个月内出 pilot。
4. **CL 三条（Idea 7/8/9）**：Idea 9 全 TF 最便宜，可作为 CL 方向的第一篇；Idea 7/8 更像"改写社区共识"的诊断论文。
5. **Idea 6 / 10 / 11 / 12**：按团队背景选；Idea 12 需要语音栈。

---

## 8. Next Steps

- [ ] 在可访问 arXiv 的环境重跑 `tools/arxiv_fetch.py` / `verify_papers.py`，核实四份 lens 报告中标 `⚠` / `[片段]` / `[未核实-记忆]` 的条目（尤其 RetentiveKV、ViSkip 是否 TF；LU-KV/MixedDimKV 的 profiling 需求；InfLoRA/SimpleCIL/ZSCL 的 ID；AdaBeam/GEPA 数字）。
- [ ] 对 top-5 运行 `/novelty-check`（重点查 2026-07 之后的 arXiv，这是本次搜索最薄弱的窗口）。
- [ ] 运行 `/research-review — reviewer: codex — difficulty: hard` 补上缺失的跨模型 jury，把 `acceptance_status` 从 provisional 升级。
- [ ] 为选中的 1–2 条调用 `/research-refine-pipeline` 生成 `FINAL_PROPOSAL.md` 与 `EXPERIMENT_PLAN.md`；pilot 按 PILOT_MAX_HOURS=2 先跑最小版本（Idea 2 的单模型 HarmBench 子集；Idea 3 的单模型 MATH-500 N=16）。
- [ ] 补搜本次未覆盖的：ACL/CVPR 2026 主会 CL 论文、OPCM/MINGLE 持续合并、Parametric RAG、MemoryLLM/Cartridges、CTTA@NeurIPS'25、2026-07 后的 KV 压缩与 TTS 新作。

---

## 附：Lens 报告索引

| 文件 | 内容 |
|---|---|
| `idea-stage/lens/lens_efficiency.md` | 60 篇效率论文卡片（8 个子方向表）、饱和度、5 lens × 2–5 gap、10 idea |
| `idea-stage/lens/lens_tts.md` | 55 篇 TTS 论文卡片（8 个子方向）、饱和度表、5 lens × 2–3 gap、10 idea |
| `idea-stage/lens/lens_cl.md` | 70 篇 CL 论文卡片（8 个子方向）、饱和度表、5 lens × 2–4 gap、10 idea |
| `idea-stage/lens/lens_other.md` | 64 篇其他方向卡片（9 个方向）、饱和度表、5 lens × 4 gap、12 idea |
