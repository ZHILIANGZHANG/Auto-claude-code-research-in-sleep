# Research Idea Report（v2，核实后重审版）：Training-free / 少训练 / 低资源的 LLM-MLLM 研究空间

**Direction**：LLM/MLLM 推理效率、Test-Time Scaling、Continual Learning，以及其他不依赖（大规模）训练、单卡或 API 即可完成的研究方向——哪里还有空间、方向是什么、潜在 idea 有哪些。
**Generated**：2026-10-17（v1）→ **v2 同日核实后重审**
**Pipeline**：ARIS `/research-lit` → `/idea-creator` Phase 1.5 lens fan-out（4 子方向 × 5 lens）→ Phase 3 机械去重 → **人工核实（Step 1.5 替代：arxiv.org/abs 逐篇 + CrossRef）→ 针对 top-12 的 novelty 探测（11 次 WebSearch + 21 篇新论文 abs 核实）→ 重排与重定位** → Phase 6 报告。
**Ideas evaluated**：42 generated → 38 after dedup → **核实后：3 条淘汰（被 2026 H2 论文覆盖）、6 条降级、2 条从 backlog 升级** → 10 recommended + 24 backlog + 4 eliminated

> **v2 相对 v1 的变化（必读）**
> - 四份 lens 报告（`idea-stage/lens/`）已逐篇核实：共 ~200 个 arXiv ID 抓 abs 页，关键数字抓 html grep；每份文件顶部新增「§0.5 核实记录」。修正包括：2 篇撤回论文（STU-PID 2506.18831 → 改引 2510.04309；PulseFocus 2603.04676）、2 个错误 ID、若干错误标题/数字、**4 篇 v1 写成 ICML/NeurIPS 主会的实为 workshop**（Audit Before You Merge、Is Our Benchmark Enough = ICML'26 CATS WS；CF-Router = ICLR'26 WS；GraphMemory = NeurIPS'26 TTCL WS）、3 篇 training-free 判断错误（RetentiveKV、DraftExpert、Prosodic Streaming 实为需训练）。
> - **仍不可用**：arXiv API / Semantic Scholar / OpenAlex 因出口 IP 限速返回 429，所以 `verify_papers.py` 的 S2 层与 `arxiv_fetch.py search` 没跑；本次核实靠 abs 页人工替代。Codex 跨模型 jury 仍不可用，排名为同家族自评，`acceptance_status: provisional`。
> - 核实与 novelty 探测的全部原始记录：`idea-stage/lens/*.md` §0.5 与 `idea-stage/novelty_probe_notes.md`。

---

## 0. 一页结论（TL;DR）

**v1 判断中被核实推翻或削弱的三点**
1. **语音 LLM-TTS 的 AR LM 侧已不是空白**：2026-08~09 连出 ECCD（[2608.00722](https://arxiv.org/abs/2608.00722)，training-free 对比解码治幻觉，CosyVoice2/3、WER −55.6%）、语速激活 steering（[2609.33810](https://arxiv.org/abs/2609.33810)，SLT'26，含 CosyVoice2）、口音中和 steering（[2603.05977](https://arxiv.org/abs/2603.05977)，SLT'26）、EmoShift/CoCoEmo。v1 的 Idea 12 两条子线均被覆盖，**淘汰**。
2. **Steering 安全税的缓解已被做掉**：[2608.08383](https://arxiv.org/abs/2608.08383)（COLM'26，primal-dual 找单方向消融）、OPIUM [2607.19806](https://arxiv.org/abs/2607.19806)、[2605.01167](https://arxiv.org/abs/2605.01167)、Forecasting Side Effects [2608.11227](https://arxiv.org/abs/2608.11227)；且 Safety Pitfalls 本身已测过"正交投影只减 15–25%"。v1 的 Idea 2 **淘汰**。
3. **"KV 压缩拉长输出"的现象已被发表**（ReCo [2608.04771](https://arxiv.org/abs/2608.04771)：R-KV 25% 压缩使 R1-Distill-Qwen-7B 长度 +38.8%、Llama-8B +79.0%；RLKV v3 [2510.08525](https://arxiv.org/abs/2510.08525) 同样观察），但 **ThinKV（ICLR'26 Oral）全文明确说"驱逐不导致长度增加、只有量化导致"**——两组顶会结论直接矛盾，机制无人解释、TF 修复无人做。v1 的 Idea 1 从"发现现象"**重定位为"解决矛盾 + 机制 + TF 修复"**，仍是效率方向的首选。

**仍成立的红海清单**（同 v1）：单图视觉 token 剪枝、prefill KV 驱逐与驱逐×量化联合、旋转式 PTQ、置信/熵早停与自适应 self-consistency、ViT prompt-pool CIL、MLLM 持续指令微调 LoRA 路由、activation steering 方法本体、文本 DoLa 系。**新增红海**：steering 安全税缓解、LM-TTS 的 TF 控制/去幻觉、反思 token 调度（CyclicReflex ICLR'26、Reflection Steering、ARLCP ICLR'26）。

**核实后仍有空间的主题**（按证据强度排序）
1. **矛盾裁决类**：驱逐是否拉长输出（ThinKV vs ReCo/RLKV）；置信选择在难题上有效还是崩溃（DeepConf vs CGES vs Consilience）；attention 剪枝 vs 随机（HoloV/TOPS vs 2512.07580）。
2. **跨社区统一类**：output-KL vs 权重空间量作为遗忘预测量（RL's Razor 只在 LLM/机器人 FM 验证，视觉 CIL 与合并未验证）；CTTA vs TF-CIL；预训练泄漏 ↔ TF-CL 优势。
3. **新 regime 迁移类**：工具调用 agent 的 KV 压缩；KV 驱逐打分在 W4A4 旋转量化模型上；同图多问的 query-agnostic 视觉 KV；MLA/MoE 上的压缩；视频 M-RoPE 的逐维处理。

**核实后 Top-5**（全部单卡 ≤60 GPU-h）：
Idea 1 *Eviction or Quantization: Who Makes Reasoning Models Talk Longer?* → Idea 2 *KL-Razor for Vision & Merging* → Idea 3 *Leakage-Stratified CL* → Idea 4 *ConfMap* → Idea 5 *One Stream, Two Communities*。

---

## 1. 方法与来源（v2 增补）

| 项目 | 内容 |
|---|---|
| Lens 子代理（v1） | ① 效率 60 篇 ② TTS 55 篇 ③ CL 70 篇 ④ 其他 64 篇；WebSearch-only |
| 核实（v2） | 4 路子代理，`curl arxiv.org/abs/<id>`（限速 3s）+ `arxiv.org/html` grep + CrossRef；共 ~200 ID；每篇 ✅/✏️/❌/⚠ |
| Novelty 探测（v2） | 主代理对 top-12 各做 1 次 extended WebSearch，对发现的 21 篇候选撞车论文抓 abs 核实；记录在 `idea-stage/novelty_probe_notes.md` |
| 未能核实 | 无 arXiv ID 的 workshop/会议论文（WAVE、FLoC、HAWK、SepPrune、TagRouter、RAS、TokenDance 等）；OpenReview 单条接口被反爬拦截；abs 未写录用状态的会议归属 |
| 去重结果 | 同 v1（42→38）；v2 另将 B11 Depth-Switch（被 2512.07580 覆盖）、B17 闭式 MLLM 路由（被 CF-Router 覆盖）标为淘汰 |

---

## 2. Landscape Summary（v2 修订处用 **[v2]** 标出）

### 2.1 LLM / MLLM 推理效率（training-free）

KV cache 侧已是"驱逐 × 量化 × 合并"红海（KVzip [2505.23416](https://arxiv.org/abs/2505.23416) NeurIPS'25、Expected Attention [2510.00636](https://arxiv.org/abs/2510.00636)、GRKV [2605.31105](https://arxiv.org/abs/2605.31105) EMNLP'26、RDKV/HqeKV/CAPKV…）。真正新的是三条诊断线：(a) 压缩与生成长度的关系——**[v2]** MLSys'25 [2503.24000](https://arxiv.org/abs/2503.24000) 与 ReCo [2608.04771](https://arxiv.org/abs/2608.04771)、RLKV [2510.08525](https://arxiv.org/abs/2510.08525) 说驱逐拉长，ThinKV [2510.01290](https://arxiv.org/abs/2510.01290)（ICLR'26 Oral，html 核实）说"驱逐不拉长、纯量化最高拉长 5.1×"；(b) 压缩的"物理"：~90% 处幻觉悬崖（[2603.01426](https://arxiv.org/abs/2603.01426)），固定预算次优（Sparse Frontier [2504.17768](https://arxiv.org/abs/2504.17768)），**[v2]** Compactor [2507.08143](https://arxiv.org/abs/2507.08143) 已提出"context-calibrated compression"推断每上下文可承受的压缩率；(c) 多指令提示下压缩选择性遗忘指令（Pitfalls, ACL'26 main [2510.00231](https://arxiv.org/abs/2510.00231)）。解码期 KV（R-KV、BeaconKV [2609.04971](https://arxiv.org/abs/2609.04971) ICML'26）揭示"重要性回流"。

视觉 token 剪枝在单图饱和且被 CVPR'26 [2512.07580](https://arxiv.org/abs/2512.07580) 根本性质疑——**[v2]** 其摘要已包含"深层随机 + 浅层 DART/DivPrune 的混合策略"，所以"按地平线切换"不再是 idea。长视频/流式、多模态 KV（MadaKV、FlashCache [2511.16786](https://arxiv.org/abs/2511.16786) CVPR'26；**[v2]** RetentiveKV [2605.04075](https://arxiv.org/abs/2605.04075) ACL'26 Findings 含可学习门控、非严格 TF）、剪枝 × PTQ（QAPruner [2604.02816](https://arxiv.org/abs/2604.02816)）刚起步。投机解码：MMSpec [2603.14989](https://arxiv.org/abs/2603.14989) 证明文本 SD 在 VLM 上失效（Lookahead 1.07×）；**[v2]** SpecVLM [2508.16201](https://arxiv.org/abs/2508.16201)（EMNLP'25 main）已证明"draft 对视频 token 剪枝不敏感"并做了 training-free SD（但用独立小 Vid-LLM 作 draft）；SparseSpec-L [2607.27735](https://arxiv.org/abs/2607.27735) 用稀疏 KV 的自己作 drafter（文本）；**[v2]** DraftExpert 实为需训练。

**蓝海（核实后仍成立）**：驱逐/量化 × 生成长度的机制；工具调用/多轮 agent 的压缩；驱逐打分在 W4A4 旋转量化模型上的迁移；MLA/MoE 上的 KV 压缩；同图多问的 query-agnostic 视觉 KV；"draft-model-free"的 VLM 自投机。

### 2.2 Test-Time Scaling（training-free）

并行采样 + 置信选择（DeepConf [2508.15260](https://arxiv.org/abs/2508.15260) ICLR'26、CGES [2511.02603](https://arxiv.org/abs/2511.02603)、SeerSC、Parallel-Probe [2602.03845](https://arxiv.org/abs/2602.03845)、Consilience [2608.09898](https://arxiv.org/abs/2608.09898)）与单轨迹 early-exit（DEER [2504.15895](https://arxiv.org/abs/2504.15895)、CGRS [2508.05337](https://arxiv.org/abs/2508.05337)、RPDI-EE、CoDE-Stop、ASAG [2606.15070](https://arxiv.org/abs/2606.15070)）饱和。**[v2]** "token ≠ 延迟"的审计论点已被 SpecExit [2509.24248](https://arxiv.org/abs/2509.24248) 与 PUMA [2605.17672](https://arxiv.org/abs/2605.17672) 发表（证明 DEER 端到端变慢），只剩"零额外前向的批同步退出"这一设计增量。**[v2]** 反思 token 调度已成小红海：CyclicReflex [2506.11077](https://arxiv.org/abs/2506.11077) ICLR'26、Reflection Steering [2608.25542](https://arxiv.org/abs/2608.25542)、ARLCP [2602.12113](https://arxiv.org/abs/2602.12113) ICLR'26、"thinking trap" [2506.23840](https://arxiv.org/abs/2506.23840)；CGRS 摘要即"当前回答高置信时压制反思"。**[v2]** "决策点"探针与 probe-based 早停已在 [2604.22266](https://arxiv.org/abs/2604.22266) 做了简单版。

置信信号的失效条件：**[v2]** 新证据支持"RLVR 轴"——[2609.32470](https://arxiv.org/abs/2609.32470)（LRM 的 verbalized confidence prior 集中于少数高值并贯穿 RL）、EACL'26 "Think Just Enough"（置信控制在 RL 模型上更伤精度）、"Variation in Verification"（ICLR'26，对 verifier 做了难度 × 模型相图）。跨方法、跨置信定义的统一相图仍无。

agent 侧：General AgentBench [2602.18998](https://arxiv.org/abs/2602.18998) 说并行/序列扩展均无增益；**[v2]** [2604.16529](https://arxiv.org/abs/2604.16529)（2026-04）用 rollout 结构化摘要 + Recursive Tournament Voting 做事后选择（Claude-4.5-Opus SWE-bench Verified 70.9→77.6）；FailFast [2608.03222](https://arxiv.org/abs/2608.03222) 训 0.6B 监视器做单轨迹早停。在线跨轨迹共识早停仍未见。

### 2.3 Continual Learning（training-free / 少训练）

ViT-CIL 上 TF 解析式方法（RanPAC [2307.02251](https://arxiv.org/abs/2307.02251)、Fly-CL [2510.16877](https://arxiv.org/abs/2510.16877) ICLR'26、AnaCP、LR-RGDA、VILA ICML'26）持续打平训练式方法；**[v2]** APER（IJCV）已因数据重叠提出新基准，必须引用。LLM 侧 RL's Razor [2509.04259](https://arxiv.org/abs/2509.04259)：遗忘由 output-KL 预测，权重量不行——**[v2]** 核实其已覆盖机器人基础模型，但视觉 CIL 与模型合并未覆盖；视觉合并线（MagMax、SFA **[v2]** v2 已改名 "Forget About It" 并改用 Qwen3-14B；**[v2]** Audit Before You Merge = ICML'26 CATS workshop）仍全靠权重空间启发式。

记忆/检索式 CL：**[v2]** "RAG or Learning?"（ACL'26 Findings，即 v1 的 Chronos）已在同一时间戳基准对比 continual FT / 编辑 / RAG 并报告遗忘 + 时间不一致；WikiBigEdit [2503.05683](https://arxiv.org/abs/2503.05683)（ICML'25）已画 1→500K 更新的编辑/WISE/RAG/LoRA 曲线；CL-Bench [2606.05661](https://arxiv.org/abs/2606.05661)、ContinualSkillBench [2608.03874](https://arxiv.org/abs/2608.03874) 说朴素 ICL 赢。"统一遗忘定义 + 等预算"仍缺但增量变小。**[v2]** MLLM-CL 的 TF 闭式路由已被 CF-Router（ICLR'26 workshop，MLLM 末层隐藏态 + RLS 闭式分类器选 LoRA）做掉；"Is Our Benchmark Enough"为 workshop 文。CTTA 与 CIL 仍未同台（TTABC [2606.14299](https://arxiv.org/abs/2606.14299) NeurIPS'26 D&B 只有 TTA 族；**[v2]** ARC [2405.14318](https://arxiv.org/abs/2405.14318) ICLR'25 是"CL 的 test-time training"，需列为相关工作）。

### 2.4 其他 training-free / 低资源方向（v2 修订）

| 方向 | v1 评级 | **v2 评级** | 核实后要点 |
|---|---|---|---|
| 合并·Thinking/Instruct 插值 | 活跃→趋饱和 | 趋饱和 | **[v2]** L2S [2503.20641](https://arxiv.org/abs/2503.20641) 已报告 1.5B–32B 规模相关性 + 按复杂度自适应长度；Revisiting v2 已补 Llama-3.1-8B |
| 合并·理论/可预测性 | 起步 | 起步（增量变小） | **[v2]** Submodule Linearity [2504.10902](https://arxiv.org/abs/2504.10902) 摘要已含"按子模块合并 + 最优权重闭式解" |
| Steering 安全税 | 活跃（反面研究刚起步） | **饱和（2026 H2）** | 2608.08383 COLM'26、OPIUM、2605.01167、2608.11227 |
| CoT 长度 steering | 活跃 | 活跃 | **[v2]** STU-PID 撤回 → 改引 2510.04309（ICLR'26）；Reflection Steering 2608.25542 |
| TF 长上下文扩展 | 活跃 | 活跃 | DPE [2504.18857](https://arxiv.org/abs/2504.18857)、MrRoPE [2601.22181](https://arxiv.org/abs/2601.22181)（录用信息 abs 未载）；**[v2]** 视频理解侧已有 INTP-Video-LLMs [2409.12963](https://arxiv.org/abs/2409.12963)（2024，TF 增帧）、GIRAFFE M-RoPE++ [2412.12735](https://arxiv.org/abs/2412.12735)（需训练）、FLEX [2602.14027](https://arxiv.org/abs/2602.14027)（视频生成，频率感知 RoPE） |
| Many-shot 压缩 / 函数向量 | 活跃 | 活跃 | **[v2]** 2505.12075 真实标题为 "Do different prompting methods yield a common task representation…"（NeurIPS'25）；AAAI'26 "Sensitivity-Aware Task Vectors for Many-Shot Multimodal ICL" 疑似与 B21 相近（未读全文） |
| TF 路由 / 难度估计 | 活跃 | 活跃 | **[v2]** ODAR [2602.23681](https://arxiv.org/abs/2602.23681) 已用难度估计在 Fast/Slow Agent 间路由 |
| 视频 KV-as-memory | 活跃 | 活跃 | FlexMem 真实标题 "Scaling the Long Video Understanding of MLLMs via Visual Memory Mechanism"；**[v2]** VisCache [2608.24063](https://arxiv.org/abs/2608.24063)（2026-08，prompt-aware） |
| 多图交错 TF | 起步 | 起步 | **[v2]** PulseFocus 已撤回，勿引 |
| **LLM-TTS（语音）TF 控制/去幻觉** | 起步 | **活跃→趋饱和** | ECCD 2608.00722、语速 steering 2609.33810、口音 2603.05977、EmoShift 2601.22873、CoCoEmo 2602.03420；PCG 2511.13732（SD 需 draft） |
| Agent TF 效率 | 起步 | 起步 | Tran & Kiela [2604.02460](https://arxiv.org/abs/2604.02460) 标题实为仅多跳推理 |

---

## 3. 跨方向结构性 gap（v2 收敛为五个主题）

### G1. 压缩/早停的真实代价与矛盾裁决（contradiction + diagnostic）
- **驱逐 vs 量化谁拉长输出**：ThinKV（驱逐不拉长、量化拉长 ≤5.1×）vs ReCo/RLKV/MLSys'25（R-KV 驱逐拉长 +38.8%/+79%）。差异可能来自模型族（R1-Llama vs R1-Qwen）、压缩率、驱逐策略（R-KV 冗余去重 vs ThinKV 思维感知）、是否统计"答案出现后的尾巴"。无人裁决、无机制、无 TF 修复（ReCo 用训练式 process-reward 估计器）。
- **early-exit 的延迟**：审计已被 SpecExit/PUMA 做；剩余是"零额外前向 + 批同步"设计。

### G2. 信号失效相图（contradiction + untested-assumption）
- 置信类 TTS 选择器：DeepConf vs CGES vs Consilience；RLVR 轴有新证据（2609.32470、EACL'26）；跨方法统一相图仍无。
- attention 剪枝 vs 随机：2512.07580 已给混合策略，"地平线随分辨率/帧数前移"只剩视频外推（价值低）。
- steering 向量在 MoE / ≤1.7B / 长上下文扩展后的漂移仍无数据。

### G3. 基准有效性与遗忘预测量（diagnostic + method-transfer）
- 预训练泄漏 ↔ TF-CL 优势（CLDyB + Data Leakage 各一半；APER 已提新基准但未量化泄漏）。
- output-KL vs 权重空间量作为遗忘预测量：LLM 已证，视觉 CIL + 模型合并未证；可反过来作合并系数准则。
- MLLM-CL 可分性跨基准（Is Our Benchmark Enough 仅 workshop + 仅 MLLM-CL）。

### G4. 新 regime 迁移（scaling-regime + method-transfer）
- 工具调用 agent 的 KV 压缩（Pitfalls 只做 IFEval；BFCL/τ-bench 未用于压缩评测）。
- 驱逐打分 × W4A4 旋转量化（QAPruner 只做了视觉剪枝 × PTQ）。
- MLA/MoE 上 KV 驱逐 × 专家跳过。
- 同图多问 prefix 复用的 query-agnostic 视觉 KV（StreamMem 只做流式、VisCache prompt-aware）。
- draft-model-free 的 VLM 自投机（SpecVLM 用独立 draft）。
- 视频 M-RoPE 的逐维有效长度（INTP-Video-LLMs 用整体插值）。

### G5. 跨社区统一（contradiction + method-transfer）
- CTTA（UL-TTA/MS-TTA/PTA）vs TF-CIL（RanPAC/RAIL）同一 CLIP 流。
- 权重插值 vs 激活 steering 的同轴性（L2S 已报规模相关，但 steer-merge 几何未做）。
- 统一遗忘定义的 ICL/RAG/LoRA/编辑对比（"RAG or Learning?" 做了一半）。

---

## 4. Recommended Ideas（v2 重排，每条含"重审后的定位"）

每条新增四个重审字段：**已被发表的部分** / **我们的 delta** / **48 小时 pilot** / **Kill criteria**。Novelty 为核实后的同家族自评。

### Idea 1：Eviction or Quantization — Who Makes Reasoning Models Talk Longer?（v1 Idea 1 重定位）
- **已被发表的部分**：现象本身（MLSys'25 [2503.24000](https://arxiv.org/abs/2503.24000)；ReCo [2608.04771](https://arxiv.org/abs/2608.04771) 给出 R-KV +38.8%/+79%；RLKV v3）；训练式修复（ReCo 的 process-reward 三组件）；ThinKV 的反向结论（驱逐不拉长、量化拉长）。
- **我们的 delta**：(1) 裁决矛盾——在同一模型族 × 同一压缩率网格 × 同一长度统计口径下复现 ThinKV 与 ReCo，定位分歧来自模型族、驱逐策略还是"答案后尾巴"的统计口径；(2) 机制——token 级归因：被驱逐/被量化的是"结论锚点"还是"重复检测"所需的 KV，用 Global Eviction Ratio 与 logit 轨迹回答；(3) TF 修复——锚点 KV 保护 + 重复熵触发收尾，与 ReCo（训练式）同台，报端到端延迟。
- **Method**：R1-Distill-Qwen-7B、R1-Distill-Llama-8B、Qwen3-8B × {SnapKV, R-KV, BeaconKV, KIVI-2bit, ThinKV 量化} × 压缩率 {25%, 50%, 75%}；逐样本记录总长度、答案首次出现位置、答案后 token 数、重复 n-gram 率。
- **Hypothesis**：长度膨胀在驱逐与量化下都存在但机制不同——驱逐丢"结论锚点"导致重复自检，量化抬高熵导致更多反思 token；ThinKV 的"驱逐不拉长"只在其思维感知驱逐 + 特定模型上成立。保护 <2% 锚点 KV 可消除驱逐侧 >70% 的膨胀。
- **48 小时 pilot**：单模型 R1-Distill-Qwen-7B，MATH-500 200 题，R-KV 25% vs KIVI-2bit vs 无压缩，三项长度指标 + 准确率；~8 GPU-h。若 R-KV 不复现 ReCo 的 +38.8%，即进入"口径差异"分析。
- **Kill criteria**：若两种压缩的长度膨胀都 <10% 且与 ReCo 不一致，说明现象依赖实现细节，转为 negative/replication 短文。
- **Minimum experiment**：AIME24/25、MATH-500、LiveCodeBench 子集；单卡 80G 约 40–60 GPU-h。
- **Novelty 6/10 | Risk MEDIUM | Contribution：diagnostic + method**。
- **Reviewer's likely objection**："ReCo 已修复"——答：ReCo 需训练 reward 估计器且未解释机制；"锚点保护 = 保留首尾 token"——需与 StreamingLLM 式 sink 保护对照。
- dedup_key：`kv-compression-output-length-inflation-anchor-protection`

### Idea 2：KL-Razor for Vision & Merging（v1 Idea 8 升至第 2）
- **已被发表的部分**：RL's Razor [2509.04259](https://arxiv.org/abs/2509.04259)（NeurIPS'25）在 LLM 与机器人 FM 上证明 output-KL 预测遗忘、权重量不行；[2605.26097](https://arxiv.org/abs/2605.26097) 说 replay 数据上约束 KL 更防遗忘（LLM）；LoRA intruder dims [2410.21228](https://arxiv.org/abs/2410.21228) 已做缩放奇异值的干预。
- **我们的 delta**：(1) 跨模态验证——ViT/CLIP 的 CIL 与 MTIL 上 8 种方法（CODA、SD-LoRA、EWC-LoRA、SLAO、MagMax、SFA、TA、全微调）的 KL / Δ-norm / 稀疏度 / 梯度秩 / intruder 数 → 实测遗忘的回归；(2) **把 KL 变成合并系数准则**——MagMax/SFA/TA 的系数由"新任务 KL-to-base 最小"选择，这是 RL's Razor 没做的干预实验；(3) 顺带回答 G5 的问题：权重平均之所以有效是否因为它在 KL 上近似最小。
- **Hypothesis**：视觉上 KL 的 r>0.8，权重量 r<0.5；KL 选系数比固定系数遗忘 ↓20%。
- **48 小时 pilot**：ViT-B/16 ImageNet-R 10 任务，仅 3 种方法（全微调、SD-LoRA、MagMax），记录 KL 与遗忘，看相关是否 >0.7；~6 GPU-h。
- **Kill criteria**：pilot 相关 <0.4 且权重量更好 → 结论反转也值得发（"RL's Razor 不跨模态"），但方法部分放弃。
- **Minimum experiment**：约 40–60 单卡小时。
- **Novelty 7/10 | Risk MEDIUM | Contribution：diagnostic + criterion**。
- **Objection**："class-incremental 下输出头在变，KL 怎么定义"——用特征空间 KL 或共享 logit 子集；"KL 与性能循环定义"——用新任务 KL 预测旧任务遗忘。
- dedup_key：`kl_forgetting_predictor_vision_merging`

### Idea 3：Leakage-Stratified CL（v1 Idea 7 升至第 3）
- **已被发表的部分**：CLDyB [2503.04655](https://arxiv.org/abs/2503.04655) 提出污染可能；Data Leakage [2508.17416](https://arxiv.org/abs/2508.17416) 量化 ImageNet/LAION 泄漏但不涉及 CL 方法；APER（IJCV）因重叠提出新基准但未分层分析；PAM [2603.21170](https://arxiv.org/abs/2603.21170) 指出基线缺失。
- **我们的 delta**：首次把"每类泄漏分"与"TF vs 训练式方法差距"交叉；给出零泄漏协议。
- **Hypothesis**：TF 优势随泄漏分单调增加；零泄漏层训练式方法反超 ≥3 pts。
- **48 小时 pilot**：只做 ImageNet-R + CUB，RanPAC vs CODA-Prompt，用 CLIP 特征对 ImageNet-21K 子集做近重复检索；~8 GPU-h。
- **Kill criteria**：泄漏分与差距的 Spearman <0.2 且 DINO 域距离解释力更强 → 改为"域距离分层"叙事。
- **Minimum experiment**：6 数据集 × 6 方法，约 30–40 单卡小时。
- **Novelty 7/10 | Risk MEDIUM | Contribution：diagnostic + protocol**。
- **Objection**：近重复阈值任意；泄漏 ≠ 记忆——报多阈值敏感性 + 用 membership-inference 式指标佐证。
- dedup_key：`cl_leakage_stratified_tf_vs_trained`

### Idea 4：ConfMap（v1 Idea 3，加 RLVR 轴）
- **已被发表的部分**：Consilience 声称置信在难题上失效；CGES v2 有校准/区分度分析；"Variation in Verification"（ICLR'26）对 verifier 做了难度 × 模型相图；[2609.32470](https://arxiv.org/abs/2609.32470) 说 LRM 置信 prior 集中；EACL'26 "Think Just Enough" 说置信控制在 RL 模型上更伤精度。
- **我们的 delta**：跨三种置信定义（DeepConf bottom-10% group conf / CGES 后验 / Consilience 时间曲线）× pass@1 难度桶 × 同一基座不同后训练检查点（base/SFT/RLVR）的统一相图 + 一条切换规则；这是 "Variation in Verification" 对 verifier 做过而对 verifier-free 选择器没做的事。
- **Hypothesis**：增益集中在 pass@1 ∈ [0.2, 0.7]；最难桶 AUROC → 0.5 并被多数投票反超；RLVR 更重的模型相图右移。
- **48 小时 pilot**：Qwen3-8B，MATH-500 N=16，三种置信定义的 AUROC 按桶；API 或 ~6 GPU-h。
- **Kill criteria**：三种定义在所有桶上 AUROC 都 >0.75 → "置信总是有效"，相图无意义；转去验证 Consilience 的"灾难性失效"是否可复现。
- **Minimum experiment**：Qwen3-4B/8B、R1-Distill-7B、GPT-OSS-20B；N=32；40–60 GPU-h。
- **Novelty 6/10 | Risk LOW | Contribution：diagnostic + rule**。
- **Objection**："benchmark 分析不是方法"；难度桶与置信相关——用独立模型的 pass@1 定义难度。
- dedup_key：`tts_confidence_difficulty_phase_map`

### Idea 5：One Stream, Two Communities（v1 Idea 9）
- **已被发表的部分**：TTABC（NeurIPS'26 D&B）统一了 TTA 族；BayesTTA 指出渐变偏移；ARC [2405.14318](https://arxiv.org/abs/2405.14318)（ICLR'25）做了"CL 的 test-time training"；RLAP-CLIP（ICLR'26）原型 CIL。
- **我们的 delta**：CTTA 方法（UL-TTA、MS-TTA、PTA、TPS）与 TF-CIL 方法（RanPAC、RAIL、Continual-CLIP）首次放进同一 CLIP 流（渐变 corruption × 类增量 × label shift），写成"原型 + 统计量更新"统一式，给混合规则。
- **Hypothesis**：两族各崩在对方擅长的维度；统一闭式更新拿到两边 ≥90%。
- **48 小时 pilot**：CLIP ViT-B/16，CIFAR-100-C 流（5 种 corruption × 10 任务），UL-TTA vs RanPAC；全 TF，~3 GPU-h。
- **Kill criteria**：任一方法在混合流上已同时接近两边最优 → 统一框架无增益，缩为 benchmark 短文。
- **Minimum experiment**：<15 单卡小时。
- **Novelty 6/10 | Risk LOW | Contribution：diagnostic + unified framework**。
- **Objection**："两个社区定义不同对比不公平"——多种流配置 + 报告每族在本族设定下的原始数字。
- dedup_key：`ctta_cil_unified_clip_stream`

### Idea 6：ReconVis（v1 Idea 11）
- **已被发表的部分**：KVzip（文本重建式打分）；StreamMem [2508.15717](https://arxiv.org/abs/2508.15717)（流式视频 query-agnostic）；VisCache [2608.24063](https://arxiv.org/abs/2608.24063)（prompt-aware）；FlashCache（attention-free 但无多问评测）。
- **我们的 delta**："同图/同视频多问 + prefix 复用"设定下的 query-agnostic 视觉 KV 打分（重建式）+ 低分 KV 岭回归合并保留迟发重要性；报告 prefix 命中率与多问退化曲线。
- **48 小时 pilot**：Qwen2.5-VL-7B，MMBench 同图多问重组 300 组，VisionZip vs 重建式打分 @20% 预算；~5 GPU-h。
- **Kill criteria**：query-aware 方法在多问设定下退化 <2 pts → 问题不存在。
- **Novelty 6/10 | Risk MEDIUM | 约 30 GPU-h**。
- **Objection**：重建前向多一次 prefill；多问基准是重组——补 VisDial / Video-MME 原生多问。
- dedup_key：`query-agnostic-visual-kv-reconstruction-multiquery`

### Idea 7：PruneDraft — Draft-Model-Free Self-Speculation for VLMs（v1 Idea 4 降级重定位）
- **已被发表的部分**：SpecVLM [2508.16201](https://arxiv.org/abs/2508.16201)（EMNLP'25 main）：draft 对视频 token 剪枝不敏感、两阶段剪枝 draft 的视频 token、2.68×/2.11×——但用独立小 Vid-LLM 作 draft；SparseSpec-L 用稀疏 KV 的自己作 drafter（文本）；MMSpec 证明文本 SD 失效。
- **我们的 delta**：无独立 draft 模型——同一 VLM 深层视觉 token 剪枝 + 浅层早退作 draft，全 KV 验证；面向图像与短视频（SpecVLM 只做视频）；在 MMSpec 六类任务上与 SpecVLM、Lookahead 同台，报 wall-time 与 p99。
- **Hypothesis**：接受率 >0.8，1.5–2.0×，无需任何 draft 模型。
- **48 小时 pilot**：Qwen2.5-VL-7B，MMSpec 子集，测"剪 80% 视觉 token 的自己"对全模型的 top-1 一致率（接受率代理）；~4 GPU-h。
- **Kill criteria**：一致率 <0.6 → 自草拟上限不足，放弃。
- **Novelty 5/10 | Risk MEDIUM | 约 30 GPU-h**。
- **Objection**："SpecVLM 的增量"——必须证明去掉 draft 模型后加速比不降且部署更简单。
- dedup_key：`visual-token-pruned-self-speculative-decoding-vlm`

### Idea 8：DPE-for-Video（v1 Idea 5 降级）
- **已被发表的部分**：INTP-Video-LLMs [2409.12963](https://arxiv.org/abs/2409.12963)（TF 增帧：token 重排 + 上下文扩展）；GIRAFFE M-RoPE++（需训练）；FLEX [2602.14027](https://arxiv.org/abs/2602.14027)（视频生成的频率感知 RoPE 调制）；DPE/MrRoPE（文本逐维）。
- **我们的 delta**：对 M-RoPE 的时间维度做 DPE 式"逐维有效长度"估计并只重映射关键维度，与 KV 压缩（FlexMem/StreamMem）正交叠加；与 INTP-Video-LLMs 的整体插值和 FLEX 的频率分组同台。
- **48 小时 pilot**：Qwen2.5-VL-7B，Video-MME 短/中子集，64→256 帧，DPE 式 vs 均匀采样 vs INTP 式；~6 GPU-h。
- **Kill criteria**：256 帧下三者差距 <1 pt → 基准对帧数饱和，换 LongVideoBench 再试一次仍无差距则放弃。
- **Novelty 5/10 | Risk MEDIUM | 约 30 GPU-h**。
- dedup_key：`dimensionwise_rope_video_mllm`

### Idea 9：ToolKV（backlog B8 升级）
- **已被发表的部分**：Pitfalls（ACL'26 main，IFEval 上多指令被选择性遗忘）；ST-Lite（GUI agent）；UltraQuant（EMNLP'26 Industry，4-bit KV，仅量化）。
- **我们的 delta**：首个工具调用 agent 的 KV 压缩基准（BFCL、τ-bench、多轮 SWE 轨迹）+ 段类型（system / tool schema / observation）驱逐诊断 + 段感知预算（schema 冻结、observation 激进）。
- **48 小时 pilot**：Qwen2.5-7B-Instruct，BFCL 多轮子集 200 条，SnapKV 50% 预算下工具调用合法率；~3 GPU-h。
- **Kill criteria**：50% 预算下合法率下降 <5 pts → agent 轨迹还不需要压缩，放弃。
- **Novelty 6/10 | Risk LOW | 约 30 GPU-h | Contribution：benchmark + method**。
- dedup_key：`tool-calling-agent-kv-compression-segment-aware-budget`

### Idea 10：Rotation-Aware KV Eviction（backlog B7 升级）
- **已被发表的部分**：QAPruner [2604.02816](https://arxiv.org/abs/2604.02816)（视觉剪枝 × PTQ 有交互）；QUOTA [2604.17320](https://arxiv.org/abs/2604.17320)（FlatQuant 扩到 VLM，无驱逐）；RDKV/HqeKV（KV 量化 × 驱逐，权重 FP16）。
- **我们的 delta**：首次系统评测权重旋转量化（QuaRot/SpinQuant/OptRot W4A4）与 KV 驱逐打分（SnapKV/K-Norm/Expected Attention/FlashCache 式）的交互；若失效，在旋转空间重估 query 分布的闭式修正。
- **48 小时 pilot**：Llama-3.1-8B QuaRot W4A4 公开权重，LongBench 子集，SnapKV 与 K-Norm 的打分排序与 FP16 版本的 Spearman；~4 GPU-h。
- **Kill criteria**：所有打分 Spearman >0.9 且精度差 <0.5 → 无交互，写 1 页负结果附在其他工作里。
- **Novelty 6/10 | Risk LOW | 约 25 GPU-h | Contribution：diagnostic + fix**。
- dedup_key：`kv-eviction-under-rotation-quantization`

---

## 5. Backlog（v2 重评；★ = 核实后仍无覆盖）

| # | Idea | 主题 | v1→v2 Novelty | 核实发现 |
|---|---|---|---|---|
| B1 | Reflect-or-Not（反思 token 2×2 分解）| G2 | 7→5 | CGRS 摘要即"高置信时压制反思"；C6 已用强制补全；CyclicReflex/Reflection Steering/ARLCP 挤满。只剩 2×2 因果分解 + 插入/压制统一比较 |
| B2 | Trajectory-Consensus Probe for Agents | G4 | 7→6, HIGH | 2604.16529 的 rollout 摘要 + RTV（事后）、FailFast（单轨迹）；在线跨轨迹共识仍无；可直接复用 2604.16529 的摘要表示 |
| B3 | Latency-Honest Early Exit（批同步零前向） | G1 | 6→5 | 审计论点已被 SpecExit/PUMA 发表；只剩设计 |
| B4 | ★ MoE-KV（MLA/MoE 上驱逐 × 专家跳过） | G4 | 6 | 无覆盖 |
| B5 | CliffGuard（每请求自适应压缩率） | G1 | 6→5 | Compactor 已有 context-calibrated compression |
| B6 | Matched-Control Audit for LLM TTS Selectors | G2 | 5 | 无直接覆盖，但负结果风险 |
| B7 | Oracle-Gap 评测协议 | G2 | 5→4 | C6 已做 probe-based 早停；只剩协议 |
| B8 | Low-Budget Regime Study（N≤4） | G4 | 5→4 | NoThinking 已做低预算 compute-matched |
| B9 | Cross-Trajectory Hidden-State Convergence | G1 | 6→4 | ST-BoN 摘要即内部状态一致性早停 |
| B10 | TF 预算/强度统一旋钮（kNN 记忆 + 口头化难度） | G5 | 6→5 | PETS、Blend-ASC（ICML'26）更强基线；ODAR 已做难度→预算路由；"记忆/近邻"部分未覆盖 |
| B11 | ★ Unified Forgetting Ledger | G5 | 7→5 | "RAG or Learning?"（ACL'26 Findings）与 WikiBigEdit 已做一半；只剩三类遗忘定义 + 等 FLOPs |
| B12 | Audited LoRA Merging Benchmark | G3 | 5 | Audit Before You Merge 为 workshop 文，扩展空间仍在；2511.21437 (TMLR'26) 结论"只有 TA 可靠" |
| B13 | Decomposing the Gain（PTM-CL 增益分解） | G3 | 6→5 | 2410.21228 已做 intruder 奇异值干预 |
| B14 | ★ Update-Count Scaling（10→10,000 次更新） | G4 | 6 | 语言侧 WikiBigEdit 已画 1→500K；视觉侧无 |
| B15 | MLLM-CL 可分性跨基准审计 | G3 | 5–6→4 | 闭式路由部分被 CF-Router 覆盖；只剩跨基准审计 + 困难版 |
| B16 | ★ Pretraining-Scale Sweep for TF-CL | G4 | 6 | 无覆盖 |
| B17 | Merge-vs-Steer 等价性 | G5 | 6→5 | L2S 已报规模相关；steer-merge 几何仍无 |
| B18 | Merge Predictability（激活几何 → 逐层 α） | G3 | 6→5 | Submodule Linearity 已有闭式最优权重；收窄为"事前可预测性 + 激活 vs 权重指标" |
| B19 | Head-Aware Many-Shot Compression | G5 | 6→5 ⚠ | AAAI'26 "Sensitivity-Aware Task Vectors for Many-Shot Multimodal ICL" 疑似相近，需读全文 |
| B20 | ★ Agent-Trace KV-Overlap Measurement | G4 | 5 | 无覆盖；0–5 GPU-h |
| B21 | ★ Steering under Context Extension | G2 | 6 | 无覆盖 |
| B22 | Cross-Tokenizer TF Merge | G4 | 6 | Darwin Family 的跨架构 Mapper 应作对照 |
| B23 | Video Token Pruning Random-Baseline Audit | G2 | 5 | 无覆盖；负结果风险 |
| B24 | 跨应用 steering 安全税审计（TTS/MLLM 擦除/CoT 压缩） | G2 | 7→3 | v1 Idea 2 残余；修复已被做，审计价值有限 |

---

## 6. Eliminated（v2 新增 4 条，附证据）

| Idea | 淘汰原因 |
|---|---|
| v1 Idea 2 Steering Safety-Tax + Refusal-Orthogonal Fix | 2608.08383（COLM'26）、OPIUM 2607.19806、2605.01167、2608.11227 已做缓解；Safety Pitfalls 已测正交投影只减 15–25% |
| v1 Idea 12(a) AR Speech-LM Prosody Steering | 2609.33810（SLT'26）在 CosyVoice2 上用单 block 激活 clamp 控语速，Seed-TTS-Eval |
| v1 Idea 12(b) TTS Attention-Contrastive Decoding | ECCD 2608.00722（CosyVoice2/3、Llasa、GLM-TTS，WER −55.6%）；attention-constrained inference 已是其基线 |
| v1 B11 Depth-Switch Pruning | 2512.07580 摘要已含"浅层 DART/DivPrune + 深层随机"混合策略 |
| v1 B17 闭式 MLLM 路由 | CF-Router（ICLR'26 WS）已是 MLLM 隐藏态 + RLS 闭式分类器选 LoRA |
| 以及 v1 §6 的全部条目（training-free latent、又一个驱逐/剪枝/旋转/早停信号/prompt-pool/知识编辑正则/steering 方向估计/文本 DoLa 变体/多 agent 调度） | 同 v1 |

---

## 7. Suggested Execution Order（v2）

1. **先跑 48 小时 pilot 的三条低成本诊断**：Idea 5（全 TF，3 GPU-h）、Idea 10（4 GPU-h）、Idea 9（3 GPU-h）。三者 kill criteria 清晰，一天内知道哪条值得投入。
2. **主线 A（效率）**：Idea 1 的 pilot（8 GPU-h）先验证 ReCo 的 +38.8% 能否复现；复现 → 全量；不复现 → "口径差异"本身就是论文的一半。
3. **主线 B（CL）**：Idea 2 与 Idea 3 共用 ViT-B/16 + ImageNet-R 的特征抽取，可合并 pilot（~10 GPU-h）；两者结论相互印证（泄漏高 → TF 强；KL 预测遗忘）。
4. **主线 C（TTS）**：Idea 4 可全部用 API 做，适合没有 GPU 的时段。
5. Idea 6/7/8（多模态三条）窗口期短、撞车风险中等，建议各做 pilot 后只保留一条。

---

## 8. Next Steps

- [x] 用 abs 页核实四份 lens 报告（v2 完成；S2/OpenAlex 层仍 429 未跑）
- [x] 对 top-12 做 novelty 探测并重排（v2 完成）
- [ ] 在能访问 arXiv API 的环境重跑 `tools/verify_papers.py`，补 S2 层；核实 §1 中未逐篇核实的 ~45 篇非 closest-work 论文（CL）与无 arXiv ID 的 workshop 论文
- [ ] 读全文划界：AAAI'26 "Sensitivity-Aware Task Vectors"（B19）、"Variation in Verification"（Idea 4）、CF-Router 与 Audit Before You Merge 正文（OpenReview 被拦截）
- [ ] `/research-review — reviewer: codex — difficulty: hard` 补跨模型 jury；把 `acceptance_status` 从 provisional 升级
- [ ] 对 Idea 1/2/3 调 `/research-refine-pipeline` 出 `FINAL_PROPOSAL.md` + `EXPERIMENT_PLAN.md`；pilot 先跑 §4 各条的 48 小时版本
- [ ] 补搜 2026-09 之后的新作（本次 novelty 探测最薄弱的窗口是 2026-09~10）

---

## 附：文件索引

| 文件 | 内容 |
|---|---|
| `idea-stage/IDEA_REPORT.md` | 本文件（v2） |
| `idea-stage/IDEA_REPORT_2026-10-17.md` | v1（核实前） |
| `idea-stage/novelty_probe_notes.md` | 主代理对 top-12 的 novelty 探测记录（21 篇新论文 abs 核实） |
| `idea-stage/lens/lens_efficiency.md` | 效率 lens（含 §0.5 核实记录，37 篇） |
| `idea-stage/lens/lens_tts.md` | TTS lens（含 §0.5 核实记录，41+12 篇） |
| `idea-stage/lens/lens_cl.md` | CL lens（含 §0.5 核实记录，~45 条） |
| `idea-stage/lens/lens_other.md` | 其他方向 lens（含 §0.5 核实记录，85 ID） |
