# 02 · MLLM / VLM / Video-LLM / Omni / VLA 推理效率（training-free 或极低训练）文献调研 + Idea 生成

> 日期：2026-10-17 ｜ 方法：ARIS `/idea-discovery` 中的 research-lit → idea-creator → novelty-check 三阶段（未使用 Codex/Gemini 外部审稿、未跑 GPU pilot）。
> 数据来源：arXiv 列表检索（按 announced date 倒序，约 40 个 query）、arXiv abs 页摘要核对、WebSearch 补充、3 篇 PDF 精读（2610.12060、2606.11576、2604.12035）。
> **所有 arXiv id 都取自 arXiv 检索结果页，或在 arxiv.org/abs 页面核对过标题**。没有 arXiv id 的论文会注明（例如 NeurIPS'25 的 "Reasoning with Fewer Eyes"）。
> 诚实声明：这个方向每周都有 10 篇以上新论文。下面的新颖性判断只覆盖 2-3 次定向检索，**不能保证**没有同期工作。

---

## 0. 一句话结论

图像 visual token pruning/merging（FastV/SparseVLM/VisionZip 的后继者们）已经**极度饱和**：仅 2026 年 9 月 15 日到 10 月 15 日，arXiv 上标题含 "visual token pruning" 的新工作就有 50 篇以上。VLA 剪枝、Omni 音视频 token 压缩、流式视频 KV 和多模态投机解码在 2026 年上半年也迅速拥挤。剩余的真空主要在以下几处：
1. **"agentic / thinking-with-images" 推理过程中的视觉上下文**：多轮 zoom/crop 带来的跨轮冗余；
2. **推理型 MLLM 的 response 级成本**：剪枝会改变 CoT 长度，而且已有论文给出了相反结论；
3. **把剪枝当作"视图生成器"而不是单纯的省算力手段**：互补子集集成；
4. **评测稳健性维度**：同义改写和跨语言 query 下 query-guided 剪枝的稳定性；
5. **新架构特有的信号**：Qwen3-VL DeepStack、Qwen3.5 混合线性注意力。

---

## 1. Landscape：子方向、代表作、饱和度

饱和度标记：🔴 极度拥挤 ｜ 🟠 拥挤但仍有缝隙 ｜ 🟢 相对空白

### 1.1 单图 visual token pruning / merging 🔴🔴
- 奠基工作（2024–2025）：FastV `2403.06764`、SparseVLM `2410.04417`、VisionZip `2412.04467`、DivPrune `2503.02175`、CDPruner `2506.10967`、VScan `2505.22654`。另有反思类工作 "Token Pruning in MLLMs: Are We Solving the Right Problem?" `2502.11501`（2025-02），结论是 random/pooling 这类基线很强。
- 2026 年近况（仅列部分）：
  - **从"重要性"转向"条件可移除性 / 覆盖率"**：CoRePrune `2609.26484`（2026-09，Qwen3.5 上用 128 token 保留 90.3%）、ACPruner `2609.34558`、MiCo `2609.34330`、QCPruner `2609.19990`、TOPS `2606.27161`、MMTok `2508.18264`。
  - **两阶段（先视觉引导、再 text 引导）**：DeFT / "When Text Matters" `2609.34861`、TwinPrune `2610.17530`、Bifocal `2610.18399`（2026-10，混合粒度）、Shallow-to-Deep `2609.25635`。
  - **Pre-LLM**：VPRune `2609.24485`、CRISP `2607.16326`、PixelPrune `2604.00886`。
  - **去偏**：D2Pruner `2512.19443`、Attention Debiasing `2508.17807`。
  - **位置编码**：Layer-Aware PE `2609.23715`、GAP `2506.21873`。
  - **样本级 relevance-diversity 权重**：SAVER `2610.16615`（2026-10，已经做了逐样本 oracle 权重分析）。
  - **不删 token 的计算旁路**：P2P `2610.03389`。
- 判断：单纯提出一个新的打分函数，已经几乎不可能在 NeurIPS/ICLR 级别站住脚。

### 1.2 视频 token 压缩 🔴
- HoliTom `2505.21334`、DyCoke `2411.15024`、FastVID/PruneVid（2025）。
- 2026 年：OTT-Vid `2605.11803`、DynaTok `2605.19322`（处理了 positional bias）、KiToke `2604.03414`、InfoMerge `2606.02161`、EarlyTom `2605.30010`、FoveaTree `2610.16744`、FlashGaze `2610.04225`。
- Codec 引导：`2610.12959`、CodecSight `2604.06036`。
- 反思类：SimpleCluster `2609.35394`（2026-09），简单聚类基线即可打平复杂方法。

### 1.3 流式视频 / 视频 KV 🔴
StreamMem `2508.15717`、InfiniPot-V `2506.15745`、StreamingTOM `2510.18269`、StreamKV `2511.07278`、V-Rex `2512.12284`、LiveVLM `2505.15269`、MuKV `2605.22269`、ViCoStream `2606.19849`、KeyRec `2609.32182`、Foresight `2610.03123`、混合架构上的 DeltaS `2609.27470`。

### 1.4 多模态 KV cache 压缩 🟠
LOOK-M `2406.18139`、MEDA `2502.17599`、MadaKV `2506.15724`、HAE `2602.02197`、HybridKV `2604.05887`、KVCapsule `2605.16439`、RetentiveKV `2605.04075`、BACON / "Last But Not Least" `2606.14782`、VisCache `2608.24063`、频域引导 outlier-KV 方法 `2511.16786`、MHA2MLA-VLM `2601.11464`、OmniKVQuant `2609.11582`、统一模型 UniCache `2609.32831`。

### 1.5 高分辨率 / 主动感知 🟠
Zoom-Refine `2506.01663`、RAP `2503.01222`、MRD `2512.02906`、LookWise `2603.00171`、Q-Zoom `2604.06912`（需要训练）、AwaRes `2603.16932`（用 0.36× token 打平 full-res）、AdaptVision `2512.03794`、VisLens `2608.30705`、FAVE `2609.04392`、Foveated Compression `2610.07729`，遥感 UHR 方向 `2609.31747`。

### 1.6 多模态投机解码 🔴
ViSpec `2509.15235`、SpecVLM `2509.11815`、HiViS `2509.23928`、DREAM `2505.19201`、MASSV `2505.10526`、SAGE `2602.00523`、TABED `2601.20357`、MMSpec benchmark `2603.14989`、Sparrow（视频）`2602.15318`、FOVEA `2608.22883`、HAWK `2610.00623`、One-Pass Block Drafting `2609.00355`、SP²ec `2610.18735`、扩散并行 drafting 综述 `2608.20743`。

### 1.7 Omni（音频 + 视频）token 压缩 🔴（2026 年 5–9 月突然爆发）
FastAV `2601.13143`、OmniZip `2511.14582`、Keep What Audio Cannot Say `2605.11605`、OmniRefine `2605.12056`、OmniDrop `2605.14458`、OmniFocus `2607.03050`、Out of Sight `2607.21179`、OmniScope `2607.23193`、Allocation Before Ranking `2608.01665`、OmniPack `2608.03812`、Deferred Audio Pruning `2608.08794`、OmniRoute `2609.37052`、AccKV `2511.11106`、OmniMem `2606.07577`。

### 1.8 VLA / 具身效率 🔴（约 25 篇）
ADP `2509.22093`、SQAP-VLA `2509.09090`、Compressor-VLA `2511.18950`、ETA-VLA `2603.25766`、BFA++ `2602.20566`、SAFE-Pruner `2605.29662`、Spatial Scaffolds `2609.36967`、Stage-Aware `2610.05273`、VLA-ACL `2610.08133`、FastVLA `2610.22146`，以及 VLA 投机解码 Spec-VLA `2507.22424`、KERV `2603.01581`、HeiSD `2603.17573`、WA-SpecDec `2608.08725`，可认证加速 CARE `2610.08917`。

### 1.9 推理型 MLLM（长 CoT over images）效率 🟠（变化最快）
- 视觉侧：
  - "Reasoning with Fewer Eyes"（NeurIPS 2025，未找到 arXiv id）：生成少量 token 后撤掉所有视觉 token。
  - DSTP "Why and When Visual Token Pruning Fails?" `2604.12358`（ECCV'26）：解码期相关视觉信息会漂移（RVIS）。
  - VisionPulse `2605.31457`：逐步动态视觉稀疏，并声称冗余视觉上下文会**拉长**推理。
  - ViMoD `2610.12060`：精读其 Fig.3 发现，FastV 剪枝会让 MathVerse 上的回答**变长**，总 decoder FLOPs 反而超过 dense。
  - AVIS `2606.11576`：联合分配看（VCS）与想（VRS）的算力。
  - Position Rebinding Cache Reuse `2606.26631`：复用视觉 KV 时会出现 stale positional binding。
- CoT 侧：V-Skip `2601.13879`、VIG `2608.21883`、MCPO `2609.04947`。
- Thinking-with-images：DeepEyes `2505.14362`、TTSP `2604.11025`。

### 1.10 统一理解 + 生成模型 🟢/🟠
Hyper-Bagel `2509.18824`、Flash-Unified `2603.15271`、UniCache `2609.32831`；扩散 MLLM 剪枝：RedVTP `2511.12428`、LLaDA-V 剪枝 `2601.20168`、dMLLM 冗余研究 `2511.15098`、DAVET `2608.01821`。

### 1.11 新架构：MoE / 混合线性注意力 / DeepStack 🟢（刚起步）
- MoE：FastMMoE `2511.17885`、VisMMOE `2605.05899`。
- 混合线性注意力（Qwen3.5 类）：V-CoLA `2610.11251`（2026-10-08，指出已有方法在线性注意力上失效）、DeltaS `2609.27470`。
- DeepStack（Qwen3-VL `2511.21631`）：没有找到把 DeepStack 当作剪枝信号的论文。仅有 OCR 路由机制分析 `2602.22918`（DeepStack 模型的 OCR 瓶颈位于中层），以及 TensorRT-Edge-LLM 文档中按同一 mask 剪 DeepStack planes 的工程做法。

### 1.12 评测陷阱 / 诊断 🟠（2026 年下半年迅速增多）
- UniPruneBench `2511.02650`：random 很强、OCR 最脆弱。
- Spatial Provenance Audit `2608.00077`：答对了，但保留的 token 里没有可追溯的 OCR 证据。
- Break-Even Study `2608.03649`：token 更少不等于更快。
- Confidence under Pruning `2604.12035`：置信度察觉不到证据被删，并已做了**置信度触发的 cascade + conformal risk control**。
- 安全：`2610.09703`（剪枝放大越狱攻击）。鲁棒性：OOD-VTP `2607.22716`。隐私：`2609.15671`。
- 多轮：MetaCompress `2603.21701`（需要训练）、SparseVILA `2510.17777`。
- 多图：mmVB `2610.18858`。
- GUI：GUIPruner `2602.23235`、GUI-KV `2510.00536`、STP-GUI `2610.20390`。

**已验证论文数：上面列出约 130 个 arXiv id（均来自 arXiv 检索或 abs 页），检索中实际浏览的相关论文超过 300 篇。**

---

## 2. 结构性 Gap 分析（按 lens）

| Lens | 观察 | 机会 |
|---|---|---|
| **论文之间的矛盾** | ViMoD `2610.12060`：剪枝（FastV）**拉长** CoT，总 FLOPs 反而上升。VisionPulse `2605.31457`：冗余视觉上下文**拉长** CoT，剪掉反而更短。 | 两者很可能都对，取决于 selector 是删掉了证据还是删掉了干扰项。目前没有人系统地解释这一点。→ Idea 2 |
| **未测试的假设** | 几乎所有方法都假设视觉输入只编码一次，并且是一张全局图。thinking-with-images agent 会在多轮中不断追加 crop，crop 区域与全局图之间的几何重叠是**已知的**（bbox）。 | 跨轮的几何去冗余，零打分成本。→ Idea 1 |
| **方法迁移** | test-time scaling / self-consistency 在 LLM 中很成熟。剪枝研究反复观察到"少看反而更少幻觉"（PruneHal、OOD-VTP、TOPS）。 | 把多个互补剪枝子集当作多视图做集成，在 iso-FLOPs 下兼顾精度和幻觉。→ Idea 3 |
| **未探索的 regime** | 多轮对话：现有方案要么需要训练（MetaCompress），要么保留完整 KV 做检索（SparseVILA，内存重）。 | 只存第 K 层隐状态作为 checkpoint，每轮按新 query 重选，并对新选中的 token 局部重算。→ Idea 5 |
| **诊断问题没人问** | query-guided 剪枝依赖 text→visual attention。同一问题换种说法或换成中文，保留集是否稳定？目前完全没有检索到相关工作。 | → Idea 4 |
| **新架构** | Qwen3-VL 的 DeepStack 在前 3 层残差注入多级 ViT 特征；Qwen3.5 采用混合线性注意力。 | 架构内生的免费信号。→ Idea 6、7 |
| **评测** | 很多 benchmark（POPE/MME/SQA-IMG）在 0 个视觉 token 时仍有很高分，"保留 95%"的说法被抬高了。 | 引入 blind-floor 归一化 retention（作为 Idea 4 的组成部分）。 |

---

## 3. 候选 Idea（按综合推荐度排序）

> 统一的实验资源假设：单卡 24–80GB；模型为 Qwen2.5-VL-7B / Qwen3-VL-8B（Instruct 和 Thinking）、LLaVA-OneVision-7B、InternVL3.5-8B；每个 idea 最小验证 ≤ 1–3 GPU-day。

### Idea 1 ★★★★★ GeoDedup：Thinking-with-Images agent 的跨轮几何去冗余（training-free）
- **假设**：DeepEyes / Mini-o3 类模型在多轮 zoom 中不断追加 crop 的视觉 token。crop 一旦加入，全局图中被该 bbox 覆盖的 token（以及被更细 crop 取代的旧 crop）对后续推理的边际价值很低。由于 bbox 已知，可以零成本定位这些冗余。
- **机制**：
  1. 在第 t 轮 crop 返回后，把全局图里与 bbox 重叠度 > τ 的 token 降为低分辨率摘要。做法是对 KV 做 2×2/4×4 池化合并，并保留 1 个"锚点"维持空间布局。
  2. 对嵌套 crop 只保留最细一级的完整 token；粗一级 crop 只保留边界外区域。
  3. 可选：对重叠区按 key-novelty（新 crop 能否重建旧 token）决定是否删除。
  4. 位置 ID 按 PRCR `2606.26631` 的经验保持不变，避免 rebinding 崩溃。
- **最小实验**：
  - 模型：DeepEyes-7B（`2505.14362`，Qwen2.5-VL-7B 底座），以及开源的同类 zoom-agent 7B 模型；
  - 基准：V* Bench（191 题）、HR-Bench 4K/8K、MME-RealWorld-lite；
  - 指标：准确率、每条轨迹的累计视觉 token 数、端到端时延、轮数是否变化；
  - 基线：不压缩；在每轮对全上下文跑 FastV / DivPrune / VisionZip；GUIPruner 风格的按时间衰减分辨率。
- **新颖性论证**：AwaRes `2603.16932` 和 AdaptVision `2512.03794` 是"按需拿 crop"并需要训练，不处理已有上下文的冗余。GUIPruner `2602.23235` 针对 GUI 历史截图，采用时间衰减而不是几何重叠。TTSP `2604.11025` 关注的是精度而不是效率。
- **Closest**：`2603.16932`、`2512.03794`、`2602.23235`、`2606.26631`、`2604.11025`。
- **风险**：
  1. 轨迹通常只有 1–3 轮，节省比例可能不大，需要先统计 token 分布；
  2. 删掉全局图后，模型可能失去"回看别处"的能力，导致轮数增加，因此必须报告**轨迹级**总成本；
  3. 开源 zoom agent 的推理代码各不相同，工程量中等。
- **Novelty 速查（3 次检索）**：没有找到"跨轮 crop 去冗余"的 training-free 方法 → **NOVEL-LIKELY**。

### Idea 2 ★★★★☆ "剪了以后想得更久？"——解释推理型 MLLM 中剪枝与 CoT 长度的矛盾，并提出 response-level 预算
- **假设**：剪枝对 CoT 长度的影响取决于 selector 删掉的是**证据**还是**干扰**。
  - 注意力类 selector（FastV）更容易删掉证据，模型会出现 hesitation，"wait/再看一下"变多，CoT 变长（对应 ViMoD 的观察）。
  - 覆盖或多样性类 selector（DivPrune/VisionZip）更多删掉干扰项，CoT 变短（对应 VisionPulse 的观察）。
  - 衡量剪枝收益应该看 response-level 总 FLOPs，而不是 prefill FLOPs。
- **机制**：
  1. 诊断部分：5 种 selector × 3 个 thinking 模型（Qwen3-VL-8B-Thinking、MiMo-VL-7B-RL、GLM-4.1V-9B-Thinking）× 保留率 {100, 50, 25, 11}% × 基准（MathVista-mini、MathVerse-vision、MMMU-Pro、HR-Bench）。记录长度、hesitation token 频率、"证据是否被删"（借用 `2604.12035` 的 object-token 覆盖率和 `2608.00077` 的 provenance）。
  2. 方法部分（training-free）：提出 Length-Calibrated Budget。用 prefill 后前 32 个生成 token 的 hesitation 或熵信号预测 CoT 膨胀，触发时一次性恢复被删的 top-k token（KV 已 offload 到 CPU）。
- **最小实验**：约 1.5–2 GPU-day，大头在 thinking 模型的长解码。
- **新颖性论证**：ViMoD 只在 FastV/MathVerse 上做了动机图，VisionPulse 只给出相反方向的观察，两者都没有做 selector 维度的系统解释。DSTP `2604.12358` 和 VisionPulse 关注的是精度和逐步稀疏，而不是 **response-level 成本和长度机制**。
- **Closest**：`2610.12060`、`2605.31457`、`2604.12358`、`2606.11576`，以及 NeurIPS'25 "Reasoning with Fewer Eyes"。
- **风险**：方法部分与 DSTP 的"解码期恢复"思路很接近，需要强调触发信号来自长度或hesitation，以及评价口径是 response-level。诊断部分的价值较高，方法部分的增量相对有限。
- **Novelty 速查**：**CROWDED-BUT-DELTA**。诊断角度新，方法与 DSTP/VisionPulse 相邻。

### Idea 3 ★★★★☆ Complementary-View Ensembling：把剪枝当作多视图生成器（iso-FLOPs 集成）
- **假设**：剪枝能减少注意力稀释和幻觉，但会丢失覆盖；集成可以把覆盖找回来。把视觉 token 划分为 K 个**互补且各自多样**的子集，例如 DPP/DivPrune 贪心交替分配，并让每个子集都带上 top 显著 token。每个子集单独前向，再做 logit 平均或投票。在总 FLOPs ≈ 单次 50% 保留（K=2 × 25%）的条件下，有望**同时**比单次 50% 剪枝更准、比 full-token 更少幻觉。
- **机制**：K 路共享文本前缀，可以批处理，仍兼容 FlashAttention。聚合方式可选 logit 平均、"对比"聚合（类 VCD），或按每路 confidence 加权。
- **最小实验**：
  - 模型：Qwen2.5-VL-7B、LLaVA-OV-7B；
  - 基准：POPE、HallusionBench、AMBER（幻觉），MMBench、GQA、TextVQA（通用和 OCR）；
  - 对照：full、单路 25%/50%（DivPrune、VisionZip）、VCD、OOD-VTP；
  - 指标：iso-FLOPs 下的 accuracy 和 CHAIR。
  - 约 1 GPU-day。
- **新颖性论证**：ParVTS `2511.18875` 按 subject/non-subject 并行分组，并在中途丢掉一路，目的是省算力而不是集成。VCD 系列使用扰动图像，而不是互补的 token 子集。OOD-VTP `2607.22716`、PruneHal `2510.19183`、TOPS `2606.27161` 都是单路方法。
- **Closest**：`2511.18875`、`2607.22716`、`2510.19183`、`2603.10360`。
- **风险**：
  1. OCR 和计数这类需要全局整合的任务可能下降；
  2. 若收益只体现在幻觉基准上，影响力有限；
  3. 文本前缀重复 K 次，短图像场景下的开销不可忽略。
- **Novelty 速查（2 次检索）**：**NOVEL-LIKELY**。但"test-time augmentation over visual tokens"属于直觉上容易想到的方向，可能有未检索到的 workshop 论文。

### Idea 4 ★★★☆☆ Query-guided 剪枝的稳定性审计：同义改写 / 跨语言 / blind floor
- **假设**：text-guided selector（SparseVLM、FastV、CDPruner、QCPruner `2609.19990` 等）的保留集对 query 表述高度敏感。具体包括：
  1. 同义改写下保留集的 Jaccard 很低，答案一致率下降；
  2. 非英文 query（中、日、阿拉伯语）的 text→visual 对齐变差，retention 系统性偏低；
  3. 扣除 0-token blind floor 之后，主流"保留 95%"的数字会大幅缩水，方法间排名也可能改变。
- **最小实验**：
  - 改写：用 LLM 为 GQA/TextVQA/MMBench 各 500 题生成 5 种改写；
  - 跨语言：MMBench-EN 与 MMBench-CN 是原生配对数据，另加机器翻译的 TextVQA；
  - 方法：8 种 selector（4 种 query-agnostic、4 种 query-guided）；
  - 模型：Qwen2.5-VL-7B、InternVL3.5-8B；
  - 指标：kept-set Jaccard、答案 flip rate、blind-normalized retention =（剪枝分数 − blind 分数）/（full 分数 − blind 分数），配对 bootstrap CI。
  - 约 1 GPU-day。
- **新颖性论证**：UniPruneBench `2511.02650`、Provenance Audit `2608.00077`、Confidence `2604.12035`、Break-even `2608.03649` 覆盖了任务、溯源、置信度和时延，但都没有覆盖 query 扰动和语言维度，也没有做 blind floor 归一化。
- **风险**：属于纯评测论文，需要配一个简单修复（例如 query 改写集成后的 relevance 分数，或 query-agnostic 保底配额）才有足够分量。评测陷阱这条线 2026 下半年已有 4–5 篇，审稿人可能疲劳。
- **Novelty 速查（2 次检索，均无命中）**：**NOVEL-LIKELY**（细分维度）/ 大类 CROWDED。

### Idea 5 ★★★☆☆ Mid-Layer Checkpoint 重选：多轮对话下的 training-free 剪枝
- **假设**：FastV 类方法在第 K 层剪枝，之后的每一轮都只能看到第一轮选出的 token。如果只缓存所有视觉 token 在第 K 层的隐状态（约 3–4 层 KV 的体积，SparseVILA 需要存全部 28 层），新一轮到来时按新 query 在第 K 层重选，只对**新增**的 token 重算 K…L 层，就能以远低于 SparseVILA 的内存恢复多轮精度。
- **最小实验**：
  - 构造多轮数据：GQA 和 TextVQA 同图多问，按图拼成 3–5 轮；另用 MMDU、ConvBench；
  - 对照：SparseVILA `2510.17777` 复现、ASAP `2603.14549`、query-agnostic（VisionZip/DivPrune）、FastV/SparseVLM；
  - 指标：第 2–5 轮准确率、峰值显存、每轮时延。
  - 约 1 GPU-day。
- **Closest**：`2510.17777`、`2603.21701`（MetaCompress，需要训练）、`2603.14549`、`2606.09916`（IntentKV，面向 agent 文本）。
- **风险**：
  1. 新增 token 的深层 KV 与早先文本的深层 KV 不一致（早先的文本没有看到这些 token），可能需要近似；
  2. 增量相对 SparseVILA 主要体现在内存，审稿人可能认为是工程优化。
- **Novelty 速查**：**CROWDED-BUT-DELTA**。

### Idea 6 ★★☆☆☆ DeepStack-Novelty：把 Qwen3-VL 多级 ViT 注入残差当作免费的细粒度信号
- **假设**：DeepStack 在 LLM 前 3 层给每个视觉位置加上浅、中层 ViT 特征。浅层特征与最终 token 之间的差异（"细节残差"范数或新颖度）可以零成本标出 OCR 和小物体 token，而这些恰好是最终层 saliency 或 [CLS] attention 容易删掉的。`2602.22918` 发现 DeepStack 模型的 OCR 瓶颈在中层，从机制上支持这一点。
- **最小实验**：Qwen3-VL-8B；TextVQA、DocVQA、ChartQA、OCRBench 与 GQA、MMB；对照 VisionZip、DivPrune、CDPruner、QuietPrune（CVPR'26）、PixelPrune `2604.00886`；把 DeepStack-novelty 作为加性项插入现有 selector。约 0.5 GPU-day。
- **风险**：架构特定，影响面窄。若 Qwen3.5 或后续版本放弃 DeepStack，价值会下降。可以作为 Idea 4 或其他工作的一个组件。
- **Novelty 速查（2 次检索）**：没有找到把 DeepStack 当作剪枝信号的工作 → **NOVEL-LIKELY**（narrow）。

### Idea 7 ★★☆☆☆ Delta-Rule Retrieval Error：混合线性注意力 MLLM 的单图/多图视觉 KV 驱逐
- **假设**：在 Qwen3.5 类混合架构里，只有 full-attention 层持有 KV。Gated DeltaNet 层在 prefill 时对每个视觉 token 计算"写入前的检索误差" ‖v − S·k‖，这个量本身就是 token 新颖度，可以免费拿到。用它在 prefill 之后驱逐 full-attn 层的视觉 KV，并用于多图跨图去冗余。
- **Closest**：DeltaS `2609.27470`（流式视频的 chunk 级状态漂移）、V-CoLA `2610.11251`（线性注意力上的 uniqueness 合并，2026-10-08）。
- **风险**：两篇近亲论文都在最近 1 个月内发表，这一线会非常快地拥挤；token 级信号相对 chunk 级的增量也未经验证。
- **Novelty 速查**：**CROWDED-BUT-DELTA**（时间窗口很短）。

### Idea 8 ★★☆☆☆ 统一模型编辑任务中源图 token 的区域感知剪枝
- **思路**：在 BAGEL 或 Show-o2 的 image editing 中，源图的 ViT/VAE token 在每个去噪步都要被注意。编辑区域以外的源图 token 可以按 edit-instruction 的 attention 合并，非编辑区由 VAE latent copy-through 保真。
- **Closest**：UniCache `2609.32831`（已对编辑任务做了 5× KV 压缩）、Flash-Unified `2603.15271`、Hyper-Bagel `2509.18824`。
- **Novelty 速查**：**CROWDED-BUT-DELTA**，且 BAGEL 实验成本偏高（14B MoT），不推荐作为首选。

### 已确认被做掉的想法（避免重复）
| 想法 | 已有工作 | 判定 |
|---|---|---|
| 低置信度时用全量 token 重跑的 cascade | Confidence under Visual Token Pruning `2604.12035`（含 conformal risk control） | **ALREADY-DONE** |
| 推理中撤掉视觉 token | VTW `2405.05803`、Reasoning with Fewer Eyes（NeurIPS'25）、DyVTE `2411.19628` | **ALREADY-DONE** |
| 解码期视觉证据漂移与恢复 | DSTP `2604.12358`、VisionPulse `2605.31457`、ViMoD `2610.12060` | **ALREADY-DONE** |
| 逐样本 relevance/diversity oracle 权重 | SAVER `2610.16615` | **ALREADY-DONE** |
| 剪枝后 grounding 崩溃源于位置 ID | GAP `2506.21873`、Layer-Aware PE `2609.23715` | **ALREADY-DONE** |
| 剪枝的安全性和鲁棒性 | `2610.09703`、`2607.22716` | **ALREADY-DONE** |
| token 更少却不更快（break-even） | `2608.03649`、UniPruneBench `2511.02650` | **ALREADY-DONE** |
| MoE MLLM 路由感知剪枝 | FastMMoE `2511.17885` | **ALREADY-DONE** |
| 多图预算分配 | mmVB `2610.18858` | **ALREADY-DONE** |

---

## 4. 推荐执行顺序
1. **Idea 1（GeoDedup）**：新颖性最好，任务是真实部署痛点（agentic 视觉推理），实验小（V* 只有 191 题）。第一步先统计 DeepEyes 在 V*/HR-Bench 上的轨迹 token 分布，确认有足够的压缩空间（目标是多轮轨迹占比 > 40%，并且 crop 与全局图有明显重叠）。
2. **Idea 2（CoT 长度矛盾）**：诊断价值高，可以和 Idea 1 共用 thinking 模型的基础设施。
3. **Idea 3（互补视图集成）**：最便宜（约 1 GPU-day），结论非黑即白，适合作为快速 pilot。
4. Idea 4、6 可以合并成一篇"新一代 MLLM 上剪枝评测稳健性"的论文；Idea 5 和 7 作为备选。

## 5. 不确定性与局限
- 2026 年 10 月中旬每天都有新论文，新颖性判断的有效期大约只有 4–8 周。
- 只精读了 3 篇 PDF，其余都依据摘要判断。Idea 2 的相反结论来自 ViMoD 的 Fig.3 和 VisionPulse 的摘要，VisionPulse 的长度实验细节尚未核对全文。
- "Reasoning with Fewer Eyes"、QuietPrune（CVPR'26）没有找到 arXiv id，引用的是会议页面。
- Mini-o3 等 zoom-agent 的 arXiv id 未核对，因此正文没有给出。
