# 01 — 文本 LLM 推理效率（training-free / 极少训练）：文献地图 + 候选 idea

> 日期：2026-10-17。检索手段：arXiv HTML 检索（按最新排序）、arXiv abs 页元数据、WebSearch；下载并阅读 2609.25809 全文（pdftotext）核对局限。
> 所有 arXiv id 均经 arXiv abs 页确认存在（2405.16444 / 2411.04975 / 2402.18096 / 2506.05345 为较早的经典，也已核验）。
> 无 Codex/Gemini 交叉评审、无 GPU pilot —— 新颖性判断仅基于检索，**存在漏检风险**（该领域 2026 年每月新增数十篇 KV 论文，arXiv 全文检索为 AND 语义，长查询易漏）。

---

## 1. 领域地图（2025 → 2026-10）

### 1.1 KV cache 驱逐 / 压缩（**极度饱和**）
仅 2026-10 前两周就有 ≥20 篇。代表：
- 打分函数持续"内卷"：DropKV (2610.22130, Oct-26)、Behavior-Preserving KV Compression (2610.06479, Oct-26，以 KL 为准则)、LORE-KV / Monte Carlo Estimation for KV Cache Eviction (2610.07643, Oct-26)、CORE coverage-calibrated (2610.02235, Sep-26)、K-VEC coverage-driven (2606.29563, Jun-26)、ECOKV (2609.06663, Sep-26)、InertiaKV (2609.03515, Sep-26)。
- query-agnostic / 可复用：KV² (2610.03198)、EchoPress (2610.00412)。
- 可逆/召回：WakeKV (2610.02713)、KVFetch（顺序遗忘 sequential forgetting，2610.08811）。
- 位置几何：RoPERelay / Post-Selection Positional Geometry (2610.21126)。
- 低秩 / 量化 KV：AttSVD (2610.06927)、iS-KV (2610.02815)、JoLT (2607.12550)、SPECTRA 2-bit (2608.07915)、VQKV (2603.16435)、XQuant (2510.11236)。

### 1.2 推理模型（long CoT）的 KV（**饱和，但有矛盾可挖**）
- R-KV (2505.24133)、LazyEviction (2506.15969)、ThinKV (2510.01290)、Crystal-KV (2601.16986)、ForesightKV (2602.03203)、LongFlow (2603.11504)、VaSE (2606.03928)、ReasonAlloc (2606.11164)、KARA (2607.01237)、BeaconKV (2609.04971)、AvoKV-E (2610.03007)、BreadthKV / Spend Bytes on Breadth (2610.05685)、KV-Rescue (2608.15797)。
- 基准/诊断：Hold Onto That Thought (2512.12008)，SkipKV (2512.07993)。
- **关键矛盾**：Hold Onto That Thought (2512.12008) 认为 H2O / SnapKV-D 等 heavy-hitter 方法占优；而 **Random Attention (2609.03430, Sep-26)** 证明"保住 prompt + 头内随机驱逐"即可匹配最强基线，原因是 (i) prompt 才是脆弱部分，(ii) 推理轨迹本身在文本层（模型复述）与头间层面都冗余。BreadthKV (2610.05685) 显示驱逐的大部分损失来自"脱轨"（跑到长度上限）。
- **未检验的假设**：Random Attention 的结论只在"短 prompt + 长 CoT 的数学/科学题"上成立；prompt 本身很长（必须压缩 prompt）、或轨迹不复述（代码、精确复制）时是否仍成立，没人测过（KVFetch 2610.08811 在长上下文非推理场景下已显示逐字复制会断）。

### 1.3 多轮 / Agent 场景 KV（**快速升温，方法层多，"推理模型多轮"有缝**）
- CommitKV (2608.07855)、IntentKV (2606.09916)、StepKV (2609.22158)、SideQuest (2602.22603, 需少量训练)、KVTether (2609.39819)、Request Order Matters (2610.05833)、ReCache (2608.19662)、AgentKVShift (2607.21604)、LoopServe（多轮稀疏 prefill 跨轮复用, 2507.13681）、HEAR protocol (2610.06597)。
- 非前缀复用：CacheBlend (2405.16444)、KVBoost (2608.21362)、ATTUNER (2609.36722)、RelaxKV (2609.33503)、Evaluating the accuracy of KV cache reuse techniques (2609.31415)、KVEraser（**学习式**跨度擦除，2606.17034）。
- **缝隙**：Qwen3 等模型的 chat template 在多轮中**剥离历史 thinking**，这会破坏前缀缓存（Tinker 文档明确指出 O(T²) 计算）。我没找到对"剥离 thinking 后复用 answer 段 KV"的 training-free 方法研究。

### 1.4 稀疏 / 动态注意力（prefill & decode）（**饱和**）
RBS-Attention (2609.20971)、Self-Indexing Attention (2609.13205)、PIVOT (2607.24593)、SpotAttention (2606.22874)、IndexCache 跨层索引复用 (2603.12201)、Kascade (2512.16391)、The Sparse Frontier (2504.17768)、FlashPrefill V2 (2608.19758)、HySparse2 (2609.26368)、CSAttention (2604.08584)、OVAL (2610.06686)。

### 1.5 投机解码（training-free drafter / self-spec）（**饱和**）
- 自投机 / 层跳：SWIFT (2410.06916)、CLaSp (2505.24196)、KNN-SSD (2505.16162)、KnapSpec (2602.20217)、ConfLayers (2604.14612)、rigor-matched audit of layer skipping (2608.28846)、Cassandra（边缘推理模型, 2605.26558）。
- 稀疏 KV 自投机：A Sparse Glimpse of the Whole (2607.27735)、S³R（RL rollout, 2610.13202）、Self-Speculation for Faster Reasoning Models / SSR (2608.20359)。
- 检索 / n-gram：SuffixDecoding (2411.04975)、LogitSpec (2507.01449)、AdaPLD (2606.05742)、RACER (2604.14885)、ToolSpec (2604.13519)、Oilbird（复用 verifier 的 key, 2608.03839）、多样本互为草稿 Speculative Decoding for Multi-Sample Inference (2503.05330)。
- 有损：Margins, Not Windows (2609.02897)、Approximate Speculative Decoding (2608.03447)、Training-free Relaxed SD 调研 (2607.08690)。

### 1.6 PTQ / 剪枝（**饱和；推理模型方向的"token 膨胀"已被做**）
- 旋转/低比特：OptRot (2512.24124)、DuQuant++ (2604.17789)、GaugeQuant (2607.20757)、Fractional Power-of-Two (2610.21257)。
- 推理模型量化：Quantization Hurts Reasoning? (2504.04823)、AYOT 自生成 CoT 校准 (2608.01078)、FAQ (2601.11200)、**Quantization Inflates Reasoning（token 膨胀, 2606.25519）**、Extreme Low-Bit…Targeted Recovery（loop rescue 回退 FP16, 2606.02011）、Quantized Reasoning Models Think They Need to Think Longer (2606.00206)。
- 剪枝：SARATHI (2610.15514)、Attention-Preserving MLP Trimming for reasoning (2610.19121)、SparseDecoding (2610.12327)。

### 1.7 MoE 推理（expert 剪枝/跳过）（**中等拥挤，"推理长度"维度空白**）
ACE 免校准 expert skipping (2609.05228)、You Only Need 2/3 of the Chosen Experts（12 个细粒度 MoE 的系统实证, 2609.25809）、RAZOR (2609.30465)、OMP-MoE (2609.31631)、SERE (2602.07616)、Super Experts (2507.23279)、Huffman-code routing (2607.20427)、共享前缀 MoE 路由局部性 (2604.17182)、AlphaQ MoE 比特分配 (2606.04980)。
**我读了 2609.25809 全文**：它报告了 Thinking 模型平均输出 8.9k token、更抗剪枝，但**只看准确率保留和 per-token 吞吐**，没有测剪枝前后 CoT 长度的变化，也没有算 per-problem 端到端成本。

### 1.8 Early exit / 层跳过（**中等，自投机化后饱和**）
River-LLM (2604.18396)、SimLens (2507.17618)、FlexEE (2609.17008)、SEED (2609.36590)、LiteStage（多阶段推理分段层预算, 2510.14211）。

### 1.9 扩散 LLM（dLLM）推理（**极度饱和且变化快**）
Fast-dLLM++ (2606.02955)、Flash-dLLM (2609.26796)、Polestar (2607.14107)、DC-Leap (2607.20467)、Mask-Guided KV Eviction in Block Diffusion (2610.06996)、SoloQ 免校准量化 (2610.07121)、Batch-Paced 批解码 (2610.17917)、BlockServe (2607.08930)、Serving Masked Diffusion LLMs (2608.23807)、Faster but Different（多模态 dLLM 内容漂移, 2607.29079）。

### 1.10 组合 / 评测
Inference Engineering Pareto Atlas (2609.17863，Qwen2.5-7B，未涉及推理模型 token 膨胀)、A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality (2609.18005)、The Pitfalls of KV Cache Compression (2510.00231，IFEval 多指令单 prompt)、Inference-Time Hyper-Scaling with KV Cache Compression (2506.05345，压缩换更多样本，DMS 需训练)。

### 1.11 小结：饱和 vs. 开放
- **饱和**（不建议进入）：单轮长上下文 KV 打分函数；推理 KV 驱逐新打分；稀疏 prefill 核；自投机/层跳；检索式 drafter；W4A4 旋转量化；dLLM 缓存+并行解码；"量化导致 CoT 变长"的诊断。
- **相对开放**：
  1. 推理模型**多轮**场景中剥离 thinking 和 KV 复用之间的冲突；
  2. **MoE expert 剪枝**带来的推理长度膨胀（量化与 KV 驱逐都已有人测过，MoE 没有）；
  3. Random Attention 结论在**长输入 × 长输出**、低冗余轨迹下的边界；
  4. 多种效率手段**叠加时的 token 膨胀是否复合**（单一手段已有人测，组合没有）；
  5. 多轮长对话下指令/系统提示随驱逐**跨轮衰减**。

---

## 2. 候选 idea（按优先级排序）

> 资源假设：单张 24–80GB GPU，开源 7–8B（或 MoE 20–30B，FP8/MXFP4），≤ 数 GPU-day。

### Idea 1 — ReThink-Reuse：多轮推理中剥离 thinking 后的 training-free KV 复用 ★★★
- **假设**：对于 template 剥离历史 thinking 的推理模型（Qwen3-8B、R1-Distill），可以不重新 prefill：直接删掉 thinking 段的 KV，保留 answer 段 KV，再把位置重排（RoPE 逆旋转 + 重旋转），必要时只对少量偏差最大的 token 做选择性重算（CacheBlend 式）。这样能在 TTFT/FLOPs 降一个量级的同时，达到接近精确重 prefill 的下一轮质量。另有一个可检验的反直觉假设："看过 thinking"的 answer KV 可能反而**更好**，即存在 latent reasoning residue。
- **机制**：对每轮依次比较四种方案。(a) 精确方案：剥离 thinking 后重 prefill；(b) 保留 thinking（交错思考，分布外）；(c) 删除 thinking KV 并压缩位置，零重算；(d) 在 (c) 的基础上，按 KV 偏差（CacheBlend 指标）或注意力对 top-r% token 重算。
- **最小验证**：模型 Qwen3-8B（thinking 模式）、DeepSeek-R1-Distill-Llama-8B。任务：多轮数学（MT-Bench-101 或自建的 AIME/MATH 多轮追问链）、BFCL multi-turn / τ-bench（开源模型版）、MT-Bench 第二轮。指标：下一轮准确率、与 (a) 的 token-level agreement / KL、TTFT、总 prefill FLOPs 随轮数的曲线。基线：(a)(b)、CacheBlend、RoPERelay（2610.21126，仅做位置部分）、KVEraser（若有开源权重）。总量约 1 GPU-day。
- **为何可能新颖**：现有工作都没有专门处理推理模型的"thinking 剥离 ↔ 前缀缓存"冲突。Tinker 文档只是指出了这一 trade-off；KVEraser 是学习式的通用跨度擦除；CommitKV / IntentKV 处理的是 agent 事件，不涉及 thinking 剥离。
- **最接近工作**：KVEraser 2606.17034；RoPERelay 2610.21126；CacheBlend 2405.16444；IntentKV 2606.09916；CommitKV 2608.07855；Evaluating KV reuse accuracy 2609.31415。
- **风险**：(c) 可能本身就"足够好"，论文会显得偏工程（这时可以把 residue 假设当作主卖点）；新一代模型（Kimi-K2-thinking、MiniMax-M2、gpt-oss）倾向保留交错思考，痛点可能只存在于部分模型族。
- **查新**：做了 5 次检索（arXiv："multi-turn reasoning KV cache"、"reasoning trace removal KV"、"KV cache reuse thinking"、"prefix caching reasoning models multi-turn"；WebSearch ×3），都没有命中。**NOVEL-LIKELY**（中等置信度）。

### Idea 2 — MoE expert 剪枝的"隐藏 token 膨胀"与相位感知 expert 预算 ★★★
- **假设**：对推理 MoE 做动态 expert 剪枝（top-k 截断、ACE 等）时，per-token 加速 1.2–1.7×，但 CoT 会变长、脱轨率上升，所以每道题的端到端加速被高估。thinking 段与 answer 段（以及"wait/验证"段）对剪枝的敏感度不同，相位感知的 k 调度能在相同平均 k 下降低膨胀。
- **机制**：(1) 测量 CoT Token Inflation Ratio，并统计 length-cap 命中率随 k 的变化；(2) training-free 调度：thinking 中低熵 token 用小 k，answer 段、高熵 token、检测到复述/循环时恢复 full-k。
- **最小验证**：GPT-OSS-20B（MXFP4，约 16GB）、Qwen3-30B-A3B-Thinking（FP8，约 32GB）。数据集 AIME24/25、MATH500、LiveCodeBench、GPQA-D。基线：uniform top-k 截断（2609.25809 的强基线）、ACE 2609.05228、top-p 动态路由。指标：accuracy、平均/尾部生成长度、每题 wall-clock 和能耗。约 2 GPU-day。
- **为何可能新颖**：量化（2606.25519）和 KV 驱逐（2512.12008）的 token 膨胀都已有人测过，MoE expert 剪枝还没有。2609.25809 全文只报告了原生输出长度，没有比较剪枝前后的长度。
- **最接近工作**：2609.25809；ACE 2609.05228；Quantization Inflates Reasoning 2606.25519；LiteStage 2510.14211（分阶段层预算，dense 模型）；Huffman routing 2607.20427。
- **风险**：如果膨胀很小，诊断部分就是负结果（可写成"MoE 剪枝不膨胀，与量化相反"，依然有信息量）；相位调度的增益可能不足 1%（2609.25809 发现保守预算下动态规则和 uniform 的差距 <1%），应聚焦在激进预算区间。
- **查新**：3 次 WebSearch + 2 次 arXiv（"MoE expert pruning output length"、"dynamic expert pruning reasoning tokens"）+ 全文核对。膨胀诊断部分 **NOVEL-LIKELY**，相位调度部分 **CROWDED-BUT-DELTA**。

### Idea 3 — 当 prompt 也必须压缩时：Random Attention 结论的边界与非对称压缩 ★★☆
- **假设**："prompt 脆弱、轨迹自保护"（2609.03430）意味着在长输入 + 长 CoT 场景下，最优分配是 prompt 用 query-aware 的一次性压缩（SnapKV / KV² 风格）、轨迹用无打分随机驱逐，并且大部分预算给 prompt。在低冗余轨迹（代码生成、逐字复制标识符）中，随机驱逐会失效。
- **最小验证**：Qwen3-8B-thinking、R1-Distill-Qwen-7B。长输入推理任务：LongBench v2（带 thinking）、RULER-QA / BABILong 加 CoT、LiveCodeBench（低冗余对照）。在不同 prompt/trace 预算比例下做网格，比较 Random、R-KV、SnapKV-D、VaSE、KVFetch。约 2 GPU-day。
- **最接近工作**：Random Attention 2609.03430；LongFlow 2603.11504；Crystal-KV 2601.16986；KVFetch 2610.08811；BreadthKV 2610.05685；ZoomR（ACL 2026，未取到 arXiv id）。
- **风险**：结果可能只是"按比例分配预算"这种工程结论，需要一个可解释的冗余度量（例如轨迹 n-gram 复述率能预测随机驱逐的损失）来撑起论文。
- **查新**：2 次 WebSearch + 3 次 arXiv，没找到专门做"长输入 × 长输出联合压缩"的工作。**CROWDED-BUT-DELTA**。

### Idea 4 — 效率手段叠加的 token 膨胀复合效应 + 统一的"最便宜旋钮回退"调控器 ★★☆
- **假设**：W4 量化、KV 驱逐/量化、expert 剪枝各自都会让 CoT 变长；叠加后膨胀是超乘性的（脱轨概率叠加），导致叠加后的端到端收益远低于单项相乘的预测。一个 training-free 的在线调控器（检测复述/熵异常后，先回退代价最小的旋钮，例如恢复 KV 精度，再恢复 expert k）能找回大部分收益。
- **最小验证**：Qwen3-8B（AWQ-W4 × KV-INT4 × R-KV/Random 驱逐的 2×2×2 析因）加 GPT-OSS-20B（expert 截断）。数据集 AIME/MATH500/GPQA。指标：每题端到端时间和 token、脱轨率、各因素交互项。约 2–3 GPU-day。
- **最接近工作**：2606.25519；loop rescue 2606.02011；KV-Rescue 2608.15797；Pareto Atlas 2609.17863（只用 Qwen2.5-7B 和 GSM8K，没有推理模型）；BreadthKV 2610.05685（只做 KV 内部的精度和数量权衡）。
- **风险**：析因实验工作量大；回退调控器与 loop rescue 相近。
- **查新**：3 次检索。**CROWDED-BUT-DELTA**（亮点在复合效应的析因诊断）。

### Idea 5 — Derail-triggered Rehydration：脱轨时从 CPU 冷层回滚并恢复被驱逐的 KV ★★☆
- **假设**：驱逐造成的损失主要来自脱轨（2610.05685、2608.15797）。只在检测到脱轨时，把被驱逐的 KV 从 CPU 冷层恢复，并回滚到上一个推理步骤边界重新解码；不需要 helper 模型（KV-Rescue 需要），也不需要整题回退 FP16（loop rescue）。这样能以很低的平均显存找回大部分精度。
- **最小验证**：Qwen3-8B、R1-Distill-Llama-8B，AIME/MATH500，预算 B=64–512。基线：R-KV、Random、VaSE、KV-Rescue、WakeKV。指标：准确率、峰值和平均显存、PCIe 流量、长度上限命中率。约 1–2 GPU-day。
- **最接近工作**：KV-Rescue 2608.15797；VaSE 2606.03928；WakeKV 2610.02713；KVFetch 2610.08811；loop rescue 2606.02011。
- **风险**：差异较小，审稿人会说这是"WakeKV + loop 检测"。
- **查新**：2 次 WebSearch。**CROWDED-BUT-DELTA**。

### Idea 6 — 长多轮对话中系统提示/指令遵从随 KV 压缩的跨轮衰减 ★★☆
- **假设**：在多轮场景下，系统提示离当前 query 越来越远，驱逐/量化导致的指令遵从衰减随轮数累积，并且是非线性的（在某个阈值后坍塌）。只需加一条"系统提示保护 + 均衡保留"的规则（类似 Random Attention 的保 prompt 思路）就能修复。
- **最小验证**：Llama-3.1-8B、Qwen3-8B；构造 IFEval 约束的系统提示加 10–30 轮对话（MT-Eval / Multi-IF 风格）；比较 SnapKV、H2O、StreamingLLM、KV-INT4、IntentKV。按轮统计约束遵守率。约 1 GPU-day。
- **最接近工作**：The Pitfalls of KV Cache Compression 2510.00231（单 prompt 多指令）；IntentKV 2606.09916；CommitKV 2608.07855；No Token Left Behind 2402.18096。
- **风险**：结论可能被视为 2510.00231 的直接延伸。
- **查新**：1 次 WebSearch（检索结果明确说没有多轮衰减研究）+ 2 次 arXiv。**CROWDED-BUT-DELTA**。

### Idea 7 — 剥离的 thinking 作为下一轮的投机解码草稿库 ★☆☆（可并入 Idea 1）
- **假设**：上一轮被剥离的 thinking 里常常已经包含下一轮会复述的推导/代码片段。把它作为 suffix-tree 草稿库（零训练）可以提高下一轮的接受长度。
- **验证**：同 Idea 1 的多轮设置，比较 SuffixDecoding（只用可见历史）与加入剥离 thinking 的版本的接受长度和加速。
- **最接近工作**：SuffixDecoding 2411.04975；SSR 2608.20359；Multi-Sample SD 2503.05330。
- **查新**：1 次 WebSearch，没有命中。**NOVEL-LIKELY** 但增益可能很小，适合当 Idea 1 的附加章节。

### Idea 8 — 文本 dLLM 加速器的"一致性而非准确率"审计 ★☆☆
- **假设**：Fast-dLLM / dLLM-Cache / Polestar 等在纯文本推理上也存在内容漂移：准确率保住了，但输出与未加速版本不一致；漂移主要来自陈旧缓存而不是并行阈值。
- **验证**：LLaDA-8B / Dream-7B，GSM8K/MATH/HumanEval，统计 agreement 与 speedup 的前沿。
- **最接近工作**：Faster but Different 2607.29079（多模态 dLLM）；Polestar 2607.14107；Flash-dLLM 2609.26796。
- **查新**：2 次 arXiv。**CROWDED-BUT-DELTA**（只是从多模态迁移到文本，增量有限）。

---

## 3. 已查实"已被做过"的方向（不要做）
| 想法 | 已有工作 |
|---|---|
| 并行样本互为 n-gram 草稿（self-consistency/BoN） | Speculative Decoding for Multi-Sample Inference 2503.05330 |
| KV 驱逐导致 CoT 变长 | Hold Onto That Thought 2512.12008；SkipKV 2512.07993 |
| 量化导致 CoT 变长 / 回退 FP16 | 2606.25519；2606.02011；2606.00206 |
| 推理模型 PTQ 用自生成 CoT 校准 | AYOT 2608.01078；2504.04823；FAQ 2601.11200 |
| 头间互补 / coverage 驱逐 | K-VEC 2606.29563；ECOKV 2609.06663；CORE 2610.02235 |
| 多轮稀疏 prefill 跨轮复用 | LoopServe 2507.13681 |
| 推理模型稀疏 KV 自投机 | SSR 2608.20359；S³R 2610.13202；Sparse Glimpse 2607.27735；Cassandra 2605.26558 |
| 压缩 KV 换更多样本/更长推理 | Inference-Time Hyper-Scaling 2506.05345；BreadthKV 2610.05685 |
| 树搜索/ToT KV 管理 | ArborKV 2605.22106 |
| dLLM 批解码 | Batch-Paced 2610.17917；BlockServe 2607.08930 |

## 4. 不确定性声明
- arXiv 检索是 AND 语义，长查询经常返回 0 条，所以"未命中"不等于"不存在"。Idea 1 和 2 在动手前应再做一次 Semantic Scholar 或 Google Scholar 的全文检索，并查看 NeurIPS 2026 / ICLR 2027 投稿。
- 本领域 2026 下半年月增速极高，从立项到投稿的 2–3 个月窗口内被抢先的概率不低，Idea 1/2 应尽快出 pilot。
- 我没有读完 2609.03430、2606.17034、2610.21126 的全文；它们的附录可能已经覆盖部分实验（例如 RoPERelay 可能测过多轮场景）。
