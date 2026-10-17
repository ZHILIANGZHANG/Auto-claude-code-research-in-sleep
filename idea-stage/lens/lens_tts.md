# Lens 报告：LLM Test-Time Scaling（training-free / 少训练方向）

日期：2026-10-17。调研者：ARIS 子代理（Phase 1 + Phase 1.5）。

## 0. 方法与数据可信度说明

- 本环境下 `arxiv.org`、`export.arxiv.org`、`huggingface.co`、`alphaxiv.org`、`semanticscholar.org` 的 HTTPS CONNECT 全部被代理拒绝（403），WebFetch 也 DNS 失败。**因此所有论文信息来自 WebSearch 返回的摘要/片段（多数直接引自 arXiv abs/html 或会议页面）**，没有逐篇读完 abs 页面。共用了 ~45 个查询式（超过要求的 12 个），覆盖 8 个子方向。
- 每篇论文都标注了 arXiv ID 与来源；带「⚠」的条目表示只看到部分摘要，数字需在后续阶段复核（2026-10-17 已按 §0.5 逐篇核实，剩余 ⚠ 仅为会议录用状态等无法从 abs 页确认的信息）。没有任何条目是编造的；拿不准的地方明确写"未确认"。
- 搜索配额在最后一轮用尽，以下几篇只拿到标题/会议信息而无摘要细节：2502.12118（Setlur et al.）、SpecExit 2509.24248、PETS 2602.16745、BrowseConf (ICLR 2026)、"Think Deep, Think Fast" (ICML 2026)、"Confidence Leaps" (EACL 2026)、"Variation in Verification" (ICLR 2026)、"Test-time Verification via Optimal Transport" (ICLR 2026)。后续若要引用需补读。（→ 已在 §0.5 补读并修正。）

## 0.5 核实记录（2026-10-17，abs 页面）

方法：`arxiv.org/abs/<id>`（标题/日期/评论/摘要）逐篇核对；需要具体数字时抓 `arxiv.org/html/<id>` grep；非 arXiv 论文用 CrossRef + OpenReview 搜索 + ACL Anthology。标记：✅ 一致 / ✏️ 修正 / ❌ 未找到 / ⚠ 仍无法核实。下面各条目已按本记录就地改正。

**§0 中只有标题的论文（已补齐）**
- 2502.12118 ✅ "Scaling Test-Time Compute Without Verification or RL is Suboptimal"（Setlur, Rajaraman, Levine, Kumar，2025-02）。理论 + 3/8/32B 实证：固定算力下 verifier-based（RL/搜索）优于 verifier-free（蒸馏/克隆搜索轨迹）；当基座对正确解分布异质且奖励分布不尖锐（anti-concentration）时，VF 的次优性随测试时预算增长而**扩大**。注意：这里的 VF/VB 指**训练范式**（蒸馏 vs RL/搜索），不是直接讨论推理时投票 vs 验证器。
- SpecExit 2509.24248 ✅ 2025-09（v2 2025-10）。用轻量 draft 模型的隐状态同时预测未来 token 与退出信号，无 probing 开销；平均生成长度 −66%，端到端延迟比 speculative-decoding 基线快 2.5×，不损精度。原文明确写：DEER 在 Qwen3-4B-Thinking-2507 上把 GSM8K/ARC-C 长度减 32%/44%，但端到端延迟反而高于 vanilla；probing 短语 "Final Answer is" 在代码任务失效。⚠ "ICML 2026" 未在 abs 页确认。
- PETS 2602.16745 ✅ 2026-02（v2 2026-05）。定义 self-consistency rate（与无限预算多数票的一致率）；离线场景把轨迹建模为众包 worker，在线流式场景按难度自适应预算，均有理论保证；GPQA 上在离线/在线分别省 75%/55% 采样预算达到完美 self-consistency。TF=是。
- BrowseConf ✏️ 会议修正：**ACL 2026 Findings**（2026.findings-acl.21）+ LLA 2026 workshop，arXiv 2510.23458，**不是 ICLR 2026**。内容：搜索 agent 在长动作序列后给 verbalized confidence；高置信时精度高、低置信接近 0；用置信决定是否重试直到达阈值，省 token 且与固定预算 TTS 相当。TF=是。
- "Think Deep, Think Fast" (ICML 2026) ⚠ 按此标题在 CrossRef/OpenReview 均无结果；最接近的是 **"Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"（ICML 2026, OpenReview PRSE0U5eGT）**：定义 deep-thinking token（深层才收敛的 token），其比例 DTR 与正确率正相关、优于长度与置信基线；Think@n 按 DTR 选样本、用短前缀提前剔除。原标题疑为记错，待确认。
- "Confidence Leaps in LLM Reasoning: Early Stopping and Cross-Model Transfer" ✅ EACL 2026 short（2026.eacl-short.43）。发现置信常在一个"顿悟时刻"突然跃升（confidence leap）；据此做 TF、模型无关的早停；leap 前的推理文本可跨模型族迁移。无具体数字（摘要）。
- "Variation in Verification: Understanding Verification Dynamics in LLMs" ✅ ICLR 2026 Poster，arXiv 2509.17995。12 benchmark × 14 开源模型 + GPT-4o 研究生成式 verifier：易题更易验证；弱 generator 的错更易被检出；验证能力与自身解题能力相关但随难度变化；同一 verifier 下弱 generator 可接近强 generator（Gemma2-9B→27B 差距缩小 75.7%）。
- "Test-time Verification via Optimal Transport: Coverage, ROC, & Sub-optimality" ✅ ICLR 2026 Poster。把可验证 TTS 建模为传输问题，sub-optimality–coverage 曲线有三个区间（transport / policy-improvement / saturation），分析 sequential 与 batched 采样；Qwen/Llama/Gemma 实证。

**带 ⚠ 的条目**
- A6 2501.18585 ✏️ "+10.7%" 未在 abs/html 找到；html 实际数字：TIP+Self-Consistency 把 QwQ-32B-Preview AIME24 Pass@4 从 43.7→51.4；TIP+Laconic Decoding 把 R1-Distill-Qwen-32B Pass@4 从 74.4→78.0。NeurIPS 2025 未在 abs 确认。
- A9 DEER 2504.15895 ✅ 数字一致（10 benchmark × 11 模型，−19.1%~−80.1%，+0.3~5.0%）。✏️ limitation 中的 "0.52–0.71× 速度" 在 SpecExit/PUMA 文本中均未找到，改为定性表述（SpecExit：DEER 在 Qwen3-4B-Thinking-2507 上延迟高于 vanilla；PUMA：DEER 在 DS-7B/14B 上比 Full CoT 更慢）。⚠ "ICLR 2026" 未在 abs 确认（html 伦理声明引用 ICLR Code of Ethics，说明投了 ICLR）。
- A17 2508.16665 ✅ 该结论原文是转述 Setlur et al. (2025b)=2502.12118，不是综述自己的发现。
- B6 2601.21619 ✅ 摘要一致；html 补充：LanBo 多数设置省 >50% 显存、延迟显著低于其他方法（因 PreAda 保留解码并行）。无统一精度数字。
- B8 PETS ✅ 见上。
- B9 2608.05643 ✅ AMC 25.0→32.5（Qwen2.5-1.5B），MATH500 58.0%；摘要确有与 verifier-based BoN / beam / lookahead 的比较，但未见 compute-matched 说明。投稿 EMNLP 2026。
- B10 Consilience 2608.09898 ✅ 论点一致；html 数字：Self-Certainty 在 OSS-20B HMMT 上 76.5→68.0，DeepConf-20% 在 OSS-120B LCB 上 65.7→61.9（均低于 Pass@1）；Consilience 把 OSS-120B LCB 提到 69.7%（Pass@1 65.7%）；n=8 时 +4.2 分、与 Self-Verify 相当但全并行（36 vs 59 分钟）。
- C3 SpecExit ✅ 见上。
- C6 2604.22266 ✏️ 摘要原文：Qwen3-4B 仅 32% 题答案中途改变；最后一次切换后平均再生成 760 token；probe-based 早停**每题省 500 token、精度降 2%**（去掉"500–1000"）。
- C8 2605.17672 ✏️ 补全：方法名 **PUMA**（Redundancy Detector + 答案级验证），5 个 LRM × 5 benchmark 平均 token −26.2%；DS-7B/14B 上 1.40×/1.28× 加速；DEER 比 Full CoT 慢；指出答案级信号（置信/试答一致）反映"答案就绪"而非"推理收敛"，会过早退出。TF=主流程 TF（detector 为轻量模块；另有 learned stopping-policy 实验）。
- D1 2512.02304 ✅ 结论补全：跨模型族验证 > 自验证/同族验证，solver 与 verifier 越像增益越小；reasoning post-training 削弱自改进但增强跨族改进；数学/逻辑任务更受益。ICLR 2026 RSI workshop。
- D3 TTSP ✏️ 会议：**ICML 2026 DL4C workshop poster**（非主会）；数字一致（LCB v6 难度校准子集无 verifier 极限 mean@8 +35% 绝对；基座 coverage≈0 时失效）。✏️ **TF=否**：它是 test-time training（非对称自博弈更新模型）。
- D5 ✅ 全名 "Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification"，ACL 2026 Findings（1243）。DeepVerifier（rubric 由失败分类学导出）meta-eval F1 比 agent-as-judge 高 12–48%；GAIA/XBench 难子集 +8–11%。TF=是。
- F4 ATLAS 2606.01667 ✅ Claude Sonnet 4.6 骨干：HLE-Verified 56.00%、LCB 82.29%、GPQA-D 85.75%；多模型版 ATLAS-MM LCB 85.63%。TF=是（prompt 级 orchestrator）。
- F5 RGV 2608.24024 ✅ 补数字：4 个搜索 agent benchmark × 5 模型，比置信投票最高 +5.4%，minority-correct 题 +35%；失效原因命名为 copy inflation（检索文档复制 token 的 logprob 虚高）。EMNLP 2026 Findings。
- F11 CoSPlay ✅ 2605.23491，2026-05。Qwen2.5-7B-Instruct 平均 BoN 22.1→33.2%，UT 准确率 14.6→78.3%，追平 RLVR 模型 CURE-7B。TF=是。
- G4 2509.15148 ✏️ 名称修正：**ATTS（Asynchronous Test-Time Scaling via Conformal Prediction）**，不是 "A1"；ICLR 2026；56.7× 加速、4.14× 吞吐 ✅。
- H6 TF-TTCL 2604.13552 ✅ 补数字（html）：数学平均 44.86%（GSM8K 87.49 / MATH-500 54.00 / AIME24 13.33 / Minerva 24.63）。
- 其他非 arXiv：C9 REFRAIN ✅ 正式标题 "Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning"（ACL 2026 long 1256），token −20~55%；F6 TUMIX ✅ ICLR 2026 Poster，49% 成本、+3.55%；F9 ✅ ACL 2026 Findings 383；G3 ✅ arXiv 2509.04474，ICLR 2026 Poster；G5 ✏️ 是 **NeurIPS 2025 ER workshop**（非主会）。

**10 个 idea 的 Closest work（ID↔标题、日期、TF、数字、摘要覆盖度）**
- B1 DeepConf 2508.15260 ✅ 2025-08；ICLR 2026 Poster（OpenReview 8LqHs0KIM7）✅；AIME25 DeepConf@512 99.9%、token −84.7% ✅；TF=是 ✅。html 自认 "confidently wrong" 情形会导致回退（某设置下 −4.69%~−0.31%）。**覆盖**：摘要只做离线过滤 + 在线早停，**未**做难度分桶或跨方法相图 → Idea 1 核心未被覆盖；Idea 7 的"随机过滤对照"未做。
- B2 CGES 2511.02603 ✅ 2025-11（v2 2026-06，扩展版）；NeurIPS 2025 Efficient Reasoning workshop ✅；TF=是 ✅。✏️ 摘要新数字：5 个 benchmark 平均 call −58%（16.0→6.7），精度与 SC 差 ≤0.4pp。GPQA="uninformative case"（附录 J）✅；"Thm 3.5" 编号未核。**覆盖**：已有"校准 vs 区分度"的信号质量分析，但只在单一方法、未按难度分桶 → Idea 1/10 部分重叠。
- B10 Consilience 2608.09898 ✅ 2026-08；TF=是 ✅。**覆盖**：摘要已明确声称"置信方法在复杂任务上灾难性失效、高置信=未探索" → Idea 1 的假设 H 的"难题桶崩溃"部分**已被其摘要陈述**，Idea 1 的增量只剩跨方法/跨难度/跨模型统一相图与切换规则。
- A9 DEER 2504.15895 ✅（见上）。**覆盖**：Idea 2/3 的基线，摘要无延迟数字 → 不覆盖。
- C3 SpecExit 2509.24248 ✅。**覆盖 Idea 2 的审计论点**：摘要已写"probing 开销限制端到端延迟增益"，正文已给出 DEER 延迟反升的实证 → Idea 2 的 Hypothesis 前半句（token 降但 wall-clock 不降）**已被 SpecExit 和 PUMA 部分证明**；Idea 2 剩余增量是"零额外前向的批同步探测"与四指标系统审计。
- C8 PUMA 2605.17672 ✅。**覆盖**：报告了 wall-clock 加速与 DEER 变慢 → 同上，Idea 2 审计部分重叠；PUMA 的 redundancy detector 不是批同步设计。
- B3 SeerSC 2511.09345 ✅ 2025-11（v2 2026-01）；−47% token、−43% 延迟 ✅；TF=是（System-1 估计用同模型快速模式）。**覆盖**：Idea 4 的"解码前按题分配预算"已有（但用熵估计，不用历史记忆）；Idea 2 的"token≠延迟"论点已有。
- C6 2604.22266 ✅ 2026-04（见上数字修正）。**覆盖 Idea 3**：摘要已包含"强制答案补全探测中间答案 + 最后一次切换位置 + 基于探针的早停（省 500 token、−2%）"→ Idea 3 的工具与简单方法**已被覆盖**；未覆盖的是把它变成跨方法评测协议（premature rate / overshoot / flip loss）与阈值校准。**覆盖 Idea 5**：强制补全得到中间答案的手段相同，但未按"Wait"位置与难度做 2×2。
- A13 NoWait 2506.08343 ✅ 2025-06；−27%~−51%（5 个 R1 系列，10 benchmark 含视觉/视频）✅；TF=是 ✅；EMNLP 2025 Findings 未在 abs 确认。**覆盖**：无条件压制 → 不覆盖 Idea 5 的条件规则。
- C1 CGRS 2508.05337 ✅ 2025-08；AAAI 2026（abs 评论）；−18.5%~−41.9%（AIME24/AMC23/MATH500/GPQA-D）✅；TF=是 ✅。**覆盖 Idea 5**：摘要已是"当前回答高置信时压制反思触发词"→ Idea 5 第 (4) 步的条件压制规则**与 CGRS 基本相同**（CGRS 用置信，Idea 5 用"中间答案已对 + 熵低"，但推理时无法知道"已对"，可操作信号仍是置信/熵）。Idea 5 的增量只剩 2×2 因果分解与"插入 vs 压制"统一比较。
- A11 α1 2505.24863 ✅ 2025-05；TF=是；html 确认需先在 10 个随机样本上估计平均思考长度，α=1.4。**覆盖**：不覆盖。
- B7 Parallel-Probe 2602.03845 ✅ 2026-02；−35.8% 顺序 token、−25.8% 总 token ✅；TF=是 ✅；"ICML 2026" 未在 abs 确认。**覆盖 Idea 6/9**：摘要是"周期性让所有分支出中间答案 + 共识早停 + 偏离剪枝"，限数学答案 → Idea 6 的迁移对象；Idea 9 的"全局停止 + 剪枝"框架相同，差别只在信号（隐状态 vs 中间答案）。
- F1 CATTS 2602.12276 ✏️ 2026-02（v2 2026-08）；v2 摘要数字：WebArena-Lite / Online-Mind2Web / GoBrowse 比多数投票最高 **+11.8%**，token 少于均匀扩展；原文 "+9.1% vs ReAct、2.3×" 在 v2 html 未找到。TF=是 ✅。LLM Arbiter 推翻高共识 ✅。**覆盖 Idea 6**：步级投票熵/margin 分配算力；附录还测了 DeepConf 变体在 GoBrowse 上 → 不做轨迹级共识，Idea 6 未被覆盖。
- F2 2602.18998 ✏️ 标题 "**Evaluating** Test-Time Scaling of General LLM Agents"（benchmark 名 General AgentBench ✅），v2 2026-09；"scaling plateau"（原文写"上下文天花板"→改为 plateau）+ "verification gap" ✅。
- F3 AggAgent 2604.11753 ✅ COLM 2026；+5.3% 平均、deep research +10.3%、成本 ≤ 单次 rollout ✅。**覆盖**：事后聚合，不在线早停 → Idea 6 未覆盖。
- G1 ST-BoN 2503.01422 ✅ NeurIPS 2025 **Spotlight**；显存 −80%+、延迟 −50%、算力省 70–80%、同成本 +3–4 分 ✅；TF=是 ✅。**覆盖 Idea 9**：摘要已是"利用模型内部状态的早期采样一致性识别最优路径并截断次优路径" → Idea 9 的核心信号（跨轨迹隐状态一致性）**已被 ST-BoN 使用**；Idea 9 唯一差别是"收敛即全体停止 + 投票"而非"保留一条继续"，novelty 应下调。
- H2 TTSV 2512.04748 ✅ AAAI 2026；Qwen2.5-Math-7B 四集平均 30.78→48.00 ✅；✏️ LLaMA-3.1-8B 平均 **+0.42**（26.13→26.55；"+0.20" 是 EM 基线的增量），MATH500 −1.20 ✅。TF=几乎（只优化前置向量）✅。**覆盖 Idea 10**：只观察未解释 → 不覆盖。
- B11 2601.06423 ✅ 2026-01；GPT-5.2 78→90%（N=5）忠实度 0.540→0.510；Claude Opus 4.5 78→74.3% 忠实度 0.270→0.891 ✅。
- 2606.15455 ✅ "Understanding Diversity Collapse in RLVR via the Lens of Overtraining"，2026-06；Pass@1 升、高 k Pass@k 降；提出 BBG。**覆盖 Idea 10**：不涉及置信/TTS 信号 → 不覆盖。
- H5 Dynamic Cheatsheet 2504.07952 ✅ 2025-04；GPT-4o Game of 24 10%→99% ✅；Claude GPQA-D +9%；TF=是；"EACL 2026" 未在 abs 确认。**覆盖 Idea 4**：记忆存策略/代码片段，不存采样预算 → 不覆盖；但 PETS 的在线流式预算分配（有理论保证）与 Blend-ASC（B5，ICML 2026，比 SC 少 4.8× 样本、无超参、支持 batching）是 Idea 4 更强的基线。
- F8 2608.01207 ✅ 2026-08；PGS 在 TextVQA 对 CoT-only 多数投票 +31.8 但 MatchedCtrl 追平 → 增益消失 ✅。**覆盖 Idea 7**：方法论相同但限 VLM；LLM 侧未做。
- A8 NoThinking 2504.09858 ✅ 700 token ACM23 51.3 vs 28.9 ✅。**覆盖 Idea 8**：摘要已做低预算 compute-matched 比较（7 个数据集）和 NoThinking×N 并行扩展，且声明"低预算/低延迟下的竞争性参考" → Idea 8 的策略空间中 "NoThinking+BoN vs Thinking" 这一对比**已被覆盖**；未覆盖的是 early-exit、refine、budget forcing 四路同台与查找表。
- A3 Snell 2408.03314 ✅ >4×、14× ✅；2024-08。
- B9 2608.05643 ✅（Idea 8 基线）。
- 其他检查过的 ID（标题均与文件一致）：2502.12215、2505.15778、2506.13752、2506.05295、2507.14417（TMLR 2025 Featured）、2503.16419（TMLR 2025）、2503.22048（EMNLP 2025）、2607.05391、2502.20379、2502.05171、2510.07358、2607.09438（ImageCLEF 2026）、2502.14382、2503.05330、2604.00510、2605.04263、2505.20633（ICML 2025）、2602.09719、2606.21803（EMNLP 2026 Findings）、2511.20857、2502.06703、2501.19393、2511.00751、2603.14251（html 确认 W=512、λ=2.0、贪心、8×H20）、2604.04930、2606.15070（ICML 2026 Spotlight ✅；"相对 +4.4%" 未找到）、2510.00071（NeurIPS 2025 FoRLM workshop；−53% token/−46.1% 延迟/−57.9% 能耗）、2511.12309（ICML 2026 ✅）。

**仍无法核实**：各 "ICLR/ICML/NeurIPS/EMNLP 2026" 录用状态凡 abs 评论未写的（DEER、SpecExit、Parallel-Probe、NoWait、Dynamic Cheatsheet、2601.21619 NeurIPS 2026、2603.14251 NeurIPS 2026、A6 NeurIPS 2025）；"Think Deep, Think Fast" 的真实标题；CGES "Thm 3.5" 编号；A6 "+10.7%" 与 DEER "0.52–0.71×" 的出处（已改为可核实表述）。

---

## 1. 论文卡片（按子方向分组）

格式：标题 | arXiv ID / 日期 / 会议 | 问题 | 方法一句话 | 关键数字 | training-free? | 自认 limitation / 后续工作

### 1.1 锚点（2022–2025 上半年）

**A1. Self-Consistency Improves Chain of Thought Reasoning in Language Models** — [2203.11171](https://arxiv.org/abs/2203.11171), 2022-03, ICLR 2023。采样多条 CoT、对最终答案多数投票。TF=是。Limitation：N 固定、成本线性增长（后续工作反复指出）。

**A2. Tree of Thoughts: Deliberate Problem Solving with LLMs** — [2305.10601](https://arxiv.org/abs/2305.10601), 2023-05, NeurIPS 2023。把推理组织成"思维"节点的树搜索（BFS/DFS + LLM 自评）。TF=是。Limitation：依赖手工分解与 LLM 自评价值，成本高。

**A3. Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters** (Snell et al.) — [2408.03314](https://arxiv.org/abs/2408.03314), 2024-08, ICLR 2025。问题：如何按 prompt 难度分配测试时计算。方法：PRM 引导搜索 + 序列修订，按难度分桶选策略。关键数字：compute-optimal 比 best-of-N 效率高 >4×；FLOPs 匹配下小模型 + TTS 可胜 14× 更大模型（仅在小模型有非零成功率的题上）。TF=否（需 PRM 与修订模型）。Limitation：难度估计用 oracle/分位数；只在 MATH；难题上 TTS 不如预训练。

**A4. Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling** (Liu et al.) — [2502.06703](https://arxiv.org/abs/2502.06703), 2025-02, ICLR 2025 workshop。reward-aware compute-optimal TTS，用绝对难度阈值替代分位数。关键数字：1B 在 MATH-500 超 405B；7B 超 o1/R1。TF=否（PRM）。Limitation：高度依赖 PRM 选择和 policy-PRM 匹配。

**A5. s1: Simple test-time scaling** — [2501.19393](https://arxiv.org/abs/2501.19393), 2025-01。1K 样本 SFT + budget forcing（强制结束 / 追加 "Wait"）。关键数字：AIME24 50%→57%。TF=budget forcing 本身是 TF，但依赖 SFT 后模型。Limitation：追加 "Wait" 收益很快饱和，过长会退化或撞上下文窗口。

**A6. Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs** — [2501.18585](https://arxiv.org/abs/2501.18585), 2025-01, NeurIPS 2025。频繁思路切换与错误相关；提出 Thought Switching Penalty (TIP) 解码惩罚。关键数字（已核实，html）：TIP+Self-Consistency 把 QwQ-32B-Preview AIME24 Pass@4 43.7→51.4；TIP+Laconic Decoding 把 R1-Distill-Qwen-32B Pass@4 74.4→78.0（原「+10.7%」未找到出处）。TF=是。Limitation：阈值/惩罚强度手调；只在数学。

**A7. Revisiting the Test-Time Scaling of o1-like Models** — [2502.12215](https://arxiv.org/abs/2502.12215), 2025-02。发现 QwQ/R1/LIMO 上正确解比错误解更短；在 AIME/Omni-MATH 上出现 inverse scaling。TF=分析性。

**A8. Reasoning Models Can Be Effective Without Thinking (NoThinking)** — [2504.09858](https://arxiv.org/abs/2504.09858), 2025-04。prompt 跳过 thinking 段 + 并行 best-of-N。关键数字：700 token 预算下 ACM23 51.3 vs 28.9；pass@k 随 k 增大优势扩大。TF=是。Limitation：依赖验证器或 confidence 选择；只在 R1-Distill 系列。

**A9. Dynamic Early Exit in Reasoning Models (DEER)** — [2504.15895](https://arxiv.org/abs/2504.15895), 2025-04, ICLR 2026。在 "Wait" 等转折点插入试答，答案置信高则退出。关键数字：10 benchmark × 11 模型，CoT 长度 −19.1%~−80.1%，准确率 +0.3~5.0。TF=是。Limitation（后续论文指出，已核实）：试答 probing 的额外前向使端到端延迟反而上升——SpecExit 报告 Qwen3-4B-Thinking-2507 上 GSM8K/ARC-C 长度 −32%/−44% 但延迟高于 vanilla，PUMA 报告 DS-7B/14B 上 DEER 比 Full CoT 慢（原「0.52–0.71×」未找到出处）；probing 短语在代码任务失效。

**A10. Soft Thinking** — [2505.15778](https://arxiv.org/abs/2505.15778), 2025-05, NeurIPS 2025。用概率加权的 embedding 混合代替离散 token 作为推理 token。关键数字：pass@1 最多 +2.48，token −22.4%。TF=是。Limitation（自认）：concept token 对模型是 OOD，长链会崩溃；Cold Stop 只是缓解，根本解需要训练。

**A11. AlphaOne (α1)** — [2505.24863](https://arxiv.org/abs/2505.24863), 2025-05。在 α-moment 前按 Bernoulli 调度插入 "wait"，之后强制结束 thinking。TF=是。Limitation（自认）：需要先跑 10 个样本估计平均思考长度；依赖 "wait" 作为有效转折 token。

**A12. Steering LLM Thinking with Budget Guidance** — [2506.13752](https://arxiv.org/abs/2506.13752), 2025-06, ACL 2026 Findings。轻量预测器建模剩余思考长度的 Gamma 分布，逐 token 引导采样。关键数字：紧预算下 MATH-500 最多 +26%。TF=主模型不训，但预测器需训。

**A13. Wait, We Don't Need to "Wait"! (NoWait)** — [2506.08343](https://arxiv.org/abs/2506.08343), 2025-06, EMNLP 2025 Findings。推理时把反思 token logits 置负。关键数字：5 个 R1 系列 CoT −27%~−51%，"不损性能"。TF=是。Limitation：是否在模型不确定的难题上损害准确率，未充分回答。

**A14. Sample Complexity and Representation Ability of Test-time Scaling Paradigms** — [2506.05295](https://arxiv.org/abs/2506.05295), 2025-06, ICLR 2026。理论：self-consistency 需 ~1/Δ² 样本，best-of-n 需 ~1/Δ；自纠正 + verifier 反馈可模拟在线专家学习。TF=理论。

**A15. Inverse Scaling in Test-Time Compute** — [2507.14417](https://arxiv.org/abs/2507.14417), 2025-07, TMLR 2025 / ICLR 2026。构造计数干扰、伪特征回归、约束追踪等任务，更长推理反而变差。TF=分析性。

**A16. Stop Overthinking: A Survey on Efficient Reasoning for LLMs** — [2503.16419](https://arxiv.org/abs/2503.16419), v4 2025-08, TMLR。三分类：model-based / output-based / prompt-based。

**A17. Trust but Verify! A Survey on Verification Design for Test-time Scaling** — [2508.16665](https://arxiv.org/abs/2508.16665), 2025-08。已核实：综述正文转述 Setlur et al. 2025b（=2502.12118）的结论——无 verifier 的 TTS 次优，且与 verifier-based 的差距随预算增大而扩大；该结论不是综述自己的发现，且 2502.12118 的 VF/VB 指训练范式（蒸馏 vs RL/搜索）。

### 1.2 并行采样 / 自适应 N / confidence 选择（2025-07 之后）

**B1. Deep Think with Confidence (DeepConf)** — [2508.15260](https://arxiv.org/abs/2508.15260), 2025-08, ICLR 2026 Poster（OpenReview 已核实）。局部 token logprob 置信（group / bottom-k / tail）做离线加权投票和在线提前终止低置信 trace。关键数字：AIME25 DeepConf@512 达 99.9%，token −84.7%（GPT-OSS-120B）。TF=是。Limitation：未发现独立复现；html 自认存在 "confidently wrong" 回退（某设置 −4.69%~−0.31%）；对"全程自信地错"无能为力（Consilience 2608.09898 正是攻击此点）。

**B2. CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency** — [2511.02603](https://arxiv.org/abs/2511.02603), 2025-11, NeurIPS 2025 workshop。用标量置信构造候选答案后验，后验质量超阈值即停。关键数字（v2 摘要）：5 个 benchmark 平均 call −58%（16.0→6.7），精度与 SC 差 ≤0.4pp。TF=是。Limitation（自认；"directional drift" 条件，定理编号未核）：只在置信信号"有信息"（正漂移）时有效；GPQA 被作者称为 "uninformative case"，置信区分不了对错。

**B3. Seer Self-Consistency (SeerSC)** — [2511.09345](https://arxiv.org/abs/2511.09345), 2025-11。快速 System-1 估计答案熵，提前决定每题采样数，保持并行。关键数字：token −47%，延迟 −43%。TF=是。核心论点：token 减少 ≠ 延迟减少（序列式 early-stop 丢失并行性）。

**B4. Self-Consistency Is Losing Its Edge / Reevaluating SC Scaling** — [2511.00751](https://arxiv.org/abs/2511.00751), 2025-11。Gemini 2.5 在 HotpotQA/MATH-500 上 20 样本只 +0.4/+1.6，高 N 下有时下降。TF=分析性。Limitation：仅 Gemini、仅两个数据集，样本小。

**B5. Optimal Self-Consistency for Efficient Reasoning** (Feng et al.) — [2511.12309](https://arxiv.org/abs/2511.12309), 2025-11, ICML 2026。mode estimation / voting theory 统一分析，推导 SC 的 power-law，提出 Blend-ASC 动态分配。TF=是。

**B6. On the Overscaling Curse of Parallel Thinking** — [2601.21619](https://arxiv.org/abs/2601.21619), 2026-01, NeurIPS 2026。全局 N 为数据集精度而设，但多数样本小 N 即饱和（"overscaling curse"）；用 latent 表征的轻量预测器在解码前分配每题预算（LanBo / PreAda）。TF=主模型不训，预测器需训。关键数字（html）：LanBo 多数设置省 >50% 显存、延迟显著低于其他方法；摘要无统一精度数字（已核实）。

**B7. Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing** — [2602.03845](https://arxiv.org/abs/2602.03845), 2026-02, ICML 2026。周期性让所有分支给中间答案，全局共识达成即停并剪掉偏离分支。关键数字：顺序 token −35.8%，总 token −25.8%。TF=是。发现：width–depth 分配非单调。

**B8. PETS: Principled Framework Towards Optimal Trajectory Allocation for Test-Time SC** — [2602.16745](https://arxiv.org/abs/2602.16745), 2026-02（v2 2026-05）。已核实：定义 self-consistency rate（与无限预算多数票的一致率）；离线场景把轨迹当众包 worker 做分配，在线流式场景按难度自适应预算，均有理论保证；GPQA 上离线/在线分别省 75%/55% 预算达到完美 self-consistency。TF=是。

**B9. Refining Over Resampling: Test-Time Self-Correction** — [2608.05643](https://arxiv.org/abs/2608.05643), 2026-08。宽度（多 rollout）+ 深度（自批评-自纠正）后投票。关键数字：Qwen2.5-1.5B AMC 25.0→32.5。TF=是。Limitation：自纠正对小模型是否可靠长期争议；摘要有与 verifier-BoN/beam/lookahead 的比较但无 compute-matched 说明（已核实）；投稿 EMNLP 2026。

**B10. Consilience for Verifier-Free Test-Time Scaling** — [2608.09898](https://arxiv.org/abs/2608.09898), 2026-08。论点：高置信常意味着"没探索"，置信选择在难题上崩溃；提出惩罚早期高置信、要求最终确定的组合指标。TF=是。关键数字（html，已核实）：Self-Certainty 在 OSS-20B HMMT 76.5→68.0、DeepConf-20% 在 OSS-120B LCB 65.7→61.9（均低于 Pass@1）；Consilience 把 OSS-120B LCB 提到 69.7%；n=8 时 +4.2 分、与 Self-Verify 相当但全并行。直接与 DeepConf 矛盾。

**B11. Does Inference Scaling Improve Reasoning Faithfulness?** — [2601.06423](https://arxiv.org/abs/2601.06423), 2026-01。4 个前沿模型 × 100 GSM8K：SC 对准确率与忠实度影响方向因模型而异（GPT-5.2 准确率升、忠实度不动；Claude Opus 4.5 反之）。Limitation：N=100 太小。

### 1.3 序列式：early exit / 思考长度控制

**C1. Efficient Reasoning via Certainty-Guided Reflection Suppression (CGRS)** — [2508.05337](https://arxiv.org/abs/2508.05337), 2025-08, AAAI 2026。当前回答高置信时压制反思触发词（"Wait"/"Alternatively"），低置信才允许反思，否则压制反思 token。关键数字：token −18.5%~−41.9%。TF=是。

**C2. ARS (Adaptive Reflection Suppression)** — [2510.00071](https://arxiv.org/abs/2510.00071), 2025-10。多 checkpoint 监测确定性，自适应调整压制强度。关键数字：token −53%、延迟 −46.1%、能耗 −57.9%。NeurIPS 2025 FoRLM workshop。TF=是。

**C3. SpecExit: Accelerating LRM via Speculative Exit** — [2509.24248](https://arxiv.org/abs/2509.24248), 2025-09（v2 2025-10），⚠ "ICML 2026" 未在 abs 确认。已核实：用轻量 draft 模型的隐状态同时预测未来 token 与退出信号，无 probing 开销；平均长度 −66%，端到端比 speculative-decoding 基线快 2.5×，不损精度。正文指出 DEER 在 Qwen3-4B-Thinking-2507 上长度减但延迟高于 vanilla。TF=否（draft 模型需训练）。

**C4. Mitigating Overthinking via Reasoning Path Deviation Monitoring (RPDI-EE)** — [2603.14251](https://arxiv.org/abs/2603.14251), 2026-03, NeurIPS 2026。局部/全局平均熵之比作为"偏离指数"，超阈值退出。TF=是。设置：512-token 窗口、阈值 2.0、贪心解码、8×H20。Limitation：阈值手调；只贪心。

**C5. Early Stopping for LRMs via Confidence Dynamics (CoDE-Stop)** — [2604.04930](https://arxiv.org/abs/2604.04930), 2026-04。正确轨迹早达高置信，错误轨迹冗长；用中间答案置信的动态决定停止。关键数字：token −25%~−50%。TF=是。

**C6. Large Language Models Decide Early and Explain Later** — [2604.22266](https://arxiv.org/abs/2604.22266), 2026-04。强制答案补全探测中间预测：Qwen3-4B 仅 32% 的题答案在中途改变；最后一次切换后平均再生成 760 token。基于探针的简单早停可省 500 token / 题，精度损 2%（已按摘要核实）。TF=分析 + 简单启发式。

**C7. Stop When Further Reasoning Won't Help: Attention-State Adaptive Generation (ASAG)** — [2606.15070](https://arxiv.org/abs/2606.15070), 2026-06, ICML 2026 Spotlight。用注意力分布推断推理状态并调整生成。关键数字：Qwen3-8B 平均 +3.2%（或相对 +4.4%），token −40%。TF=是。自认先前 TF 方法"依赖手工 prompt 或不可靠置信"。

**C8. Stop When Reasoning Converges: Semantic-Preserving Early Exit** — [2605.17672](https://arxiv.org/abs/2605.17672), 2026-05。已核实：方法 PUMA = 轻量 Redundancy Detector（语义冗余）+ 答案级验证；5 个 LRM × 5 benchmark 平均 token −26.2%，DS-7B/14B 上 1.40×/1.28× 加速；批评答案级信号（置信/试答一致）反映"答案就绪"而非"推理收敛"，会过早退出；报告 DEER 比 Full CoT 慢。TF=主流程 TF。

**C9. REFRAIN（"Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning"）** — ACL 2026 ([aclanthology 2026.acl-long.1256](https://aclanthology.org/2026.acl-long.1256/))。bandit 控制器按难度调停止阈值；token −20%~−55%。TF=是（bandit 在线）。

**C10. ThinkEdit** — [2503.22048](https://arxiv.org/abs/2503.22048), 2025-03。中层存在"推理长度方向"，权重编辑缓解过短思考。TF=否（权重编辑，但无梯度训练）。

### 1.4 验证器替代

**D1. When Does Verification Pay Off?** — [2512.02304](https://arxiv.org/abs/2512.02304), 2025-12。37 模型 × 9 benchmark；提出 verifier gain 预测 rejection sampling 增益；结论（已核实）：跨模型族验证 > 自验证/同族验证，solver 与 verifier 越相似增益越小；reasoning post-training 削弱自改进但增强跨族改进；数学/逻辑任务更受益。ICLR 2026 RSI workshop。

**D2. LLM-as-a-Verifier** — [2607.05391](https://arxiv.org/abs/2607.05391), 2026-07。评分 token logits 的期望给连续分数，无需训练，面向 agentic 任务。TF=是。

**D3. Test-Time Scaling with Weak Verifiers via Self-Play (TTSP)** — ICML 2026 DL4C **workshop** poster（已核实，非主会）。非对称自博弈的 test-time training；LiveCodeBench v6 难度校准子集无 verifier 极限 mean@8 +35%（绝对）；自认当基座 coverage≈0 时增益消失。TF=否（测试时更新模型）。

**D4. Multi-Agent Verification** — [2502.20379](https://arxiv.org/abs/2502.20379), 2025-02。多个 aspect verifier 投票。TF=是。

**D5. Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification (DeepVerifier)** — ACL 2026 Findings（已核实）。rubric 由自动构建的失败分类学导出；verifier meta-eval F1 比 agent-as-judge 高 12–48%；GAIA/XBench 难子集 +8–11%。TF=是。

### 1.5 隐式 / latent 推理

**E1. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** — [2502.05171](https://arxiv.org/abs/2502.05171), 2025-02, NeurIPS 2025。3.5B 模型、800B token 训练的循环块。TF=否。自认优点：零样本 per-token 自适应计算、KV 共享、可 speculative。
**E2. Encode, Think, Decode (ETD)** — [2510.07358](https://arxiv.org/abs/2510.07358), 2025-10。把现有模型的关键中间层训练成循环块。TF=否。
**E3. Soft Thinking**（见 A10）是目前唯一明确 training-free 的 latent 方案，且自认 OOD。
→ **此子方向 training-free 可做的部分极少，且搜索中未找到 2026 年 training-free 的 layer-looping 工作**（搜索配额用尽前未能确认）。

### 1.6 Compute-optimal / 迁移到 MLLM、agent、代码

**F1. Agentic Test-Time Scaling for WebAgents (CATTS)** — [2602.12276](https://arxiv.org/abs/2602.12276), 2026-02。每步投票分布的熵 / top-1–top-2 margin 决定是否追加计算。关键数字（v2 摘要，已核实）：WebArena-Lite / Online-Mind2Web / GoBrowse 比多数投票最高 +11.8%，token 少于均匀扩展（原"+9.1% vs ReAct、2.3×"未在 v2 找到）；LLM Arbiter 有时推翻高共识决定。TF=是。
**F2. Evaluating Test-Time Scaling of General LLM Agents** — [2602.18998](https://arxiv.org/abs/2602.18998), 2026-02（v2 2026-09；标题已修正）。提出 General AgentBench；10 个领先 agent 在通用设置下大幅退化；序列扩展遇 "scaling plateau"，并行扩展遇 "verification gap"，两者都没有一致增益。
**F3. Agentic Aggregation (AggAgent)** — [2604.11753](https://arxiv.org/abs/2604.11753), 2026-04, COLM 2026。聚合 agent 用轻量工具检查并行轨迹；平均 +5.3%，deep research +10.3%；成本 ≤ 一次 rollout。
**F4. ATLAS: Agentic Test-time Learning-to-Allocate Scaling** — [2606.01667](https://arxiv.org/abs/2606.01667), 2026-06。LLM orchestrator（Claude Sonnet 4.6）通过 explore 动作控制搜索循环；HLE-Verified 56.00%、LiveCodeBench 82.29%、GPQA-D 85.75%（已核实）。TF=是。
**F5. Beyond Confidence: TTS for Multi-Turn Search Agents via Retrieval Grounding (RGV)** — [2608.24024](https://arxiv.org/abs/2608.24024), 2026-08。投票层加入与检索证据的词汇重叠信号；论点：token-logprob 置信投票在多轮搜索 agent 上迁移差，原因是 copy inflation。关键数字（已核实）：4 个 benchmark × 5 模型，比置信投票最高 +5.4%，minority-correct 题 +35%。EMNLP 2026 Findings。TF=是。
**F6. TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture** — ICLR 2026 Poster（已核实）。Gemini-2.5 上比最佳基线 +3.55%；多 agent 不同工具策略互相精炼，置信够高即停；49% 成本保性能。
**F7. Test-Time Scaling for Small VLMs on Multilingual Visual MCQ** — [2607.09438](https://arxiv.org/abs/2607.09438), 2026-07。Qwen2.5-VL-7B / Qwen3.5-4B：答案可解析性和每链 token 预算最重要；PRM beam search 比纯 SC 低 0.39pp 却贵 8×。
**F8. It's the Decoding Format, Not the Perturbation (VLM 审计)** — [2608.01207](https://arxiv.org/abs/2608.01207), 2026-08。图像扰动一致性选择（PGS）的增益在匹配解码格式与预算的对照下消失；多数投票打败基于自验证的选择。
**F9. Test-Time Scaling in Multimodal Foundation Models: A Comprehensive Survey of Generation and Reasoning** — ACL 2026 Findings（已核实）。
**F10. S\*: Test Time Scaling for Code Generation** — [2502.14382](https://arxiv.org/abs/2502.14382), 2025-02, EMNLP 2025 Findings。并行采样 + 序列调试 + 生成区分性输入做选择。GPT-4o-mini+S* 超 o1-preview 3.7%。TF=是。自认只做竞赛代码。
**F11. CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test** — [2605.23491](https://arxiv.org/abs/2605.23491), 2026-05（已核实）。Qwen2.5-7B-Instruct 平均 BoN 22.1→33.2%、UT 准确率 14.6→78.3%，追平 RLVR 模型 CURE-7B。自生成单元测试噪声大、与错误代码虚假耦合；代码与 UT 协同自博弈。TF=是。

### 1.7 TTS × 推理效率

**G1. Sampling-Efficient TTS: Self-Estimating Best-of-N in Early Decoding (ST-BoN)** — [2503.01422](https://arxiv.org/abs/2503.01422), 2025-03 (v3 2025-11), NeurIPS 2025 Spotlight。用内部状态一致性早停次优候选，无 reward model。关键数字：动态显存 −80%+，延迟 −50%，计算省 70–80%。TF=是。
**G2. Speculative Decoding for Multi-Sample Inference** — [2503.05330](https://arxiv.org/abs/2503.05330), 2025-03。并行路径间后缀匹配构造 draft 池，无 draft 模型。TF=是。
**G3. Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling** — [2509.04474](https://arxiv.org/abs/2509.04474), ICLR 2026 Poster（已核实）。N-gram 方法在 best-of-N / 多轮思考里出奇有效（重复模式多）。
**G4. ATTS: Asynchronous Test-Time Scaling via Conformal Prediction** — [2509.15148](https://arxiv.org/abs/2509.15148), 2025-09, ICLR 2026（名称已修正，原写 "A1"）。TTS 下 spec decoding 的同步/显存瓶颈；在线校准 + 三阶段拒绝采样；56.7× TTS 加速、4.14× 吞吐、不损精度（已核实）。
**G5. Efficient Sparse Decoding for TTS with KV Cache Disaggregation and Asynchronism** — NeurIPS 2025 ER workshop（非主会，已核实）。高并发下块稀疏注意力退化。
**G6. Adaptive Parallel MCTS** — [2604.00510](https://arxiv.org/abs/2604.00510), 2026-04。negative early exit 剪无进展轨迹 + adaptive boosting；vLLM 集成降 p99 延迟。自认严格停止条件在 MATH500 只触发 ~5%。
**G7. PARSE: Parallel Prefix Verification for Speculative Generation** — [2605.04263](https://arxiv.org/abs/2605.04263), 2026-05。语义级 spec（SpecReason / Lookahead Reasoning）的并行验证。

### 1.8 轻量 TTT / TTA

**H1. Test-Time Learning for LLMs (TLM)** — [2505.20633](https://arxiv.org/abs/2505.20633), 2025-05, ICML 2025。无标签测试数据上最小化输入困惑度，LoRA 更新。≥20% 域适应增益。TF=否（LoRA）。
**H2. Model Whisper: Test-Time Steering Vectors (TTSV)** — [2512.04748](https://arxiv.org/abs/2512.04748), 2025-12, AAAI 2026。冻结模型，前置可学习向量，目标为输出熵最小化。Qwen2.5-Math-7B 4 个数学集平均 30.78→48.00；**LLaMA-3.1-8B 仅 +0.42（26.13→26.55；原"+0.20"是 EM 基线的增量），MATH500 −1.20**。TF=几乎（只训几个向量）。Limitation：模型依赖性极强；熵最小化已知会放大自信错误。
**H3. Unsupervised Layer-Wise Dynamic TTA** — [2602.09719](https://arxiv.org/abs/2602.09719), 2026-02。超网络预测每层每步学习率乘子，只更新 LoRA。TF=否。
**H4. Test-Time Training with Next-Token Prediction** — [2606.21803](https://arxiv.org/abs/2606.21803), 2026-06。长上下文 TTT 内环目标选择。TF=否。
**H5. Dynamic Cheatsheet** — [2504.07952](https://arxiv.org/abs/2504.07952), 2025-04, EACL 2026。黑盒模型 + 持久可编辑记忆；GPT-4o Game of 24 10%→99%。TF=是。
**H6. Training-Free Test-Time Contrastive Learning (TF-TTCL)** — [2604.13552](https://arxiv.org/abs/2604.13552), 2026-04, ACL 2026 Findings。多角色生成轨迹 → 强弱轨迹对比蒸馏成文本规则 → 检索应用。TF=是。关键数字（html）：数学平均 44.86%（GSM8K 87.49 / MATH-500 54.00 / AIME24 13.33 / Minerva 24.63）。
**H7. Evo-Memory / ExpRAG** — [2511.20857](https://arxiv.org/abs/2511.20857), 2025-11。流式自进化记忆 benchmark。

---

## 2. Landscape map：饱和度判断

| 子方向 | 2025-07 后代表作 | 饱和度 | 判断 |
|---|---|---|---|
| 1 并行采样 + 自适应 N + 置信选择 | B1–B10 | **做烂** | 置信/熵/后验/共识早停已有 ≥8 篇；增量空间只剩"置信信号何时失效"(B2, B10) 与"token≠延迟"(B3) |
| 2a 序列 early exit / 反思压制 | A9, C1–C9 | **做烂** | 信号已经穷举：试答置信、熵比、注意力、反思 token、答案切换；差异在 probing 开销与跨域泛化 |
| 2b 思考长度主动控制（延长） | A5, A11, A12 | 中 | "延长"方向（budget forcing、α1）比"缩短"少很多；但 A15 等显示延长常无益 |
| 2c 树搜索 / MCTS training-free | G6, DPTS | 中偏冷 | 2026 只见系统层优化；无 PRM 的树搜索价值被 F7 等质疑 |
| 3 验证器替代 | D1–D5, A17 | 中 | 理论说 verifier-free 次优 (A17/2502.12118)，实证说 SC 常打败自验证 (F8)；矛盾未解 |
| 4 over/under-thinking 诊断 | A6, A7, A15, C6 | 中 | 诊断层面有新意（C6 "decide early"）；但 underthinking 自 A6 后几乎无 training-free 跟进 |
| 5 latent / 隐式 training-free | A10 | **刚起步 / 近乎空白** | 只有 Soft Thinking 一篇 TF，且自认 OOD |
| 6 compute-optimal & 迁移 | A3, A4, F1–F11 | agent 侧**刚起步** | F2 说 agent 上两种扩展都无增益；VLM 侧 F7/F8 刚开始做审计 |
| 7 TTS × 效率 | G1–G7 | 中 | 系统层工作多，但算法层"把早停设计成不破坏并行/批处理"只有 B3 提到 |
| 8 轻量 TTT/TTA | H1–H7 | 中偏冷 | 梯度版多；真正 training-free 的只有记忆/规则类 (H5/H6)，缺与采样类 TTS 的交叉 |

---

## 3. 五个 lens 的结构性 gap

### Lens 1: Method-transfer（方法迁移）

**G1.1 把"并行共识探针"从数学迁移到 agent 的 step-level**。Parallel-Probe (B7) 在数学上用 2D 探针（所有分支定期出中间答案）做全局早停；CATTS (F1) 在 web agent 上用每步投票熵分配算力，但只在单步内做，没有跨 rollout 的"轨迹级共识收敛"概念。F2 指出 agent 并行扩展卡在验证缺口。→ 把 B7 的"跨分支共识 + 偏离剪枝"搬到多步 agent 轨迹（以状态/动作序列而非答案做一致性）未被做。基于：B7, F1, F2, F3。

**G1.2 把 ST-BoN 的"内部状态一致性早停"迁移到序列 early-exit**。ST-BoN (G1) 用并行候选间的隐状态一致性在早期剪枝；序列 early-exit 一族 (A9, C4, C5, C7) 全部用单条轨迹内的信号（熵、置信、注意力）。中间地带——"一条轨迹的隐状态与同题其他轨迹的隐状态是否收敛"作为停止信号——未见。基于：G1, C4, C7。

**G1.3 把 Dynamic Cheatsheet / TF-TTCL 的文本规则记忆作为 TTS 的"预算先验"**。H5/H6 用记忆改善单次回答；B3/B6 用预测器估计每题预算。没有工作让"过去相似题的采样历史（N 到多少才收敛）"作为新题 N 的先验——即 training-free 的 compute-optimal 分配。基于：H5, H6, B3, B6。

### Lens 2: Contradiction（矛盾）

**G2.1 置信选择到底在难题上有效还是崩溃？** DeepConf (B1) 报告 AIME25 99.9%；CGES (B2) 承认 GPQA 上置信无信息；Consilience (B10) 声称置信在难题上"灾难性失效"，高置信=未探索。三者数据集、模型、置信定义不同。没有一篇在同一 benchmark 套件上按难度分桶、固定置信定义，系统画出"置信信号有效性 vs 题目难度 vs 模型"的相图。基于：B1, B2, B10, D1。

**G2.2 verifier-free TTS 是否次优？** 理论侧 (A17 引用 2502.12118) 说无 verifier 的 TTS 差距随预算扩大；实证侧 F7 说 PRM beam search 比 SC 低 0.39pp 贵 8×，F8 说多数投票打败自验证选择，F2 说 agent 并行扩展卡在验证缺口。矛盾的可能解释：verifier 质量阈值。D1 的 verifier gain 指标是对的工具，但没人用它预测"何时该花预算在 verifier 而非更多样本"。基于：A17, F7, F8, D1, B5。

**G2.3 token 减少 ≠ 延迟/成本减少**。DEER (A9) 报告 token −19~80%，但 SpecExit/C8(PUMA) 已实证 DEER 端到端比 vanilla 更慢（已核实）；SeerSC (B3) 指出序列早停破坏并行。几乎所有 early-exit 论文只报 token，不报 wall-clock 或 GPU-seconds。基于：A9, C3, C8, B3。

### Lens 3: Untested-assumption（未验证假设）

**G3.1 假设"正确轨迹早达高置信"跨模型族成立**。C5/C6/A7 在 Qwen/R1-Distill 上观察到；但 B11 显示 SC 的效果在 GPT-5.2 与 Claude 上方向相反，H2 显示 TTSV 在 Qwen 上 +17 分、LLaMA 上 0 分。熵/置信类信号很可能依赖 RLVR 后训练带来的校准特性（RLVR 已知造成 diversity collapse，见 2606.15455）。没有工作测"RLVR 后训练程度"与"置信早停有效性"的关系。基于：C5, C6, H2, B11, 2606.15455。

**G3.2 假设反思 token（"Wait"）是冗余的**。NoWait (A13)、CGRS (C1) 声称压制反思不损性能，而 A5/A11 靠插入 "Wait" 提分，A6 说过度切换有害。没人按"题目难度 × 模型是否已有正确中间答案"做 2×2 分解，回答"反思何时有用"。基于：A13, C1, A5, A11, A6, C6。

**G3.3 假设 early-exit 对贪心解码的结论能推广到采样 + 投票**。RPDI-EE (C4) 只在 greedy 做；而真正的部署场景是 SC/DeepConf 式并行采样。early exit 会改变答案分布（截断后的分布更集中），其与投票的交互未知。基于：C4, B1, B7。

### Lens 4: Scaling-regime（规模/预算区间）

**G4.1 极低预算区间（N≤4、thinking ≤ 1K token）几乎无人系统研究**。A8 NoThinking 显示低预算下跳过 thinking 更好；B4 显示高 N 收益消失。多数方法从 N=8~512 报告。对"单卡/API 成本敏感"的用户，N∈{1,2,3,4} 下哪种 TF 策略最优（NoThinking+BoN? 短 CoT+refine? 单条长 CoT+early exit?）没有 compute-matched 的系统比较。基于：A8, B4, B9, A3。

**G4.2 小模型 (≤4B) 上 verifier-free TTS 的"地板效应"**。A3/A4 的"小模型+TTS 胜大模型"结论依赖 PRM；TTSP (D3) 承认基座几乎不出正确答案时增益消失；B9 在 1.5B 上报告增益但未分难度。"小模型 + 纯 TF 方法"在哪个 pass@k 地板之上才值得扩展，缺少可操作的判据。基于：A3, A4, D3, B9, B5。

### Lens 5: Diagnostic（诊断/审计）

**G5.1 TTS 方法的"受控基线审计"刚在 VLM 上出现 (F8)，LLM 侧没有**。F8 证明图像扰动一致性的增益实际来自解码格式与预算。同样的质疑可以对准 DeepConf 的 tail-confidence、Consilience 的时间曲线、RPDI 的熵比：在"匹配 token 预算 + 匹配解码格式 + 随机过滤"对照下，增益还剩多少？基于：F8, B1, B10, C4。

**G5.2 "决策点"诊断 (C6) 未被用来评估 early-exit 方法的上界**。C6 用强制答案补全找到答案最后一次切换的位置，等价于 oracle early-exit 点。任何 early-exit 方法都应报告"距 oracle 点多远 / 多早退出造成多少翻转"，但目前没有统一的 oracle-gap 指标。基于：C6, A9, C4, C5, C7。

---

## 4. 候选 idea（8–10 个）

### Idea 1: ConfMap — 置信选择失效相图
- **Method**：(1) 固定三种置信定义（DeepConf 的 bottom-10% group conf、CGES 的答案后验、Consilience 的时间曲线分）；(2) 在 MATH-500 / AIME24-25 / GPQA-Diamond / LiveCodeBench 上，用 pass@1 把题分成 5 个难度桶；(3) 对每桶计算置信-正确 AUROC、oracle 选择 vs 置信选择 vs 多数投票的差距；(4) 画出"置信有效性 × 难度 × 模型（RLVR 程度）"相图并给出一条简单切换规则。
- **Hypothesis**：置信选择的增益集中在中等难度桶（pass@1 ∈ [0.2, 0.7]）；在最难桶上 AUROC → 0.5 且被多数投票反超；RLVR 更重的模型相图整体右移。
- **MVE**：Qwen3-4B/8B (thinking)、DeepSeek-R1-Distill-7B、GPT-OSS-20B；每题 N=32 采样；~2.5K 题 × 32 = 80K 条轨迹，vLLM 单张 A100/H100 约 40–60 GPU 时；指标 AUROC、accuracy@N、token。
- **Closest work**：B1 DeepConf、B2 CGES（v2 有"校准 vs 区分度"分析、称 GPQA 为 uninformative case，但没画相图）、B10 Consilience（摘要已声称置信方法在复杂任务上"灾难性失效"——Hypothesis 的"最难桶崩溃"部分已被其陈述）、"Variation in Verification"（ICLR 2026，对 verifier 做了难度 × generator × verifier 的类似相图）。区别：第一个统一跨方法、跨难度、跨模型的受控诊断，并产出可操作切换规则。
- **Novelty 6/10**；**Risk**：结论可能"显而易见"；需要 oracle 分析撑起贡献。
- **Contribution type**：诊断/分析 + 简单规则。
- **Reviewer objection**："这是 benchmark 分析不是方法"；"难度桶由 pass@1 定义与置信本身相关"。
- **dedup_key**：tts_confidence_difficulty_phase_map

### Idea 2: Latency-Honest Early Exit — 把 early-exit 重新设计成批友好
- **Method**：(1) 复现 DEER/RPDI/CoDE-Stop/ASAG，在 vLLM 下报告 token、wall-clock、GPU-seconds、吞吐四项；(2) 把 probing 改为"同步槽位"：在批内固定 step（每 256 token）统一做一次轻量探测（只用已有 logits 的熵比，无额外前向）；(3) 对比"每轨迹异步退出"与"批同步退出"的延迟-精度 Pareto。
- **Hypothesis**：≥半数现有 TF early-exit 方法在真实批处理下 wall-clock 不降甚至升；批同步 + 零额外前向的变体能在 token 节省 80% 的同时真正降延迟 ≥30%。
- **MVE**：Qwen3-4B/8B-Thinking；MATH-500、AIME24、GPQA、LiveCodeBench 子集；批大小 32/128；单卡 A100 约 30 GPU 时。
- **Closest work**：A9 DEER、C3 SpecExit（用 draft 模型，需训练）、C8 PUMA（已报告 wall-clock 与 DEER 变慢）、B3 SeerSC / B6 PreAda（并行采样侧，"解码前分配以保留并行"）。区别：面向序列 early-exit 的四指标延迟审计 + 零开销批同步设计。**核实后风险**：SpecExit 摘要与 PUMA 正文已证明"token 降但延迟不降/反升"，Hypothesis 前半句不再新。
- **Novelty 5/10（核实后下调）**；**Risk**：系统实现细节多；vLLM 版本差异；审计结论已被 SpecExit/PUMA 部分抢先。
- **Contribution type**：审计 + 方法改进。
- **Objection**："工程性强"；"SpecExit 已解决 probing 开销"（反驳：它需要 draft 模型）。
- **dedup_key**：tts_early_exit_latency_audit_batchsync

### Idea 3: Oracle-Gap — 用"决策点"统一评测 early-exit
- **Method**：(1) 对每条轨迹用强制答案补全（C6）每 128 token 探测一次中间答案，定义 oracle 退出点 = 答案最后一次切换位置；(2) 把所有 TF early-exit 方法的退出点与 oracle 点比较，定义 premature rate / overshoot tokens / flip loss；(3) 提出一个基于 oracle-gap 校准阈值的无参方法（在 50 道开发题上校准）。
- **Hypothesis**：现有方法主要错在"过早退出难题"而非"过晚退出易题"；按 oracle-gap 校准的阈值在相同 token 下精度更高。
- **MVE**：Qwen3-8B、R1-Distill-7B；MATH-500 + AIME + GPQA；探测成本每轨迹 ~10 次短补全；单卡 ~25 GPU 时。
- **Closest work**：C6（摘要已含：强制补全探针 + 最后切换点 + probe-based 早停省 500 token/−2%，即已有简单方法）、A9/C4/C5、"Confidence Leaps"（EACL 2026，置信跃升早停）。区别：把 C6 的探针变成跨方法评测协议（premature rate / overshoot / flip loss）和阈值校准工具；C6 的 probe-based stopping 本身不能再作为本 idea 的方法贡献。
- **Novelty 5/10**；**Risk**：强制补全本身可能改变模型行为。
- **Contribution type**：评测协议 + 轻量方法。
- **Objection**："oracle 依赖探针，探针不完美"。
- **dedup_key**：tts_early_exit_oracle_gap_metric

### Idea 4: Budget-from-Memory — training-free 的 compute-optimal 分配
- **Method**：(1) 维护一个 cheatsheet 式记忆：每做完一题记录（题目 embedding、SC 收敛所需 N*、最终是否一致）；(2) 新题来时检索 k 个最近邻，用它们的 N* 分布设定初始 N 与早停阈值；(3) 和 SeerSC（熵估计）、Overscaling-curse 的训练预测器、固定 N 比较 token-精度曲线。
- **Hypothesis**：无需训练预测器，近邻记忆即可恢复 ≥70% 的"训练预测器"节省；且随流式题数增加节省上升。
- **MVE**：Qwen3-4B；MATH-500 + GSM8K + AIME 流式 2K 题；embedding 用小 encoder；API 模型 (gpt-4o-mini) 也可做；单卡 ~20 GPU 时或 ~$50 API。
- **Closest work**：H5 Dynamic Cheatsheet（记忆存策略不存预算）、B3 SeerSC、B6 LanBo（需训）、**B8 PETS（在线流式按难度分配预算、有理论保证、TF）**、**B5 Blend-ASC（ICML 2026，动态分配、无超参、支持 batching、比 SC 少 4.8× 样本）**。区别：第一个把跨题记忆（近邻 N* 历史）用于预算分配；但必须以 PETS-online 与 Blend-ASC 为基线。
- **Novelty 6/10**；**Risk**：题目相似度与难度相关性可能弱。
- **Contribution type**：方法（TF）。
- **Objection**："流式设定不常见"；"近邻检索本身是额外成本"。
- **dedup_key**：tts_budget_allocation_memory_knn

### Idea 5: Reflect-or-Not — 反思 token 何时有用的 2×2 分解
- **Method**：(1) 在轨迹的每个 "Wait" 处用强制补全得到当前中间答案；(2) 标注该反思是"有益"（改对）、"有害"（改错）、"冗余"（不变）；(3) 按题目难度 × 中间答案正确性做 2×2 统计；(4) 得到一个条件性压制规则：仅在"中间答案已对且熵低"时压制，与 NoWait/CGRS/α1 比。
- **Hypothesis**：反思在"中间答案错 + 中难题"上净有益（翻正率 > 翻错率），在其余三个象限净冗余或有害；条件压制能同时超过 NoWait 的 token 节省与 α1 的精度。
- **MVE**：R1-Distill-7B、Qwen3-8B；MATH-500/AIME/GPQA；单卡 ~20 GPU 时。
- **Closest work**：A13 NoWait、C1 CGRS、A11 α1、A6 underthinking。区别：首次因果式分解反思价值，并统一"插入"与"压制"两派。
- **Novelty 6/10（核实后下调）**；**Risk**：CGRS 摘要即"当前回答高置信时压制反思触发词"，与第 (4) 步的条件压制规则几乎相同；C6 已用强制补全取中间答案。增量必须落在 2×2 因果分解与"插入(α1) vs 压制(NoWait/CGRS)"的统一比较。
- **Contribution type**：分析 + 方法。
- **Objection**："CGRS 的增量"。需强调 2×2 发现本身。
- **dedup_key**：tts_reflection_token_value_decomposition

### Idea 6: Trajectory-Consensus Probe for Agents — 把 Parallel-Probe 搬到多步 agent
- **Method**：(1) 并行跑 K 条 agent 轨迹；(2) 每 T 步把各轨迹的"状态摘要"（当前页面/文件/工具调用序列哈希 + 自述子目标）做一次聚类；(3) 共识簇占比超阈值时停止其他分支并把算力转给共识簇继续；偏离分支剪枝；(4) 与 CATTS（步级）、均匀 best-of-K、AggAgent 比。
- **Hypothesis**：轨迹级共识早停能在 agent 任务上恢复并行扩展的增益（F2 说目前没有），同时比均匀 K 省 ≥40% token。
- **MVE**：WebArena-Lite 或 SWE-bench Verified 小子集（50 题）；Qwen3-32B 或 API 模型；K=8；~$100 API 或 40 GPU 时。
- **Closest work**：B7 Parallel-Probe（数学答案共识）、F1 CATTS（步级投票；附录已试 DeepConf 变体）、F3 AggAgent（事后聚合）、BrowseConf（ACL 2026 Findings，verbalized confidence 重试）、F5 RGV（检索 grounding 投票）。区别：轨迹级在线共识 + 算力重分配（核实后仍未被任何摘要覆盖）。
- **Novelty 7/10**；**Risk**：状态摘要难以对齐；agent 评测噪声大。
- **Contribution type**：方法迁移。
- **Objection**："agent 轨迹不可比较，共识定义模糊"。
- **dedup_key**：tts_agent_trajectory_consensus_probe

### Idea 7: Matched-Control Audit for LLM TTS Selectors
- **Method**：(1) 对 DeepConf、Consilience、RPDI-EE、ST-BoN 各设计一个"匹配对照"：同 token 预算、同解码格式、但选择/过滤规则替换为随机或长度启发式；(2) 配对 bootstrap 检验增益显著性；(3) 报告哪些方法的增益在对照下消失。
- **Hypothesis**：至少一半的报告增益在匹配对照下不显著，剩余增益集中在"在线终止省 token"而非"选对答案"。
- **MVE**：Qwen3-8B、GPT-OSS-20B；AIME24/25、MATH-500、GPQA；N=64；单卡 ~40 GPU 时。
- **Closest work**：F8（VLM 侧同类审计）、B4。区别：首次对 LLM 置信类选择器做受控审计。
- **Novelty 5/10**；**Risk**：负面结果难发；需要强的正面副产物（如 Idea 1 的规则）。
- **Contribution type**：审计。
- **Objection**："负面结果，无新方法"。
- **dedup_key**：tts_selector_matched_control_audit

### Idea 8: Low-Budget Regime Study — N≤4 下哪种 TF 策略最优
- **Method**：(1) 固定总 token 预算 B∈{1K, 2K, 4K, 8K}；(2) 枚举 TF 策略：单条长 thinking+early exit、NoThinking×N+置信选、短 CoT×N+一次自纠正（B9 式）、budget forcing；(3) 在 3 个模型规模 × 4 个任务上画 compute-matched 曲线；(4) 给出"预算–模型–任务 → 策略"查找表。
- **Hypothesis**：在 B≤2K 时 NoThinking+BoN 占优；B∈[4K,8K] 时单条 thinking+early exit 占优；小模型 (≤4B) 的交叉点更靠右。
- **MVE**：Qwen3-1.7B/4B/8B；MATH-500、GPQA、LiveCodeBench、BBH；单卡 ~30 GPU 时。
- **Closest work**：A8 NoThinking（摘要已做 7 个数据集的低预算 compute-matched 比较 + NoThinking×N 并行扩展，并自称"低预算/低延迟下的竞争性参考"）、A3 Snell、B9。区别：NoThinking 只比 Thinking vs NoThinking 两路；本 idea 把 early-exit、refine、budget forcing 同台并给出查找表。
- **Novelty 5/10**；**Risk**：结论可能平淡。
- **Contribution type**：实证研究。
- **Objection**："又一个 benchmark 研究"。
- **dedup_key**：tts_low_budget_strategy_comparison

### Idea 9: Cross-Trajectory Hidden-State Convergence as Exit Signal
- **Method**：(1) 并行采 K 条轨迹；(2) 每 128 token 取各轨迹最后一层隐状态均值，计算跨轨迹余弦相似度矩阵；(3) 当相似度簇收敛（谱间隙超阈值）时触发全体早停并投票；(4) 与 Parallel-Probe（需生成中间答案）和 DeepConf 在线模式比。
- **Hypothesis**：隐状态收敛先于答案收敛出现（提前 ≥20% token），且无需生成中间答案。
- **MVE**：Qwen3-4B/8B，需要能读隐状态（HF/vLLM hook）；MATH-500、AIME；K=8；单卡 ~20 GPU 时。
- **Closest work**：G1 ST-BoN（摘要原文："利用模型内部状态的早期采样一致性识别最优路径并截断次优路径"——核心信号已相同）、B7 Parallel-Probe（全局共识早停 + 偏离剪枝框架已相同，只是用中间答案）、SpecExit（单轨迹隐状态预测退出）。区别仅剩：跨轨迹隐状态收敛 → 全体停止并投票（而非保留一条）。
- **Novelty 4/10（核实后下调）**；**Risk**：隐状态相似度可能对表面词汇敏感；审稿人会视为 ST-BoN × Parallel-Probe 的拼接。
- **Contribution type**：方法（TF）。
- **Objection**："ST-BoN 的扩展"。
- **dedup_key**：tts_hidden_state_convergence_global_exit

### Idea 10: RLVR-Calibration Dependence of Confidence Signals
- **Method**：(1) 取同一基座的不同后训练检查点（Qwen3 base / SFT / RLVR 版；DeepSeek-R1-Distill vs 原 Qwen2.5）；(2) 测量 token 置信、答案熵与正确性的 AUROC；(3) 测量 DeepConf/CGES/CoDE-Stop 的节省与精度；(4) 回归"后训练强度（或 pass@k 多样性塌缩度）→ 置信类 TTS 有效性"。
- **Hypothesis**：置信类 TF 方法的有效性随 RLVR 强度单调上升（更校准但多样性更低），因此在 base/SFT 模型上近乎无效；这解释了 H2 的 Qwen vs LLaMA 差异。
- **MVE**：≥6 个公开检查点；MATH-500/GPQA；N=16；单卡 ~25 GPU 时。
- **Closest work**：H2 TTSV（观察到差异未解释）、B2、2606.15455（多样性塌缩，不涉及 TTS 信号）、D1 2512.02304（37 模型 base vs post-trained：reasoning post-training 削弱自验证——是 verifier 侧的同类结论）。区别：首次把置信类 TTS 的适用性与后训练谱系挂钩。
- **Novelty 7/10**；**Risk**：检查点可得性；混杂因素多。
- **Contribution type**：分析。
- **Objection**："相关非因果"。
- **dedup_key**：tts_confidence_signal_posttraining_dependence

---

## 5. 综合判断

- **最饱和**：子方向 1（置信/熵驱动的自适应 SC 与早停）和 2a（单轨迹 early-exit）。2025-08 到 2026-06 之间至少 15 篇用几乎同一组信号（token logprob、答案熵、试答置信、注意力）做几乎同一件事；审稿人对新信号会非常疲劳。
- **最有空间的 3 个 gap**：G2.1（置信选择的难度/模型相图：三篇顶会结论互相矛盾却无人统一）、G2.3+G5.1（early-exit 与选择器的延迟/受控对照审计：几乎全部论文只报 token）、G1.1（并行共识探针迁移到 agent 轨迹：F2 明确指出 agent 上并行 TTS 卡在验证缺口）。
- **最看好的 3 个 idea**：Idea 1 ConfMap、Idea 5 Reflect-or-Not、Idea 6 Trajectory-Consensus Probe。Idea 2 是最稳的"保底"（审计 + 简单改进）。**核实后调整（§0.5）**：Idea 9 的核心信号已被 ST-BoN 摘要覆盖（降至 4/10）；Idea 2 的审计论点已被 SpecExit/PUMA 部分证明（降至 5/10）；Idea 5 的条件压制规则与 CGRS 摘要几乎相同（降至 6/10）；Idea 6 与 Idea 10 仍未被任何摘要覆盖。
- 子方向 5（training-free latent）几乎空白但非"有空间"——Soft Thinking 自认 OOD，无训练很难走远，不建议投入。
