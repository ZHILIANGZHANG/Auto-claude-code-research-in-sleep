# Lens 报告：Continual Learning（training-free / 少训练 / 低资源，预训练模型时代）

日期：2026-10-17。子代理：ARIS idea-creator Phase 1 + 1.5（CL 子方向）。

## 0. 方法与可信度说明（必读）

- 环境限制：arxiv.org / export.arxiv.org / alphaxiv / huggingface / semanticscholar 的直接 fetch 全部被代理 403 拦截；WebSearch 可用但有 200 次/turn 的共享预算，本次用掉 38 次后预算耗尽。
- 因此：**所有条目均来自搜索返回的摘要片段 + 会议页面（iclr.cc / icml.cc / neurips.cc / proceedings）**，未能逐篇读 intro / limitation。下文「自述 limitation」只在搜索片段中明确出现时才填写，否则标「未读到」。
- 所有 arXiv ID 与标题都是搜索结果中直接出现的；凡是我凭记忆补充但本次未核实的锚点论文，统一标注 `[未核实-记忆]`，请主代理在后续 turn 复核。
- 日期窗口：重点 2025-07 → 2026-10，但为锚点也纳入 2023-2025H1 的奠基工作。

---

## 1. 论文清单（按子方向分组，共 ~70 篇）

格式：标题 | arXiv/venue | 日期 | 问题 | 方法一句话 | 关键数字 | training-free? | 自述 limitation

### 1.1 预训练 ViT 上的 prompt / adapter / LoRA CL（最饱和）

锚点：
- Learning to Prompt (L2P) | arXiv 2112.08654, CVPR 2022 | prompt pool + key-query 选择 | 非 TF（训 prompt）
- DualPrompt | arXiv 2204.04799, ECCV 2022 | G-prompt + E-prompt | 非 TF
- CODA-Prompt | arXiv 2211.13218, CVPR 2023 | 注意力加权的 prompt 分解，软匹配 | 非 TF；自述 average forgetting 略高于 DualPrompt
- Hierarchical Decomposition of Prompt-Based CL | arXiv 2310.07234 (NeurIPS 2023) | 复现 CODA-Prompt 官方代码数字与原文有明显差异；CODA 需要更小 lr + cosine decay | 诊断类
- One-stage Prompt-based CL (OS-Prompt) | arXiv 2402.16189 | L2P/DualPrompt 的 top-k 硬匹配比 CODA 的软匹配更稳 | 非 TF
- Prompt Customization | arXiv 2404.18060 | CODA-P 训练时间 ≈ DualPrompt 的 4× | 非 TF
- Additive Prompt Tuning | arXiv 2503.07979 (2025-03) | prompt pool 开销大 → 单一共享 prompt 集 | 非 TF
- GAP-Prompt: Gated Adaptive Prompting | arXiv 2608.23782 (2026-08) | class-IL、任务 id 未知下的门控自适应 prompt | 非 TF | venue 未确认
- SD-LoRA | arXiv 2501.13198, ICLR 2025 Oral | LoRA 幅度/方向解耦，rehearsal-free，推理无需选组件；声称收敛到各任务共享低损区 | 非 TF
- InfLoRA `[未核实-记忆]` | arXiv 2404.00228, CVPR 2024 | 干扰最小化子空间 LoRA | 非 TF
- SAFE: Slow and Fast PEFT | arXiv 2411.02175 | 第一会话训 slow learner，之后 fast learner；slow learner 使最终准确率 +3.2 | 非 TF
- PLASTIC: Enhancing Plasticity for First-Session Adaptation | arXiv 2310.11482 v3 (2025-07), PMLR v330 (2026) | FSA 在异质任务序列上弱 → 测试时用无标签数据微调 LayerNorm | 半 TF（无标签 TTA） | 自述：FSA 在异质序列上塑性不足
- Revisiting Weight Regularization for Low-Rank CL (EWC-LoRA) | arXiv 2602.17559, ICLR 2026 | EWC 在 PEFT-CL 里被忽视；对单个共享低秩更新做 EWC，存储/推理成本不随任务数增长 | 非 TF
- Merge before Forget (SLAO) | arXiv 2512.23017, ICLR 2026 | 单 LoRA 持续合并：新任务从旧 LoRA 的正交基起步 + time-aware scaling；内存 O(1)；Llama 系列 | 非 TF（少训练）
- Pruned Adaptation Modules (PAM) | arXiv 2603.21170, CPAL 2026 (PMLR 328)；近似同文 Slim Adaptation Modules @ NeurIPS 2025 | **诊断/基线**：领域转向 FM-CIL 时没有和强轻量卷积基线对比；冻结 ResNet + 稀疏任务层，可训参数 ↓2-5×、总参数 ↓2-6×，仍超 SOTA FM-based CIL | 少训练 | 自述：方法学 gap——无法判断近期进展是真实还是基线弱
- 综述：Parameter-Efficient Continual Fine-Tuning: A Survey | arXiv 2504.13822 (2025-04)

### 1.2 完全 training-free / 解析式（analytic）/ 原型式 CL

- RanPAC | arXiv 2307.02251, NeurIPS 2023 | 冻结随机投影 + 非线性 + 原型去相关（Gram 矩阵逆）；7 个 CIL 基准相对以往 PTM 方法错误率 ↓10-62%，无 rehearsal | **TF**（仅闭式）
- RanDumb | arXiv 2402.08823, NeurIPS 2024 | 随机 RBF 特征 + 线性分类器即可超越在线 CL 的表示学习方法；用预训练特征时线性分类器超过多数 prompt-tuning | **TF** | 结论限定 low-exemplar / online 场景
- SimpleCIL / APER `[未核实-记忆]` | arXiv 2303.07338 | 原型 + cosine 分类器的强基线 | TF
- AOCIL | arXiv 2403.15751, ICML 2024 | 冻结 ViT + RLS 更新的解析分类器，在线 exemplar-free | TF
- REAL | arXiv 2403.13522 | 自述解析 CL 的表示无效/利用不足 → 双流 base pretraining + 表示增强蒸馏 | 非 TF
- RAIL | arXiv 2406.18868 | CLIP 上递归岭回归 adapter + training-free 融合模块；提出 X-TAIL（跨域、无域提示、含未见域）；零样本能力「绝对保持」 | TF | venue 未确认
- AnaCP: Analytic Contrastive Projection | arXiv 2511.13880, NeurIPS 2025 | 自述 limitation：解析法「无法把 PTM 特征适配到下游，性能次优」→ 解析对比投影 | 近 TF
- LR-RGDA | arXiv 2602.00144 (2026-02) | ViT-CIL 的分类器重建多数靠迭代 SGD → 低秩因子化解析分类器（Woodbury） | TF
- VILA | arXiv 2602.13670 (2026-02) | 解析 CIL 的瓶颈是「表示僵化 (representation rigidity)」+ 累积误差/特征不兼容 → 双分支 | 部分 TF
- Fly-CL | arXiv 2510.16877 (2025-10-19), ICLR 2026 | 预训练特征在相似度匹配阶段存在多重共线性；果蝇 PN→KC 式高维稀疏随机投影 + top-k + 流式岭分类；训练时间大幅下降、性能≥SOTA | **TF**（闭式）| 自述动机：RanPAC 类方法对低延迟场景太贵
- F-OAL (forward-only online analytic learning) | 二手页面，ID 未抓到 | 最小二乘闭式映射 | TF
- Continual-CLIP | arXiv 2210.03114 | 冻结 CLIP 零样本就是强 CL 基线 | TF
- PTA: Prototype-Based TTA of VLMs | ICML 2026 poster | 类原型积累测试样本知识，CLIP 10 个跨域基准 65.64→69.38 | TF

### 1.3 模型合并 / task arithmetic 用于 CL

- MagMax | arXiv 2407.06322, ECCV 2024 | 顺序微调减少符号冲突 + 最大幅度选择；**发现：普通权重平均/随机选择就很强，常胜复杂合并** | 少训练
- Sequential Fine-tuning with Averaging (SFA) | arXiv 2501.05559 (2025-01) | 训练中与早期 checkpoint 平均，免存数据；胜 TA / TIES / L2 / EWC（图像+语言）
- Scalable Strategies for CL with Replay | arXiv 2505.12512 (2025-05) | 比较并行合并(TA)、EMA、顺序合并用于长任务流；DARE-TIES 倾向保留预训练知识
- Layer-Aware Task Arithmetic (LATA) | arXiv 2502.20186 | 按层区分任务知识 vs 指令遵循知识加权 task vector；WikiText-2/GSM8K/HumanEval
- Closed-form LoRA merging for federated CL | arXiv 2410.17961, ICLR 2025 | 闭式合并使合并模型匹配各模块输出
- Aligned model merging for VLM CL (DeepMind) | arXiv 2506.03189 (2025-06) | 新任务参数与旧参数对齐合并，多任务顺序下遗忘减少
- Unraveling LoRA Interference | arXiv 2505.22934 | 正交子空间减少 LoRA 合并干扰
- Audit Before You Merge | ICML 2026 (无 arXiv ID) | **诊断**：把每次合并公共 LoRA 当作顺序任务（Task 0 = base 能力）；公共 hub 中大量 adapter base 版本不匹配或 Δ≈0，**会使测得的遗忘失真**；经审计的 4-adapter 集上朴素合并 GSM8K −10.5、HumanEval −18.3（Llama-3.1-8B-Instruct）；Mistral-7B 上 4 个候选中 2 个不过审 | TF（合并）
- arXiv 2511.21437 | 公开 checkpoint 上系统基准 TA/TIES/Model Stock + 子空间法（Llama-3、Qwen3，16 个 lm-eval 任务）
- MeMo: Memory as a Model | arXiv 2605.15156 (2026-05) | 把合并视为避免顺序微调分布干扰的手段
- LoRA vs Full FT: An Illusion of Equivalence | NeurIPS 2025 | LoRA 产生「intruder dimensions」（新的高奇异值方向），全参微调不产生 | 对合并/遗忘的几何有含义

### 1.4 LLM 的持续学习（continual pretraining / instruction tuning / 知识编辑 / 轻量知识注入）

- Simple and Scalable Strategies to Continually Pre-train LLMs | arXiv 2403.08763 | re-warm + re-decay 增加适应也增加遗忘；replay 缓解；提出 infinite LR schedule
- Learning Dynamics in Continual Pre-Training | arXiv 2505.07796, ICML 2025 spotlight | CPT scaling law：分离分布偏移与 LR 退火效应，可预测任意步 loss，覆盖 peak LR / loss potential / replay ratio
- Spurious Forgetting in CL of LMs | arXiv 2501.13453 (ICLR 2025) | TRACE 上部分「遗忘」在后续任务后反弹——遗忘指标需谨慎
- Forget Forgetting: CL in a World of Abundant Memory | arXiv 2502.07274 | 在 TRACE 上研究 exemplar memory 充足时的策略
- TRACE | arXiv 2310.06762 | 8 域 LLM CL 基准，含 RCL（推理增强）
- GoD-MoE (Grow-on-Demand) | AAAI 2026 | attention 层插入 LoRA 专家，按任务动态激活小子集，胜 replay 基线
- Adaptation-aware probing | arXiv 2605.31025 (2026-05) | 持续微调中的任务专属参数发现；指出 replay / 任务 adapter 都有额外开销
- CLoRA | arXiv 2410.16801, ACL 2025 | LoRA 更新的零空间正交正则，限制输出漂移
- LoRA-Null | arXiv 2503.02659 | 从预训练知识激活的零空间构造 adapter
- RL's Razor | arXiv 2509.04259, NeurIPS 2025 | **同等新任务性能下 on-policy RL 比 SFT 遗忘少；遗忘由新任务上「与 base 的 KL」预测；权重变化幅度、稀疏度、梯度秩都不能稳定解释** | 诊断
- Continual Learning via Sparse Memory Finetuning | arXiv 2510.15103 (2025-10-16, Meta FAIR+Berkeley) | memory-layer 模型，仅更新相对预训练用量被新知识高度激活的 memory slot（TF-IDF 排名）；学新事实后 NQ F1：全参 −89%、LoRA −71%、本法 −11% | 少训练 | **自述：只评 factual QA；承认 RAG 是当下 factual learning 的自然方案**
- Nested Learning / HOPE | NeurIPS 2025 (Google) | 模型 = 多频率嵌套优化问题；Continuum Memory System；HOPE 是 Titans 改版 | 架构类 | 自述：camera-ready 被大幅压缩
- In-Place TTT | ICLR 2026 | 推理中更新权重子集 → 走向 LLM 持续学习
- Test-Time Learning for LLMs (TLM) | arXiv 2505.20633, ICML 2025 | 用测试输入困惑度最小化适应；LoRA 比全参更能缓解遗忘
- Agentic TTT (aTTT) | arXiv 2607.03441 (2026-07) | 多轮 agent 的持续 TTT 会形成自训练回路放大漂移；token 级重加权
- Mechanistic Analysis of Catastrophic Forgetting (2601.18699) | 2026-01 | v1 vs v2 作者/机构/内容严重不一致（v2 含可疑的 1.5T 模型、94.2% 等数字）——**不建议引用**
- 知识编辑持续性：
  - WilKE | arXiv 2402.10987 | ROME/MEMIT 终身编辑「毒性累积 / 毒性闪现」
  - Rebuilding ROME (r-ROME) | arXiv 2403.07175 | 解决顺序编辑崩溃；自述下游退化在规模化时仍存在
  - Lifelong KE requires Better Regularization | arXiv 2502.01636, EMNLP 2025 Findings | 退化来自激活过优化 + 编辑矩阵范数增长 → 早停 + Frobenius 约束
  - WikiBigEdit | arXiv 2503.05683 | 50 万+ 真实 Wikidata 编辑；局部编辑法（MEMIT/ROME）**250 次更新内趋近零**；专门编辑法不如简单持续 LoRA 微调；RAG 用推理时间换性能
  - Towards a Principled Evaluation of Knowledge Editors | arXiv 2507.05937 | string-match 评估产生假阳性；ICL/RAG 也是一种编辑
  - Selective Knowledge Suppression | arXiv 2604.19089 (2026-04) | 可扩展终身编辑
- 综述：2402.01364；2404.16789；2603.12658 (2026-03, CL in LLMs: Methods, Challenges, Opportunities)

### 1.5 记忆 / 检索式 CL、in-context CL、agent memory

- SelfAug | arXiv 2509.03934, EMNLP 2025 Findings | RAG 场景 SFT 的遗忘 = 分布偏移 → logit 自对齐
- Chronos | ACL 2026 Findings (2026.findings-acl.546) | 带时间戳真实事件基准；vanilla RAG 和多种学习法都受「遗忘 + 时间不一致」困扰；提出无训练的 time-aware retrieval
- Memento 2: Stateful Reflective Memory | arXiv 2512.22716 (2025-12) | 对 episodic memory 做反思即可持续适应，不改参数；给出收敛证明
- Decoupling Memory from Context (GraphMemory) | arXiv 2610.02687 (2026-10-06), NeurIPS 2026 | 共享上下文不断追加 → token 成本/窗口/退化；图记忆只检索相关子图；记忆构建 token ↓81-85% | TF
- Modular Memory is the Key to CL Agents | arXiv 2603.01761 | 立场文：ICL 快适应 + in-weight 慢更新
- ContinualSkillBench | arXiv 2608.03874 (PKU) | 5 域×100 链式子任务；**ICL 平均上与显式技能库维护相当**；弱模型会积累更碎片化的技能
- CL-Bench (Continual Learning Bench) | NeurIPS 2026 (Snorkel) | 6 域真实有状态环境；**朴素 ICL 胜过专门记忆系统**
- AGENTCL | arXiv 2606.02461 | 现有基准偏向长上下文检索/推理，提出更严格的 agent CL 评测
- Letta「Continual Learning in Token Space」(博客, 2025-12) | 非论文
- Spark memory (2607.22157) | 二手摘要，未核实

### 1.6 MLLM / VLM 持续学习

- CoIN | NeurIPS 2024 D&B | 10 数据集 8 任务；**遗忘主因是指令对齐失败而非推理知识遗忘**
- Continual Instruction Tuning for LMMs | arXiv 2311.16206 | replay 与模型扩展有效；正则法只在联合指令微调过的模型上有效
- Improving MLLMs Using CL | arXiv 2410.19925, CoLLAs 2025 | 把 LLaVA 集成当作 CL：文本能力退化 ↓15%
- HiDe-LLaVA | arXiv 2503.12941 (2025-05) | 按层 CKA 相似性决定任务专属扩展 vs 通用融合
- MLLM-CL | arXiv 2506.05453 (2025-06) | 域 CL（遥感/医疗/自驾/科学/金融，IID）+ 能力 CL（OCR/数学/感知/GUI，非 IID）；MR-LoRA 参数隔离 + MLLM 路由
- MLLM-CTBench | arXiv 2508.08275 (2025-08) | 7 任务 6 域；**同时评分 CoT 轨迹**；GRPO（有 KL 控制）比 SFT 跨任务保持更稳，去掉 KL 加剧遗忘
- Routing-Based CL for MLLMs | arXiv 2511.01831 (2025-11, v3 2026-04, Amazon) | token 级路由专家；2B-8B 上匹配多任务上界且计算固定；跨模态知识迁移
- CF-Router | ICLR 2026 | 多模态 agent 终身学习中专家选择的闭式解 | 近 TF
- φ-DPO | arXiv 2602.22601 (2026-02) | 公平 DPO 用于 LMM CL；在 MLLM-CL 上 RS 85.58 / Finance 95.28 等
- **Is Our Benchmark Enough? An Analysis of CL for MLLMs** | ICML 2026 (TCD + TU/e) | **诊断**：(1) training-free、replay-free 原型路由 RePro（冻结特征 + 任务原型）匹配 MR-LoRA 的 MLLM 路由器；(2) 共享专家无益；(3) MLLM-CL 任务在表示空间高度可分、单一固定任务顺序 → 基准奖励「孤立学习」而非持续迁移；建议重叠任务流形、多任务顺序、细粒度域偏移、forward-transfer 指标
- C-CLIP | ICLR 2025 | 多域 image-caption 的多模态 CL 基准，追踪零样本遗忘
- ZSCL `[未核实-记忆]` | arXiv 2303.06628, ICCV 2023 | 特征蒸馏 + 权重平均保护零样本；MTIL 基准
- MG-CLIP: Mind the Gap | arXiv 2507.09118, ICCV 2025 | 模态 gap 大小追踪预训练知识保留程度；gap 保持 + gap 补偿，无 replay
- Open-domain VLM CL (SMU, 2025-08) | 流中含已见/未见域的新类
- Generative prompt learning (VAE replay) | arXiv 2407.15793
- Dual-phase CL with unlabeled test data | arXiv 2406.13653, TMLR 2026 | 教师-学生 + 稀疏更新，用无标签测试数据减少 VLM CIL 遗忘

### 1.7 评估问题 / 诊断

- CLDyB | arXiv 2503.04655 (2025-03) | **PTM 预训练可能污染 CL 基准**，静态基准饱和 → MDP + MCTS 动态生成任务序列
- Data Leakage in Visual Datasets | arXiv 2508.17416 (2025-08) | ImageNet test 内部泄漏 1.54%(hard)/1.95%(soft)；LAION 等预训练集对基准的泄漏可达数个百分点
- Hierarchical Decomposition (2310.07234)：CODA-Prompt 复现差异
- PAM / Slim Adaptation Modules：基线缺失
- Is Our Benchmark Enough (ICML 2026)：MLLM-CL 可分性
- Audit Before You Merge (ICML 2026)：adapter 来源污染遗忘测量
- Spurious Forgetting (2501.13453)：遗忘反弹
- RL's Razor：遗忘预测量
- WikiBigEdit：编辑法 vs LoRA vs RAG
- Rethinking 类：「Towards a Principled Evaluation of Knowledge Editors」(2507.05937)

### 1.8 与 TTA / online / test-time CL 的交叉

- TTABC / What Drives TTA for CLIP | arXiv 2606.14299 (2026-06), NeurIPS 2026 | 20+ 方法统一基准；**增益主要来自测试时证据 + 可靠代理而非重优化；跨/当前样本证据 + 轻量原型更新即可有竞争力；无普适最优范式**（norm 层更新适合 corruption，stream-aware 适合 label shift）；在线累积参数更新带来效率与稳定性问题
- BayesTTA | arXiv 2507.08607 (2025-07) | CLIP 在渐变偏移（光照/季节）下退化；现有持续 TTA 假设突变剧烈偏移
- MS-TTA | arXiv 2507.00462 | 单步 kNN Mean-Shift，TF
- UL-TTA | arXiv 2511.09101 (2025-11) | 冻结骨干，只调 logit 级参数（原型/先验/温度），TF、无反传，流式
- MINT | arXiv 2506.03190 | 记忆 prompt bank（可学习）
- BPRE | arXiv 2503.09394 | 原型-奖励共进化
- Just Shift It (TPS) | arXiv 2403.12952 | 测试时原型平移
- PLASTIC（见 1.1）：FSA + TTA
- Dual-phase（见 1.6）
- NeurIPS 2026 workshop「Towards Test-Time Continual Learning Agents」已成立——方向正在被制度化

---

## 2. Landscape map：饱和 vs 起步

| 子方向 | 状态 | 证据 |
|---|---|---|
| ViT 上 prompt-pool / LoRA-pool CIL（CIFAR-100、ImageNet-R、CUB） | **做烂**（饱和，且被诊断论文围攻） | L2P→DualPrompt→CODA→OS-Prompt→Additive→GAP-Prompt；SD-LoRA/InfLoRA/EWC-LoRA/SLAO；RanPAC/RanDumb/PAM 反复证明 TF 或轻量基线即可打平 |
| 解析式 / 随机投影 TF-CIL | **饱和中但仍有空间**：方法已多（RanPAC、Fly-CL、AnaCP、LR-RGDA、VILA、AOCIL），空间在「表示僵化」与「特征不适配」问题（AnaCP、VILA 自述） | |
| ViT-CIL 基准有效性（预训练泄漏） | **刚起步**：CLDyB、Data Leakage 只给了问题，没给「泄漏量 ↔ TF 方法优势」的因果证据 | |
| 模型合并用于 CL（视觉） | 中等饱和：MagMax、SFA、2505.12512 已把「平均就很强」说透 | |
| 模型合并用于 CL（LLM、公共 adapter） | **刚起步**：Audit Before You Merge 刚指出测量污染；2511.21437 刚做公开 ckpt 基准 | |
| LLM 顺序 LoRA/MoE CL（TRACE 类） | 饱和：O-LoRA/CLoRA/LoRA-Null/GoD-MoE/SLAO | |
| LLM 遗忘的「可预测量」 | **刚起步、高价值**：RL's Razor 只给 KL 一个量，且只在 RL vs SFT 对比中验证 | |
| memory-layer / sparse-slot 持续注入 | **刚起步**：SMF 只在 factual QA；需要 memory-layer 架构 | |
| 终身知识编辑 | 饱和于「ROME/MEMIT 崩溃」；WikiBigEdit 已宣判「不如 LoRA」 | |
| RAG/ICL vs 参数化 CL 的正面对比 | **起步且结论混乱**：CL-Bench、ContinualSkillBench 说朴素 ICL 赢；WikiBigEdit 说 RAG 换推理时间；Chronos 说 vanilla RAG 时间不一致；SMF 承认 RAG 是自然方案 | |
| MLLM 持续指令微调 | 方法饱和（HiDe-LLaVA、MR-LoRA、Routing、φ-DPO），**基准被诊断为无效**（Is Our Benchmark Enough） | |
| MLLM CL 的推理过程 / RL 视角 | 起步：MLLM-CTBench 一篇 | |
| continual TTA ↔ CL 交叉（CLIP） | 起步：TTABC 刚统一；BayesTTA 指出渐变偏移未被覆盖；无「test-time CIL」标准设定 | |
| agent memory 作为 CL | 热但混乱：多基准互相矛盾，缺统一的「遗忘」定义 | |

---

## 3. 五个 lens 的结构性 gap

### Lens A：Method-transfer（跨域方法迁移）

**A1. 「闭式/随机投影分类头」从 ViT-CIL 迁移到 MLLM 的持续指令微调路由与能力保持。**
依据：Fly-CL (2510.16877)、RanPAC (2307.02251) 证明冻结特征 + 闭式头在 ViT 上足够；Is Our Benchmark Enough (ICML 2026) 证明 training-free 原型路由 RePro 可匹配 MR-LoRA 的 MLLM 路由；Routing-Based CL (2511.01831) 仍用可训练 token 路由。缺口：没有人把 RanPAC/Fly-CL 式的去相关闭式头直接当作 MLLM 的 LoRA-expert 路由器或作为「能力 CL」里输出头，并在非 IID 的 ability track 上验证。

**A2. 「KL-to-base 作为遗忘预测量」（RL's Razor）迁移到视觉 PTM-CL 与模型合并。**
依据：RL's Razor (2509.04259) 发现 LLM 遗忘由新任务上相对 base 的 KL 预测，而权重变化幅度/稀疏度/梯度秩都不行；MagMax (2407.06322) 与 SFA (2501.05559) 用权重空间启发式（幅度、平均）选参数；Audit (ICML 2026) 用 Δ-norm 做审计。缺口：视觉 CL 与合并领域仍用权重空间量作为「干扰」代理，没有人检验 output-distribution KL 是否在 ViT/CLIP 上同样是更好的预测量，也没人用它作为合并系数的选择准则。

**A3. 「时间感知检索」(Chronos) 与「Spurious forgetting」思想迁移到 agent memory 基准的遗忘定义。**
依据：Chronos (ACL 2026 Findings) 指出 vanilla RAG 时间不一致；Spurious Forgetting (2501.13453) 指出 LLM CL 中遗忘会反弹；CL-Bench / ContinualSkillBench 对 ICL vs memory 结论不一致。缺口：agent memory CL 基准没有「遗忘 vs 暂时对齐丢失」的区分，也没有时间一致性维度。

### Lens B：Contradiction（文献内矛盾）

**B1. 「RAG/ICL 是无遗忘的 CL」 vs 「RAG 本身会遗忘/不一致」。**
CL-Bench (NeurIPS 2026) 与 ContinualSkillBench (2608.03874) 说朴素 ICL ≥ 专门 memory 系统；SMF (2510.15103) 承认 RAG 是自然方案；但 Chronos (ACL 2026) 说 vanilla RAG 有遗忘与时间不一致，GraphMemory (2610.02687) 说追加式 context 性能退化，SelfAug (2509.03934) 说 RAG 场景 SFT 仍遗忘，WikiBigEdit (2503.05683) 说 RAG 用推理时间换。矛盾核心：各家「遗忘」的操作定义不同（能力遗忘 vs 事实过期 vs 技能碎片化）。无一篇在同一任务流、同一模型、同一预算下用统一的遗忘定义对比 ICL / RAG / LoRA-CL / 编辑法。

**B2. 「KL 控制减少遗忘」 vs 「合并/平均减少遗忘」——两种解释是否同源？**
RL's Razor (2509.04259)、MLLM-CTBench (2508.08275, GRPO 的 KL 控制更稳)、EWC-LoRA (2602.17559, 参数空间正则) 、SFA (2501.05559, checkpoint 平均)、MagMax（平均即可）。RL's Razor 明确说权重空间量不预测遗忘，但 MagMax/SFA 的成功完全是权重空间操作。未被解释：权重平均之所以有效，是否恰恰因为它在 output KL 上最小？目前没有把两类方法放在同一 KL 坐标系下的实验。

**B3. 「解析/TF 方法够用」 vs 「解析方法特征僵化」。**
RanDumb (2402.08823)、PAM (2603.21170)、Fly-CL、RePro (ICML 2026) 说 TF/轻量即可；AnaCP (2511.13880)、VILA (2602.13670)、REAL (2403.13522) 自述解析法因无法适配特征而次优。两边都对，但没有人回答：在什么「预训练-下游域距离」或「泄漏比例」下 TF 开始落后？CLDyB (2503.04655) 与 Data Leakage (2508.17416) 提供了测这件事的工具却未结合。

### Lens C：Untested-assumption（未检验假设）

**C1. 假设：ViT-CIL 基准（CIFAR-100、ImageNet-R、CUB、Cars）上 TF 方法的胜利反映的是方法优劣，而非 ImageNet-21K/LAION 预训练对这些数据集的覆盖。**
依据：CLDyB 只提出「可能污染」；Data Leakage 只测了 ImageNet 自身与 LAION→基准的泄漏比例；PAM 指出基线缺失。无人做过「按泄漏/相似度分层后 TF 与可训练方法差距如何变化」的分析。

**C2. 假设：公共 LoRA 合并实验里测到的「遗忘」来自干扰。**
Audit Before You Merge 已指出大量 adapter base 版本不匹配或 Δ≈0——即过去用公共 adapter 做的合并-遗忘结论（含 2511.21437 一类的公开 ckpt 基准）可能部分是伪影。未检验：审计前后，TIES/DARE/TA 的相对排名是否改变。

**C3. 假设：MLLM CL 基准中「任务可分」是普遍现象还是 MLLM-CL 特有。**
Is Our Benchmark Enough 只分析了 MLLM-CL；CoIN、MLLM-CTBench、2311.16206 是否同样高度可分（从而同样被原型路由「解决」）未检验。若是，整条 MLLM-CL 方法线（HiDe-LLaVA、MR-LoRA、Routing、φ-DPO）的增益需要重估。

**C4. 假设：continual TTA 的「渐变偏移」与 CL 的「离散任务」是不同问题。**
BayesTTA 指出 CTTA 假设突变；TTABC 指出 stream-aware 设计适合 label shift。CL 社区的 class-incremental 本质是极端 label shift。无人在同一 CLIP 骨干、同一数据流上同时跑 CTTA 方法（UL-TTA、MS-TTA）和 TF-CIL 方法（RanPAC、RAIL），看它们是不是同一方法族的两种参数化。

### Lens D：Scaling-regime（规模/资源区间）

**D1. 「任务数 → 数百/数千」区间下 TF 闭式头 vs 单 LoRA 合并的内存/精度曲线缺失。**
SLAO、EWC-LoRA 强调 O(1) 内存；RanPAC/Fly-CL 的 Gram 矩阵在类数极大时需要正则（Fly-CL 自述多重共线性）；WikiBigEdit 显示编辑法 250 次内崩溃。无人把「更新次数」作为主轴在 10→10,000 的范围内画 TF vs 合并 vs 编辑的曲线（视觉与语言各一条）。

**D2. 小模型/单卡区间下的 LLM CL：SMF (memory layer)、Routing (2B-8B)、RL's Razor 都在 ≥1B 模型；TF 式 LLM CL（例如用冻结 LLM 的隐藏态 + 闭式头做分类型持续任务）在 0.5B-3B 的 Qwen/Llama 上几乎没有系统研究。而 TRACE 类基准多数任务是分类/短生成，恰好适合闭式头。**

**D3. 预训练数据规模 → TF 优势：Continual-CLIP (2210.03114) 与 PTA (ICML 2026) 暗示零样本能力越强 TF 越占优；Data Leakage 说明 LAION 规模决定泄漏。没有「同一架构、不同预训练数据量（CLIP 400M / LAION-2B / DataComp）下 TF-CL 与训练式 CL 差距」的扫描。**

### Lens E：Diagnostic（诊断类）

**E1. 「训练式 PTM-CL 方法到底学到了什么」：** 把 CODA-Prompt/SD-LoRA/InfLoRA 学到的更新投影到「冻结特征 + 闭式头」能解释的子空间，看残差贡献。依据：Hierarchical Decomposition (2310.07234) 已发现复现差异；PAM 发现基线缺失；RanDumb 说表示学习贡献小；LoRA intruder dimensions (NeurIPS 2025) 给出几何工具。无人做「可训练方法的增益分解 = 分类头效应 + 特征适配效应 + 调参效应」。

**E2. 「MLLM 持续指令微调的遗忘到底是什么」：** CoIN 说是指令对齐失败；MLLM-CTBench 说 CoT 轨迹揭示隐藏遗忘；Spurious Forgetting 说 LLM 的遗忘可逆；Is Our Benchmark Enough 说基准可分。缺一个把「对齐丢失 / 知识丢失 / 路由错误」三者定量拆开的诊断（例如用 RePro 式原型路由作为对照：若原型路由恢复了大部分性能，则遗忘主要是路由/对齐问题）。

**E3. 「权重空间量 vs 输出分布量」作为遗忘预测器的系统评估（跨视觉/语言/合并）**：RL's Razor 只在 RL-vs-SFT 中验证了 KL；Audit 用 Δ-norm；SFA/MagMax 用幅度。缺一个跨 8-10 种 CL 方法、2 模态的「预测器 vs 实测遗忘」相关性表。

---

## 4. 候选 idea（10 个）

### Idea 1：Leakage-Stratified CL — 预训练泄漏能解释多少 training-free 方法的优势？
- **Method**：(1) 对 CIFAR-100 / ImageNet-R / CUB / Cars / ObjectNet / 新采集的 2025-后数据，用 CLIP/DINO 特征对 ImageNet-21K 与 LAION 子集做近重复检索（复用 Data Leakage 2508.17416 的 hard/soft 定义），给每个类打「泄漏分」；(2) 在同一 ViT-B/16 (IN21K) 上跑 RanPAC、Fly-CL、SimpleCIL（TF）与 CODA-Prompt、SD-LoRA、EWC-LoRA（训练式）；(3) 按泄漏分分层报告 TF−trained 差距；(4) 构造「零泄漏」子集作为新协议。
- **Hypothesis**：TF 方法的优势随泄漏分单调增加；在零泄漏层训练式方法反超 ≥3 pts。
- **MVE**：ViT-B/16 IN21K；6 数据集；指标：分层 Last-Acc 差、与泄漏分的 Spearman；GPU：特征提取 + 6 方法×6 数据集 ≈ 30-40 单卡小时（A100/4090）。
- **Closest work**：CLDyB (2503.04655)：提出污染但未量化；Data Leakage (2508.17416)：量化泄漏但不涉及 CL 方法；PAM (2603.21170)：基线批评但不涉及数据。区别：首次把泄漏量与方法类型交叉。
- **Novelty 7/10**；**Risk**：泄漏分与「域相似度」混淆，需要用 DINO 域距离做协变量。
- **Contribution type**：诊断 + 评测协议。
- **Reviewer objection**：「近重复检索阈值任意」；「泄漏≠记忆」。
- dedup_key：`cl_leakage_stratified_tf_vs_trained`

### Idea 2：KL-Razor for Vision — 输出分布 KL 是否也是 PTM-CL/合并的遗忘预测量？
- **Method**：(1) 在 ViT/CLIP 的 CIL（ImageNet-R、CUB）和 CLIP 的 MTIL 上跑 8 种方法（CODA、SD-LoRA、EWC-LoRA、SLAO、MagMax、SFA、TA、全微调）；(2) 每步记录：新任务数据上相对 base 的 KL、权重 Δ-norm、更新稀疏度、梯度秩、intruder-dimension 数；(3) 回归实测遗忘；(4) 把 KL 作为 MagMax/SFA 的合并系数选择准则做干预实验。
- **Hypothesis**：KL 在视觉上同样是最强预测量（r>0.8），权重量 r<0.5；用 KL 选合并系数比固定系数遗忘 ↓20%。
- **MVE**：ViT-B/16；ImageNet-R 10 任务 + MTIL；指标：预测器-遗忘相关性、干预后 AvgAcc/Forgetting；GPU：约 40-60 单卡小时。
- **Closest**：RL's Razor (2509.04259，仅 LLM、RL vs SFT)；MagMax (2407.06322，权重启发式)；LoRA intruder dims (NeurIPS 2025)。区别：跨模态验证 + 作为合并准则的干预。
- **Novelty 7/10**；**Risk**：视觉分类的 KL 在 class-incremental 下因输出头变化难以定义，需用特征空间 KL 或 logit 子集。
- **Contribution**：诊断 + 方法准则。
- **Objection**：「KL 与性能是循环定义」——需用新任务 KL 预测旧任务遗忘来避免。
- dedup_key：`kl_forgetting_predictor_vision_merging`

### Idea 3：Closed-Form Ability Router — 用 RanPAC/Fly-CL 式闭式头替代 MLLM 路由器与输出头做 MLLM 持续能力学习
- **Method**：(1) 冻结 MLLM（Qwen2-VL-2B / InternVL-2B），取视觉+文本池化隐藏态；(2) 高维稀疏随机投影 + 去相关（Fly-CL）构造任务路由器，闭式更新；(3) 每任务只训 rank-8 LoRA，由路由器选择；(4) 在 MLLM-CL ability track（非 IID）和 CoIN 上评估，并与 RePro（原型路由）、MR-LoRA 比较。
- **Hypothesis**：闭式去相关路由在非 IID ability track 上比原型路由高 ≥3 pts，且无需 MLLM 推理做路由（延迟 ↓10×）。
- **MVE**：2B 模型；MLLM-CL ability + CoIN；指标：AvgAcc、BWT、路由准确率、延迟；GPU：LoRA 训练 ~20h + 评估 ~10h 单卡。
- **Closest**：Is Our Benchmark Enough (ICML 2026, RePro 原型路由)；Fly-CL (2510.16877, 仅 ViT 分类)；Routing-Based CL (2511.01831, 训练路由)。区别：首次把去相关闭式头作为 MLLM 路由并在非 IID 轨道验证。
- **Novelty 6/10**；**Risk**：Is Our Benchmark Enough 若已指出 ability track 也可分，增益可能小。
- **Contribution**：方法（轻量）。
- **Objection**：「只是换了路由器」；需展示在多任务顺序和重叠任务流形上仍成立。
- dedup_key：`closed_form_router_mllm_cl`

### Idea 4：One Stream, Two Communities — CTTA 与 TF-CIL 在同一 CLIP 数据流上的统一对照
- **Method**：(1) 构造 CLIP 数据流：ImageNet-C 渐变 corruption（BayesTTA 风格）× 类增量到达（CIL）× 标签偏移；(2) 跑 CTTA 侧 UL-TTA、MS-TTA、PTA、TPS 和 CL 侧 RanPAC、RAIL、Continual-CLIP；(3) 把两族都写成「原型 + 统计量更新」的统一公式，分析差异仅在更新规则；(4) 提出混合更新规则。
- **Hypothesis**：两族在混合流上各自崩在对方擅长的维度；统一的「原型 + 协方差 + 先验」闭式更新同时拿到两边 ≥90% 的最优值。
- **MVE**：CLIP ViT-B/16；ImageNet-C/R + CIFAR-100-C 流；指标：流式准确率、Forgetting、延迟；GPU：全部 TF，<15 单卡小时。
- **Closest**：TTABC (2606.14299, 只有 TTA 族)；BayesTTA (2507.08607, 渐变偏移)；RAIL (2406.18868, X-TAIL)。区别：首次把 CTTA 与 CIL 方法放进同一流并给统一形式。
- **Novelty 7/10**；**Risk**：流设计被批「人造」。
- **Contribution**：诊断 + 统一框架 + 轻量方法。
- **Objection**：「两个社区问题定义不同，对比不公平」——需多种流配置。
- dedup_key：`ctta_cil_unified_clip_stream`

### Idea 5：Audited Merging Benchmark — 审计前后 TIES/DARE/TA 的持续合并排名是否改变？
- **Method**：(1) 复用 Audit Before You Merge 的两项检查（base 匹配、Δ-norm）扫描 HF 上 Llama-3.1-8B / Qwen2.5-7B 的 200+ 公共 LoRA；(2) 构造「审计通过」与「未审计」两个 8-adapter 序列；(3) 顺序合并（TA、TIES、DARE-TIES、SFA 式平均、SLAO 式正交合并）；(4) 测 GSM8K/HumanEval/MMLU/IFEval 的遗忘，比较两组排名。
- **Hypothesis**：未审计集里 ≥30% 的「低遗忘」来自 Δ≈0 伪影；审计后方法排名发生显著变化（Kendall τ<0.5）。
- **MVE**：7-8B 模型，仅推理；GPU：评估 5 方法×2 组×4 基准 ≈ 25 单卡小时（vLLM）。
- **Closest**：Audit Before You Merge (ICML 2026, 只做 4 adapter、未比较合并算法)；2511.21437（公开 ckpt 合并基准，无审计）；MagMax（视觉）。
- **Novelty 5/10**；**Risk**：公共 adapter 质量过差导致无可用序列。
- **Contribution**：评测 + 诊断。
- **Objection**：「增量小」——需补充 adapter-hub 规模统计与自动审计工具。
- dedup_key：`audited_lora_continual_merging_ranking`

### Idea 6：Decomposing the Gain — 训练式 PTM-CL 方法的增益 = 分类头 + 特征适配 + 调参
- **Method**：(1) 对 CODA-Prompt、SD-LoRA、EWC-LoRA、InfLoRA 各取训练后模型；(2) 三步消融：a) 冻结其特征，换成 RanPAC 闭式头；b) 用其特征但把更新投影掉 intruder dimensions；c) 用原始冻结特征 + 其分类头；(3) 对每个方法报告三者贡献；(4) 用 Hierarchical Decomposition 的复现设置控制调参效应。
- **Hypothesis**：≥60% 的增益来自分类头 + 调参；特征适配贡献 <2 pts，且在高泄漏数据集上趋零。
- **MVE**：ViT-B/16；CIFAR-100、ImageNet-R、CUB；指标：各分量贡献；GPU：~30 单卡小时。
- **Closest**：RanDumb (2402.08823，只给总体结论)；PAM/Slim (2603.21170)；LoRA intruder dims。区别：逐方法的增益分解。
- **Novelty 6/10**；**Risk**：分解不唯一。
- **Contribution**：诊断。
- **Objection**：「Shapley 式分解更严谨」。
- dedup_key：`ptm_cl_gain_decomposition`

### Idea 7：Unified Forgetting Ledger — 同一任务流下 ICL / RAG / LoRA-CL / 编辑 / sparse-memory 的统一遗忘定义对比
- **Method**：(1) 用 WikiBigEdit 子集 + TRACE 的 2 个任务 + Chronos 式时间戳事件构造一条 2,000 更新的流；(2) 定义三种遗忘：能力遗忘（MMLU/GSM8K 掉分）、事实过期（旧答案仍被输出）、暂时对齐丢失（Spurious 式可恢复）；(3) 方法：ICL（全上下文）、RAG、持续 LoRA、WISE/AlphaEdit、SMF 式稀疏更新；(4) 固定推理+训练总 FLOPs 预算。
- **Hypothesis**：按三种遗忘拆开后，CL-Bench「ICL 赢」与 Chronos「RAG 不一致」不再矛盾：ICL 零能力遗忘但事实过期最高。
- **MVE**：Qwen2.5-1.5B/3B；指标：三类遗忘 + 等 FLOPs；GPU：LoRA/编辑训练 ~30h，推理 ~15h 单卡。
- **Closest**：WikiBigEdit (2503.05683，编辑 vs LoRA vs RAG，但无能力/时间维度)；CL-Bench (NeurIPS 2026)；Chronos (ACL 2026)。区别：统一三维遗忘定义 + 等预算。
- **Novelty 7/10**；**Risk**：工作量大，需严格裁剪到 2 域。
- **Contribution**：评测协议 + 诊断。
- **Objection**：「每条线都有更强的专门方法」。
- dedup_key：`unified_forgetting_ledger_icl_rag_lora_edit`

### Idea 8：Scaling the Update Count — 10→10,000 次更新下 TF 闭式头 vs 单 LoRA 合并 vs 编辑的退化曲线
- **Method**：(1) 视觉：ImageNet-21K 子集按 1 类/任务切成 1,000 任务，跑 RanPAC、Fly-CL（闭式）与 SLAO、EWC-LoRA（单 LoRA）；(2) 语言：WikiBigEdit 1→10,000 编辑，跑 LoRA-CL、SMF 式稀疏更新、AlphaEdit；(3) 记录精度、内存、Gram 条件数 / 编辑矩阵范数；(4) 拟合退化曲线并找拐点。
- **Hypothesis**：闭式头在类数 >2,000 时因 Gram 病态开始退化（需 Fly-CL 式稀疏化救回），而单 LoRA 合并在 ~200 任务后遗忘超越闭式头；两条曲线交点给出「何时该训练」的实用规则。
- **MVE**：ViT-B/16 + Qwen2.5-1.5B；GPU：视觉几乎 TF（~10h），语言 ~40h。
- **Closest**：WikiBigEdit（编辑 250 次崩溃）；SLAO（O(1) 内存，任务数 ≤ 几十）；Fly-CL（多重共线性）。区别：以更新次数为主轴的跨方法族曲线。
- **Novelty 6/10**；**Risk**：1,000 任务视觉基准太人造。
- **Contribution**：scaling 诊断。
- **Objection**：「极端任务数无现实意义」——用 WikiBigEdit 的真实时间流回应。
- dedup_key：`update_count_scaling_tf_vs_lora_vs_edit`

### Idea 9：Is Separability the Whole Story? — 跨 4 个 MLLM-CL 基准的任务可分性与「原型路由上界」
- **Method**：(1) 对 MLLM-CL、CoIN、MLLM-CTBench、2311.16206 基准，用冻结 CLIP/MLLM 特征计算任务可分性（线性探针准确率、Fisher 比）；(2) 对每个基准跑 RePro 式原型路由 + per-task LoRA 作为「路由上界」；(3) 与各基准上报告的 SOTA 方法比较；(4) 用任务顺序随机化与任务混合（overlapping manifold）构造「困难版」。
- **Hypothesis**：4 个基准中 ≥3 个可分性 >95%，原型路由达到 SOTA 的 ≥95%；困难版上 SOTA 方法增益缩减一半以上。
- **MVE**：LLaVA-1.5-7B 或 Qwen2-VL-2B；GPU：特征提取 + LoRA ~30h 单卡。
- **Closest**：Is Our Benchmark Enough (ICML 2026，仅 MLLM-CL)；CoIN (NeurIPS 2024)；MLLM-CTBench (2508.08275)。区别：跨基准泛化 + 困难版构造。
- **Novelty 5/10**；**Risk**：与 ICML 2026 文章重叠度高，必须靠跨基准 + 困难版差异化。
- **Contribution**：诊断 + 新基准变体。
- **Objection**：「增量工作」。
- dedup_key：`mllm_cl_benchmark_separability_audit`

### Idea 10：Pretraining-Scale Sweep for TF-CL — 预训练数据量如何决定 TF 与训练式 CL 的差距
- **Method**：(1) 固定 ViT-B/16 架构，取 OpenCLIP 不同预训练集（CC12M / LAION-400M / LAION-2B / DataComp-1B）与 IN1K/IN21K 监督模型；(2) 对每个骨干跑 Continual-CLIP、RanPAC、RAIL（TF）与 CODA、SD-LoRA（训练）；(3) 在 ImageNet-R、CUB、Aircraft、EuroSAT、DTD（含远离自然图像的域）上报告 TF−trained 差距随预训练规模的曲线；(4) 关联 Data Leakage 的泄漏估计。
- **Hypothesis**：TF 优势随预训练规模单调增长，但在远域（EuroSAT、DTD、医学）上饱和甚至反转；差距可由「预训练集-下游域距离」解释。
- **MVE**：5 骨干 × 5 数据集 × 5 方法；GPU：~60 单卡小时（训练式方法占多数）。
- **Closest**：Continual-CLIP (2210.03114)；PTA (ICML 2026)；Data Leakage (2508.17416)。区别：首次做预训练规模主轴的 TF vs trained 扫描。
- **Novelty 6/10**；**Risk**：骨干间架构/训练细节差异混淆。
- **Contribution**：scaling 诊断。
- **Objection**：「骨干不可控」——用 OpenCLIP 同代系列缓解。
- dedup_key：`pretraining_scale_sweep_tf_cl_gap`

---

## 5. 未完成 / 建议后续核实
- 搜索预算在第 38 次后耗尽，以下未能核查：ACL 2026 CL 主会论文、CVPR 2026 CL 论文、OPCM/MINGLE 类持续合并、Parametric RAG (DyPRAG)、MemoryLLM/M+/Cartridges、ModalPrompt/LLaVA-c/SEFE/CL-MoE、CTTA 的 NeurIPS 2025 论文、CLIP-CL 的 ICLR 2026 新方法。
- 2601.18699 两个版本内容矛盾，勿引用。
- InfLoRA、SimpleCIL/APER、ZSCL 为记忆补充，ID 需复核。
- 所有「关键数字」来自摘要片段，未读正文表格。
