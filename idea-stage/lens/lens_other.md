# Lens-Other：Training-free / 低资源 LLM-MLLM 研究方向广度扫描（排除推理效率、TTS-scaling、CL 三块）

- 日期：2026-10-17
- 代理：ARIS Phase 1 + 1.5 子代理（lens_other）
- 环境限制说明：arxiv.org / export.arxiv.org / alphaxiv / papers.cool / openreview / aclanthology / semanticscholar 的直连全部被代理以 403 拦截（WebFetch 报 ENOTFOUND，curl 报 CONNECT 403）。因此所有核实均通过 WebSearch 对 arXiv ID / 标题的定向查询完成（约 40 次查询；本轮共享搜索预算在第 41 次时耗尽）。每篇论文标注核实级别：**[核实]** = 定向查询返回了 arXiv 摘要/日期/作者；**[片段]** = 仅来自首轮搜索摘要片段，细节未逐一核对。所有引用均来自真实搜索结果，未编造。
- "llm tts" 两种解读：(a) test-time scaling 由另一位代理负责，本报告不重复；(b) 语音 TTS 在第 8 节覆盖。

---

## 0. 查询清单（≥15 个不同查询式，实际约 40）

首轮广度（20 条）：model merging survey/reasoning-instruct；activation steering ICLR 2026；training-free long context extrapolation；function/in-context vectors many-shot；RouteLLM/cascade/training-free router；MLLM hallucination decoding/high-res；LLM-based TTS acceleration CosyVoice/F5；agent KV reuse/memory/token budget；decoding-time alignment weak-to-strong；training-free jailbreak defense；prompt optimization GEPA/DSPy/TextGrad；training-free dLLM acceleration；training-free unlearning；training-free video memory；training-free grounding/zoom；speculative decoding for TTS；evolutionary/heterogeneous merging；training-free personalization；contrastive decoding DoLa successors；tokenizer transfer；function vectors mechanism；merging theory；ExPO/merge-instead-of-RLHF；zero-shot TTS emotion control；multi-agent token budget；multi-image training-free；Self-Extend/DCA/LongRoPE2；self-distillation/data selection；2026 routers；thinking/instruct interpolation；LoRA extremes；LLM-TTS KV/streaming；reasoning-length steering。
二轮核实（20 条）：按 arXiv ID 逐篇查询（见下表）。

---

## 1. 论文卡片（Top 30+，按方向）

格式：ID｜日期｜问题｜方法一句话｜关键数字｜training-free?｜自述 limitation / future work｜核实级别

### 1.1 模型合并（非 CL 用途）

1. **The Thinking Spectrum: An Empirical Study of Tunable Reasoning in LLMs through Model Merging** — [arXiv:2509.22034](https://arxiv.org/abs/2509.22034)｜2025-09-26 (v2 09-29)｜Lan, Zheng, Cao, Li｜开源模型只在"纯推理"与"直答"两端，中间缺位｜把通用模型与推理专家做加权合并，扫描合并强度画 accuracy-vs-token 曲线｜发现部分合并点同时在精度与 token 上优于父模型（Pareto 改进）；即使父模型权重空间差异很大也有效｜**是**｜未见 limitation 段（摘要页未给）；隐含：仅经验研究，无机制解释｜[核实]
2. **Revisiting Model Interpolation for Efficient Reasoning** — [arXiv:2510.10977](https://arxiv.org/abs/2510.10977)（ACL 2026）｜2025-10｜Wu, Yang, Liu, Wang, Ngai｜Thinking/Instruct 合并用于高效推理，但复杂方法需要配对 base 模型｜直接线性插值两组权重，不需 base｜发现三阶段动态（Instruct 占主导阶段 Pass@k 与输出 token 快速上升……）；Qwen3-4B / Qwen3-30B-A3B；第三方评述称 MI-0.9 在 AIME'25 Pass@64 达 100；TA/TIES 对 base 选择敏感｜**是**｜消融覆盖层/模块/解码；未覆盖非 Qwen 家族与跨架构｜[核实]
3. **Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions** — [arXiv:2603.09938](https://arxiv.org/abs/2603.09938)｜2026-03 (v2 03-30)｜Song & Zheng｜此前综述未把理论-方法-应用-生态串起来｜FUSE 四分类（Foundations/Unification/Scenarios/Ecosystem）｜—｜—｜明确列出 "theoretical gaps"（LMC/同盆地假设缺证明）为开放挑战｜[片段]
4. **Leveraging Submodule Linearity Enhances Task Arithmetic Performance in LLMs** — [arXiv:2504.10902](https://arxiv.org/abs/2504.10902)（ICLR 2025）｜子模块内的线性度比全局线性度更关键；基于此的合并稳定优于标准 task arithmetic｜**是**｜[片段]
5. **Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of LM Reasoning** — [arXiv:2605.14386](https://arxiv.org/abs/2605.14386)｜2026-05｜14 维合并基因 + MRI-Trust Fusion + Architecture Mapper 做跨架构"繁殖"｜**是**｜[片段]
6. **AutoMerge** — [arXiv:2601.22748](https://arxiv.org/abs/2601.22748)｜2026-01｜把模型切成异构块，对每块搜索合并算法及超参｜**是**（搜索需评估开销）｜[片段]
7. **Training-free Heterogeneous Model Merging** — [arXiv:2501.00061](https://arxiv.org/abs/2501.00061)｜2024-12/2025-01｜深度差异用层段切分，宽度差异用 elastic neuron zipping 投影到公共维度｜**是**｜[片段]
8. 锚点：**Model Extrapolation Expedites Alignment (ExPO)** — [arXiv:2404.16792](https://arxiv.org/abs/2404.16792)（ACL 2025, v5 2025-05）｜放大 SFT→aligned 权重差；DPO 训 20% 步数 + ExPO 超过全训；1.8B–70B 均有效，仅一个超参 α｜**是**｜[片段]；**Unlocking Efficient Long-to-Short Reasoning via Model Merging** — [arXiv:2503.20641](https://arxiv.org/abs/2503.20641)｜Huawei｜合并缓解 overthinking，动机是训练法需配对长短答案｜[片段]
9. LoRA 层面：**Compress-then-Merge** [arXiv:2606.03723](https://arxiv.org/abs/2606.03723)（先压缩再合并到单个 rank-r）；ACL 2026 Findings 的 SVD+CUR 无训练 LoRA 合并（Lu 等）；**Exploring Sparse Adapters for Scalable Merging** [arXiv:2507.07140](https://arxiv.org/abs/2507.07140)（ICLR 2025；20 任务，held-out 仍是难点）｜[片段]

### 1.2 解码期/激活期对齐与控制

10. **COLD-Steer: Steering LLMs via In-Context One-step Learning Dynamics** — [arXiv:2603.06495](https://arxiv.org/abs/2603.06495)（ICLR 2026）｜用一步梯度动力学近似 in-context 样本会导致的表示位移，兼顾样本效率与强度｜评测 CAA(7 任务)、BiPO(4 任务)、OpinionsQA；刻意排除 AxBench｜**是**｜自述：未覆盖 AxBench 式"忽略指令"任务｜[核实-部分]
11. **ODESteer** — [arXiv:2602.17560](https://arxiv.org/abs/2602.17560)（ICLR 2026）｜从正负激活的 log-density ratio 推导 ODE 流作为 steering 方向｜TruthfulQA +5.7%，UltraFeedback +2.5%，RealToxicityPrompts +2.4%（相对先前 steering）｜**是**｜[片段]
12. **GER-steer** — [arXiv:2603.12298](https://arxiv.org/abs/2603.12298)｜用跨层表示稳定性精炼原始 steering 向量｜**是**｜[片段]
13. **Understanding (Un)Reliability of Steering Vectors in Language Models** — [arXiv:2505.22637](https://arxiv.org/abs/2505.22637)｜2025-05｜Braun 等｜7 种提示类型均净正向，但样本间方差大、常反向；训练激活差的 cos 相似度与正负可分性预测可 steer 性；结论：目标行为不对应一致方向时 steering 不可靠｜诊断类｜[核实]
14. **The Rogue Scalpel: Activation Steering Compromises LLM Safety** — [arXiv:2509.22067](https://arxiv.org/abs/2509.22067)（ICLR 2026）｜2025-09｜随机方向 steering 把有害顺从率从 0% 抬到 1–13%（v1 报 2–27%）；良性 SAE 特征同样；20 个随机向量组合成通用攻击｜诊断类｜结论"对内部的精确控制 ≠ 对行为的精确控制"｜[核实]
15. **Analysing the Safety Pitfalls of Steering Vectors** — [arXiv:2603.24543](https://arxiv.org/abs/2603.24543)｜2026-03｜CAA steering 可使 jailbreak ASR 上升 57% 或下降 50%，归因于与 refusal 方向重叠｜诊断类｜[片段]
16. **Activation Steering for Chain-of-Thought Compression (ASC)** — [arXiv:2507.04742](https://arxiv.org/abs/2507.04742)（NeurIPS 2025 workshop 页 / ACL 2026 Findings）｜50（ACL 版 100）对冗长-简洁样本抽 steering 向量｜MATH500/GSM8K CoT 长度最多 −67.43%（ACL 版 −69.4%，加 KL 信赖域），8B 端到端 2.73×｜**是**｜[核实]
17. **Activation Control for Efficiently Eliciting Long CoT** — [arXiv:2505.17697](https://arxiv.org/abs/2505.17697)（NeurIPS 2025）｜末几层少量高影响激活决定长度与自反思；放大 + 插 "wait" 即可无训练诱发长推理｜**是**｜[片段]
18. **STU-PID** — [arXiv:2506.18831](https://arxiv.org/abs/2506.18831)｜PID 控制器在解码中动态调 steering 强度抑制 overthinking｜**是**｜[片段]
19. **Alignment-Aware Decoding (AAD)** — [arXiv:2509.26169](https://arxiv.org/abs/2509.26169)（ICML 2026, PMLR 306）｜ETH｜只用 DPO 模型 + 其 SFT 参考模型在推理期做隐式奖励优化，无额外训练｜跨基准跨规模一致优于基线；可反哺迭代 DPO 生成高价值合成样本｜**是**（需已有 DPO 对）｜[核实]
20. **AdaBeam** — [arXiv:2510.23334](https://arxiv.org/abs/2510.23334)｜2025-10｜奖励引导自适应 beam，无权重更新｜[片段，定向查询未命中，细节未核实]
21. **Refusal Activation Steering (RAS)**（AAAI 2026 学生摘要）｜对比激活在推理期推向拒绝，宣称避免过度拒绝｜**是**｜[片段]；对照：**Fail-Closed Alignment** [arXiv:2602.16977](https://arxiv.org/abs/2602.16977) 为训练法；**Silencing the Guardrails** [arXiv:2604.07835](https://arxiv.org/abs/2604.07835) 为解码期消融 refusal 低秩子空间的攻击。
22. **LayerCake: Token-Aware Contrastive Decoding within LLM Layers** — [arXiv:2507.04404](https://arxiv.org/abs/2507.04404)｜2025-07｜标点 token 主导早层、概念 token 主导中层；按深度抑制对应 token 注意力做受控退化，作为对比信号｜**是**｜注意：新版本已被作者撤回，仅 v1 可用；无定量结果可见｜[核实]
23. 视觉侧对比解码进展（ICML 2026）：**Beyond Logits: Attention Contrastive Decoding (ACD)**（logit 级 CD 伤连贯性，改在注意力层做自适应减法，可与 VCD 叠加）；**Disentangled Visual Rectification (DVR)**（VCD 的噪声对比会误压有效视觉内容）；**AttnReal** [arXiv:2503.08342](https://arxiv.org/abs/2503.08342)（把输出 token 吸走的注意力拨回视觉 token，近零开销）；2026 新作 **DeP** [arXiv:2604.12424](https://arxiv.org/abs/2604.12424)、**Spotlight and Shadow** [arXiv:2604.10071](https://arxiv.org/abs/2604.10071)｜[片段]

### 1.3 长上下文 training-free 扩展

24. **Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation (DPE)** — [arXiv:2504.18857](https://arxiv.org/abs/2504.18857)（投 COLM 2025）｜Fudan｜逐 RoPE 维度估计"有效长度"，只重映射 top 50–75% 关键维度（Llama3-8B 选 48 维，分 8 组）｜Llama3-8B 8k→128k 无训练；Llama3.1-70B RULER +18；优于 YaRN/Self-Extend；兼容 FA2｜**是**｜[核实]
25. **MrRoPE: Mixed-radix Rotary Position Embedding** — [arXiv:2601.22181](https://arxiv.org/abs/2601.22181)（ICLR 2026 oral）｜2026-01-28｜把各类 RoPE 扩展统一为混合进制转换；派生 MrRoPE-Uni / MrRoPE-Pro 两个 training-free 变体｜MrRoPE-Pro 128K NIAH 召回 >85%，InfiniteBench 检索/对话子集精度 >2× YaRN；理论上抬高 RoPE 可编码长度上界｜**是**｜[核实]
26. **GALI (Greedy Attention Logit Interpolation)** — [arXiv:2502.02659](https://arxiv.org/abs/2502.02659)（EMNLP 2025）｜复用预训练位置区间并插值注意力 logit 消除离群｜**是**｜[片段]
27. **Jet-Long: Dynamic Bifocal RoPE** — [arXiv:2607.07740](https://arxiv.org/abs/2607.07740)｜2026-07｜按序列长度动态缩放因子，短输入保原行为；对比 NTK/YaRN/Self-Extend/DCA｜**是**｜[片段]
28. **RiPRA**（ACL 2026 Findings）：相关性感知的位置资源分配；**DoPE: Denoising RoPE** [arXiv:2511.09146](https://arxiv.org/abs/2511.09146)（仅见标题）；锚点 LongRoPE2 [arXiv](https://arxiv.org/abs/2502.20082)（需 10B token 训练，非 training-free，但其"高维训练不足导致 OOD"的诊断是本方向共识）｜[片段]

### 1.4 小/少数据适应：ICL 高效化、函数向量、prompt 优化、个性化

29. **Compressing Many-Shots in In-Context Learning (MemCom)** — [arXiv:2510.16092](https://arxiv.org/abs/2510.16092)｜2025-10-17｜Khatri … Jain, Kusupati｜现有 prompt 压缩在 many-shot 下失效｜更强压缩器 + 逐层压缩｜8× 压缩精度降通常 <10%，基线可降 >70%｜否（压缩器需训练）但目标是少资源｜[核实]
30. **Instruction function vectors** — [arXiv:2505.12075](https://arxiv.org/abs/2505.12075)｜从短指令抽 FV 提升零样本；指令 FV 与示范 FV 依赖**不同注意力头**｜**是**｜[片段]
31. **Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens** — [arXiv:2604.02608](https://arxiv.org/abs/2604.02608)（ICML 2026）｜12 任务 × 6 模型 × 4032 跨模板对；FV steering 常在 logit lens 任何中间层都解不出答案时仍成功｜诊断类｜[片段]
32. **What Makes a Good Function Vector?**（NeurIPS 2026 列表）｜把归因限制在 intensional frame 使 steering 精度平均 +18%；LRP 归因约 500× 快于基线｜[片段]；**Task Vectors, Learned Not Extracted** [arXiv:2509.24169](https://arxiv.org/abs/2509.24169)｜[片段]；对比研究 **Bottom-Up vs Top-Down Steering on ICL** [arXiv:2411.07213](https://arxiv.org/abs/2411.07213)（ICV 擅长行为偏移，FV 擅长精确任务）。
33. **GEPA: Reflective Prompt Evolution Can Outperform RL** — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)（ICLR 2026 oral, v2 2026-02）｜读自然语言轨迹而非标量、维护 Pareto 前沿｜厂商博客引用：vs MIPROv2 约 +13%，vs GRPO 约 +20% 且 rollouts 少约 35×（数字来自二手源）；社区观察：metric 只给 float 时退化到 MIPRO 水平｜无梯度｜[片段]
34. **Amulet** — [arXiv:2502.19148](https://arxiv.org/abs/2502.19148)｜SJTU/PKU/BIGAI｜把每个 token 生成视为在线学习问题，闭式更新做实时 training-free 偏好适配｜**是**｜[片段]
35. **Differential Preference Steering (DPS)**（ACL 2026）｜因果掩码定位 "Preference Heads"，解码时对比有/无这些头｜**是**｜[片段]；**CLIPer** [arXiv:2605.07162](https://arxiv.org/abs/2605.07162)（分类器引导，训练成本 O(1)）；**HyperTrace** [arXiv:2609.09835](https://arxiv.org/abs/2609.09835)（自然语言假设追踪短/长期偏好，PRISM/PersonaMem-v2）｜[片段]

### 1.5 路由 / 级联 / 模型选择

36. **A Unified Approach to Routing and Cascading for LLMs** — [arXiv:2410.10347](https://arxiv.org/abs/2410.10347)（ICML 2025）｜统一 routing 与 cascading；质量估计器是关键；最高 +14%｜[片段]
37. **TagRouter**（ACL 2025 Findings）｜训练-free 标签路由做开放域生成多模型集成；13 个基线，接受率 +6.15%，成本 −17.2%｜[片段]
38. **ZeroRouter: Breaking Model Lock-in** — [arXiv:2601.06220](https://arxiv.org/abs/2601.06220)（AAAI 2026）｜2026-01-09｜USTC/iFLYTEK｜路由器加新模型需重训｜解耦"查询刻画"与"模型画像"，查询映射到通用难度潜空间；双模式优化器权衡精度/成本/延迟｜宣称精度优于全部基线且更低成本延迟｜零样本加模型（预测器需训练）｜[核实]
39. **VDAR-Router: Verbalized Query Difficulty Analysis Retrieval** — [arXiv:2607.18098](https://arxiv.org/abs/2607.18098)｜2026-07-20｜NYCU｜现有路由器看表层语义忽略难度｜LLM 先显式生成难度分析（默认 7 个可解释维度），检索相似难度画像的历史样本，用性能-成本奖励选模型｜RouterBench/LLMRouterBench 上四种难度表征均优于最强基线；维度数无统一最优｜**是**｜[核实]
40. 其他：**ODAR** [arXiv:2602.23681](https://arxiv.org/abs/2602.23681)（主动推理路由）、**VL-RouterBench** [arXiv:2512.23562](https://arxiv.org/abs/2512.23562)（VLM 路由基准）、锚点 RouteLLM（ICLR 2025，需训练）｜[仅标题]

### 1.6 Agent 与工具调用的 training-free 效率

41. **Phase-Scheduled Multi-Agent Systems** — [arXiv:2604.17400](https://arxiv.org/abs/2604.17400)｜2026-04｜控制 agent 何时激活而非只控上下文；宣称生产管线 29–38% 上下文冗余（自报）｜[片段]
42. **Single-agent vs multi-agent under matched compute**（Tran & Kiela）— [arXiv:2604.02460](https://arxiv.org/abs/2604.02460)｜2026-04｜固定推理 token 预算下 SAS 可匹配/超过 MAS；MAS 仅在单 agent 有效上下文利用退化或花更多算力时占优｜诊断类｜[片段]
43. **REDEREF** — [arXiv:2603.13256](https://arxiv.org/abs/2603.13256)｜training-free 多 agent 递归委派路由控制器（Thompson 采样）｜[片段]
44. KV/上下文层：**SparseX** [arXiv:2606.01751](https://arxiv.org/abs/2606.01751)（段级 KV 共享，附录含 AutoGen/辩论负载）、**StepKV** [arXiv:2609.22158](https://arxiv.org/abs/2609.22158)（步感知 agent KV 压缩）、**Agentic Context Management** [arXiv:2607.21503](https://arxiv.org/abs/2607.21503)（五原语生命周期；只有"验证过的压缩"才线性成本）、KVCOMM / TokenDance（跨前缀 KV 对齐、同步多 agent 轮次 All-Gather KV 共享；仅在片段中出现，无 ID）、Ramp Labs "Latent Briefing" 博客（orchestrator↔worker KV 压缩，LongBench v2 worker token −65%，非论文）、SJTU token 压缩综述（2026-07）｜[片段]

### 1.7 MLLM training-free 能力增强

45. **HiDe: Rethinking the Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling** — [arXiv:2510.00054](https://arxiv.org/abs/2510.00054)（ICML 2026）｜2025-09-28｜先前认为小目标感知受限需 zoom｜发现主因是**背景干扰而非目标尺寸**；Token-wise Attention Decoupling 定位区域 + Layout-Preserving Decoupling 保空间结构剔背景｜Qwen2.5-VL-7B / InternVL3-8B 在 V*Bench 92.1 / 91.6，超 RL 方法；内存比前一 training-free 方法 −75%；HRBench4K/8K SOTA｜**是**｜[核实]
46. **Zoom-Refine** [arXiv:2506.01663](https://arxiv.org/abs/2506.01663)（预测 bbox→裁剪→再答）；**LazyMCoT** [arXiv:2606.16158](https://arxiv.org/abs/2606.16158)（一次前向的首 token 统计估不确定性，按难度决定是否裁剪/缩放）；**ZoomUI** [arXiv:2603.14448](https://arxiv.org/abs/2603.14448)（GUI 坐标生成时的注意力分布做 zoom）；锚点 Zoom Eye（树搜索）｜[片段]
47. **FlexMem: Scaling Long Video Understanding via Visual Memory Mechanism** — [arXiv:2603.29252](https://arxiv.org/abs/2603.29252)（CVPR 2026, pp.31877–31888）｜2026-03-31｜Xiamen U / NUDT｜视觉 token 的 KV cache 作为记忆；双通路压缩写入；MemIndex 快速索引读出｜单张 3090 处理 >1k 帧；LLaVA-Video / LLaVA-OneVision 上部分基准追平或超 GPT-4o、Gemini-1.5 Pro｜**是**｜[核实]
48. **StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding** — [arXiv:2508.15717](https://arxiv.org/abs/2508.15717)｜2025-08-21｜Meta AI/NYU｜用通用 query token 的注意力分数压 KV，固定大小记忆，不需预知问题｜LLaVA-OV-7B 离线基准总分 59.4 vs 基线 56.9；与 query-aware 方法可比｜**是**｜[核实]
49. **FreshMem** [arXiv:2602.01683](https://arxiv.org/abs/2602.01683)（频率记忆 + 空间缩略图记忆；Qwen2-VL 上 StreamingBench/OV-Bench/OVO-Bench +5.20/+4.52/+2.34）；**Recurrent Attention-based Token Selection** [arXiv:2510.17364](https://arxiv.org/abs/2510.17364)（指出 ReKV 类 training-free 方法 KV 过大）｜[片段]
50. **Decoding the Pulse of Reasoning VLMs in Multi-Image Understanding (PulseFocus)** — [arXiv:2603.04676](https://arxiv.org/abs/2603.04676)｜2026-03-04｜Cornell 单作者｜CoT 中文本→图像注意力分散且有跨图位置偏差｜把 CoT 拆成 plan/focus 交错块，解码期把注意力门控到当前讨论的图｜BLINK +3.7，MuirBench +1.07；标注"ongoing project，结果可能变"｜**是**｜[核实]
51. **FOCUS / Mitigating Cross-Image Information Leakage** [arXiv:2508.13744](https://arxiv.org/abs/2508.13744)（用噪声遮住其他图）；锚点 **ICoT** [arXiv:2411.19488](https://arxiv.org/abs/2411.19488)（CVPR 2025，仅靠注意力图插入图像区域）；**SDR-MCoT**（AAAI 2026）｜[片段]
52. **MLLMEraser** — [arXiv:2510.04217](https://arxiv.org/abs/2510.04217)｜2025-10｜training-free 测试时 activation steering 做多模态知识擦除｜**是**｜[片段]

### 1.8 基于 LLM 的语音 TTS：training-free 加速 / 控制 / 零样本克隆

53. **EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable TTS via Activation Steering** — [arXiv:2508.03543](https://arxiv.org/abs/2508.03543)｜2025-08-05 (v3 10-25)｜Xie, Yang, Li, Yu, Liu｜在 flow-matching TTS 内部激活上做推理期 steering，支持情感转换/插值/擦除｜适用 F5-TTS、CosyVoice2、E2-TTS；自建多说话人情感数据集抽向量；自称首个 training-free 连续细粒度情感控制｜**是**｜局限：只作用于 flow-matching 声学部分，未触及 LLM 自回归语义 token 阶段｜[核实]
54. **TED-TTS (Segment-Aware Conditioning for Training-Free Intra-Utterance Emotion and Duration Control)** — [arXiv:2601.03170](https://arxiv.org/abs/2601.03170)（ACL 2026）｜2026-01-06｜NUS｜句内多情感/时长控制依赖非公开数据与多阶段训练｜因果掩码 + 单调流对齐过滤做分段情感条件；局部 duration embedding steering + 全局 EOS logit 调制控时长｜句内一致性 SOTA、质量保持｜**主体 training-free**，但为自动构造分段 prompt 微调了 Qwen3-8B（30k 样本）｜[核实]
55. **Fast F5-TTS** — [arXiv:2505.19931](https://arxiv.org/abs/2505.19931)（Interspeech 2025）｜非均匀采样调度裁剪冗余步｜7 步，RTX 3090 RTF 0.030，约 4× F5-TTS｜**是**｜[片段]
56. **Selective Classifier-free Guidance for Zero-shot TTS** — [arXiv:2509.19668](https://arxiv.org/abs/2509.19668)（Interspeech 2025）｜在 F5-TTS 与 CosyVoice2 上研究 CFG 选择性施加｜**是**｜[片段]
57. **Accelerating Autoregressive Speech Synthesis with Speech Speculative Decoding (SSD)** — [arXiv:2505.15380](https://arxiv.org/abs/2505.15380)（Interspeech 2025）｜小 draft 模型 + 目标模型并行验证｜约 1.4×，感知质量保持｜需 draft 模型｜[片段]
58. **Principled Coarse-Grained Acceptance for Speculative Decoding in Speech (PCG)** — [arXiv:2511.13732](https://arxiv.org/abs/2511.13732)｜2025-11｜Apple (Yanuka, Giryes 等)｜语音 token 精确匹配过严，很多 token 声学可互换｜从目标模型 embedding 空间导出 "Acoustic Similarity Groups"，在组级接受并给出组级精确性保证｜LibriTTS 上接受率与吞吐高于标准 SD 与此前语音松弛法，可懂度/说话人相似度保持（具体倍数未见）｜**是**（需 draft）｜[核实]
59. **VADUSA** [arXiv:2410.21951](https://arxiv.org/abs/2410.21951)（draft heads + 采样容忍）；**Prosodic Boundary-Aware Streaming for LLM-TTS with Streaming Text** [arXiv:2603.06444](https://arxiv.org/abs/2603.06444)（TTFA 1296 ms 最低，限制上下文防 KV 无界增长；是否需训练未确认）；**TLDR** [arXiv:2606.09019](https://arxiv.org/abs/2606.09019)（codec token 打包成 patch，1.8×、KV −75%，但用 LoRA，非 training-free）｜[片段]

### 1.9 其他新兴低资源方向

60. **Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding** — [arXiv:2505.22618](https://arxiv.org/abs/2505.22618)（ICLR 2026）｜NVIDIA/MIT｜块级近似 KV cache + 置信度阈值并行解码｜LLaDA/Dream 吞吐最高 27.6×｜**是**｜后续 **Flash-dLLM** [arXiv:2609.26796](https://arxiv.org/abs/2609.26796)（MBZUAI，2026-09；指出 cache 复用与并行验证叠加时的 I/O 成本被孤立研究）｜[片段]（注：dLLM 加速与"推理效率"代理有重叠，此处仅列锚点）
61. **Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit** — [arXiv:2506.06607](https://arxiv.org/abs/2506.06607)｜Goddard 等（mergekit-tokensurgeon）｜新 token 表示为共享锚 token 的稀疏组合，系数复用到 base embedding 空间｜Llama→Mistral NeMo、Qwen→Llama 零样本保持最好；自述：数字 token 化不匹配伤数学推理｜**是**｜EACL 2026 SRW 有 30 语言对的独立复现对比（FVT vs OMP vs 微调）｜[片段]
62. 自蒸馏/低标注（非 training-free 但低资源）：**OPSD** [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)（同一模型不同上下文自当师生）、**Neuron-OPSD** [arXiv:2607.02460](https://arxiv.org/abs/2607.02460)（神经元激活选数据+建教师上下文，无标签）、**RGSD** [arXiv:2606.12507](https://arxiv.org/abs/2606.12507)（rubric 条件教师，无验证器）、警示性研究 [arXiv:2603.24472](https://arxiv.org/abs/2603.24472)（自蒸馏为何有时损害推理）｜[片段]
63. 遗忘：锚点 **In-Context Unlearning** [arXiv:2310.07579](https://arxiv.org/abs/2310.07579)（ICML 2024）；**DUET** [arXiv:2601.21283](https://arxiv.org/abs/2601.21283)（指出 ICU 可被去掉 prompt 或逆向攻破，改用蒸馏）；MLLMEraser（见 52）｜[片段]
64. 小适配器：**BoostLoRA** [arXiv:2604.27308](https://arxiv.org/abs/2604.27308)（每轮在失败样本上训极小适配器并折入权重）｜[片段]

---

## 2. Landscape map 与饱和度评级

| 方向 | 评级 | 一句话理由 | 代表论文 |
|---|---|---|---|
| 2.1a 合并·Thinking/Instruct 可调推理 | **活跃→趋饱和** | 2025-09 至 2026 已有 Thinking Spectrum、Revisiting Interpolation、Long-to-Short 三条线得出一致结论"简单插值即可调 token-精度曲线"；剩余空间在机制解释与跨家族/跨模态 | 1, 2, 8 |
| 2.1b 合并·异构/进化/跨架构 | **活跃** | Darwin Family、AutoMerge、异构 zipping 2026 密集出现，但评测分散、无统一基准，跨 tokenizer 合并几乎空白 | 5, 6, 7, 61 |
| 2.1c 合并·理论 | **起步** | 综述明确列 "theoretical gaps"；仅子模块线性度一条可操作理论 | 3, 4 |
| 2.2a Activation steering 方法 | **饱和** | ICLR 2026 一批（COLD-Steer、ODESteer、GER-steer），均在 CAA/BiPO/TruthfulQA 小基准上比 1–6 个点 | 10–12 |
| 2.2b Steering 可靠性/安全副作用 | **活跃（刚起步的反面研究）** | Braun、Rogue Scalpel、Safety Pitfalls 三篇 2025-05 至 2026-03 连续揭示不可靠与安全税，但尚无缓解方案与跨应用审计 | 13–15 |
| 2.2c 推理长度 steering | **活跃** | ASC、Activation Control、STU-PID 2025 集中出现；与合并插值的关系未研究 | 16–18 |
| 2.2d 解码期对齐（ARGS 系） | **活跃** | AAD（ICML 2026）、AdaBeam；但都需 DPO 对或奖励模型，"无参考模型"的 training-free 对齐仍少 | 19, 20 |
| 2.2e 文本幻觉对比解码（DoLa 系） | **饱和/衰减** | LayerCake 新版撤回；视觉侧（ACD/DVR）比文本侧更活跃 | 22, 23 |
| 2.3 长上下文 training-free 扩展 | **活跃** | DPE→MrRoPE（ICLR 2026 oral）→Jet-Long 形成"逐维/进制"新分析范式；但评测仍止于 128K NIAH/RULER，与其他 training-free 干预（steering、合并）的交互未知 | 24–28 |
| 2.4a Many-shot ICL 压缩 / 函数向量 | **活跃** | MemCom 把压缩推到 8×；FV 机制研究（Steerable-not-Decodable、What Makes a Good FV）2026 出现；二者尚未交汇 | 29–32 |
| 2.4b Prompt 优化 | **饱和（工程化）** | GEPA 成为 DSPy 默认，后续多为应用；学术空间在 metric 文本反馈设计 | 33 |
| 2.4c Training-free 个性化 | **活跃** | Amulet、DPS、CLIPer、HyperTrace 路线各异，缺统一基准与长期一致性评测 | 34, 35 |
| 2.5 路由/级联 | **活跃** | 2026 转向 training-free/零样本（ZeroRouter、VDAR）；难度估计是共识瓶颈；路由对象仍是"固定模型集合" | 36–40 |
| 2.6 Agent training-free 效率 | **起步（混乱）** | 大量系统/博客（KV 共享、压缩原语）但学术化程度低；Tran&Kiela 的"算力匹配"诊断显示基础假设未被检验 | 41–44 |
| 2.7a MLLM 高分辨率/zoom | **活跃→趋饱和** | HiDe 改写了问题归因（背景干扰），LazyMCoT 做难度自适应；GUI 分支扩散 | 45, 46 |
| 2.7b 视频 training-free 记忆 | **活跃** | FlexMem(CVPR 2026)、StreamMem、FreshMem 皆为 KV-as-memory；位置编码处理从未被当作变量 | 47–49 |
| 2.7c 多图/交错推理 training-free | **起步** | PulseFocus 单作者 ongoing、FOCUS 噪声遮罩，均小增益 | 50, 51 |
| 2.8a LLM-TTS 加速（投机解码） | **活跃** | SSD→PCG（Apple）把接受准则语音化；但 draft 模型仍需训练，training-free 的自投机（层跳/早退）在 TTS 未见 | 57–59 |
| 2.8b TTS training-free 控制 | **起步** | EmoSteer（flow-matching 侧）、TED-TTS（条件/EOS logit）各一篇；AR 语义 token LM 侧的 steering 空白 | 53, 54 |
| 2.9 dLLM 加速 | 活跃（归他人） | Fast-dLLM 27.6× 已成基线 | 60 |
| 2.9 tokenizer 迁移 | **起步** | OMP 一篇 + EACL 复现；与合并/路由的联动未探索 | 61 |
| 2.9 training-free 遗忘 | **起步** | ICU 被证脆弱，steering 式遗忘仅 MLLMEraser | 52, 63 |

---

## 3. 五 lens 结构性 gap

### Lens A：Method-transfer（方法迁移）
- **A1. 逐维位置分析（DPE/MrRoPE）从未迁移到视频 MLLM 的视觉 token 位置。** FlexMem/StreamMem/FreshMem 全部靠 KV 压缩/驱逐处理 >1k 帧，把 RoPE 当常量；DPE 指出只有部分维度决定有效长度，MrRoPE 给出无训练扩展上界。视频 token 的时间维度位置分布（Qwen2.5-VL 的 M-RoPE）与文本不同，逐维重映射可能让"不驱逐"也可行。基于：24, 25, 47, 48。
- **A2. CoT 长度 steering（ASC/Activation Control）→ LLM-TTS 的自回归语义 token LM。** EmoSteer 只在 flow-matching 声学模型做 steering；TED-TTS 用 EOS logit 调时长。CosyVoice2/3 的 LM 阶段与文本 LLM 同构，ASC 式"冗长→简洁"向量对应 TTS 的"语速/停顿/强调"，尚无人做。基于：16, 17, 53, 54。
- **A3. Braun 等的"cos-sim 预测 steering 可靠性"→ 合并成功率预测。** 合并理论只有子模块线性度；steering 领域已有可计算的几何预测量（训练激活差的 cos 相似度、正负可分性）。把它搬到 task vector 的逐层 cos / 子空间重叠上，可成为 training-free 的逐层 α 选择器。基于：13, 4, 2。
- **A4. VDAR 的"口头化难度分析"→ 作为 steering 强度 / 插值系数 / zoom 深度的统一控制旋钮。** 目前 STU-PID 用 PID、LazyMCoT 用首 token 不确定性、Revisiting Interpolation 用固定 α；难度估计在路由领域已被证明是瓶颈且可 training-free 获取。基于：39, 18, 46, 2。

### Lens B：Contradiction（矛盾）
- **B1. Steering 既是安全工具又是安全漏洞。** RAS/ODESteer 宣称用 steering 降毒性/提拒绝；Rogue Scalpel 与 Safety Pitfalls 证明任意方向（含随机、良性 SAE 特征、CAA）都会抬高有害顺从率（1–13%，ASR ±57%）。没有任何**非安全用途**的 steering 论文（ASC、COLD-Steer、DPS、EmoSteer、MLLMEraser）报告过安全副作用。基于：11, 14, 15, 16, 21, 35。
- **B2. 高分辨率失败归因：目标尺寸 vs 背景干扰。** Zoom-Refine/Zoom Eye 系假设"看不清小物体"，HiDe 实验表明主因是背景干扰；LazyMCoT 则以首 token 不确定性当难度。三者对"何时该 zoom"的假设互斥，尚无在视频/多图上的裁决。基于：45, 46, 50。
- **B3. 合并是否需要共同盆地/base 模型。** 综述理论把 LMC/同盆地当前提；Thinking Spectrum 称父模型权重空间"相当不同"也能合并；Revisiting Interpolation 称 TA/TIES 对 base 敏感而纯插值不需 base。谁对取决于未明说的"差异度量"。基于：1, 2, 3。
- **B4. 多 agent 是否值得。** Phase-Scheduled MAS / REDEREF 继续优化多 agent 协调；Tran & Kiela 显示算力匹配后 SAS 不输 MAS。KV 共享类工作（SparseX/KVCOMM）的收益前提——agent 间上下文大幅重叠——与 Tran&Kiela 的"单 agent 有效上下文退化"论点互相解释又互相削弱。基于：41–44。

### Lens C：Untested-assumption（未检验假设）
- **C1. Steering 向量在长上下文/位置扩展后仍有效。** 所有 steering 论文在 ≤4k 上下文抽取与评测；DPE/MrRoPE 改写了注意力几何。向量是否随 RoPE 重映射漂移从未测。基于：10–12, 24, 25。
- **C2. 路由难度是模型无关的。** ZeroRouter 把难度映射到"通用潜空间"并声称可零样本加入新模型；但合并/插值产生的中间模型（MI-0.3…0.9）在难度空间中的位置是否可由父模型插值预测，无人验证。基于：38, 2。
- **C3. TTS steering 向量与说话人/语言无关。** EmoSteer 用多说话人数据抽向量，但未报告跨语言（中英）与零样本克隆目标说话人上的迁移；TED-TTS 的 EOS 调制假设对所有说话速率一致。基于：53, 54。
- **C4. Many-shot 压缩是任务无关的。** MemCom 逐层压缩器一视同仁；FV 研究显示指令式与示范式任务信息走不同注意力头，且 FV "可 steer 但不可解码"，暗示压缩应按头/按层而非均匀。基于：29, 30, 31。

### Lens D：Scaling-regime（规模域）
- **D1. Steering 与 FV 几乎全在 7–9B dense 模型。** MoE（Qwen3-30B-A3B）上插值已被证有效，但 steering 在 MoE 路由分支上会不会改变专家选择、小模型（0.6–1.7B）上 steering 向量是否过于低维，均无数据。基于：2, 13, 16。
- **D2. Thinking/Instruct 插值的三阶段动态只在 4B/30B-A3B 上观察。** 边缘规模（≤1.7B）和超大规模（70B+）是否仍有三阶段、是否有相同的 α 拐点，未知；这决定其是否可作为通用"思考预算"旋钮。基于：1, 2。
- **D3. Training-free 长上下文方法评测止于 128K 且多在 8B。** MrRoPE 给出理论上界，但 1M 级与 ≤1B 模型（维度更少，DPE 的"关键维度"占比可能不同）未测。基于：24, 25。
- **D4. Training-free 路由对象规模。** VDAR/ZeroRouter 在 RouterBench（约十个商用模型）上评测；在"几十个开源小模型 + 合并变体"的长尾池上，难度表征是否仍区分得开，未测。基于：38, 39。

### Lens E：Diagnostic（诊断/基准缺口）
- **E1. 没有"training-free 对齐税"统一基准。** steering、解码期对齐（AAD）、ExPO/合并三类方法从未在同一模型、同一预算（额外前向次数、额外内存）下比较对齐增益与安全/能力损失。基于：8, 14, 19.
- **E2. 没有合并"可预测性"诊断。** 合并成败只能事后评测；缺少类似 Braun 等的事前几何指标。基于：3, 4, 13。
- **E3. 语音侧缺少 training-free 控制/加速方法的联合代价评测。** EmoSteer/TED-TTS/PCG/Fast F5 各用不同指标（MOS、SIM、WER、RTF），没有"控制强度 × 说话人相似度 × WER × 延迟"的四维曲线。基于：53–58。
- **E4. Agent 跨上下文重叠度无实测。** KV 共享类方法的收益假设未用真实 trace（SWE-bench、WebArena、多 agent 辩论日志）量化。基于：44, 42。

---

## 4. 候选 idea（12 个，跨方向）

### Idea 1：Merge-vs-Steer 等价性——权重插值 α 与激活 steering 强度是否同一条轴
- **Method**：(1) 取 Instruct 与 Thinking 模型（如 Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-7B，或 Qwen3 的 think/no-think 两份权重），做 α∈{0.1…0.9} 插值；(2) 用 ASC 协议从同一对模型抽"冗长→简洁"steering 向量，扫描强度 λ；(3) 对每个 (α, λ) 记录 MATH500/GSM8K 精度与 token 数，并计算各层激活差 Δh(α) 与 steering 方向 v 的余弦；(4) 检验 Revisiting Interpolation 的三阶段拐点是否对应 cos 的突变/饱和。
- **Hypothesis**：插值的三阶段动态在激活空间上表现为一条低维曲线，steering 只能复现其中一段（Instruct 占主导的阶段），因此两者互补而非替代；在拐点处组合二者得到更优 Pareto 前沿。
- **MVE**：单卡 A100/4090 24GB（7B bf16）；MATH500 + GSM8K + AIME24；指标：acc、平均 token、Pareto 面积；约 25 GPU h。
- **Closest work**：Revisiting Model Interpolation (2510.10977)——只做权重；ASC (2507.04742)——只做激活；区别：首次把两者放到同一几何框架下并给出组合策略。
- **Novelty 6/10**｜**Risk**：若 cos 关系平凡（线性）则只是"又一个经验表"｜**Contribution type**：分析 + 轻方法｜**Reviewer objection**："Qwen 家族特有，换 Llama 是否成立？"（需加第二家族）｜dedup_key：`merge_steer_equivalence_cot_length`

### Idea 2：Steering 安全税审计 + refusal-正交化修复
- **Method**：(1) 复现 Rogue Scalpel 协议（HarmBench/AdvBench 有害顺从率）；(2) 对五类**非安全用途**steering（ASC CoT 压缩、COLD-Steer 行为、DPS 个性化、指令 FV、MLLMEraser 式擦除）逐一测量安全税；(3) 提出 training-free 修复：先用对比法估计 refusal 子空间（1–3 维），把任意 steering 向量投影到其正交补再施加；(4) 报告"目标效果保留率 vs 安全税消除率"。
- **Hypothesis**：安全税主要来自 steering 向量与 refusal 方向的非零投影（Safety Pitfalls 的解释）；正交化可消除 >80% 的安全税而目标效果损失 <10%。
- **MVE**：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct；HarmBench 子集 + 各 steering 的原任务集；约 15 GPU h（单卡）。
- **Closest work**：The Rogue Scalpel (2509.22067)、Analysing the Safety Pitfalls (2603.24543)——只诊断不修；区别：跨应用审计 + 最简修复 + 验证修复的能力代价。
- **Novelty 7/10**｜**Risk**：refusal 子空间非线性，正交化不足（Rogue Scalpel 显示随机方向也有税，暗示非方向性因素）｜**Contribution type**：诊断 + 防御｜**Objection**："正交化会削弱 steering 本身"——需报告效果保留率｜dedup_key：`steering_safety_tax_refusal_orthogonal`

### Idea 3：AR 语音 LM 的激活 steering 做语速/停顿/强调控制（CosyVoice2 LM 阶段）
- **Method**：(1) 用 CosyVoice2-0.5B 的 LLM 阶段，对同一文本合成快/慢（或有/无强调）配对语音（可用 TED-TTS 的 EOS 调制或 SSML 外部工具生成对照）；(2) 在 LM 隐状态上按 ASC 法抽"快→慢"、"平→强调"向量；(3) 推理时按 token 段施加（段级 steering 复用 TED-TTS 的 segment 思路）；(4) 评测 WER、SIM、时长误差、UTMOS。
- **Hypothesis**：语义 token LM 内存在与文本 LLM "CoT 长度"同构的"时长方向"，steering 可实现连续语速/强调控制且不伤说话人相似度。
- **MVE**：CosyVoice2-0.5B（已开源）；Seed-TTS-eval（zh/en）+ LibriTTS test-clean 子集；单卡 12 GPU h。
- **Closest work**：EmoSteer-TTS (2508.03543) 只在 flow-matching 侧；TED-TTS (2601.03170) 用 logit/embedding 而非隐状态 steering；区别：首次在 LLM-TTS 的 AR 语义 token 阶段做 activation steering，且覆盖韵律而非情感。
- **Novelty 7/10**｜**Risk**：语义 token 可能不编码时长（时长主要在声学 token/flow 阶段）｜**Contribution type**：方法迁移｜**Objection**："与直接 prompt 控制（CosyVoice 的指令模式）比优势何在？"——需对比指令模式｜dedup_key：`ar_speech_lm_prosody_steering`

### Idea 4：视频 MLLM 的逐维位置重映射（DPE-for-Video）
- **Method**：(1) 在 Qwen2.5-VL-7B / LLaVA-OneVision-7B 上，用 DPE 的 2-范数注意力贡献度量，分别对 M-RoPE 的时间/空间维度估计"有效长度"；(2) 只对关键时间维度做重映射以把可用帧数从 ~64 扩到 ≥512，不做 KV 驱逐；(3) 与 FlexMem/StreamMem（驱逐/压缩）比较，并试验"重映射 + 轻度压缩"的组合；(4) 分析哪些维度在视频里是关键维度，与文本结论对比。
- **Hypothesis**：视频 token 的时间维度在预训练中"训练不足"的维度比例高于文本，因此 DPE 式重映射的收益更大；并且与 KV 压缩正交可叠加。
- **MVE**：Video-MME、LongVideoBench、MLVU；单卡 4090/3090（FlexMem 已证单卡可行）；约 30 GPU h。
- **Closest work**：DPE (2504.18857)、MrRoPE (2601.22181)——文本；FlexMem (2603.29252)、StreamMem (2508.15717)——压缩不改位置；区别：第一次把"位置"当作视频 training-free 扩展的变量。
- **Novelty 7/10**｜**Risk**：M-RoPE 的多维结构使"有效长度"估计不直接；视频基准对帧数饱和｜**Contribution type**：方法迁移 + 分析｜**Objection**："收益可能被更简单的帧采样抵消"——需与均匀/自适应采样基线比｜dedup_key：`dimensionwise_rope_video_mllm`

### Idea 5：合并可预测性——用 steering 领域的几何指标做 training-free 逐层 α 选择
- **Method**：(1) 收集同 base 的 6–8 个 finetune（math/code/chat/safety 等，Llama-3-8B 或 Qwen2.5-7B）；(2) 对每对/每组计算逐层 task vector 余弦、子空间主角度、以及 Braun 式"探针数据上激活差的 cos 一致性"；(3) 在 50+ 随机合并组合上做回归，看哪些指标预测合并后平均增益；(4) 用最好的指标设计逐层 α（高一致层多合并，低一致层退回单一父模型），与 TIES/DARE/线性比较。
- **Hypothesis**：激活层面的一致性指标比权重层面的 cos 更能预测合并成败（因 FV 研究表明"可 steer 但不可解码"，权重/logit 视角不足）。
- **MVE**：mergekit + lm-eval；8B 模型 CPU 合并 + 单卡评测；约 20 GPU h。
- **Closest work**：Submodule Linearity (2504.10902)；Braun et al. (2505.22637)；区别：把 steering 的可靠性预测量迁移到合并，并产出一个无需搜索的逐层 α 规则。
- **Novelty 6/10**｜**Risk**：指标相关但解释力弱（R² < 0.3）｜**Contribution type**：诊断 + 方法｜**Objection**："AutoMerge/进化搜索也能找到 α"——卖点是零搜索开销｜dedup_key：`merge_predictability_activation_geometry`

### Idea 6：口头化难度作为统一控制旋钮——按难度选插值模型/steering 强度/zoom 深度
- **Method**：(1) 用 VDAR 式提示让一个小模型（或目标模型自身）输出 5–7 维难度画像；(2) 把画像映射到三种 training-free 控制：Instruct↔Thinking 插值 α（预先合成 3–4 个模型）、ASC 强度、或（MLLM）是否 zoom；(3) 与固定 α、PID、首 token 不确定性（LazyMCoT）比较成本-精度曲线。
- **Hypothesis**：显式难度画像在跨数据集迁移上优于内部不确定性信号（后者对分布漂移敏感），且一次难度估计的开销 <5% 总 token。
- **MVE**：MATH500/GSM8K/AIME 的混合流 + V*Bench；Qwen3-4B 系列 + Qwen2.5-VL-7B；API 或单卡；约 15 GPU h。
- **Closest work**：VDAR-Router (2607.18098) 只做模型选择；STU-PID (2506.18831)、LazyMCoT (2606.16158) 用内部信号；区别：把路由领域的难度表征当作通用 training-free 控制接口并跨三种机制验证。
- **Novelty 5/10**｜**Risk**：难度自评不可靠（LLM 自我校准差）｜**Contribution type**：系统/统一框架｜**Objection**："本质是 router 的 trivial 扩展"——需展示跨机制一致增益｜dedup_key：`verbalized_difficulty_control_knob`

### Idea 7：头感知的 many-shot ICL 压缩——用函数向量替代均匀逐层压缩
- **Method**：(1) 复现 MemCom 的 8× 压缩设置；(2) 对同一 many-shot prompt，按 2505.12075 分别抽取示范 FV 与指令 FV（不同头集合），把两者作为"压缩后上下文"注入；(3) 混合方案：MemCom 只压缩非 FV 头的 KV，FV 头保留/替换为向量；(4) 在 What-Makes-a-Good-FV 的 LRP 头选上做消融。
- **Hypothesis**：many-shot 的信息集中在少数头，头感知压缩在 8–16× 压缩率下精度下降明显小于均匀逐层压缩，且无需训练压缩器。
- **MVE**：Llama-3.1-8B / Gemma-2-9B；many-shot 分类、翻译、抽取任务（沿用 Many-Shot ICL 论文任务）；单卡 12 GPU h。
- **Closest work**：MemCom (2510.16092)——需训练压缩器；Multimodal Task Vectors (NeurIPS 2024)——多模态、不区分头类型；区别：training-free、头类型感知、与压缩器可组合。
- **Novelty 6/10**｜**Risk**：FV 在生成式/开放任务上弱（Brumley 等）｜**Contribution type**：方法｜**Objection**："FV 只适合分类式任务"——需包含生成任务｜dedup_key：`head_aware_manyshot_fv_compression`

### Idea 8：Agent trace 的跨上下文重叠实测——training-free KV 共享收益上界
- **Method**：(1) 收集公开 agent 轨迹（SWE-agent/OpenHands 的 SWE-bench 日志、多 agent 辩论日志、WebArena）；(2) 对每条轨迹计算 token 级前缀重叠、段级（工具输出/系统提示/历史）重叠与跨 agent 重叠；(3) 模拟 KVCOMM/SparseX 式共享可省的 prefill token 与 Tran&Kiela 算力匹配下的 SAS/MAS 比较；(4) 给出"什么样的任务/框架值得做 KV 共享"的决策表。
- **Hypothesis**：真实轨迹的跨 agent 重叠远低于论文合成负载（<30% vs >60%），training-free KV 共享的收益主要来自系统提示与工具 schema，而非对话历史。
- **MVE**：纯 CPU 文本分析 + 少量前向估算；0–5 GPU h。
- **Closest work**：SparseX (2606.01751)、ACM (2607.21503)、Tran & Kiela (2604.02460)；区别：第一个基于真实轨迹的重叠度测量与收益上界。
- **Novelty 5/10**｜**Risk**：公开轨迹代表性不足｜**Contribution type**：诊断/测量｜**Objection**："没有新方法"——定位为 analysis paper｜dedup_key：`agent_trace_kv_overlap_measurement`

### Idea 9：LLM-TTS 的注意力级对比解码治"跳字/重复"幻觉
- **Method**：(1) 在 CosyVoice2 的 AR LM 中定位文本-语音对齐注意力头（单调性最强的头）；(2) 构造"受损"前向：按 LayerCake/ACD 思路抑制这些头对文本 token 的注意力；(3) 用对比解码（正常 − β·受损）调整语音 token logit；(4) 在 Seed-TTS-eval hard 集与长句集测 WER/插入删除率、SIM、RTF。
- **Hypothesis**：跳字/重复对应对齐头注意力崩塌，注意力级对比可在不伤 SIM 的前提下把 hard 集 WER 降 15–30%。
- **MVE**：CosyVoice2-0.5B；Seed-TTS-eval（zh/en hard）；单卡 10 GPU h。
- **Closest work**：ACD（ICML 2026，VLM）、LayerCake (2507.04404)、Selective CFG (2509.19668)；区别：首次把注意力对比解码用于 LLM-TTS 的对齐幻觉。
- **Novelty 7/10**｜**Risk**：AR 语音 LM 的跳字可能来自 tokenizer/流匹配阶段而非 LM 注意力｜**Contribution type**：方法迁移｜**Objection**："CFG/重复惩罚已能缓解"——需与之对比并展示互补｜dedup_key：`tts_attention_contrastive_decoding_hallucination`

### Idea 10：插值三阶段的规模与模态外推——0.6B→70B 与 VLM
- **Method**：(1) 在 Qwen3 0.6B/1.7B/4B/8B/14B/32B 与 30B-A3B 上做 Thinking↔Instruct 插值扫描；(2) 在 VLM 上（如 Qwen2.5-VL-Instruct vs 其推理变体）重复；(3) 在 MoE 上额外记录每个 α 下专家路由分布的变化；(4) 拟合"拐点 α 随规模变化"的经验律。
- **Hypothesis**：拐点 α 随规模单调右移（大模型需要更多 Thinking 权重才切换），MoE 的切换由少数专家主导（可只插值这些专家）。
- **MVE**：数学/多模态推理基准（MATH500、MathVista）；小模型单卡、32B 需 2 卡或 API 权重托管；约 40 GPU h（可裁剪到 ≤8B 为 15 h）。
- **Closest work**：Revisiting Interpolation (2510.10977)、Thinking Spectrum (2509.22034)；区别：规模律 + 模态扩展 + MoE 专家级分析。
- **Novelty 5/10**｜**Risk**：纯经验扩展｜**Contribution type**：分析｜**Objection**："增量"——靠 MoE 专家级发现与可预测拐点加分｜dedup_key：`interpolation_three_stage_scaling_law`

### Idea 11：Steering 向量在 training-free 长上下文扩展下的漂移与重校准
- **Method**：(1) 在 Llama-3-8B 上抽取 CAA/ASC 向量（短上下文）；(2) 施加 DPE / MrRoPE-Pro / YaRN 扩展到 32K–128K；(3) 在长上下文任务（目标行为嵌于 NIAH 式长文）中测 steering 成功率随位置/长度衰减；(4) 提出重校准：在扩展后的模型上用相同 50 对样本重抽向量，或按 DPE 关键维度做向量投影修正。
- **Hypothesis**：位置重映射改变中层表示的几何，steering 成功率在 >32K 处显著下降；重校准可恢复大部分。
- **MVE**：单卡 8B + 长文合成基准；约 15 GPU h。
- **Closest work**：Braun et al. (2505.22637)（可靠性但短上下文）、MrRoPE/DPE（不涉及 steering）；区别：首个交互研究。
- **Novelty 6/10**｜**Risk**：若无漂移则只是负结果（仍可发为 finding）｜**Contribution type**：诊断｜**Objection**："应用场景小众"——可联系长上下文个性化/安全｜dedup_key：`steering_under_context_extension`

### Idea 12：跨 tokenizer 的 training-free 合并——OMP 移植 + 异构块合并
- **Method**：(1) 用 mergekit-tokensurgeon 的 OMP 把模型 B 的 embedding/unembedding 移植到模型 A 的 tokenizer；(2) 对齐后用 Training-free Heterogeneous Merging / AutoMerge 的块级策略合并非 embedding 层；(3) 评测合并模型在 A、B 各自专长任务上的保持率；(4) 专门测 OMP 自述的弱点（数字 token 化不一致→数学）。
- **Hypothesis**：tokenizer 差异是跨家族合并的主要障碍之一；OMP 对齐后，同架构不同 tokenizer 的模型（如 Llama-3 与 Mistral-NeMo 尺寸相近者）可在保持 >85% 各自能力下合并。
- **MVE**：Llama-3.1-8B ↔ Mistral-NeMo-12B 不等宽，改选同宽对（如两个 Qwen2.5-7B 派生但换过 tokenizer 的社区模型）；lm-eval；约 15 GPU h。
- **Closest work**：OMP transplant (2506.06607)、Heterogeneous Merging (2501.00061)、Darwin Family (2605.14386)；区别：把 tokenizer 当作合并的第一类障碍处理。
- **Novelty 6/10**｜**Risk**：非 embedding 层的表示空间不对齐，合并直接崩｜**Contribution type**：方法组合｜**Objection**："合适的模型对太少，实用性存疑"｜dedup_key：`cross_tokenizer_training_free_merge`

---

## 5. 关于 "llm tts" 两种解读的覆盖确认
- test-time scaling：由另一代理负责；本报告中与之相邻的内容只有 Idea 1/6/10 的"思考预算"旋钮，定位为 training-free 控制而非 compute scaling。
- 语音 TTS：第 1.8 节（7 篇）+ Idea 3、9；主要发现是 training-free 控制（EmoSteer、TED-TTS）与投机解码（SSD、PCG）两线各 1–2 篇，AR 语义 token LM 侧的 steering 与对比解码空白。

## 6. 方法论局限
- 无法打开 arXiv 页面，limitation/future work 字段多来自摘要与搜索片段；AdaBeam、GEPA 具体数字来自二手页面，使用前需核对。
- 搜索预算耗尽前未完成的核实：Fast-dLLM、Darwin Family、AutoMerge、OMP、Amulet、ODESteer、GEPA、Steerable-not-Decodable、MLLMEraser、Tran&Kiela、Jet-Long、Safety Pitfalls、SSD、合并综述的 future-directions 段。
