# 免训练 / 低资源方向顶会调研报告（2025-10 → 2026-10）

日期：2026-10-18 · 全部由我自己检索与阅读，未使用子 agent · 不以 arXiv 未录用论文为依据（arXiv 只用于检查“是否已被抢发”）

## 0. 数据与方法

| 来源 | 论文数 | 获取方式 |
|---|---|---|
| ICLR 2026 | 5,691 | iclr.cc 官方 virtual JSON（含作者机构） |
| NeurIPS 2025 | 5,982 | neurips.cc 官方 JSON（含摘要） |
| ICML 2026 | 6,735 | icml.cc 官方 JSON |
| NeurIPS 2026 | 9,139 | neurips.cc 官方 JSON |
| CVPR 2026 / ICCV 2025 / ECCV 2026 | 4,042 / 2,765 / 2,862 | CVF Open Access、ECVA |
| ACL 2026 / EMNLP 2025 | 主会 2,752 + Findings 2,164 / 主会 2,217 + Findings 1,406 | ACL Anthology |
| 大厂一手资料 | — | HuggingFace 官方 config.json；Nemotron-H、Nemotron Nano 2、DeepSeek-V4 技术报告；Qwen Gated Attention（NeurIPS'25 Oral） |

方法：标题关键词分桶 → 逐桶通读标题和机构 → 对关键论文抓取摘要（共 6,547 篇带摘要，含 NeurIPS'25 全部和其余会议效率/TTS 类约 700 篇）→ 对候选 idea 在全库做标题 + 摘要检索，再到 arXiv 查抢发。

局限：标题正则会漏检（例：ICML'26 的 “Token-Pruning-Induced Vulnerabilities in VLMs” 标题里没有 safety，第一次没检到，后来补查才发现）；ICLR/ICML/NeurIPS'26 的摘要只抓了效率与 TTS 相关的部分；OpenReview 需要人机验证，没法读审稿意见。

## 1. 各方向顶会论文量（标题关键词计数，主会）

| 方向 | ICLR'26 | NeurIPS'25 | ICML'26 | NeurIPS'26 | CVPR'26 | ECCV'26 | ACL'26 | 判断 |
|---|---|---|---|---|---|---|---|---|
| KV cache | 23 | 28 | 35 | 62 | 3 | 1 | 19 | 极度拥挤 |
| 投机解码 | 33 | 20 | 39 | 48 | 9 | 5 | 19 | 拥挤 |
| 量化 / 剪枝 | 86 | 57 | 93 | 117 | 36 | 27 | 32 | 极度拥挤 |
| 视觉 token 压缩 | 22 | 27 | 21 | 31 | 58 | 32 | 9 | 极度拥挤（CV 会议尤甚） |
| Test-time scaling | 41 | 35 | 37 | 47 | 9 | 10 | 17 | 拥挤 |
| 高效推理 / overthinking | 29 | 23 | 25 | 20 | 3 | 5 | 26 | 拥挤（多数需要 RL 训练） |
| 持续学习 / 遗忘 | 66 | 73 | 112 | 139 | 73 | 40 | 32 | 拥挤 |
| 模型合并 | 14 | 14 | 23 | 26 | 13 | 4 | 4 | 拥挤 |
| Steering / SAE | 48 | 32 | 65 | 87 | 16 | 18 | 30 | 拥挤 |
| 幻觉 | 50 | 52 | 74 | 82 | 50 | 30 | 61 | 极度拥挤 |
| 扩散 LLM | 48 | 19 | 70 | 55 | 7 | 3 | 20 | 免训练加速部分极度拥挤 |

一句话：**在顶会上，“针对某个固定模型和固定场景再提一个免训练打分规则”的空间已经基本没有了**。还有空间的地方都来自结构性变化：模型架构变了、使用场景变了、评价口径变了。

## 2. 各方向的研究思路与缺口

### 2.1 文本 LLM 推理效率
- **主流思路**：KV 驱逐和量化从“重要性打分”转向三个方向：一是理论化（The risk of KV cache compression，NeurIPS'26，Stanford/RWTH 给出 minimax 风险；Rethinking KV Eviction via Information-Theoretic Objective，ICML'26）；二是学习式（Learning to Evict，ICML'26 Apple；IndexMem，ICML'26 ByteDance/Tencent）；三是面向推理模型（ThinKV，ICLR'26 NVIDIA；R-KV，NeurIPS'25 Microsoft/ByteDance；BeaconKV，ICML'26 Apple；TriAttention，ICML'26 MIT）。
- **新出现的主线：效率的副作用**。Alignment Collapse Under KV Cache Quantization（NeurIPS'26，Microsoft MAI/Caltech）发现低比特 KV 量化会悄悄破坏拒答，原因是安全特征落在一个对量化噪声敏感 100–1000 倍的低维子空间；The Pitfalls of KV Cache Compression（ACL'26）发现多指令下 system prompt 会被忽略；When Efficiency Meets Safety（ACL'26）研究 KV 压缩与越狱；DefensiveKV（ICLR'26）指出驱逐方法依赖的“稳定性假设”很脆弱。
- **缺口**：副作用这条线在文本 KV 上已经有 4 篇顶会论文，空间在收窄。

### 2.2 MLLM 效率
- **主流思路**：单图剪枝的打分规则仍在大量产出（CVPR'26 有 58 篇），但顶会开始出现“反思型”工作：When Token Pruning is Worse than Random（CVPR'26，深层剪枝不如随机）、Why and When Visual Token Pruning Fails（ECCV'26，KAIST/NVIDIA，推理时相关视觉信息会漂移）、Are We Using the Right Benchmark（ACL'26，简单下采样胜过多数压缩方法）、FEATHER（ICCV'25，Stanford，浅层注意力剪枝偏向图像底部）。
- **安全方向已经有人做了**：Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs（ICML'26）、On the Adversarial Robustness of LVLMs under Visual Token Compression（ICML'26）。
- **新场景**：omni 音视频（OmniZip CVPR'26，OmniSIFT/OmniFit/EchoingPixels ICML'26）、VLA（ICLR'26/ECCV'26 多篇）、DeepSeek-OCR 剪枝（RTPrune，ICML'26 JD）。
- **缺口**：混合架构 MLLM（见 2.6）。

### 2.3 Test-time scaling
- **主流思路**：
  - 选择与聚合：Self-Certainty（NeurIPS'25）、Beyond Majority Voting（ICML'26 MIT/Harvard）、Optimal Self-Consistency（ICML'26 Yale/Huawei）、Position: It's Time to Optimize for Self-Consistency（ICML'26 MIT/Anthropic）、Same Pool Different Answer（ECCV'26：似然式 BoN 换个温度就有 41–47% 的题答案会变）。
  - 多样性坍塌是瓶颈：Mode-conditioning（ICLR'26 CMU+Anthropic）、Diversity Matters（ICML'26，单模型投票的收益取决于预测多样性）、Less Diverse Less Safe（ICML'26）。
  - 反向缩放：Inverse Scaling in Test-Time Compute（ICLR'26）、Does Thinking More Always Help?（NeurIPS'25）。
  - 成本视角：**Kinetics（NeurIPS'25 CMU）**、Overscaling Curse of Parallel Thinking（NeurIPS'26 SJTU）。
- **缺口**：Kinetics 的成本模型建立在纯注意力模型上，而主流模型已经换成混合架构（见第 4 节主推 idea）。

### 2.4 持续学习 / 后训练遗忘
- **主流思路**：
  - RL 比 SFT 遗忘少，而且原因已经被解释：Retaining by Doing（ICML'26 Princeton，归因于 on-policy 数据）；RFT Naturally Mitigates Forgetting（ICML'26，Qwen2.5-VL-7B）。
  - 度量方法：Mapping Post-Training Forgetting at Scale（ICLR'26 Tübingen+Google，用样本级 1→0 计数和机会校正）；CapTrack（NeurIPS'26，以能力为中心，最大到 80B）。
  - 免训练 agent 持续学习：JitRL（ICML'26 NUS，WebArena，用记忆调制 logits）；Darwinian Memory（ICML'26 DiDi）。
  - 在记忆充足的设定下，CL 的重点从稳定性转向可塑性：Forget Forgetting（ICLR'26 NYU）。
- **缺口判断**：“RL 为什么遗忘少”已经被 ICML'26 解释；免训练 CL 主要是合并（每个会议 14–26 篇）和 agent 记忆（非常拥挤）。**这个方向我没找到经得起核验的低资源免训练空白**。

### 2.5 其他免训练方向
- **Steering / SAE**：ICML'26 有 65 篇、NeurIPS'26 有 87 篇。已有基准（SteeringSafety，ICML'26）、理论（Towards Understanding Steering Strength，ICML'26）、对抗（Endogenous Resistance to Activation Steering，ICML'26），也有 Anthropic 参与的工作（Steering LMs with Weight Arithmetic，ICLR'26）。拥挤。可以利用的资源：Qwen 官方 2026-04 发布了 Qwen3.5 全系 SAE（SAE-Res-Qwen3.5-*），目前顶会上还没人用。
- **幻觉**：极度拥挤。
- **扩散 LLM 免训练加速**：极度拥挤（Fast-dLLM、dLLM-Cache、SparseD、ES-dLLM、Dynamic-dLLM、DC-Leap 等），不建议。

### 2.6 三个交叉趋势
1. **Agent 场景 × 效率**（NeurIPS'26 才开始在顶会出现）：TriAxialKV（UIUC/Cambridge，按 用户/工具调用/观测/推理 角色做混合精度 KV）、SKIM（KAIST/Meta，结构化生成下的 agent 剪枝）、TaskKV、AgentKVShift（UCSD，免训练）、ACON（ICML'26 KAIST/Microsoft）。但 arXiv 上 2026 年 6–10 月已经涌现大量同类预印本（ActKV、IntentKV、CommitKV、ReFold、FOCUS、GUI-KV、STaR-KV、AQuaUI 等），**下一轮投稿会很挤**。
2. **效率的副作用**（见 2.1、2.2）：文本 KV 和视觉剪枝的安全问题都已经有顶会论文。
3. **主流模型换成了混合架构，而顶会上的免训练效率研究还停留在纯注意力模型**（见第 3 节）。

## 3. 大厂动向（一手资料核实）

HuggingFace 官方 config.json（2026-10-18 抓取）：

| 模型 | 架构 |
|---|---|
| Qwen3.5 全系（0.8B/2B/4B/9B/27B/35B-A3B/122B/397B）、Qwen3.6-27B、Qwen3.8-27B | 每 4 层：3 层 Gated DeltaNet 线性注意力 + 1 层完整注意力；`attn_output_gate: True`；`partial_rotary_factor: 0.25`；原生多模态。Qwen3.5-9B 下载量 826 万 |
| Qwen3.8-Flash-Next | 同样的 3:1 混合 + 512 专家 MoE |
| Kimi K3 | KDA 线性注意力 + 每 4 层一层 MLA |
| GLM-5.3-Flash | KDA 线性注意力 34 层 + DeepSeek Sparse Attention 11 层 |
| DeepSeek V4.1-Flash | DSA（index_topk 512）+ 滑动窗口 + MLA |
| Olmo-Hybrid-7B（Ai2） | 与 Olmo 3 同数据训练的混合模型，有 Think-SFT 版本 |

技术报告：Nemotron Nano 2（NVIDIA）报告推理场景下吞吐是 Qwen3-8B 的 3–6 倍；DeepSeek-V4 在引言中明确把降低 TTS 成本作为架构动机；Gated Attention（NeurIPS'25 Oral，Qwen Team）说明门控注意力没有 attention sink。**这些报告都只给出吞吐数字，没有做 TTS 的等成本分配分析。**

## 4. 主推 idea：Kinetics 在混合架构上还成立吗？

**一句话**：Kinetics（NeurIPS'25 CMU）证明，TTS 的成本主要由注意力的 KV 显存访问决定，所以小模型被高估、TTS 应该靠稀疏注意力。但它的成本模型和实验只覆盖 Qwen3 和 R1-Distill 这类纯注意力模型（论文 Limitations 原文：“Our experiments primarily focus on Qwen3 … and DeepSeek-R1-Distilled-Qwen”），而 2026 年的主流开源模型已经是混合线性注意力架构。混合架构会改写 Kinetics 的核心论据，**这件事目前没有人系统做过**。

**缺口证据**：
- 顶会：6,547 篇带摘要的论文里，研究 TTS 成本或显存模型的只有 Kinetics 自己；混合架构相关的效率论文（STree、HyPIC、DTP、SambaY、NVIDIA 混合剪枝）都不涉及 TTS 的计算最优分配。
- 大厂：Nemotron、DeepSeek-V4 只报告吞吐，没有做规模 × 采样数 × 长度的等成本分析。
- arXiv：用“hybrid / linear attention / test-time scaling / best-of-N / memory access”等组合检索，没有相关预印本。

**用官方 config 算出的零成本证据**（bf16，递归状态按 bf16 计）：

| 模型 | 参数 | 完整注意力层 | KV/token | 32K 的 KV | 递归状态/序列 | KV 与状态 I/O 持平的长度 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | 28/28 | 112 KB | 3.50 GB | — | — |
| Qwen3-8B | 8B | 36/36 | 144 KB | 4.50 GB | — | — |
| Qwen3-32B | 32B | 64/64 | 256 KB | 8.00 GB | — | — |
| Qwen3.5-0.8B | 0.8B | 6/24 | 12 KB | 0.38 GB | 9 MB | ~1.5K token |
| Qwen3.5-9B | 9B | 8/32 | 32 KB | 1.00 GB | 24 MB | ~1.5K token |
| Qwen3.5-27B | 27B | 16/64 | 64 KB | 2.00 GB | 72 MB | ~2.3K token |

（Qwen3-0.6B 的 3.50GB 和 Kinetics 论文里的数字一致，说明计算口径正确。）

**可检验的假设**：
1. **“小模型的 KV 不成比例地大”这个论据不再成立**：Qwen3-0.6B 的 KV/参数显存比约 2.9，Qwen3.5-0.8B 约 0.24。预测：在混合模型上，Kinetics 的“14B 门槛”会大幅下移，2B–9B 加更多采样会成为帕累托最优。
2. **存在一个交叉长度**：输出短于约 1.5–2.3K token 时，每个 token 都要读写的递归状态成为主要访存，混合模型反而可能不如纯注意力。预测：no-think 或短答 TTS 不占优，长 CoT TTS 才占优。（如果状态以 fp32 存储，交叉点会翻倍，需要实测。）
3. **Kinetics 开出的“稀疏注意力”药方在混合模型上收益递减**：混合模型只有 1/4 的层在做注意力，瓶颈会转移到参数和状态 I/O。
4. **理论成本和 vLLM 实测之间的差距**：线性注意力 kernel 的成熟度可能吃掉理论优势。

**实验设计**（只做推理，免训练）：
- **因果对照**：Olmo-3-7B-Think-SFT vs Olmo-Hybrid-Think-SFT-7B，同数据，只有架构不同，用来排除“混合模型只是训得更好”。
- **规模阶梯**：Qwen3.5 {0.8B, 2B, 4B, 9B, 27B} vs Qwen3 {0.6B, 1.7B, 4B, 8B, 14B, 32B}，同一厂商。可加 Nemotron Nano 2 9B 作为第二个混合模型家族。
- **任务**：AIME24/25、GPQA-Diamond、LiveCodeBench，与 Kinetics 保持一致。
- **指标**：eFLOPs（Kinetics 口径并加入状态 I/O 项）、vLLM 实测延迟与吞吐、等成本下的准确率帕累托前沿。

**算力与 pilot**：
- 第 0 步（零 GPU，已完成一部分）：用官方 config 算出成本曲线和交叉长度。
- 第 1 步（约 0.5 GPU-day）：vLLM 实测 Qwen3.5-2B/9B vs Qwen3-1.7B/8B，N ∈ {1…32}、输出长度 2K–32K 的吞吐和延迟，验证假设 2、4。
- 第 2 步（约 2–3 GPU-day，单张 H100/A100-80G）：AIME/GPQA 上 N 最多到 32 的准确率曲线，加上 Olmo 对照。
- **止损**：如果混合与纯注意力两个家族的“等成本最优模型规模”相差不到一个尺寸档，或者 Olmo 对照显示优势完全来自准确率差异而非成本，就停。

**风险**：
- 结论“混合更省”本身有些直观，论文必须靠假设 2（交叉长度）和假设 3（稀疏注意力收益递减）这类非直观结论撑起来。
- Kinetics 原作者（CMU）可能也在做后续。
- Qwen3.5 和 Qwen3 的训练数据不同，所以 Olmo 这对模型是必不可少的。

**适合的会议**：NeurIPS / ICLR（scaling 分析类论文），或 MLSys。

## 5. 其他还有条件可做的方向（核验程度较低）

| 方向 | 顶会锚点 | 空间判断 |
|---|---|---|
| 在 Olmo 3 / Olmo Hybrid 同数据对上，系统比较 KV 驱逐、稀疏注意力、投机解码在两种架构上的表现，解释哪些方法失效、为什么 | Comparing Transformers and Hybrid Models at the Token Level（NeurIPS'26 Ai2：混合模型里注意力层负责复制和层级结构）；HyPIC（NeurIPS'26 Lenovo） | 方法层面 arXiv 已很挤（GateSight、DASC、TreeWY、Bole、V-CoLA），但“同数据对照的因果分析”还没人做；适合作为主推 idea 的一节 |
| 用 Qwen 官方 Qwen3.5 SAE 做混合架构上的 steering / 可解释性 | SteeringSafety（ICML'26）等 | 资源是新的，但 steering 方向整体拥挤，需要找到混合架构特有的问题 |
| Agent 场景下按角色区分（用户/工具/观测/推理）的免训练压缩 | TriAxialKV、TaskKV、AgentKVShift（NeurIPS'26） | 顶会刚起步，但 arXiv 已经涌现 10 篇以上，风险高 |

## 6. 已排除（顶会上已被做过）

| 想法 | 已发表的顶会论文 |
|---|---|
| 视觉 token 剪枝的安全 / 越狱 | Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs（ICML'26） |
| 视觉 token 压缩的对抗鲁棒性 | On the Adversarial Robustness of LVLMs under Visual Token Compression（ICML'26） |
| KV 量化破坏安全对齐 | Alignment Collapse Under KV Cache Quantization（NeurIPS'26） |
| KV 压缩 × 越狱、多指令遵循 | When Efficiency Meets Safety；The Pitfalls of KV Cache Compression（ACL'26） |
| 剪枝让推理型 MLLM 想更久 / 解码时相关视觉信息漂移 | Why and When Visual Token Pruning Fails（ECCV'26，KAIST/NVIDIA） |
| 剪枝评测基准的噪声 | Are We Using the Right Benchmark（ACL'26） |
| RL 为什么比 SFT 遗忘少 | Retaining by Doing（ICML'26 Princeton） |
| Mamba 类 MLLM 的视觉 token 剪枝 | DTP（ICLR'26） |
| 混合模型的位置无关缓存复用 | HyPIC（NeurIPS'26 Lenovo） |
| 混合 SSM 的树形投机解码 | STree（NeurIPS'25） |
| TTS 多样性坍塌 | Mode-conditioning（ICLR'26 CMU+Anthropic） |
| 扩散 LLM 免训练加速 | Fast-dLLM（ICLR'26）及 ICML'26 十余篇 |

另外，之前会话里的 C1/C2（剪枝稳定性）和 ReThink-Reuse 已按同样标准核验并降级，见 `../lit/10_c1c2_verification.md`、`../lit/11_rethink_reuse_verification.md`。
