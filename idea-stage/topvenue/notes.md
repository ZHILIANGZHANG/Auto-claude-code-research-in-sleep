# 顶会调研工作笔记（自己检索，不用子 agent，不用 arXiv 无录用论文）

数据源（2026-10-18 下载）：
- ICLR 2026 (5691), NeurIPS 2025 (5982, 含摘要), ICML 2026 (6735), NeurIPS 2026 (9139) —— 各会议官网 virtual site 的 orals-posters JSON（含作者机构）
- CVPR 2026 (4042, CVF Open Access), ICCV 2025 (2765), ECCV 2026 (2862) —— CVF / ECVA virtual JSON
- ACL 2026 main 2752 + Findings 2164, EMNLP 2025 main 2217 + Findings 1406 —— ACL Anthology
- 摘要：NeurIPS 2025 自带；其余从会议 virtual 页面 / Anthology 页面抓取
- OpenReview API 需要人机验证，不可用

## 标题关键词计数（主会，含 Findings 另列）
见 buckets.py 输出：KV 194、投机解码 208、量化/剪枝 536、稀疏注意力/长上下文 378、MoE 338、视觉 token 244、TTS 249、高效推理 171、持续学习 627、合并 128、编辑/遗忘 346、TTA 181、steering/SAE 363、幻觉 552、不确定性 1067、training-free 475、dLLM 238

## 主题 1：Agent 场景 × 推理效率（NeurIPS'26 刚起步）
- TriAxialKV (NeurIPS'26, UIUC/Cambridge/Imperial)：agent 上下文按 时间/模态/语义角色(用户/工具调用/观测/推理) 三轴混合精度 KV 量化；Qwen3-VL-32B-Thinking 做 computer-use
- SKIM (NeurIPS'26, KAIST/Meta)：structured generation 下 agent 剪枝+蒸馏，决策位置最关键
- TaskKV (NeurIPS'26)：agent 步级 KV 保留，AgentBench/ALFWorld，20% 预算保 87–89%
- AgentKVShift (NeurIPS'26, UCSD)：免训练，agent 记忆单元 KV 复用的残差 = 共享偏移 + 小波动
- ASQ (NeurIPS'26, LG AI)：语义感知 PTQ
- ACON (ICML'26, KAIST/Microsoft)：长程 agent 上下文压缩
- Pitfalls of KV Cache Compression (ACL'26)：多指令 / system prompt 泄露
- GUI/computer-use 的视觉 token 效率：顶会只有 FocusUI (CVPR'26, 单步 UI grounding)；投机解码 × agent：Speculative Actions (ICLR'26), SpecEyes (ECCV'26)

## 主题 2：效率方法的副作用（安全/对齐/指令遵循）
- Alignment Collapse Under KV Cache Quantization (NeurIPS'26, Microsoft MAI/Caltech)：11 个模型，低比特 KV 量化静默破坏拒答；安全特征在低维子空间，对量化噪声敏感 100–1000 倍
- When Efficiency Meets Safety (ACL'26)：KV 压缩 × 越狱，双刃剑
- The Pitfalls of KV Cache Compression (ACL'26)
- On the Adversarial Robustness of LVLMs under Visual Token Compression (ICML'26)
- When Reasoning Meets Compression (ICLR'26, PSU/Columbia)
- DefensiveKV (ICLR'26)：驱逐的"稳定性假设"脆弱
- Less Diverse, Less Safe (ICML'26)：TTS 候选多样性下降 → 不安全输出

## 主题 3：主流开源模型已转向混合架构，而免训练效率方法仍停留在纯注意力模型（一手证据）
HuggingFace 官方 config（2026-10-18 抓取）：
- Qwen3.5 全系（0.8B/2B/4B/9B/27B/35B-A3B/122B/397B）、Qwen3.6-27B、Qwen3.8-27B：`layer_types` = 3 层 linear_attention(Gated DeltaNet) : 1 层 full_attention；`attn_output_gate: True`（门控注意力）；`partial_rotary_factor: 0.25`；原生多模态（image-text-to-text）。Qwen3.5-9B 下载量 826 万。
- Qwen3.8-Flash-Next：同样 3:1 混合 + 512 专家 MoE
- Kimi K3：KDA 线性注意力 + 每 4 层一层 MLA
- GLM-5.3-Flash：KDA 线性注意力 34 层 + DeepSeek Sparse Attention 11 层
- DeepSeek V4.1-Flash：DSA（index_topk 512）+ 滑动窗口 + MLA
- Qwen 官方 2026-04 发布了 Qwen3.5 全系 SAE（SAE-Res-Qwen3.5-*）
顶会证据：
- Gated Attention for LLMs (NeurIPS'25 Oral, Qwen Team)：门控注意力 → attention-sink-free（Qwen3.5 的设计来源）
- 顶会里针对混合架构的免训练效率工作极少：DTP（ICLR'26，Mamba-MLLM 视觉 token 剪枝）、HyPIC（NeurIPS'26 Lenovo，混合模型位置无关缓存复用）、STree（NeurIPS'25，混合 SSM 树形投机解码）、NVIDIA 混合模型剪枝（NeurIPS'25，需训练）、Comparing Transformers and Hybrid Models at the Token Level（NeurIPS'26 Ai2：注意力层负责复制/层级结构，混合优势在重复 n-gram 上消失）
- Kinetics (NeurIPS'25 CMU)：TTS 成本由注意力（KV 访存）主导 → 小模型被高估；该结论基于纯注意力模型
- FEATHER (ICCV'25 Stanford)：FastV 式浅层注意力剪枝偏向图像底部 token（RoPE 衰减导致）
- When Token Pruning is Worse than Random (CVPR'26)：深层（>20 层）剪枝不如随机，"信息视界"

## 主题 4：TTS（顶会主线）
- 多样性坍塌是并行采样瓶颈：Mode-conditioning (ICLR'26 CMU+Anthropic)、Diversity Matters VLM TTC (ICML'26 UZH/ETH)、Less Diverse Less Safe (ICML'26)
- 选择/聚合：Self-Certainty BoN (NeurIPS'25 HKU/Berkeley)、Beyond Majority Voting 高阶信息 (ICML'26 MIT/Harvard)、Optimal Self-Consistency (ICML'26 Yale/Huawei)、Position: Optimize for Self-Consistency (ICML'26 MIT+Anthropic)、Same Pool Different Answer（ECCV'26，似然式 BoN 不稳定）
- 反向缩放：Inverse Scaling in Test-Time Compute (ICLR'26)；Does Thinking More Always Help? (NeurIPS'25)
- 成本：Kinetics (NeurIPS'25)、On the Overscaling Curse of Parallel Thinking (NeurIPS'25 SJTU)

## 主题 5：持续学习 / 后训练遗忘（顶会主线）
- RL 遗忘少于 SFT：Retaining by Doing (ICML'26 Princeton，on-policy 数据是原因)；RFT Naturally Mitigates Forgetting (ICML'26，Qwen2.5-VL-7B)
- 度量：Mapping Post-Training Forgetting at Scale (ICLR'26 Tübingen+Google，样本级 1→0 计数)；CapTrack (NeurIPS'26，能力中心，含 80B)
- 免训练 agent 持续学习：JitRL (ICML'26 NUS，WebArena)；Darwinian Memory (ICML'26 DiDi)
- 记忆充足时的 CL：Forget Forgetting (ICLR'26 NYU)
- VLA 不太遗忘：ICML'26 UT Austin

## 主题 6：GUI/computer-use agent 的历史冗余
- 已被顶会认识但都是训练方法：SimpAgent (ICCV'25，历史高度冗余)、HiconAgent (CVPR'26)、GUI-Rise (NeurIPS'25)、iSHIFT (CVPR'26)
- 效率攻击：SlowBA (ECCV'26)

## 主题 7：扩散 LLM 免训练加速 —— 顶会极度拥挤（Fast-dLLM、dLLM-Cache、SparseD、ES-dLLM、Dynamic-dLLM、DC-Leap……），不建议
