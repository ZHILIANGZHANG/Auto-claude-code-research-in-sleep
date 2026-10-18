# 待读论文清单

本清单汇总了这段时间“越推理越脱离证据”、感知视界、implicit caption、role/turn 开关这几条讨论里提到、但还没读完的论文。下载链接单独放在 `to_read_links.md`；读过的和正在读的论文、读到哪一节，统一记在 `reading_status.md`。

**会议标注规则**：只写已核实的来源，包括 OpenReview decision、会议页面、arXiv comment 或 journal_ref。查不到的写“未核实”。只有 preprint 的单独标出。

**状态**：⏳ 进行中 · ⬜ 未读 · ✅ 已读（笔记在 NOTES.md）

共 45 篇：进行中 2 篇，未读 43 篇。

---

## 0. 正在读

| # | 论文 | 会议 / 机构 | 状态 | 备注 |
|---|---|---|---|---|
| 1 | On Test-Time Scaling for Vision-Language Models（2606.28864） | ECCV'26（arXiv comment）；VUB / imec | ⏳ | 读到哪、带着的问题见 `reading_status.md` |
| 2 | Look Light, Think Heavy（2606.22565；NOTES #94） | ACL'26；中科院自动化所 | ⏳ | 读到哪见 `reading_status.md` |

## A. 多模态：推理越长，越不看图

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 3 | Mitigating Visual Forgetting via Take-along Visual Conditioning (TVC)（2503.13360） | ACL'25 | ⬜ | 推理到一半把图拿掉，MathVista-hard 只掉约 2%，说明后半段推理基本不用图。修补方法是周期性重新注入图像 |
| 4 | More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models（2505.21523） | NeurIPS'25；UCSC / Stanford 等 | ⬜ | 提出 RH-AUC 和 RH-Bench；推理越长，对视觉 token 的 attention 越低、对指令 token 的越高 |
| 5 | More Thought, Less Accuracy? On the Dual Nature of Reasoning in VLMs（2509.25848） | ICLR'26；ANU / Melbourne | ⬜ | 称之为 visual forgetting，提出训练方法 VAPO 把推理锚定在视觉上 |
| 6 | Deeper Thought, Weaker Aim（2603.14184） | CVPR'26（依据 poster 页） | ⬜ | 解释为“注意力发散”；VRGA 是 training-free 方法，挑出视觉 head 重加权 |
| 7 | Reflection Anchors for Propagation-Aware Visual Retention（2605.09614） | preprint（Under Review） | ⬜ | 同一现象的修补方法，查新时找到 |
| 8 | Cognitive Pivot Points and Visual Anchoring（2604.10219） | preprint（TPAMI under review） | ⬜ | 同上 |

## B. 文本 / agent：推理越长，越不用外部信息

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 9 | Inverse Scaling in Test-Time Compute（2507.14417） | TMLR'25（Featured）；Anthropic Fellows | ⬜（已选为第 7 篇精读） | 推理越长越差：被干扰项带偏、抓住虚假特征、丢掉约束 |
| 10 | AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions（2506.09038） | NeurIPS'25；Meta FAIR / NYU | ⬜ | 推理微调让弃权下降 24%；推理预算越大越不弃权；问题出在 RLVR 阶段 |
| 11 | The Danger of Overthinking: Reasoning-Action Dilemma in Agentic Tasks（2502.08235） | 未核实；Berkeley | ⬜ | 推理模型偏好内部模拟而不是环境反馈，分析了 4018 条 SWE-bench 轨迹 |
| 12 | The Hallucination Tax of Reinforcement Finetuning（2505.13988） | Findings EMNLP'25；USC | ⬜ | RFT 让不可解题的拒答率下降 80% 以上，混入约 10% 不可解题数据就能恢复 |
| 13 | Mind Your Step (by Step)（2410.21333） | ICML'25；Griffiths 组 | ⬜ | 外围参考：在“人类越想越差”的任务上，CoT 同样伤害模型 |
| 14 | When to Trust Tools? Adaptive Tool Trust Calibration（2604.08281） | 未核实 | ⬜ | 工具结果被推理覆盖（tool ignored） |
| 15 | Judged Useless, Queried Anyway（2610.06191） | 未核实 | ⬜ | agent 自己判断证据没用，却仍然继续查，判断不转化为行动 |
| 16 | Agents Explore but Agents Ignore: LLMs Lack Environmental Curiosity（2604.17609） | 未核实 | ⬜ | agent 看到了环境信息但不用 |

## C. 机制与反面视角

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 17 | Characterizing and Mitigating Reasoning Drift in LLMs | ICLR'26；中科院自动化所 | ⬜ | 这里的 drift 指锁死在错误轨迹上（前期可塑、后期锁定），用 RAAS 激活 steering 修正。和“脱离证据”不同，但方法上可借鉴 |
| 18 | Position: Reasoning After Perception Means Reasoning Without Vision（2507.16863） | ICML'26 position；国科大 / 中科院计算所 | ⬜ | 和你的“视觉 CoT 只是拿到信息后的逻辑推理”观点最接近 |

## D. 感知视界 / 视觉信息流 / implicit caption

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 19 | What Keeps Vision-Language Models Looking at the Image?（2607.12815） | 未核实；东大松尾研 | ⬜ | 因果分析：属性一旦在文本里被说出来，对图像的依赖就下降。和感知视界撞车最直接 |
| 20 | Pathways of Visual Information Flow in VLMs（2607.03358） | NeurIPS'26；Copenhagen | ⬜ | 区分 direct 和 text-mediated 两条视觉信息通路，对应 implicit caption 的“读出”命题 |
| 21 | Attention-guided Fine-tuning of MLLMs Improves CoT Reasoning（2606.01558） | 未核实 | ⬜ | 也测了推理过程中对图像 attention 的衰减 |
| 22 | From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of VLMs（2605.20177） | ICML'26；UCSC VLAA | ⬜ | 先感知后推理的解耦训练，和“视觉 CoT ≈ 先抽信息再做逻辑”有关 |
| 23 | What's in the Image? A Deep-Dive into the Vision of VLMs（2411.17491） | CVPR'25 | ⬜ | 视觉信息在哪些层、哪些 token 被读取 |
| 24 | Towards Interpreting Visual Information Processing in VLMs（2410.07149） | ICLR'25 | ⬜ | 视觉 token 表征的可解释性分析，对应命题 A（表征里其实有信息） |
| 25 | Why are Visually-Grounded Language Models Bad at Image Classification?（2405.18415） | NeurIPS'24；Stanford | ⬜ | 信息在表征里但读不出来，原因是数据而不是架构，对应命题 B |
| 26 | Eyes Wide Shut? Exploring the Visual Shortcomings of MLLMs（2401.06209） | CVPR'24；NYU | ⬜ | CLIP-blind pairs，视觉编码器本身的上限 |
| 27 | Rethinking Causal Mask Attention for Vision-Language Inference（2505.18605） | 未核实 | ⬜ | 视觉 token 的 causal mask 设计 |
| 28 | Chain-of-Visual-Thought (CoVT)（2511.19418） | 未核实 | ⬜ | 用连续视觉 token 推理，是“非 caption 式推理”的一种尝试 |
| 29 | v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning（2505.18842） | COLM'26（journal_ref） | ⬜ | 推理中主动回指视觉 token，对应命题 C（是否只读一次） |
| 30 | Can Visual Generation Expand What Unified Multimodal Models Know?（2610.21459） | 未核实 | ⬜ | 和“统一模型 vs VLM”的检验相关（Janus、BAGEL 的理解通路本身是解耦的） |

## E. 解码阶段的视觉 KV 淘汰

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 31 | Make Each Token Count: Long-Context KV Cache Eviction（2605.09649） | NeurIPS'26；Yale / CUHK | ⬜ | 可学习的 KV 淘汰方法 |
| 32 | RetentiveKV: Uncertainty-Aware Multimodal KV Cache Eviction（2605.04075） | 未核实 | ⬜ | 提出“延迟重要性”：视觉 token 可能晚些时候才重新被用到 |
| 33 | MadaKV: Adaptive Modality-Perception KV Cache Eviction（2506.15724） | ACL'25 | ⬜ | 按模态区分的 KV 淘汰 |

## F. benchmark 对视觉的依赖程度

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 34 | The Role of Visual Modality in Multimodal Mathematical Reasoning（2503.04167） | 未核实 | ⬜ | 数学类多模态题到底需不需要图 |
| 35 | VCBench: Multimodal Math Reasoning with Explicit Visual Dependency（2504.18589） | NeurIPS'26；阿里达摩院 | ⬜ | 同上，构造了显式依赖视觉的题目 |

## G. role / 多轮 / steering（“开关”这一侧）

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 36 | Hesitation Has a Geometry: Hyperbolic Probes for Sparse Activation Steering (HEST)（2610.02391） | 未核实 | ⬜ | 只在分叉 token 上做 steering，查新时发现已被做 |
| 37 | On the Loss of Context-awareness in General Instruction Fine-tuning（2411.02688） | 未核实 | ⬜ | 指令微调让模型更不读 user 上下文，和 role accessibility 有关 |
| 38 | User-Assistant Bias in LLMs（2508.15815） | 未核实 | ⬜ | 模型偏向信 user 还是信 assistant |
| 39 | LLMs Get Lost In Multi-Turn Conversation（2505.06120） | 未核实；MSR / Salesforce | ⬜ | 多轮对话中性能下降，和 #95 BOULDER 相关 |
| 40 | What if LLMs Ate Their Words: Causal History Effects in Multi-Turn Interaction（2609.05882） | 未核实 | ⬜ | 对话历史对后续行为的因果影响 |
| 41 | Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence（2604.08169） | 未核实 | ⬜ | 做 steering 时如何不破坏连贯性 |

## H. 原始候选池里剩下没读的

| # | 论文 | 会议 / 机构 | 状态 | 为什么在清单上 |
|---|---|---|---|---|
| 42 | Why Do Some Language Models Fake Alignment While Others Don't?（2506.18032） | NeurIPS'25 | ⬜ | 在 25 个模型上做 alignment faking，只有少数模型会伪装 |
| 43 | Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers（2512.15674） | ICML'26；Owain Evans / Sam Marks 等 | ⬜ | 训练 LLM 读另一个模型的激活并用自然语言解释 |
| 44 | Large language models suffer from a curse of ambiguity（2608.15448） | NeurIPS'26 spotlight；Stanford Linderman 组 | ⬜ | 理论方向 |
| 45 | Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality（2506.06522） | NeurIPS'25 | ⬜ | 后训练数据质量的对比研究 |

---

## 已读完的论文

见 `reading_status.md`。一篇读完后，把它从本清单移到那里。
