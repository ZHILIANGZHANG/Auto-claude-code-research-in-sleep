# "带符号视觉 token 重要性"：查新与可行性深度调研

> 原设想（来自 DEEP_READING 的"缝 B"）：attention 质量是"符号盲"的。CRG（ICML'26）附录显示约 50% 的高 attention 视觉 route 在压低模型当前的答案。如果改用**带符号的视觉 token 贡献**来打分，就能同时得到：（a）更好的剪枝（效率）；（b）剪掉"反证 / 有害"token，从而提升准确率或减少幻觉。
>
> 本文做三件事：
> 1. 在顶会语料中查新：在约 45k 篇的基础上，补了 CVPR'25、ICLR'25、ICML'25、NeurIPS'24、ACL'25，共约 6 万篇，并用 arXiv 排查撞车。
> 2. 精读最接近的 10 篇原文，逐一核对"信号有没有符号、相对于什么目标、要不要标签、开销多大"。
> 3. 在 CPU 上跑了一个小规模 pilot 实验（Qwen2-VL-2B，POPE-adversarial），直接检验这个方向的核心假设。

---

## 0. 结论先行

（见文末第 6 节；实验数据在第 5 节）

---

## 1. 查新范围与方法

- **顶会语料**：
  - ICLR'26、ICML'26、NeurIPS'25/'26、CVPR'26、ECCV'26、ICCV'25、ACL'26、EMNLP'25（原有约 45k 篇）；
  - 新补 CVPR'25、ICLR'25、ICML'25、NeurIPS'24、ACL'25（约 1.7 万篇）。
- **筛选**：先按标题筛出视觉 token 压缩 / 剪枝 / 合并（288 篇）和多模态幻觉（172 篇），再补 94 篇。
- **摘要检索**：抓取全部摘要，用 gradient / attribution / causal / counterfactual / harmful / distract / "beyond attention" / value norm / "surpass unpruned" 等关键词检索。
- **arXiv**：只用于排查撞车，不作为证据。共做了 20 余组查询，例如 "visual token pruning gradient / attribution"、"harmful visual tokens"、"token pruning mitigates hallucination"、"query-agnostic / multi-turn pruning"。
- **精读原文 PDF**：GACD、EAZY、ZOO-Prune、HARM（Same Attention, Different Truths）、DIPrune、DCP、Dual-Pathway Circuits、LeaF（Learning to Focus）、Visual Token Compression Enhances Robustness；另外复用此前读过的 CRG、PAS、When Pruning is Worse than Random。

---

## 2. 最接近的工作：逐篇核对（重点：有没有符号？相对什么目标？）

### 2.1 效率侧（视觉 token 剪枝）

| 工作 | 会议 | 打分信号 | 有无符号 | 参照目标 | 备注 |
|---|---|---|---|---|---|
| **ZOO-Prune** | CVPR'26 | 在 projector 上加随机扰动，用零阶估计的梯度范数 | **无符号**（范数） | projector 输出本身（与问题无关） | 原文明确说："用梯度算敏感度既昂贵，又**需要 ground-truth 输出来定义 loss，而推理时没有**"，所以退而求其次只看 projector。在 POPE 和 MMMU 上，projector 与 encoder 的排序 Spearman 只有 0.49–0.55 |
| **DIPrune** | arXiv 2610（未见录用） | 对"最终 task loss 失真"做 Taylor 上界，得到逐 token 的梯度能量 | **无符号**：原文用 Cauchy–Schwarz 把交互项界住，"**removing any dependence on their relative orientation**" | 推理时用层间近似 | 作者主动丢掉了方向 |
| **DCP** | EMNLP'25 | Grad-CAM：在 CLIP 视觉编码器中算 ReLU(A⊙∇A) | **只保留正部分**（ReLU） | **CLIP 的图文相似度**，不是 LLM 的决策 | 用到了符号，但目标是外部的 CLIP |
| AD-TP | NeurIPS'25 | attribution 驱动的自适应剪枝 | — | — | 面向通用 Transformer，不是 MLLM |
| CDPruner / MoB / D2Pruner / AgilePruner / SVD-Prune | NeurIPS'25 / NeurIPS'25 / arXiv / ICLR'26 / arXiv | 条件 DPP、覆盖、去位置偏置的 attention、erank、SVD leverage | 无 | — | 都在修补 attention 的偏置或补多样性，不涉及符号 |
| STAR-Pro、SparseVILA（MIT Han Lab） | arXiv | 重要 token 随层深变化；prefill 剪枝与 decode 检索解耦 | 无 | — | 已经指出"一次性剪枝不可靠"和"多轮对话中查询未知" |
| FlashVLM、OmniSIFT | ECCV'26、ICML'26 | 文本引导选择；训练式压缩 | 无 | — | 声称超过未剪枝模型（"beyond-lossless"） |
| When Pruning is Worse than Random | CVPR'26 | 去掉 token 后 GT 首 token 概率的变化 | 有符号 | **GT**（oracle，有标签泄漏） | 作为分析工具使用，不是方法 |

**小结**：效率侧确实**没有**"相对 LLM 决策、带符号、免标签"的剪枝方法。但原因不是没人想到。ZOO-Prune 和 DIPrune 都**明确绕开或丢弃了符号**，ZOO-Prune 还把"推理时没有目标"写成了动机。这说明社区已经意识到这个障碍，这一点非常关键（见第 4 节）。

### 2.2 幻觉侧（token 级去除 / 抑制视觉证据）

| 工作 | 会议 | 做法 | 有无符号 / 参照目标 |
|---|---|---|---|
| **EAZY** | ICCV'25（Amazon / Rutgers） | 约 1.5% 高 attention 的"幻觉 image token"驱动了 object 幻觉，把它们**置零**即可消除。先生成，再对每个 object 词取 top-K attention 的 token 置零，再重新生成 | 无符号（attention）；需要先生成出 object 词，属于**事后**处理 |
| **HARM**（Same Attention, Different Truths） | CVPR'26 **Highlight** | 真实 object 与幻觉 object 得到的视觉 attention 一样多。对高 attention 区域用 logit-lens 解码，看语义是否与输出一致，**不一致就把区域 mask 掉** | 用"语义是否一致"代替符号；参照的是**模型自己生成的词** |
| **GACD** | CVPR'26 | 一阶 Taylor 梯度估计每个视觉 / 文本 token 对**当前输出 token** 的影响，抑制与输出 object 虚假相关的视觉特征，并重新平衡视觉与文本 | 原文 Eq. 6：影响 = **梯度的 L1 范数**，**无符号**；每个解码步都要算梯度 |
| **CRG** | ICML'26 | head 内拆出 visual route 和 text route，计算 do-effect，只压冲突 head 的 text route | **有符号**，但在 head/route 层面，不在 token 层面；参照**当前解码 token** |
| CausalLens | CVPR'26 | head 拆成视觉、文本、系统三条通路，按敏感度调中层 hidden state | head 层面 |
| Dual-Pathway Circuits | NeurIPS'26（UIUC / Meta） | activation patching 找出"grounding 通路"与"幻觉通路"；**grounding 组件在正确样本上支持 GT，在错误样本上转为支持幻觉答案（polarity flip）** | **组件层面的符号已被研究** |
| One Token, Two Fates | CVPR'26 | 去掉部分视觉 token 构造潜空间负样本做校正 | — |
| HaloProbe | ICML'26 | attention 型幻觉检测被**token 位置和 object 重复**两个混淆因素主导（Simpson 悖论） | 证实了此前对 PAS 的位置质疑：**那个切入点已被占** |
| PruneHal、Visual Token Compression Enhances Robustness | arXiv、MM'26 | 用剪枝来缓解幻觉 / 提升鲁棒性（剪掉 OOD 视觉 token） | "剪枝即去噪"的叙事**已经有人提出** |
| GrAInS | ACL'26（UNC Bansal 组） | 用 Integrated Gradients 找出**正向和负向**贡献最大的 token，构造 steering 向量（LLM 与 VLM 都适用） | **有符号的 token attribution**；参照"偏好输出 vs 非偏好输出"（**需要成对标签**） |
| LeaF（Learning to Focus） | NeurIPS'25 | 纯文本场景：去掉"干扰 token"后，数学准确率 +20% | 干扰 token 由**更强的教师模型和训练集标准答案**的梯度对比找出，然后蒸馏 |

**小结**：幻觉侧的 token 级去除 / 抑制已经**相当拥挤**，至少有 EAZY、HARM、GACD、PruneHal、MM'26 等五篇以上。组件级的"符号翻转"也已被 Dual-Pathway 发表。**所有能"去掉 token 后变准"的工作，都依赖一个外部参照**：GT 标签（When Pruning is Worse than Random）、教师模型（LeaF）、成对偏好（GrAInS）、logit-lens 一致性检查（HARM），或者需要先生成再回溯（EAZY）。

---

## 3. 新意空间评估

- 字面上的空位是"**免标签、相对 LLM 决策、带符号**的视觉 token 剪枝"。效率侧没有人做；幻觉侧做的是无符号或组件级的版本。
- 但这个空位之所以空着，是因为存在下一节所说的**根本性障碍**。审稿人很可能会问："符号是相对谁的？"

---

## 4. 根本问题：符号相对于什么？（参照目标问题）

推理时没有 GT。可选的参照只有三类：

**(R1) 模型自己的预测 ŷ**（CRG、GACD 采用此类）。
- 设 l 为 ŷ 的 margin，s_i 为 token i 对 l 的一阶贡献。去掉 s_i<0（"反证"）的 token，一阶近似下**只会让 l 朝 ŷ 的方向增大**。
- 对单决策任务（POPE、选择题），这在一阶意义下**永远不会把错误答案翻对**，只会让模型对已有答案更自信。
- 这就是**确认偏差**：反证恰好是模型在出错时唯一能纠错的信息。
- 这个推理只依赖一阶近似，可以直接在实验里检验（第 5 节）。

**(R2) GT / 教师 / 偏好对**：有效，但要标签，属于 oracle 或训练式方法（LeaF、GrAInS、When Pruning is Worse than Random）。

**(R3) 与任务无关的参照**（projector 输出、CLIP 相似度、OOD 距离）：这就是 ZOO-Prune、DCP、MM'26 的做法，信号退化成"敏感度大小"或"相似度"，不再有决策意义上的符号。

此外：
- **符号随目标变化**：同一张图的不同问题，以及同一段生成中的不同步，token 的符号可能都会变。prefill 时一次性剪枝的方法本质上无法利用一个随目标变化的符号（第 5 节实验 [4] 和生成实验）。
- **剪枝理论的先例**：权重剪枝领域（OBD/OBS、Molchanov 等人的 Taylor 剪枝）早就知道，一阶有符号项求和会相互抵消、不可靠，所以都用平方或绝对值。DIPrune 用 Cauchy–Schwarz 去掉方向，是同一个思路。
