# 顶会论文逐篇精读：细节、作者自述局限与可切入点

读法：arXiv 全文 PDF（录用版本或 camera-ready 前版本），逐段读正文 + 附录；只记录（1）论文里容易被忽略的细节发现，（2）作者自己承认的局限 / future work，（3）由此看出的切入点。切入点是我的判断，标注为【切入点】。

---

## 1. LessIsMore（ICML'26；Princeton, CMU, MSR）arXiv 2508.07101

**核心**：推理模型解码时，各 attention head 的 top-k 重要 token 高度重合（同一 KV 组内和跨所有 head 都重合），且“最近 token 在关键 token 中所占比例”在整个推理过程中近似恒定。于是在第 2 层或第 12 层做一次跨 head 统一选择（并集后全局排序），25% 预算固定给最近窗口，选出的 token 集合供后续所有稀疏层复用。

**细节里值得注意的**：
- 稀疏注意力在推理任务上会导致**推理链变长**：Qwen3-8B、AIME-24、2K 预算下，全量注意力平均 14.8K token，Quest 30.0K（+103%），SeerAttention-r 19.8K（+34%），TidalDecode 17.4K（+18%）。作者解释为“选择错误累积，迫使模型进入冗余推理”，但没有任何分析证明。
- **稀疏后准确率反而高于全量注意力**：Qwen3-8B AIME-24 在 4K/6K/8K 预算下为 75.6/76.5/76.7，全量是 74.5；Qwen3-14B 也类似。论文标题就叫 “Less Is More”，但正文完全没解释为什么“看得少反而更准”。
- 在 NIAH 检索任务上同样预算召回率很高，但在 AIME 上召回率随生成长度一路下降（TidalDecode 约 75%，StreamingLLM 低于 65%）：**检索任务和推理任务对稀疏注意力的需求本质不同**。
- 选择层的位置很关键，作者用 TidalDecode 的 NIAH 搜索来定（Qwen3-8B 用第 12 层）：完全是经验性的。
- 只做 r=0.25 的消融：0% 最近窗口或 100% 最近窗口都答错。

**作者没做 / 局限**：只在 GQA/MHA 的纯注意力模型上做（Qwen3、R1-Distill-Llama）；没有混合架构、MLA；只有数学和 GPQA，没有代码、agent、多轮；没有解释跨 head 高度重合的原因，也没解释为什么推理任务和检索任务不同。

【切入点】
1. **“稀疏注意力为什么能让推理更准”的机制研究**：是否因为屏蔽了旧推理段，减少了回看和反复验证（overthinking 的来源之一）？可测：稀疏前后“wait / double-check”类反思词频率、重复验证段的数量。若成立，可以把“选择性遗忘”当作免训练的抗过度思考手段。与 Make Each Token Count（NeurIPS'26，长上下文中驱逐反而提升性能）可以互相印证。
2. **把“推理长度膨胀”当作选择出错的在线信号**：一旦检测到反复、绕圈，就触发重新选择或回退到全量注意力。
3. **跨 head 一致性本身是否是难度或不确定性信号**：head 之间重合度低的时刻，是否对应推理卡住、出错。

---

## 2. BeaconKV（ICML'26；汉阳大学, 成均馆大学）arXiv 2609.04971

**核心**：推理过程中，少数解码步会产生“思维回访 token”（Thought Revisiting Tokens，TRT）：它们的注意力跳回很远的早期上下文，例如开头制定的解题计划、题目约束。这些“全局 query”在 pre-RoPE 的 query 空间里聚成少数几簇，而且与相邻的局部 query 相似度低。于是用在线的最远点采样（Continual FPS）保留 16 个几何上最分散的历史 query 作为“信标”，加上 16 个最近 query 一起给 KV 打分，打分时取 max 而不是 mean。

**细节里值得注意的**：
- 全局 query 不集中在某层某头，而是分散出现在很多层和头里（附录 B：top-150 注意力位置的平均距离超过 200 就算全局）。
- 只保留“开头的 query + 最近 query”不行（Initial+Recent 基线更差），说明回访目标在推理过程中是不断变化的。
- 取 mean 会稀释稀疏的回访信号（256 预算下 max 为 23.3，mean 为 18.8）。
- 信标给太多也不行：(nrecent=1, nbeacon=31) 时准确率更低，延迟还翻倍。
- 回访的内容是“解题计划和题目约束”：从附录 E 的逐 token 可视化看，回访发生在推理阶段切换处。

**作者自述局限（附录 C）**：只测了推理任务，没测检索、摘要和一般长上下文任务；超参数（信标数、最近 query 数、预算）是固定的，自适应调整留作 future work。

**与 LessIsMore 的张力**：LessIsMore 认为“重要 token 是全局稳定的”；BeaconKV 认为“回访是零星且不可预测的，最近 query 预测不了”。两篇都是 ICML'26，而且都在 Qwen3 / R1-Distill 上做。两者可能都对：总体重要集合稳定，但少量回访事件决定成败。这个矛盾没人解释。

【切入点】
1. **回访事件本身是一个行为信号**：TRT 出现在推理阶段切换处（回看计划或约束）。可以检验 TRT 的频率是否对应“重新规划”或“反复验证”，并据此免训练地检测 overthinking 或卡住的推理，与 ConPress 发现的“答题后的验证被大量压缩”相呼应。
2. **把信标思路搬到推理型 MLLM 的视觉 token 上**：推理过程中模型会周期性“回看图像”（ECCV'26 的 DSTP 发现解码时相关视觉信息会漂移）。用信标 query 预测哪些视觉 token 将被回看，而不是一开始就剪掉。DSTP 是最接近的工作，需要区分。
3. **调和 LessIsMore 与 BeaconKV**：做一个统一分析，量化“稳定全局集合”和“零星回访”各占多少注意力质量、各自对准确率的贡献。

---

## 3. Make Each Token Count / DBTrimKV（NeurIPS'26；Yale, CUHK）arXiv 2605.09649

**核心**：长上下文里，无关 token 会在 softmax 分母中稀释注意力（attention dilution），所以选对了的驱逐不只是近似全量注意力，还能提升效果。方法是给每个 token 学一个保留分数 β（轻量门控，只占 0.43% 参数，冻结主模型训练），各层各头共享最后一层投影使分数可比，然后用一个全局预算统一排序驱逐，不再给每层每头分固定预算。

**细节里值得注意的**：
- 驱逐后**超过全量 KV**：文本推理上最多高 3.75%（AIME24 等长生成任务收益最大，GSM8K 这类短推理几乎没差别）；**MMDU 多轮视觉对话上比全量高 14%（M=512）**。
- 理论：只要“近乎平手”的干扰 token 数量足够多，注意力稀释就不可避免（命题 3.1）；只要有用 token 的保留率高于干扰 token，稀释就会下降（推论 3.2）。
- 在 Qwen3-VL-4B 上测 token 的“存活”：某个头里只有 13.8% 的 token 在 4096 步后仍被 top-256 选中，16K 后只剩 3.2%；视觉 token 衰减速度和文本差不多。
- 可视化发现：图像细节 patch 大量被驱逐，但 **<|vision_start|> 和 <|vision_end|> 这两个边界 token 在很多头里一直被保留**，作者推测它们是“压缩后的结构锚点”。视觉 token 在开头保留得多，推理转向文本后迅速减少；长期记忆集中在中间层的少数头里。
- 用 token 的上下文嵌入 x_t 做门控输入，比用 K、V 更好。

**作者的展望**：自适应的保留目标、与检索结合的缓存策略、把驱逐机制放进训练里。

【切入点】
1. **“注意力稀释导致长推理和多轮对话退化”的免训练诊断**：DBTrimKV 是学出来的，LessIsMore 是免训练的，两者都观察到“看得少反而更好”。可以用 oracle 驱逐（事后知道哪些 token 有用）给出上界，量化长推理错误中有多大比例来自稀释。多轮对话上 +14% 这个数尤其大，可能对应“多轮对话中迷失”问题的一部分成因。
2. **视觉边界 token 作为锚点**：只保留 vision_start/vision_end 加少量 patch 时，模型还剩多少“看图”能力？这些边界 token 是否像 attention sink 一样承载了全局图像摘要？可做免训练的机制分析和极限压缩。
3. **视觉 token 随推理推进而失效的时间曲线**：推理型 MLLM 何时不再需要图像，可以据此在解码中途免训练地释放视觉 KV。

---

## 4. IVC-Prune（ICLR'26；武汉大学, 小红书）arXiv 2602.03060

**核心**：从 RoPE 的数学结构出发，当 key 的旋转矩阵接近单位阵或 90° 旋转时，注意力就把 query 的“绝对位置”分量分离出来。作者据此提出，某些位置的视觉 token 充当了隐式视觉坐标（IVC token）：取“各维 cos 之和最大”和“各维 sin 之和最大”的位置，各占约 10%。剪枝时保留 IVC token 和前景 token；前景用 value 向量相似度来选，以避开注意力里的位置偏差。只在中间某一层选一次，保留原始 position ID，并同时裁掉前面各层的 KV。

**细节里值得注意的**：
- **只留前景 token 时，视觉定位崩到 58.0（RefCOCO testA，Qwen2.5-VL-7B），基线是 92.2**。再加 15% 随机 token 回到 79.3，加角点和中心 73.5，加滑动窗口 89.0，加对角线 89.8，加 IVC 10% 回到 92.8。也就是说，**任何“空间上铺开”的模式都能恢复大部分性能，IVC 只比对角线多 3 个点**。“RoPE 坐标”这个解释只覆盖了最后一小段差距。
- 作者声称：前人报告的“浅层剪枝很敏感”，其实是因为删掉了 IVC token，而不是剪枝本身的问题。
- SparseVLM 的聚类合并会重建 position ID，在 RefCOCO 上只剩基线的 15.3%；关掉聚类恢复到 76.1%。但通用 VQA 几乎不受影响，说明**语义保住了，空间结构丢了**。
- OCRBench 上所有剪枝方法都大幅掉点，集中在文字识别和手写公式识别这类“前景就是全部”的图像上。
- 中间层的注意力对 prompt 最敏感，浅层和最后几层对 prompt 依赖较弱。

**作者自述局限**：剪枝比例固定，不随任务复杂度调整；选择层靠在小验证集上试；IVC 依赖 RoPE，换成可学习 2D 位置编码就不成立。

【切入点】
1. **“坐标 token”还是“空间覆盖”？** 表 6 显示规则的空间覆盖就能拿回大部分性能。一个干净的机制研究：在相同 token 数下比较 IVC、均匀网格、蓝噪声采样等，再做注意力去除 / 激活修补，看模型是否真的把这些位置当坐标轴用。如果结论是“空间覆盖 + 保留原始 position ID”就够了，那就是对一篇 ICLR 论文核心解释的修正。
2. **新一代位置编码下 IVC 理论是否成立**：Qwen3-VL 和 Qwen3.5 用的是交错式 MRoPE，Qwen3.5 只对 25% 维度做旋转。IVC 的位置计算公式在这些模型上应该完全不同。论文只在 Qwen2.5-VL、InternVL2.5、DeepSeek-VL2、LLaVA 上做，没有新模型。
3. **位置 ID 是否被保留，是所有剪枝方法在定位任务上分化的主因**：SparseVLM 关掉聚类就从 15% 回到 76% 这个结果，可以推广成“剪枝 / 合并对空间任务的影响主要取决于 position ID 处理方式”的系统研究。

---

## 5. PriorTR（ECCV'26；浙江大学, 港理工）arXiv 2606.24156

**核心**：MLLM 在**没有任何指令**时也会把注意力集中在某些固定区域（模型先验 Q）。指令下的注意力 P 是“先验 × 语义提升”的混合，所以按 P 的大小剪枝会保留先验区域，丢掉真正与指令相关但先验低的 token。方法是用图像后面那个换行符 `\n` 的注意力当先验：因果掩码下它看不到指令，所以是“无指令探针”。打分用 PVI = P·log(P/Q)，一次前向就能同时拿到 P 和 Q，在第 2 层剪枝。

**细节里值得注意的**：
- 先验区域在各层都很一致（图 1），而且**和问题无关**。
- 预算越紧，先验修正的收益越大（LLaVA-1.5，K=64 时比第二名高 2.1%）。
- TextVQA 在极端压缩下是例外：比带空间多样性约束的 VisPruner 低（55.9 vs 57.7）。作者明确说，把空间正则化整合进打分框架是 future work。
- 只用 LLaVA-1.5 / Next、InternVL2.5、Qwen3-VL-8B；在 Qwen3-VL 上只和 FastV 比。

**论文没回答的**：先验到底是什么？是 FEATHER 发现的 RoPE 位置偏差（偏向图像底部）？是高范数的 sink / register token？还是视觉显著区域？三种解释对应完全不同的处理方式。

【切入点】
1. **拆解“模型先验”的来源**：用 position ID 打乱、遮掉高范数 token、换成纯色图等免训练实验，分离位置偏差、sink token、视觉显著性三种成分。CVPR'26 的 “What Do Visual Tokens Really Encode?” 已发现视觉 token 分为 sink、dead、alive 三类，可以直接对接。
2. **先验区域是不是“全局摘要”**：如果先验 token 是 sink 一类、承载全局信息，那么像 PriorTR 这样把它扣掉，对需要全局理解的任务可能有害。这是一个可检验的反向假设。
3. **先验与幻觉**：无指令时模型固定关注的区域，是否就是幻觉时模型依赖的“语言先验”在视觉侧的对应物？可以和 PAS（CVPR'26）、Causal Route Gating（ICML'26）这些幻觉机制论文连起来。

---

## 6. R2R（NeurIPS'25；清华, 无问芯穹）arXiv 2505.21600

**核心**：让 R1-1.5B 沿着 R1-32B 的推理路径走，逐 token 比较两者的下一 token 预测：只有 11% 不同，其中很多只是措辞差异（"let's" 与 "let us"），真正改变推理走向的“分歧 token”更少。用“各自续写到句末，再让 Qwen2.5-72B 判断语义是否分叉”的方式，自动标出 760 万个 token 标签，训练一个 56M 参数的 FFN 路由器，只把分歧 token 交给大模型。

**细节里值得注意的**：
- 分歧 token 的小模型熵是非分歧 token 的 3.8 倍；低频 token 更容易分歧。
- 只按“预测不同”就路由（不区分中性和分歧），召回率明显下降：**语义分歧和表面不一致是两回事**。
- 固定大模型、换更小的小模型，帕累托前沿反而更好。
- **R2R 的输出比纯小模型更短**，但正文没有分析原因。
- 小模型答对的 7 道 AIME 题，大模型也全答对：大模型严格优于小模型，不是各有所长。
- 路由器需要训练：生成标签用了 8 张 A800 跑 2.3 天。

**作者自述局限**：主要在贪心解码下调和测，其他采样策略只做了有限验证；需要系统层面的进一步优化。**Future work**：MoE 用不同大小的专家，按 R2R 式的“分歧难度”来监督路由；把 query 级路由和 token 级路由组合起来。

【切入点】
1. **免训练版 R2R**：消融里没有“只用小模型熵 + token 频率阈值”这一纯规则基线，可是熵差 3.8 倍已经说明信号很强。直接做免训练路由，看能保留多少收益。需要先查最近有无同类顶会工作。
2. **分歧 token 和 RL 里的“分叉 token”是同一批吗**：RL 文献里有“少数高熵 token 决定推理走向”的发现。如果 R2R 的分歧 token 和这些分叉 token、RPDI 里的转折词高度重合，就能把“推理关键 token”统一成一个可免训练检测的对象，同时用于路由、早停和采样分配。
3. **多模态版本**：视觉推理中，分歧是更多发生在感知 token（读图）上，还是推理 token 上？小 VLM 加大 VLM 的 token 级协作目前还没人做。

---

## 7. SpecReason（NeurIPS'25；Princeton, CMU）arXiv 2504.07891

**核心**：推理链里少数步骤（问题分析、拆解、规划）是关键，大多数步骤只是执行计算；而且一步推理的价值在语义，不在逐字措辞。于是让小模型按“步”（句子或逻辑步）先写，大模型只做一次 prefill 给出 0–9 的效用分，超过阈值就接受，否则大模型重写这一步。

**细节里值得注意的**：
- 比原始大模型快 1.4–3.0 倍，**准确率反而高 0.4–9.0%**。作者给出的解释是：**小模型更不啰嗦，在固定 token 预算下更短的推理链来得及答完**。AIME 上 4096 预算时差距是 16.2%，预算放大后差距缩小。
- 让大模型先写前 n 步来定规划，效果不如调接受阈值。
- 大模型作为评审只靠一个 prompt 打分，没有用 logprob。

**作者的 future work**：用 logprob 置信度或动态阈值替代静态打分；把小模型解码和大模型验证流水线化。

【切入点：横向看到的共同模式】
LessIsMore（稀疏后比全量更准）、DBTrimKV（驱逐后比全量更准）、R2R（混合生成比小模型短）、SpecReason（准确率提升来自更短）、ConPress（推理变短但准确率不掉）——**很多效率论文报告的“准确率反而提升”，可能都来自同一个混杂因素：推理变短了，而评测的 token 上限是固定的**。
1. **元分析 / 复现型研究**：把这些方法放到“不设上限”和“等长度”两种条件下重新评测，看准确率提升还剩多少。如果大部分消失，这是一篇对整个效率方向都有提醒意义的分析论文；如果仍在，就说明“少看 / 少想”确实有正则作用，可以进一步解释机制。
2. **“步骤难度不均匀”的免训练利用**：SpecReason 的关键假设是只有少数步骤难，这和 R2R 的“少数分歧 token”、BeaconKV 的“少数回访 token”是同一个现象在不同粒度上的体现，可以统一刻画。

---

## 8. RPDI-EE（NeurIPS'26；国科大, 中科院信工所）arXiv 2603.14251

**核心**：过度思考时，推理链会偏离正确路径，表现为高熵的转折词（"Wait"、"But"）频繁出现。在 R1-Distill 系列生成的 4590 万个 token 上统计，熵呈长尾分布：绝大多数 token 熵接近 0，少数转折词贡献了大部分总熵。于是定义 RPDI = 局部滑窗平均熵 / 全局平均熵，超过阈值就强制结束思考。

**细节里值得注意的**：
- 图 1 的典型案例：**模型先得到了正确答案，随后进入有缺陷的验证循环，把对的改成了错的**。过度思考不只是浪费 token，还会直接改错答案。
- 正常推理时 RPDI 在 1 附近波动；走偏时局部熵相对全局熵升高。
- 现有早停方法（基于答案一致性、探针、代理模型）在大模型上会“过度截断”，把还没完成自我纠错的推理提前掐断。

【切入点】
1. **“改对为错”有多普遍**：统计思考过程中中间答案从正确翻转到错误的频率，以及发生在哪类转折词之后。如果比例可观，就可以做免训练的“答案锁定”，例如出现过一次稳定正确的答案后，再出现转折词就提高结束概率。
2. **统一几类“关键 token”**：R2R 的分歧 token（小模型熵高 3.8 倍）、RPDI 的高熵转折词、BeaconKV 的回访 token、ConPress 里被大量压掉的“答题后验证”，很可能是同一批位置。用一个统一的熵 / 注意力分析把它们对齐，能同时服务于路由、早停、KV 保留三类方法。

---

## 9. DenseSteer（ICML'26；NCSU / Rutgers）arXiv 2605.29247

**核心**：在 Qwen2.5 同家族里，大模型推理**步数更少、每步信息密度更高**（作者叫 Reasoning Density = Ntokens/Nsteps 类指标）。如果直接把小模型往大模型的 hidden state 方向 steer，NLL 会暴涨（分布不匹配）。所以他们用 GPT-5.1 把**小模型自己的** trace 改写成更"密"的版本作正样本，再用目标模型自身的 NLL 筛选"兼容"样本。之后取**最后一个 token** 在第 l* 层的 h_pos − h_neg 均值作 steering vector，推理时 h̃ = h + λv。只需 50 对样本。

**细节里值得注意的**
- 实验设置：greedy 解码，**max 2048 tokens**，模型只有 1B–8B 的 instruct 版本（不是 reasoning model）。
- **Table 1 的真实增益很小**：
  - Qwen2.5-3B 平均分 61.2→62.5（+1.3）；Llama-3.2-3B 52.5→52.7（+0.2）。
  - Llama 的 MATH500 反而 46.0→43.4。
  - AIME "0→10%" 只相当于 30 题里多对 3 题。
  - "Prompt Engineering（Dense Style）"基线跌到约 20%，作者解释为小模型读不懂复杂 prompt，直接给出答案。这个基线本身是坏的，不能用来证明"steering 优于 prompt"。
- 层敏感性（Fig. 4/5/6）：
  - 早层 L6 无反应。
  - 中层 L16/L17 让步数和 token 同时下降，准确率单调上升。
  - 晚层 L35 **反而让输出变长**，"与已形成的推理轨迹冲突，产生补偿式冗长"。
  - λ>10 时 token 方差暴增。
- steering 后 NLL 下降，作者解读为"更符合模型先验"。但 NLL 下降也可能只是输出更短、更模板化。
- OOD：LogiQA EM 44.2→58.2/60.2（+14 个点，远大于数学上的增益）。这很可疑，多半是**修复了格式或长度截断**，而不是"推理变密"。
- InFamilySteer 对层和 λ 在 [−20, 20] 上做了 sweep，调参量不小。

**作者自述局限 / future work**
- 只能重组模型已有的潜在推理行为，不能补知识或技能。
- 主要在数学上验证。更广的领域、**更大的模型家族**和**更机制化的归因分析**留作 future work。

【切入点】
1. **长度混淆的又一例证**：2048 截断加小模型，"密度 steering"带来的增益可能大部分来自少被截断。只要按"未截断子集"或"等 token 预算"重评，就能检验这一点。这与 LessIsMore/DBTrimKV 的"压缩反而更准"属于同一类问题，可以合并成一篇 **"efficiency 方法的准确率增益归因"** 分析，把截断率、输出长度和格式错误三项拆开。
2. **作者没做 reasoning model（R1-distill / Qwen3-thinking）**。在长 CoT 模型上，"密度方向"是否等价于已有的 "overthinking steering"（如 SEAL；附录 B 有比较）？两者的方向向量 cos 相似度是多少？如果高度共线，DenseSteer 的新意就塌了。这是一个很便宜的检验。
3. **晚层 steering 让输出变长**是个有意思的现象：同一个向量在中层压缩输出、在晚层拉长输出。这说明"长度控制"在网络里存在**符号翻转的层结构**，可以用于解释为什么各家 steering 论文对层的选择很敏感。
4. **"最后一个 token 的 hidden state 当整条 trace 的摘要"**这个设计很粗糙（最后一个 token 往往就是答案或 EOS）。换成 step-boundary token（"\n\n"、"Wait" 位置）的平均做对比，可能就是一个更合理的版本，并能与 R2R 的 divergent token、BeaconKV 的 TRT 连起来。

---

## 10. PAS: Prelim Attention Score（CVPR'26；Los Alamos NL / U Florida）arXiv 2511.11502

**核心**：object 幻觉发生时，模型生成该 object token 时**不看图，而看自己前面生成的输出（prelim）**。
- 作者用"在给定 prelim 条件下，图像与 object 的互信息"来形式化这一点：互信息越低，越可能是幻觉。
- 落地成本最低的版本：object token 对所有 prelim token 的 attention 之和（PAS），**取第 0 层**、所有 head 平均。不需要额外 forward。
- 检测效果：
  - LLaVA-1.5-7B / MSCOCO AUROC 84.2，SVAR 81.5，IC 75.1。
  - 与 s_img 相比，s_prel 平均 84.8 vs 82.1（Table 5）。
  - 不同解码方式下都稳定（Table 4）。

**细节里值得注意的**
- **最佳层是 layer 0**（Fig. 5），作者用"浅层收集信息、深层处理信息"一笔带过。但 layer-0 的 attention 几乎只由位置和词嵌入决定，很难承载"模型切换到不看图的运行模式"这种语义解释。
- s_BOS、s_ins、s_img 与幻觉负相关，s_prel 与幻觉正相关。四者之和 = 1（softmax），所以 s_prel 高就**机械地**意味着其他三项低。作者在 Fig. 6 承认它们相关，但认为"不冗余"。
- 设置是 max_new_tokens=512 的开放式描述（"describe the image"），模型只有 LLaVA-1.5、MiniGPT-4、Shikra，**全部是 2023 年的老架构**。没有 Qwen-VL 系列，也没有 MRoPE。
- **没有位置基线**。在长描述里，越靠后的 object 越容易是幻觉，这是 LURE（ICLR'24）等早已报告的现象。而 prelim 越长，第 0 层分给 prelim 的 attention 质量自然越大。PAS 很可能大部分只是在测**"这个 object 出现在第几个 token"**。

**作者自述局限 / future work**
- 只适用于标准 full self-attention。对稀疏、线性等变体，"输出 token 不再 attend 到所有 token 类型，假设可能不成立"。
- 只做了 object existence 幻觉；relation/attribute 幻觉留作 future work。
- 只做了开放式描述。VQA 中"不同问题带来不同的 prelim 动态"是 future work。
- head 选择（只用部分 head）留作 future work。

【切入点】
1. **位置混淆检验（成本极低、杀伤力大）**：用 token index k 和 caption 内 object 序号作单变量检测器，看 AUROC 能到多少；再做"按位置分桶后的 PAS AUROC"。如果分桶后掉到 60 左右，就说明这类"attention-ratio 幻觉检测器"（PAS、SVAR）都可能被位置主导。这与 FEATHER 的 bottom bias、IVC/PriorTR 的位置先验是同一条线：**MLLM 里大量"attention 信号"其实是位置信号**。
2. **在 MRoPE / 混合架构上 PAS 是否成立**：作者自己说不确定。Qwen3.5 是 3 linear : 1 full，并有 attn_output_gate、无 sink（Gated Attention）。在这种架构里，"sum-to-1 的 attention 分配"本身就不再成立：gate 可以把整个 head 的输出关掉。可以直接问：**gated attention 下幻觉检测该用什么信号？**（例如 gate 值本身）这比"把方法搬到新架构"更进一步，因为是信号定义本身失效。
3. **从检测到 training-free 缓解**：既然幻觉对应"对 prelim 过度依赖"，可以在第 0 层（或检测层）PAS 超阈值时，对当前 step 做 prelim-attention 抑制或 image re-injection。这类似 OPERA，但触发是 token 级、零额外 forward 的。需要先确认切入点 1 不成立。
4. **推理型 MLLM 的长 CoT**：thinking 模型生成几千 token 的 prelim，PAS 的"prelim 依赖"在长思维链里会被系统性放大，正好可以与 BeaconKV 的"回看视觉 token"现象连起来：**reasoning MLLM 在 CoT 后段是否逐渐"不看图"**，以及这是否就是视觉 reasoning 后段幻觉的来源。

---

## 11. Causal Route Gating, CRG（ICML'26；西南财经大学统计学院 / NUS）arXiv 2605.24024

**核心**：幻觉的一个关键机制是 **route competition**：即使视觉 token 得到了 attention，最终 token 的决策仍可能被 text 通路主导。
- 把每个 head 的输出按 key 类型**精确拆成** visual route 和 text route（attention 输出对 value 线性，所以可以精确拆分）。
- 对当前 token 的 decision score ℓ 做 do-intervention：Δvis = ℓ(1,1) − ℓ(0,1)，Δtxt 同理。用"一次 forward + 一次梯度"的一阶近似来估计。
- 两者符号相反的 head 叫 conflict head。在 conflict head 里，按 VRI（视觉相对依赖）选 top-k 最小的 head，**只压它的 text route**，视觉 route 保持不变：
  - Conflict-B（视觉降 ℓ、文本升 ℓ）强压；
  - Conflict-A（视觉升 ℓ、文本降 ℓ）轻压。

**细节里值得注意的**
- **最有价值的发现（附录 A.4，Table 9）**：
  - POPE-popular 的 answer-critical token 上，**约 50.3% 的 head–example 对里，visual route 的效应是负的**（在降低当前决策分数）。
  - 即使只看 VAR（视觉 attention 占比）最高的 10% head，这个比例仍有 51.5%。
  - VAR 与 |Δvis| 只有中等程度的 Spearman 相关。
  - 结论是 **attention 质量是"符号盲"的**：一个 head 看起来高度关注视觉，但它的视觉 route 常常在提供反证。
- 作者的理论（Thm A.6）：只有当被 attend 的 visual value 与 score 的敏感方向大致对齐（没有符号翻转、没有严重抵消）时，VAR 才与真实因果效应相关。
- 层范围 ablation：
  - LLaVA-1.5 在 (8, 19) 层最好，Qwen-VL-Chat 在 (10, 24)，都是中到中上层的连续块。
  - Fig. 3：浅层很多 head 给视觉很高的 attention，但**中层 VAR 几乎是平的**。
- 开销：每步延迟约 2×（与 VCD 相当），OPERA 为 7×。需要 eager attention 和 autograd。
- 模型：LLaVA-1.5、Qwen-VL-Chat、Qwen2.5-VL-7B。POPE 准确率 +4~6%，CHAIR_S 52.8→更低。

**作者自述局限 / future work**
- 一阶、局部估计，在证据模糊或跨层强交互时有噪声。多个 gate 同时施加时的**高阶交叉项**没有被刻画。
- 只能处理"视觉与文本冲突"型幻觉。**同向偏置**（如粉色香蕉在视觉编码里就塌缩成黄色香蕉）、OCR 和多步推理错误不在范围内。
- 开销随生成长度线性增长，长文本更贵。
- 需要白盒，**黑盒版本需要用图像/文本扰动敏感度这类输出级代理**（作者点名的方向）。
- 会让模型更保守、降低描述丰富度。faithfulness 与 helpfulness 的平衡是 future work。

【切入点】
1. **"attention 质量符号盲"对 visual token pruning 是一个直接的打击**：FastV、SparseVLM、PriorTR、IVC 等几乎都用 attention 质量（或其变体）给视觉 token 打分。如果 50% 的高 attention 视觉贡献是"反证"，那么：
   - (a) 按 attention 剪枝会**系统性地保留一些在推高错误答案的 token**；
   - (b) 用带符号的 **value·gradient 或 DLA 分数**（prefill 时算一次）做剪枝打分，可能比 attention 好。
   这也给 "When Token Pruning is Worse than Random"（下文第 17 篇）提供了一种机制解释候选。
2. **三路分解：image / instruction / prelim**：CRG 只分了 visual 与 text 两路，PAS 只看了 prelim 的 attention 质量。把 text route 再拆成 instruction 和 prelim，用 CRG 的因果效应替代 PAS 的 attention 质量，就能直接检验 PAS 的"过度依赖 prelim"假说是否在**因果意义**上成立，还是只是位置效应。
3. **去掉梯度的近似**：用 direct logit attribution（route 输出 × unembedding 投到 ℓ 的方向上）代替 autograd，可以把 2× 的开销降到约 1×，并能与 FlashAttention 兼容（route 拆分只需要按 key 段分别算 softmax 加权的 value 和）。这是作者点到的"开销"局限的直接工程化延伸，但要验证跨层传播被忽略后符号还准不准。
4. **reasoning MLLM 的长 CoT**：作者自承"多步推理错误不在范围内"，长 CoT 下开销也会线性增长。可以只在 PAS / 熵 触发时调用 CRG 的检测（稀疏触发），形成"检测–干预"两级的 training-free 流水线。

---

## 12. Inverted Detection and Control in Steering Vectors（NeurIPS'26；Northeastern, Jennifer Dy 组）arXiv 2608.02957

**核心**：
- Steering 的默认假设是：一个向量能区分概念（正样本与它更对齐），所以加上它就促进该概念。
- 作者发现存在 **inverted-steering vectors（ISV）**：区分度很高（AUC 可达 0.97），**加上去却稳定地产生相反效果**。
- 几何解释：steer 某个上游 head 时，RSV 会让下游判别性 head 表现得"概念存在"，ISV 则让它们表现得"概念不存在"。
- 由此定义 **spoof score**：在**生成前**（问题最后一个 token），测 steer 上游 head 对所有下游判别 head（AUC ≥ 0.85）内积的影响。它区分 ISV 和 RSV 的 AUC 为 0.91，准确率 81%。
- 把它用于 ITI：先对 spoof score 为负的 head 翻转符号（ITI-RRF），30 组实验里有 27 组变好，最多 +138%。

**细节里值得注意的**
- 模型：Gemma 3 12B、Qwen2.5 14B、Olmo 3 7B。概念：财富倾向、可纠正性、短视、拒答、TruthfulQA。
- 以往文献把"anti-steerable"样本（3–50%）归因于伪相关。本文说明这是**向量层面的系统性性质**，不是样本噪声。
- Qwen 上只有 7/10 变好，作者"留作进一步研究"。
- 暗示：ITI 在不同模型和概念上的高方差，可能部分来自**方向符号错了**。

**作者自述局限 / future work**
- 只研究平移式 steering（mean-diff）。affine 和非线性 intervention 中是否也有 inversion 留作 future work。

【切入点】：这篇与 CRG、DenseSteer 形成一个**跨论文的共同主题："检测信号 ≠ 控制方向"**
1. **CRG**：约 50% 的高 attention 视觉 route 在降低决策分数，即 attention 质量对符号是盲的。
2. **DenseSteer**：同一个密度向量，在 L17 压短输出，在 L35 **拉长输出**，即层间符号翻转。
3. **本文**：高 AUC 的 probe 方向有一部分是反向控制器。

→ 可做的方向：
- **(a) reasoning steering 里的 ISV**：现在大量 training-free "overthinking / 长度 / 自反思" steering（DenseSteer、SEAL 等）都用 mean-diff 并只挑一层。在推理模型里批量挖"长度 ISV"，再用 spoof score 在**零生成成本**下给每层每 head 定符号。如果 ISV 普遍存在，就能解释为什么这类方法对层选择极其敏感。
- **(b) 视觉 token 剪枝 / 幻觉检测里的"符号"问题**：把 spoof score 的思路搬到 token 层面，即"保留这个视觉 token 会让下游判别 head 偏向 present 还是 absent"。这是 CRG 的 Δvis 的一个**无梯度版本**。
- (c) 理论上更干净的问题：DenseSteer 的晚层反转是不是 ISV 现象在**层维度**上的体现？用 spoof score 能否预测出 DenseSteer 的最佳层（L17）和反转层（L35）？这是一个只需几十对样本的便宜实验。

---

## 13. LinearPatch（NeurIPS'25；清华深研院 / 华为诺亚 Haoli Bai, Lu Hou）arXiv 2505.24680

**核心**：layer pruning 掉点的主因是**剪枝界面处的激活量级失配**。
- 第 ℓ* 层的输出与第 ℓ*+n 层期望的输入在通道尺度上不一致。
- 失配被 **BOS、分隔符等特殊 token 上 >10³ 的 massive outlier** 放大：同一通道里，单个 channel scale 无法同时拟合所有 token。实测 token-wise scale 的标准差 σ_d = 2137.75。
- 解法：先 Hadamard 旋转压平 outlier，再在旋转空间做 channel-wise scaling，两者合成一个实对称矩阵，插在剪枝界面。training-free 版本只需校准统计；可选 5K 无标注样本、30 分钟单卡蒸馏这一个矩阵。
- 结果：LLaMA-3-8B 剪 5/32 层，保留 94.15%（LLM-Streamline 为 90.84%）。

**细节里值得注意的**
- Fig. 1(b)：把 scale 偏离最优值 d（α≠1）会让 PPL 急剧变差，说明**"量级"本身就是信息瓶颈**。
- 评测只有 PPL、MMLU 和常识 QA（多为 log-likelihood 选择题）。附录 Table 16 也加了 DeepSeek-R1-Distill-Qwen-7B / Llama-8B，**但仍只测 QA，不测数学或长 CoT 生成**。
- 剪枝率限制在 30% 以下，剪枝度量用 cosine similarity（ShortGPT/Streamline 风格）。

**作者自述局限 / future work**
- "layer pruning 可能对任务不均匀地造成损伤。QA 也许稳健，但**复杂推理或依赖上下文的任务可能受损**。未来应建立评估效率与任务性能权衡的框架。"

【切入点】
1. **推理模型上的层剪枝是一个被作者点名、但没有测的空白**：R1-distill 只测了 QA。长 CoT 生成中，界面失配会在几千 token 上**自回归累积**，这在 PPL/QA 上完全看不到。可以直接测 LinearPatch/ShortGPT 在 AIME/MATH 上的表现、输出长度变化（是否引起 overthinking 或循环）。这与"量化伤害推理"（COLM 那篇，作者自己也引了）是同一类问题，但对象是层剪枝。
2. **机制在 sink-free 架构上会失效还是变简单？** LinearPatch 的核心难点来自 BOS/分隔符上的 massive activation。Qwen 的 Gated Attention（NeurIPS'25 Oral）表明 output gate **消除了 massive activation 和 attention sink**，而 Qwen3.5/3.6 正采用 attn_output_gate=True。预测：在 gated 模型上，Hadamard 那一半几乎没用，简单的 channel scaling 就够了，**层剪枝本身也更容易**。这是一个有明确可证伪预测的分析实验，而不是"搬架构"：它检验的是"层剪枝的难度来源于 sink 机制"这一因果假设。
3. **动态层跳过（per-token early exit / layer skip）也有同样的界面失配**，而且是**每个 token 跳的层不同**，没法用一个固定 patch。可以按"跳过段"预存一组 LinearPatch（只是 C×C 矩阵），配合 RPDI/熵这类 token 级难度信号，做 training-free 的自适应深度解码。
4. 注意 3:1 混合架构（linear attention + full attention）中，剪掉 full-attention 层与剪掉 linear 层的代价应当非常不对称。这个不对称性可以作为一个 observation 起点。

---

## 14. ExpertWeaver（ICML'26；浙大 / 上海 AI Lab / ByteDance Seed / 复旦）arXiv 2602.15521

**核心**：dense LLM 的 GLU 门控激活中隐含着 MoE 结构。
- Takeaway 1：用 |gate| 做 top-50% 的 AbsTopK-GLU 几乎不掉点，说明 GLU 自带 token 级稀疏。
- Takeaway 2：存在一组跨任务始终激活的 **universal neurons**，可以作为 shared expert。
- Takeaway 3：其余是 task-specific 的 **specialized neurons**，可按共激活模式聚类成 routed experts。
- Takeaway 4：各层 universal 与 specialized 的比例不同（用 CV 度量），所以**每层配置不同**。
- 做法：router 也直接由 GLU gate 权重构造，完全 training-free。既可作为 25% 稀疏的动态结构剪枝，也可作为 downcycling 的初始化。

**细节里值得注意的**
- **附录 E，Table 5（作者在正文里写成"maintains strong performance"）**：
  - Qwen2.5-7B 的 GSM8K 在 25% 稀疏下只有 **34.9**（dense 约 85+），12.5% 稀疏下 18.9（原文如此，比 25% 还低，疑为不同设置）。
  - DeepSeek-R1-Distill-Qwen-7B 在 12.5% 稀疏下，GPQA-D 为 37.3（dense 约 49），LiveCodeBench 为 **16.8**（dense 约 37）。
  - LLM-Pruner、FLAP 更差（GSM8K 掉到 2.0 或直接为 0）。
  - 也就是说：**training-free 的 dense→MoE 和结构剪枝，在常识 QA 上看着好，一到数学或长推理就崩**，即使只稀疏 12.5%。
- vLLM 实测：H100 上 TPOT 从 18.5ms 降到 9.8ms，TTFT 从 1404 降到 542ms，确实有真实加速。
- 校准数据是 Flan-v2 多任务，没有推理 trace。

**作者自述局限 / future work**：正文没有专门的 limitation 段。附录 E 自己暴露了复杂任务上的大幅掉点。

【切入点】
1. **"效率方法在推理任务上的系统性崩塌"是第二个跨论文共性**。LinearPatch 只测 QA，并自承"复杂推理可能受损"；ExpertWeaver 的附录直接显示推理任务掉一半。一个有价值的分析问题是：**崩塌发生在哪些 token 上？** 预测是集中在 R2R 发现的那约 10% 的 divergent token，以及 "Wait"、"So"、"Therefore" 这类高熵分叉 token（这两类在第 6 篇 R2R 和第 2 篇 BeaconKV 的 TRT 都出现过）。
2. **方法切入：token 自适应稀疏度（"Self-R2R"）**。
   - 用当前 token 的熵或 GLU gate 的集中度做零成本判别器，对少数分叉 token 走 **dense FFN**，对其余 token 走 ExpertWeaver 的稀疏专家。
   - 这是 R2R 的"单模型版本"：不需要 SLM，也不需要训练 router，权重共享，KV cache 也共享（因为 attention 不变）。
   - 期望：用很少的额外计算换回大部分推理精度。这比 R2R 部署简单很多（R2R 需要两个模型和两套 KV）。
   - 新意风险：需要检查 2026 年有没有"entropy-gated dynamic sparsity"的顶会工作。arXiv 上大概率已有类似做法，需要核查。
3. **校准分布问题**：用推理 trace 代替 Flan 做 universal/specialized 划分，看推理神经元是否被错误地划进了 routed experts 而经常不被激活。如果是，这本身就是一个 observation，即"推理能力依赖于低频激活神经元"。

---

## 15. Do Thinking Tokens Help with Safety?（NeurIPS'26；Princeton PLI）arXiv 2606.25013

**核心**：推理模型的 safety 决策**并不是在思考中做出的**。
- 在 GPT-OSS、Qwen、Olmo、Phi 上，用**第一个 thinking token 的 hidden state** 训一个线性 probe，就能预测最终是拒绝还是服从：AUROC 0.84–0.95，balanced accuracy 约 88%。而用该 token 本身的文本做 TF-IDF 接近随机。
- 信号在 **prefill 结束时陡增**。
- 前缀实验：在自然 trace 长度的 20/40/60/80% 处截断，每个前缀采样 8 个续写。**B=20% 时续写间方差已经低于 0.2**，之后继续下降。
- 开启与关闭 thinking 相比，只有不到 30% 的 prompt 发生多数标签翻转，而且翻转只能改善 ASR 或 ORR 之一，不能两者兼得。
- 文本上看似"自我审议"的摇摆只出现在 15–34% 的 rollout 中，其中 71–92% 是表演性的。**72.8–76.6% 的摇摆发生在结果已锁定时**（拒绝率 ≤0.05 或 ≥0.95）。
- 现有防御：
  - 推理时方法（SafeChain primer、PSR 周期性自省、SafeRemind 按熵插入提醒）要么没效果，要么以 ORR 上升换 ASR 下降，还会**压制本来就少的审议信号**。
  - 训练方法同样主要把模型推向过度拒绝。

**作者自述局限 / future work**
- 只研究了中等规模的开源 LRM。
- 只看了拒绝与服从，没有覆盖事实性、欺骗、隐私、多轮。
- 首 token 可解码性来自预训练、SFT 还是后训练，留作 future work。
- §3.2 表明"真正的审议罕见但存在"，可以设计奖励真审议的训练目标。

【切入点】
1. **这其实是一个"推理效率"结论**：在 safety 类 prompt 上，**70–97% 的 thinking 是纯浪费的 token**。training-free 方法可以是：prefill 末端的 probe 置信度高时，直接跳过或截断 thinking（若锁定为拒绝，直接给出拒绝模板）；只有置信度不高时才让模型思考。
   - 难点：probe 需要训练（只是一个 logistic regression，属于"少训练"的范畴）。
   - 风险：这和"answer is decodable early → early exit"的推理效率工作（数学上已有不少）思路相同，新意在于把 safety 作为对象，并且有 ASR/ORR 两个指标。要核查 2026 年顶会是否已有"safety-aware adaptive thinking"。
2. **与第 8 篇 RPDI-EE、第 16 篇 Early Decisions 连接**：三篇都在说"决策在早期就已经锁定，后续 token 只是在合理化"。可以合成一个更一般的问题：**对于什么类型的任务，thinking 的后段是在"计算"，什么时候是在"合理化"？** 用本文的前缀续写方差作为通用诊断指标（只需采样，不用训练），跨数学、代码、safety、VQA 做系统性测量。这是一篇有 insight 的分析论文，且计算成本可控（7–8B 模型、采样为主）。
3. **反向利用**：SafeRemind 在熵指示"决策锁定"时插入提醒，但本文显示锁定发生得极早（20%）。把干预点**提前到 prefill 末端**（例如在首 token 处对 probe 方向做 steering），可能比在 thinking 中间插入文本更有效。需要注意第 12 篇的 ISV 问题：probe 方向不一定是正确的控制方向。

---

## 16. Early Decisions Matter（ICML'26；KAIST / LG AI Research）arXiv 2604.10567

**核心**：扩散语言模型（LLaDA-8B、Dream-7B）在**完全非自回归（NAR）的 confidence 解码**下有一个内在失败模式：**proximity bias**。
- 去 mask 的顺序会集中在空间上相邻的 token 上，误差沿空间传播，整条轨迹严重依赖**第一步在哪里去 mask**。
- 叠加 **EOS dominance**：早期最自信的 token 往往是 EOS、标点等，被去 mask 的 token 中超过 50% 是 EOS，压缩了可用的推理长度。这与 EOS overflow（Rainbow Padding，需要微调）是同一现象。
- 方法：一个 2 层 Transformer 的轻量 planner，只在**第 1 步**选择去 mask 位置；加上对 EOS logit 的时间退火温度。backbone 不训练。
- 结果（T=32）：平均 44.8→49.1，优于 semi-AR（27.0）。

**细节里值得注意的**
- **反直觉 1**：NAR 下**步数越少反而越好**（Fig. 2），扩散的"更多推理计算换更好质量"的单调性被破坏。原因是高计算预算下每步只去 mask 极少几个 token，EOS 这种高概率 token 先被选中；低计算预算下每步去 mask 很多 token，非 EOS 候选也能被纳入。
- **反直觉 2**：**只随机化第一个去 mask 位置**带来的 pass@k，远高于在所有步上做温度采样。把随机化推迟到中间步，性能严重下降。即"不确定性在时间上高度不对称"。
- planner 只看候选位置的最终层 hidden state，不看全序列。

**作者自述局限 / future work**：没有在开放式生成（创意写作、长摘要）上验证；proximity bias 和初始轨迹塑形在更多语言场景下是否成立，留作 future work。

【切入点】
1. **"早期决策锁定"是第三个跨论文共性**（第 15 篇 safety 在首 token 锁定，第 8 篇 RPDI-EE，本篇在 dLLM 第 1 步锁定）。对于 TTS，这意味着：**多样性应当只注入在早期**。在自回归 LLM 上的对应实验是："只在前 k 个 token 用高温，之后 greedy" vs "全程温度采样"，比较 pass@k 和 majority vote 的样本效率。如果早期注入同样更优，就是一个 training-free、零开销的 TTS 改进，并且有本文的机制作为支撑。需要核查 2025–26 顶会是否已有"前缀多样化采样"的工作（arXiv 上可能已有）。
2. **dLLM 的 training-free 改进**：本文 planner 需要训练（虽然很轻）。能否用 training-free 规则替代（例如第一步强制在"问题后第一个位置"或"推理段中部"开局，或者用 attention 熵选择位置）？作者 Table 1 有 Uniform（U）对比，可以看看规则式选择与 learned planner 的差距有多大。
3. **dLLM 的效率 × TTS 交叉**：既然低步数更好、早期多样性更值钱，dLLM 的 best-of-N 可以**共享前缀以外的所有步骤**，只在第 1 步分叉，计算节省很明显。这是 dLLM 特有的 TTS 计算结构，在 AR 模型上无法做到。

---

## 17. When Token Pruning is Worse than Random（CVPR'26；同济 / UCSC / Amazon / 华东师大）arXiv 2512.07580

**核心**：在 LLaVA-1.5-7B 和 Qwen2.5-VL-7B 的**深层**（LLaVA 约 16–20 层后，Qwen 在 MME 上约 21 层后、TextVQA 上约 24 层后）剪视觉 token 时，FastV、SparseVLM、DART、DivPrune 等方法**与随机剪枝一样甚至更差**。
- 作者的解释是 **"vanishing token information"**。定义信息 I_i(V_k) = 在第 i 层去掉 V_k 后，**ground-truth 答案首 token 概率**的变化。
- 随着层数加深，视觉 token 的信息先变得均匀，然后在某一层整体消失。这一层叫 **information horizon**。
- horizon 是动态的：
  - OCR 等视觉密集任务的 horizon 更深；
  - 更强的模型（Qwen2.5-VL）使用更深层的视觉 token：在 MME 上 Qwen 在第 20 层仍利用视觉 token，LLaVA 在第 16 层就已经不用。
- 方法：在深层改用随机剪枝，与现有方法组合（DivPrune+Random 在 Qwen2.5-VL 剪 50% 时保留 96.9%）。

**细节里值得注意的**
- **Fig. 6 的"剪掉 75% 低信息 token 反而比不剪更好"**（第 10 层，MME +27.8%，TextVQA +6.1%）。注意这个打分用到了 **ground-truth 答案**，属于 oracle 选择，本身带有标签泄漏，不能当方法看。但它给出了一个**上界**：**存在一个视觉 token 子集，比全部 token 更好**。也就是说，部分视觉 token 在"帮倒忙"。这与第 11 篇 CRG 的发现（约 50% 的视觉 route 效应为负）方向一致。
- 信息只按**第一个生成 token** 度量（MME 是 yes/no，TextVQA 是短答案）。**没有覆盖长 CoT 或长答案生成**。
- 与 VTW、PDrop 的区别：VTW 在固定层撤掉所有视觉 token。本文指出 horizon 随任务和模型变化，所以固定层会伤害 OCR 类任务。
- Fig. 7：第 1–7 层高信息 token 集中在与问题相关的区域；第 9 层以后变得均匀，第 16 层接近 0。

**作者自述局限 / future work**：正文没有专门的 limitation 段。结论里只说"希望为利用信息量大的视觉 token 提供经验洞见"。

【切入点】
1. **per-sample 的 horizon 检测（training-free 方法）**：作者证明 horizon 随任务和模型而变，却仍然用固定配置加随机剪枝。如果能用一个**无标签的代理量**（例如视觉 token 上 attention 或 value-norm 分布的方差、熵，即 Fig. 7a 的"均匀化"本身）在推理时逐层检测"是否已经越过 horizon"，就可以做**逐样本自适应的视觉 token 撤除层**。这是对 VTW 和本文最直接的推进。需要核查 2026 年 CVPR/ICCV/ECCV 是否已有 "adaptive visual token withdrawal"。
2. **推理型 MLLM 里 horizon 是否随生成时间变化**：本文只看首 token。BeaconKV（第 2 篇）表明推理过程中模型会回看视觉 token，PAS（第 10 篇）表明后段可能"不看图"。合起来的问题是：**horizon 是"层的函数"还是"(层, 生成步) 的函数"？** 如果长 CoT 中某些步（例如验证步、"let me look at the image again"）horizon 会突然变深，那么所有"在 prefill 一次性剪枝"的方法在 reasoning MLLM 上都会系统性失效。这是一个有 insight 的分析点，直接连接效率与推理。
3. **"有害视觉 token"的无标签识别**：oracle 实验说明存在一个比全集更好的子集。用 CRG 风格的带符号 route 效应，相对于模型自己 top-2 候选的 margin（而不是 GT），能否找回一部分增益？这就把"剪枝"从"保性能的效率手段"变成"提升性能的去噪手段"。不过要非常小心：任何"剪了更准"的结论，都要排除第 9 篇里那类长度和格式混淆。
4. 与 IVC-Prune、PriorTR 的位置或空间覆盖发现合起来看：深层"随机 ≈ 最优"意味着深层只需要**空间覆盖**，而浅层需要**语义选择**。一个"浅层语义选、深层均匀覆盖"的两段式剪枝在机制上是有依据的。需要核查是否已被 DivPrune+Random 这种组合覆盖。

---

# 跨论文综合：17 篇读下来反复出现的四条"缝"

## 缝 A：效率方法"反而更准"，但没人拆开原因（长度/截断/格式混淆）
- **证据**：
  - LessIsMore：稀疏后 AIME 比全量更准，但链变长。
  - DBTrimKV：驱逐后更准。
  - DenseSteer：在 2048 截断下增益只有 +0.2~1.3，LogiQA 却 +14。
  - PAS 与 Pruning-worse-than-random 的 oracle 剪枝：剪了更准。
  - ConPress：压缩后更准。
- **作者们的态度**：都把"更准"当作卖点写进标题或摘要，**没有一篇做等长度、按截断分层、按格式错误分层的归因**。
- **可做**：一篇系统性的分析论文，即 "Accuracy gains of efficiency methods: how much is length?"。
  - 对 5–8 个代表性方法统一测：截断率、输出长度、格式错误率、"wait"类反思次数，以及**在未截断子集上的准确率差**。
  - 成本：只做推理，7–8B 模型，单卡或少卡即可。
  - 风险：属于"rethinking"类分析，用户之前对"搬架构"的 rethinking 不感兴趣；但这一篇的价值在于**推翻一批论文的核心卖点**，属于另一种性质。需要用户判断是否喜欢。

## 缝 B：attention/probe 信号是"符号盲"的，检测方向 ≠ 控制方向
- **证据**：
  - CRG：约 50% 的高 VAR 视觉 route 在降低决策分数。
  - Inverted SV：高 AUC 的 probe 方向会反向控制。
  - DenseSteer：同一向量在 L17 与 L35 的效果符号相反。
  - PAS：最佳层是 layer 0，很可能是位置信号而非语义。
  - Pruning-worse-than-random 的 oracle：存在比全集更好的视觉子集。
- **可做（方法类）**：**带符号的视觉 token 重要性**（value·gradient / DLA / spoof-score 风格的无梯度近似），用于：
  - (i) 剪枝打分，替代 attention 质量；
  - (ii) 去掉"反证"token，作为 training-free 的幻觉缓解。
  - 核心对照组：FastV、SparseVLM、DivPrune 和随机，加上 CRG 的 Δvis（作为上界）。
  - **先做的便宜实验**：在 POPE/MME 上统计 attention 高分 token 中带符号效应为负的比例，及剪掉这些 token 对准确率的影响。如果效应明显，就是一个 observation→method 的完整故事。
  - 风险：需要检查 2026 年是否已有"gradient/attribution-based visual token pruning"的顶会工作（arXiv 上肯定有 attribution 剪枝，但多数不是 MLLM 或不是带符号的）。

## 缝 C：决策早期锁定，后段 token 在"合理化"
- **证据**：
  - Thinking-Safety：首 token probe 的 AUROC 达 0.84–0.95，20% 前缀后续写方差 <0.2，74% 的摇摆发生在已锁定之后。
  - Early Decisions：dLLM 第一步决定轨迹，只随机化第一步的 pass@k 优于全程采样。
  - RPDI-EE：熵比早退。
  - R2R：只有约 10% 的 token 是分叉点。
- **可做**：
  - (i) **TTS 的"早期多样性"采样**：只在前 k 个 token 或前若干个分叉 token 用高温，之后 greedy。比较同样计算量下 pass@k 和 majority vote 的效果。这是 training-free、零开销、实验便宜的方案，有两篇顶会论文的机制作支撑。需核查是否已有。
  - (ii) **前缀续写方差**作为通用"思考是否还在计算"的诊断，跨数学、代码、safety、VQA 测量，从而得出"什么任务的后段 thinking 是浪费"的结论。

## 缝 D：效率方法在推理或长 CoT 上崩，而作者只测 QA/PPL
- **证据**：
  - LinearPatch 只测了 PPL/QA，自承"复杂推理可能受损"。
  - ExpertWeaver 的附录：GSM8K 85→35，LiveCodeBench 37→17。
  - Pruning-worse-than-random、PAS：只看首 token 或短答案。
- **可做（方法类）**：**Self-R2R：单模型内按 token 难度切换 dense/sparse**。
  - 只对分叉 token（按熵或 gate 集中度判别）走 dense FFN 或全层，其余 token 走稀疏专家或剪层模型。
  - 优势：不需要 SLM，不需要训练 router，KV 共享。直接回应 ExpertWeaver 和 LinearPatch 在推理任务上的崩塌。
  - 风险：最可能在 arXiv 上已有"entropy-gated dynamic depth / sparsity"，必须核查。

## 我的排序（仅基于阅读，尚未做新意核查）
1. **缝 B 的"带符号视觉 token 重要性"**：observation 直接来自 CRG 附录 Table 9 加 oracle 剪枝实验，两篇都是顶会；方法清晰，训练-free，单卡可做，可以同时讲效率与幻觉两个故事。
2. **缝 C(i) 的"早期多样性采样"**：实验最便宜，有机制支撑；但 idea 很简单，新意风险最大，需要先查。
3. **缝 D 的 Self-R2R**：工程价值高、结果可能好看；新意风险中等偏高。
4. **缝 A 的长度归因分析**：最稳，能出结论，但属于分析类。

下一步：对 1–3 分别做顶会范围的新意核查（ICLR'26 / ICML'26 / NeurIPS'25–26 / CVPR'26 / ECCV'26 / ACL'26 的 JSON 语料加关键词检索），再决定主攻方向。

## 初步新意核查（顶会语料约 45k 篇，按标题和摘要关键词检索；只是粗查）
- **缝 B，带符号视觉 token 重要性**：没有找到在 MLLM 视觉 token 上做"带符号或因果效应剪枝"的顶会工作。最接近的两篇：
  - NeurIPS'25 *Learning to Focus*（人大 / 清华）：在**纯文本 LLM** 上用梯度找出"干扰 token"，发现去掉它们后推理更准，然后**蒸馏训练**。observation 同向，但模态和方式都不同，可以当作强 related work 和动机支撑。
  - NeurIPS'25 *AD-TP*（山东师大）：attribution 驱动的自适应剪枝，针对通用 Transformer，不针对 MLLM，也不涉及符号。
  - 结论：**空间存在**，需要再到 arXiv 上精查（只用来排查撞车，不作为依据）。
- **缝 C(i)，早期多样性采样**：发现**反证**。ICLR'26 *Global Forking Tokens*（多伦多 / Amazon）指出，在难题上"触发多样且正确推理模式的 forking token 往往在采样树深处"，并且温度缩放的多样性–准确性权衡会变差。这与"只在早期注入多样性"的直觉冲突。这个方向需要先做小实验验证，**优先级下调**。
- **缝 D，Self-R2R**：ECCV'26 *GeMoE*（浙大）用 gating entropy 做 MoE-LVLM 的自适应专家数，ICML'26 *DTop-p MoE* 是预训练阶段的方法。它们都是"原生 MoE 内的自适应 k"，不是"dense→稀疏的 token 级回退"。部分重叠，**新意中等**。
