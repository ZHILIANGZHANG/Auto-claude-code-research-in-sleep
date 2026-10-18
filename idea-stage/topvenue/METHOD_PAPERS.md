# “观察 → 方法”型顶会论文（以免训练 / 轻量方法为主）

筛选方式：在约 9,000 篇带摘要的顶会论文（NeurIPS'25 全部，以及 ICLR'26 / ICML'26 / NeurIPS'26 / CVPR'26 / ECCV'26 / ICCV'25 的效率、TTS、MLLM、幻觉、steering、合并、持续学习、量化剪枝等方向）里，自动抽取摘要中的“我们发现……”句和“因此提出……”句，并要求出现免训练、即插即用、推理时、闭式解等字样；再人工逐篇筛掉与 LLM/MLLM 无关或观察不够具体的。每条都按摘要原意改写，没有凭标题猜测。

格式：**观察** → **方法**。机构空着的是会议页面没给机构信息。

## 1. KV cache 与稀疏注意力

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| ICML'26 | LessIsMore (Cross-Head Unified Sparse Attention) | Princeton, CMU | 推理时关键 token 是全局且稳定的：各 head 共享、随解码步不变 → 跨 head 统一选 token 的免训练稀疏注意力 |
| ICML'26 | BeaconKV | Hanyang, Apple | 推理模型中同类 query 在嵌入空间里聚成少数几簇 → 每簇保留一个“信标 query”，用它预测哪些 KV 会被再次访问 |
| ICML'26 | xKV | Cornell, Meta | KV 的主奇异向量在各层之间高度对齐（CKA） → 多层共享一个低秩子空间来压缩 KV |
| ICML'26 | PatternKV | BIT, 小红书 | K 有稳定、随上下文演化的结构，V 有潜在语义规律，两者都聚成共享模式 → 减去模式后再量化残差，扩大量化余量 |
| NeurIPS'25 | AsymKV (Homogeneous Keys, Heterogeneous Values) | 上海财大 | 相邻 key 获得的注意力相近，相邻 value 却分布各异 → key 合并、value 无损压缩 |
| ICLR'26 | QuoKA | Qualcomm, KAIST | 与“平均 query”余弦相似度低的 query 和更多 key 交互、对 logits 贡献最大 → 按这类 query 来选 KV，加速 chunked prefill |
| ICLR'26 | RESA | 哈工大 | 注意力 logits 本质上低秩、高度冗余 → 用低秩估计补回稀疏注意力丢掉的部分 |
| NeurIPS'26 | FFTSparse | 哈工大 | 长上下文注意力常出现对角条纹：特定相对距离的 query-key 一直高相关 → 用 FFT 找出这些距离来构造块稀疏 |
| NeurIPS'26 | Make Each Token Count | Yale, CUHK | 全量 KV 并非最优：无关 token 会稀释注意力 → 驱逐反而能提升长上下文表现（学习式保留） |
| NeurIPS'26 | Cross-Model KV Cache Transfer | NVIDIA, Cornell, AMD | 同家族不同尺寸模型的 KV 是线性可映射的（14B→32B 单层能解释 56% 的 key 方差） → 用闭式线性映射跨模型复用 prefill |
| ICML'26 | RelayCaching | 北邮, 清华 | 多 agent 中相同内容的 KV 在不同阶段高度一致，前缀造成的偏差只集中在少数层和位置 → 直接复用上一个 agent 解码阶段的 KV |
| ICML'26 | LRAgent | 首尔大学 | 多 LoRA agent 之间的 KV 差异主要来自 adapter 输出，基座部分几乎相同 → 把 KV 拆成共享的基座部分和 adapter 部分 |
| ICML'26 | Fast KV Compaction via Attention Matching | Thinking Machines, MIT | 把上下文压缩写成“注意力匹配”问题后，它分解成若干有闭式解的子问题 → 快速在隐空间压缩上下文 |
| NeurIPS'25 | Evaluator Heads（提示词压缩） | 华为 | 某些 attention head 天生能挑出长输入里最重要的 token → 只跑浅层这些 head 来压缩 prompt |
| NeurIPS'25 | Language Models (Mostly) Know When to Stop Reading | Duke, Rice | 特定 head 编码了“信息已经足够”的信号 → 读到足够信息就提前截断上下文 |

## 2. 视觉 token 压缩（MLLM / VLA）

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| ICLR'26 | IVC-Prune | 武大, 小红书, 腾讯 | LVLM 通过 RoPE 隐式建立了视觉坐标系，有些位置的 token 是空间推理必需的“坐标 token” → 保留坐标 token 加上语义相关的前景 token |
| ECCV'26 | PriorTR (Prior-Corrected Token Reduction) | 浙大, 港理工 | 即使没有文本指令，MLLM 也会盯住某些与任务无关的区域（模型先验） → 从注意力里扣掉这个先验再剪枝 |
| CVPR'26 | One Layer's Trash is Another Layer's Treasure | — | 不同层关注的视觉区域不同，最优 token 子集随层变化 → 逐层自适应选 token |
| CVPR'26 | TransPrune | — | token 表示在层间的“变化量”携带语义信息 → 按变化量剪枝 |
| NeurIPS'25 | HoloV | HKUST | 只追“高注意力 token”会丢掉整体上下文 → 保留整体覆盖 |
| ICML'26 | CLIP Tricks You | 世宗大学 | 指代区域内的视觉 token 与文本的 CLIP 相似度反而低 → 针对像素级定位重新设计文本引导剪枝 |
| ICML'26 | Position Is All You Need | 华科 | 指代分割任务里，位置编码和局部空间邻域比语义内容更关键 → 保位置的免费压缩 |
| ICML'26 | RTPrune（DeepSeek-OCR） | 京东, 清华 | DeepSeek-OCR 解码时先读高范数 token，再把注意力分给其余 token → 两阶段剪枝 |
| ICML'26 | SpecPrune-VLA | 上交 | VLA 相邻时间步的输入图像高度相似 → 结合局部和全局信息的两级免训练剪枝 |
| ECCV'26 | SAFE-Pruner（VLA） | 清华 | VLA 在多个控制步里始终把注意力放在同一个语义实体上 → 预见未来步的剪枝 |
| NeurIPS'26 | TeMPO（视频） | 滑铁卢等 | 现有视频压缩器不记得前面帧已经交给 LLM 的 token → 帧因果地追踪“已提交支持集” |
| CVPR'26 | MoDES（MoE MLLM） | — | 为纯文本 LLM 设计的专家跳过方法搬到 MLLM 上会掉点很多 → 区分模态的动态专家跳过 |
| ECCV'26 | RegCache（视觉编码器激活量化） | POSTECH, Google | 视觉编码器的离群值行为与语言模型不同 → 在中间层加前缀寄存器、删掉离群 token |
| ICCV'25 | Representation Shift | Korea Univ, Adobe | 拿不到注意力图时（FlashAttention）无法打分 → 用表示变化量当 token 重要性，兼容 FlashAttention |

## 3. 推理效率与 test-time scaling

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| ICML'26 | ConPress | 国科大, 微软 | 同一 prompt 里放多道题，每题推理自动变短，加一道“1+1=?”就压掉一半 → 采多题轨迹再蒸馏回单题（已精读全文） |
| NeurIPS'25 | R2R (Roads to Rome) | 清华, 北大 | 大小模型的推理路径只在极少数 token 上真正分叉 → 只在这些 token 上调用大模型 |
| NeurIPS'25 | SpecReason | Princeton, CMU | 推理过程对近似很宽容：每一步的价值在于语义而不是逐字 → 小模型投机地写推理步，大模型只做审核 |
| NeurIPS'25 | Certaindex | UCSD, UIUC, CMU | 推理算法的中间答案过了某点就不再变化 → 用“答案稳定度”决定何时停止 |
| NeurIPS'25 | Hogwild! Inference | Yandex | 推理模型不微调就能共享 KV、并行协作 → 多个 worker 共享一个并发更新的注意力缓存 |
| NeurIPS'26 | RPDI-EE | 国科大 | 推理跑偏时，高熵的转折词会异常激增 → 用局部/全局熵比做免训练早停 |
| ICML'26 | Think Deep, Not Just Long | Google, UVA | “深层预测大幅修正的 token”比 token 数更能反映推理努力 → 据此选样本和分配预算 |
| ICML'26 | DenseSteer | NCSU, Rutgers | 推理越熟练，步数越少但每步信息密度越高 → 免训练地把小模型 steer 向“稠密推理” |
| ICML'26 | VIA-SD | 浙大, A*STAR | 很多被拒的草稿 token，其实用验证模型的一个精简子网就能判对 → 多层级验证的投机解码 |
| NeurIPS'26 | STeP (Speculative Tensor Parallelism) | AMD, IISc | 按显著性排序 FFN 通道后，取前 k 个就能逼近完整输出 → 子网放 GPU 当草稿、其余放 CPU 并行验证 |

## 4. 幻觉缓解（MLLM）

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| CVPR'26 | PAS (Prelim Attention Score) | — | 很多幻觉发生时模型其实没看图，而是依赖自己前面已经生成的 token 来推新物体 → 用对已生成 token 的注意力做免训练检测 |
| ICLR'26 | Hallucination Begins Where Saliency Drops | 上交 | 当之前生成的 token 对下一个 token 的显著性很低时，就会出现幻觉（上下文记忆失效） → 用显著性来诊断和干预 |
| CVPR'26 | Same Attention, Different Truths | — | 用 logit-lens 读视觉注意力，区分出两种幻觉：“看错了易混区域”与“没看图” → 分别处理 |
| ICML'26 | Causal Route Gating | 西南财大 | 视觉 token 即使被注意到，最终决策也可能被文本通路主导 → 把每个 head 拆成视觉路和文本路，按因果效应调权 |
| ICML'26 | Fox (Dismantling Pathological Shortcuts) | 电子科大 | 幻觉不是因为注意力不够强，而是在关键决策步上特定 head 与视觉证据脱钩 → 免训练修正 |
| ICML'26 | ILVAD | 东南大学 | 模型生成过程中会逐渐“忘记”正确的视觉证据 → 用层间视觉注意力差异来增强证据 |
| ICML'26 | Steer Where It Matters (TLVS) | 华南理工 | 图像对生成的影响只稀疏地发生在少数解码步，整句平均会稀释它 → 按 token 的视觉敏感度做 steering |
| ICML'26 | AFIP（注意力分散导致视觉模糊） | NUS, 哈工大 | 幻觉与类似人类“分心”的注意力分散有关 → 跨 head 注意力富集与历史注意力增强 |
| CVPR'26 | First Logit Boosting | — | 第一个 token 携带的视觉信息会随生成衰减，“The” 这类 token 有稳定作用 → 把首个 token 的 logits 加回后续步 |
| CVPR'26 | Vision-Guided Attention | — | 模型能从视觉 token 正确提取语义，但后续推理没用上 → 先做视觉定位，再引导注意力 |
| ICLR'26 | Dynamic Multimodal Activation Steering | 华东师大 | 真实性和视觉感知用的是不同的 head 子集 → 分开动态 steering |
| NeurIPS'25 | ZoomText + Grounded Layer Correction | Trento 等 | 对场景文字注意力越强的层，越不容易产生语义幻觉 → 用这些层的表示来修正输出 |
| NeurIPS'26 | HALS (Do Robust LVLMs Hallucinate More?) | FAU | 提升对抗鲁棒性会让视觉特征同质化，反而加重幻觉 → 在解码器隐状态上 steer 离开同质化方向 |
| NeurIPS'26 | USteer | Qualcomm | 置信度不仅能用来检测幻觉，还能用来控制 → 用置信度对激活的梯度做免训练 steering |
| ICML'26 | RAUQ | MBZUAI | 某些 head 的循环注意力模式反映了不确定性 → 无监督、单次前向的幻觉检测 |

## 5. Steering / 安全

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| ICLR'26 | COLD-Steer | Georgia Tech, MIT | 少量样本微调的效果可以在推理时近似出来 → 用上下文样本的一步梯度动态来 steer 激活，不更新参数 |
| NeurIPS'26 | Inverted Detection and Control in Steering Vectors | Northeastern | 有些判别力很强的 steering 向量，加上去反而产生相反的行为 → 不用生成就能识别这类反向向量 |
| NeurIPS'26 | Minimally Invasive Steering | JHU | steering 前后每 token 的 KL 可二阶展开成 Fisher 二次型 → 在提升奖励的同时尽量少改基座行为 |
| CVPR'26 | ARGUS（间接提示注入防御） | — | MLLM 的“指令遵循”行为编码在一个子空间里 → 在表示空间 steer，防御跨模态注入 |
| ECCV'26 | MoRAS (Attention Misses Visual Risk) | SNU, UC | 多模态安全失败的原因之一是没注意到图里的危险区域 → 增强对风险区域的视觉注意力 |
| NeurIPS'26 | DualSteer | 南理工 | 越狱时注意力从安全推理转移到恶意对齐的 token 上 → 双空间 steering 恢复安全推理 |
| NeurIPS'25 | One Head to Rule Them All | 南理工 | LVLM 的安全能力集中在少数几个 head 上 → 只放大一个关键 head 就能防御 |
| NeurIPS'25 | SafePTR | 电子科大 | 不到 1% 的 token（在浅中层）导致了不安全行为 → 先删后恢复，免安全微调 |
| ICLR'26 | Multilingual Routing in MoE | UCLA, 复旦 | MoE 在浅层和深层按语言路由，中间层跨语言对齐 → 在中间层 steer 路由器激活英语常用的专家，提升多语言表现 |
| NeurIPS'26 | DISCA（文化对齐） | 胡志明市大学 | 国内不同人群的分歧（而非共识）才是 steering 信号 → 用 persona 分歧做免训练文化对齐 |

## 6. 量化 / 剪枝 / 合并 / 持续学习

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| NeurIPS'25 | LinearPatch | 清华, 华为 | 层剪枝后掉点主要来自剪枝接口处激活幅度不匹配 → Hadamard 加通道缩放的一个线性补丁 |
| NeurIPS'26 | Ghosted Layers | USC | 现有剪枝恢复方法只在受限的算子子空间里找解 → 在边界做无约束的激活对齐，闭式求解 |
| NeurIPS'25 | Lost Component Compensation | NTU | 剪枝丢掉的信息反映在注意力激活上 → 有选择地把这部分补回去 |
| ICML'26 | SINQ | 华为, ETH | 仅从权重矩阵结构就能恢复出预测激活大小的列缩放 → 免校准数据的双轴（Sinkhorn）归一化量化 |
| ICML'26 | OSAQ | 中科院自动化所, Berkeley | Hessian 在不同输入下有一致的低秩结构，某些方向一直很弱 → 把离群值“吸收”到这些方向里 |
| NeurIPS'26 | Quantization-Induced Flips Are Not Random | CEA 等 | 量化导致的预测翻转来自隐式正则、低通滤波和谱偏差 → 事后保留好的翻转、修正坏的翻转 |
| ICML'26 | ExpertWeaver | 浙大, 字节 Seed | 稠密模型 GLU 的激活模式里藏着一个天然的 MoE：常亮的通用神经元 + 动态的专门神经元 → 免训练地把稠密模型改造成 MoE |
| ICML'26 | SPON (Resting Neurons) | Stony Brook, IBM | 激活稀疏化掉点是因为破坏了预训练学到的输入依赖激活，导致分布漂移 → 加“自发神经元”对齐表示 |
| ICML'26 | HARC (When Model Merging Breaks Routing) | 中山大学 | 合并 MoE 时路由器会失效 → 用二阶曲率免训练地重新校准路由器 |
| ICML'26 | SVC (Spectral Over-Accumulation) | 南京大学 | 多个任务共享同一谱方向时，合并会让奇异值被过度累加 → 免数据地重新缩放奇异值 |
| ICML'26 | Core Subspace（预训练权重在合并中的作用） | 北交大 | 预训练权重在主奇异子空间里对多任务损失近乎最优 → 合并时保留这部分 |
| CVPR'26 | ACE-Merging | — | 在局部线性近似下，每个任务的输入协方差可以直接从它的微调更新里估出来 → 免数据的闭式合并 |
| ICML'26 | Prompt Reinjection（多模态 DiT） | 复旦 | 文本分支里 prompt 的语义随深度逐渐被遗忘 → 把浅层的 prompt 表示重新注入深层 |
| ICML'26 | KSTER (Reverse-Engineering Model Editing) | 上海期智, 清华 | 知识编辑的参数更新是低秩的，会泄露被编辑的数据 → 两阶段攻击从更新中恢复原数据 |

## 7. 扩散 LLM

| 会议 | 论文 | 机构 | 观察 → 方法 |
|---|---|---|---|
| ICML'26 | Early Decisions Matter | KAIST, LG AI | 基于置信度的解码有很强的邻近偏差：总是先解相邻位置 → 只引导早期的 token 选择 |
| NeurIPS'26 | TAPS (Denoising Time Matters) | UNC, NUS | 早期去噪步决定语义走向，后期步只决定措辞 → 在早期按时间退火加扰动来提升多样性 |
| ICML'26 | CORE（重掩码） | UCF, UMD | 静态置信度不可靠，应该看 token 对上下文扰动是否敏感 → 重掩码“上下文脆弱”的 token |
| ICML'26 | DyLLM | 首尔大学 | 各去噪步之间大部分 token 的表示稳定，只有少数显著 token 在变 → 只重算显著 token |
| ICLR'26 | Diffusion LM Knows the Answer Before It Decodes | 港理工 | 走到一半步数时答案就已经可以确定 → 提前收敛、提前停止 |
