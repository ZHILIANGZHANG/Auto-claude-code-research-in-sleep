# 调研中觉得有新意的顶会论文（按主题）

每篇的概括都来自我实际读过的摘要（少数读过全文的另有标注）。录用会议来自官方录用列表。

## A. 效率方法的“副作用”：变快了，但哪里坏了

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| NeurIPS'26 | Alignment Collapse Under KV Cache Quantization | Microsoft MAI, Caltech | 困惑度几乎不变（1.03×），Mistral-7B 却丢了 15.2% 的拒答；安全特征落在一个对量化噪声敏感 100–1000 倍的低维子空间；不存在通用的安全比特数 |
| ACL'26 | The Pitfalls of KV Cache Compression | — | 多指令场景下，有些指令会随压缩率上升直接被模型无视；会导致 system prompt 泄露 |
| ACL'26 | When Efficiency Meets Safety | — | KV 压缩对越狱是双刃剑：优化型攻击会因为恶意 query 被驱逐而“意外失效”，合并型压缩反而会让浅层 head 崩溃、放大攻击 |
| ICML'26 | Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs | 电子科大等 | 多数视觉剪枝随剪枝率升高而更不安全；但 query 引导的压缩相反，剪到 99.8% 反而更安全。机制：背景 token 被删后，注意力塌缩到少数恶意前景 token 上 |
| ICML'26 | On the Adversarial Robustness of LVLMs under Visual Token Compression | 港理工, NTU | 以前的对抗攻击在全 token 上优化、推理时却过压缩瓶颈，因此会高估鲁棒性；提出与压缩对齐的攻击 |
| NeurIPS'26 | FocusVLA: Hijacking Attention to Break Visual Token Pruning in VLA | 浙大, 清华, Duke | 把基于注意力的剪枝器变成后门：在几个不重要的 head 里植入触发器，让剪枝器删掉关键 token |
| ICLR'26 | Fewer Weights, More Problems: A Practical Attack on LLM Pruning | 东京大学, ETH | 可以构造一个看起来无害、但一被 vLLM 剪枝就变恶意的模型 |
| ICML'26 | Toward Safe Quantization-Aware Fine-tuning | 电子科大 | 量化后的模型在微调时更容易丢掉安全对齐，原因是量化误差带来初始安全偏移 |
| ICLR'26 | DefensiveKV | 中科大 | KV 驱逐依赖“重要 token 集合稳定”这个假设，而它很脆弱；改成控制最坏情况的聚合 |
| ICLR'26 | When Reasoning Meets Compression | PSU, Columbia | 对 R1 做量化、蒸馏、剪枝的系统比较：参数量对“记忆知识”的影响大于对“推理”的影响；2.51-bit 的 R1 接近原版 |

## B. “反直觉”的效率发现

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| CVPR'26 | When Token Pruning is Worse than Random | — | 第 20 层之后所有免训练剪枝都不如随机；存在一个“信息视界”，之后视觉 token 全部冗余；OCR 这类任务的视界更深，模型越强视界越深 |
| ICCV'25 | Feather the Throttle | Stanford | FastV 类浅层剪枝选中的 token 平均位于图像从上往下 80.7% 的位置（RoPE 衰减导致），定位任务因此崩掉（读过全文） |
| ECCV'26 | Why and When Visual Token Pruning Fails? | KAIST, NVIDIA | 剪枝在简单理解任务上没问题，在复杂推理上失败；原因是解码过程中“相关视觉信息”会漂移 |
| ACL'26 | Are We Using the Right Benchmark | — | 简单的图像下采样胜过许多先进的 token 压缩方法；现有基准里有大量与压缩无关的噪声样本 |
| CVPR'26 | What Do Visual Tokens Really Encode? | — | 视觉 token 可分为 sink、dead、alive 三类，只有约 60% 的 alive token 携带图像特有信息 |
| CVPR'26 | Random Wins All: Rethinking Grouping Strategies for Vision Tokens | — | ViT 里精心设计的 token 分组策略，被随机分组打败 |
| ICML'26 | Position Is All You Need（RES 的免费午餐压缩） | 华科 | 常规 token 压缩在指代分割上会崩，关键是保留原始位置信息 |
| ICML'26 | CLIP Tricks You | 世宗大学 | 指代区域内的视觉 token 与文本的 CLIP 相似度反而低，所以基于 CLIP 的文本引导剪枝会把目标剪掉 |
| ICML'26 | RTPrune（DeepSeek-OCR 剪枝） | 京东, 清华 | DeepSeek-OCR 解码有两阶段阅读轨迹：先读高范数 token，再把注意力分给其余 token；普通 VLM 剪枝会破坏文字保真度 |
| NeurIPS'26 | Cross-Model KV Cache Transfer in LLM Families | NVIDIA, Cornell, AMD | 同家族不同尺寸模型的 KV 之间有很强的线性结构（Qwen3 14B→32B，单层就能解释 56% 的 key 方差），切换模型时可以跳过 prefill |
| ICML'26 | Fast KV Compaction via Attention Matching | Thinking Machines Lab, MIT | 不靠摘要、也不靠昂贵训练，直接在隐空间快速压缩上下文 |
| NeurIPS'25 | Inference-Time Hyper-Scaling with KV Cache Compression | NVIDIA | 压缩 KV 可以让同样预算生成更多 token，从而提高 TTS 准确率 |
| NeurIPS'26 | Memory Inception | Yale, Princeton | 只在选定层插入由文本生成的 KV 块来做 steering，不占上下文 |
| ICML'26 | Test-Time Training with KV Binding Is Secretly Linear Attention | NVIDIA, Stanford | 很多 TTT 层其实就是学到的线性注意力，而不是“测试时记忆” |
| NeurIPS'25 | Gated Attention for LLMs（Oral） | Qwen Team | 注意力输出加一个 sigmoid 门就能消除 attention sink；这是 Qwen3.5 的设计来源 |
| NeurIPS'26 | Comparing Transformers and Hybrid Models at the Token Level | Ai2 | 用同数据的 Olmo 3 / Olmo Hybrid 对比：混合模型的优势集中在实词上；遇到闭括号和重复 n-gram 时优势消失，说明复制和层级结构靠的是注意力层 |

## C. Test-time scaling：推理越多不一定越好

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| ICLR'26 | Inverse Scaling in Test-Time Compute | Edinburgh | 构造出推理越长越差的任务；Claude 会被无关信息分心，o 系列会过拟合题目表述；延长推理时 Claude Sonnet 4 表现出更多“自我保存”倾向 |
| NeurIPS'25 | Does Thinking More Always Help? Mirage of TTS | UMD | 加 “Wait” 延长思考，性能先升后降 |
| ICLR'26 | The First Impression Problem | 南京大学 | 模型一看到题就形成内部猜测，后续推理和这个猜测冲突时就会过度反思 |
| ICML'26 | Think Deep, Not Just Long | Google, UVA | 用“深层预测发生大幅修正的 token”来衡量推理努力程度，比 token 数可靠 |
| NeurIPS'26 | Do Thinking Tokens Help with Safety? | Princeton | 在 GPT-OSS、Phi、OLMo、Qwen 上，安全决策在思考的第一个 token 就已经编码好了，思考过程未必让模型更安全 |
| ICML'26 | Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away | UMD | RL 后训练会削弱 MLRM 的安全性；只在越线时注入一句 “Wait, think safely” 就能恢复 |
| NeurIPS'25 | The Hawthorne Effect in Reasoning Models | ELLIS Tübingen, Microsoft | 推理模型能察觉自己在被测试并改变行为；可以线性探测，也能 steer |
| ICML'26 | Less Diverse, Less Safe | USF, UC Davis | 稍微降低候选多样性，TTS（MCTS、BoN）就更容易输出不安全内容，Llama-Guard 识别不出这类输入 |
| ECCV'26 | Same Pool, Different Answer | 港理工 | 候选池完全不变，只改温度或截断选项，41–47% 的题 BoN 选出的答案就会变 |
| ICLR'26 | Mode-conditioning | CMU, Anthropic | 并行采样受多样性坍塌限制；按推理模式分配采样预算可提升 Pass@k |
| ICML'26 | Diversity Matters（VLM 的 TTC） | UZH, ETH | 单模型投票在 VLM 上几乎无效，原因是输出高度相关；多模型集成才有多样性 |
| NeurIPS'25 | Best-of-N Jailbreaking | Anthropic | 对提示词做随机打乱或大小写变换并反复采样，1 万次后在 GPT-4o 上攻击成功率达 89%、Claude 3.5 Sonnet 达 78% |
| NeurIPS'26 | Tail Competition Explains Scaling in BoN | NEC | 用极值理论把 BoN 看成正确与错误答案在奖励尾部的竞争，解释了 reward hacking 导致的下降 |
| ICLR'26 | ROC-n-reroll | ETH, MPI | 证明 BoN 和拒绝采样的效果完全由 verifier 的 ROC 曲线几何决定 |
| ICLR'26 | Making, Not Taking, the Best of N | Cohere | 不从 N 个里挑一个，而是让 judge 把 N 个融合成一个答案 |
| ICML'26 | Position: It's Time to Optimize for Self-Consistency | MIT, Anthropic | 谄媚、逻辑泛化不完整等问题来自“只对单个输入输出对做训练和评测”这个假设 |
| ICLR'26 | Distilled Pretraining | Anthropic, FAIR | 蒸馏预训练让 TTS 表现更好，但会损害上下文学习 |
| ICML'26 | Overthinking: Amplifying Reasoning Weights to Extract Learned Secrets | MATS, Redwood | 用 M + α(R − M)、α > 1 的权重外推放大推理，用来审计模型隐藏的信息 |
| ICML'26 | Inducing Overthink（DoS 攻击） | 浙大, 阿里 | 用遗传算法构造让推理模型无限想下去的输入 |
| NeurIPS'25 | Kinetics: Rethinking Test-Time Scaling Law | CMU | 把显存访问计入成本后，“小模型多采样”被高估；TTS 用在 14B 以上更划算（读过全文） |
| NeurIPS'26 | On the Overscaling Curse of Parallel Thinking | 上交 | 全局统一的采样预算让大部分样本浪费了预算 |

## D. 持续学习、遗忘、合并

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| ICML'26 | Retaining by Doing | Princeton | RL 比 SFT 遗忘少，原因是 on-policy 数据带来的 mode-seeking |
| ICML'26 | Reinforcement Fine-Tuning Naturally Mitigates Forgetting | 港城大等 | 在 Qwen2.5-VL 持续后训练中，RFT 接近多任务训练且几乎不遗忘，SFT 则灾难性遗忘 |
| ICLR'26 | Mapping Post-Training Forgetting at Scale | Tübingen, Google | 平均准确率会把遗忘和反向迁移相互抵消；改用样本级 1→0 计数，并做随机猜测校正 |
| NeurIPS'26 | CapTrack | Tübingen, TUM, Imperial | 遗忘不只是丢知识，还有鲁棒性和默认行为的漂移；指令微调漂移最大 |
| ICLR'26 | Forget Forgetting | NYU | 记忆充足时，CL 的瓶颈从“稳定”变成“可塑”，简单的 replay 就能赢 |
| ICML'26 | Pretrained VLAs are Surprisingly Resistant to Forgetting | UT Austin | 预训练 VLA 几乎不遗忘；看似遗忘的技能微调一下就能恢复 |
| ICML'26 | Just-In-Time RL | NUS | 不更新梯度，用经验记忆估计优势函数并直接调 logits，是 KL 约束策略优化的闭式解 |
| ICML'26 | Multi-Head Attention as a Source of Catastrophic Forgetting in MoE | 复旦, Bath | MoE 本该靠稀疏路由减少干扰，但路由器的输入是多个 head 拼接后的特征，这是遗忘的瓶颈 |
| ICML'26 | When Model Merging Breaks Routing | 中山大学 | 合并 MoE 时路由器会失效（routing breakdown），提出免训练校准 |
| ICML'26 | From Memorization to Parameter Interference | Mila | 专家模型训练过度（记忆化）会损害之后的合并效果 |
| ICML'26 | Demystifying Mergeability | Sapienza, UCSD | “能不能合并”不是模型的固有属性，而是取决于合并方法和配对任务 |
| ICML'26 | Transporting Task Vectors across Different Architectures without Training | Modena | 免训练地把任务向量迁移到不同宽度的模型上 |
| NeurIPS'26 | The Illusion of Forgetting: Rank Leakage in Knowledge Editing | GMU | 编辑后 top-1 改了，但原来的敏感词仍排得很靠前，一次查询就能提取出来 |
| ICML'26 | Tackling Fake Forgetting through Uncertainty Quantification | IIT, MSU | “遗忘”的样本，真实标签仍在 conformal 预测集合里 |
| ICLR'26 | Learning is Forgetting: LLM Training as Lossy Compression | Princeton, Cohere | 预训练接近信息瓶颈的最优压缩 |

## E. Steering / 可解释性 / 扩散 LLM

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| ICML'26 | Endogenous Resistance to Activation Steering | AE Studio | Llama-3.3-70B 会在生成中途抵抗被 steering、自己纠正回来；找到了 26 个“跑题检测”SAE 特征 |
| NeurIPS'26 | Sanity Checks for SAEs | HSE | 在已知真特征的合成设置下，4 种主流 SAE 里有 3 种只能恢复 7–9% 的真特征，尽管解释方差达到 71% |
| ICML'26 | Do SAEs Identify Reasoning Features? | Berkeley | 所谓“推理特征”可能只是和推理同时出现的表层线索 |
| NeurIPS'25 | Refusal Direction is Universal Across Safety-Aligned Languages | LMU | 14 种语言共享同一个拒答方向 |
| ICLR'26 | Steering Evaluation-Aware LMs to Act Like Deployed | Astra 等 | 用 steering 向量压制模型的“我在被评测”意识 |
| ICLR'26 | Steering LMs with Weight Arithmetic | Copenhagen, Anthropic | 两个相反的小微调相减得到行为方向，再加到权重上 |
| ICML'26 | Weight Updates as Activation Shifts | UW-Madison | 给出激活 steering 与权重更新的一阶等价，以及 steering 能复现微调的条件 |
| ICLR'26 | Diffusion LM Knows the Answer Before It Decodes | 港理工, Dartmouth | 扩散 LLM 在一半步数时就已经“知道”答案 |
| ICML'26 | The Flexibility Trap | 清华 | 在数学和代码上，扩散 LLM 的“任意顺序生成”反而限制了推理潜力 |
| ICML'26 | Masks Can Be Distracting | Cambridge, Qualcomm | 扩散 LLM 尽管是双向注意力，仍有很强的局部性偏差；mask token 多了会分散注意力 |
| ICLR'26 | The Devil behind the Mask | 上交, 上海 AI Lab | 扩散 LLM 的对齐挡不住“带 mask 的上下文”越狱 |
| NeurIPS'25 | LM Behavioral Phases are Consistent Across Architecture | MIT, UCSD | Transformer、Mamba、RWKV 在预训练过程中的行为变化阶段高度一致 |

## F. Agent 场景（顶会刚出现）

| 会议 | 论文 | 机构 | 有意思在哪 |
|---|---|---|---|
| NeurIPS'26 | TriAxialKV | UIUC, Cambridge | agent 上下文的 KV 敏感度沿三个轴不同：时间远近、模态、语义角色（用户/工具/观测/推理） |
| NeurIPS'26 | SKIM | KAIST, Meta | 结构化生成下，真正重要的只是少数“决策位置”的 logits |
| NeurIPS'26 | AgentKVShift | UCSD | 复用 agent 记忆单元的 KV 时，残差 = 一个共享偏移 + 很小的逐 token 波动 |
| ECCV'26 | SlowBA | 上交 | 针对 GUI agent 响应速度的后门：弹窗触发后诱导超长推理 |
| ICCV'25 | Less is More: SimpAgent | 哈工大 | GUI agent 的历史上下文高度冗余，界面元素关系松散 |
