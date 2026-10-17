# Lens: Training-free / 少训练 LLM & MLLM 推理效率 —— 文献地图、结构性 gap 与候选 idea

日期：2026-10-17 | 子代理：ARIS idea-creator Phase 1 + 1.5 (efficiency lens)

## 0. 方法与可信度说明

- 共执行 36 次 WebSearch（中途触发会话搜索上限），查询覆盖 KV cache 驱逐/合并/量化、视觉 token 剪枝（图像/高分辨率/视频/流式）、投机/并行解码（含 dLLM）、PTQ/旋转量化/MLLM 量化、稀疏注意力/prefill、层跳过/早退/MoE 专家剪枝与卸载、reasoning 模型 over-thinking、多轮/agent/端侧场景、评测与诊断类论文。
- 本环境 arxiv.org / huggingface / openreview / alphaxiv 全部被 egress proxy 拒绝（CONNECT 403），WebFetch 亦 DNS 失败，因此**无法读取 abs/HTML 全文**；每篇论文的「问题/方法/数字」来自搜索引擎返回的 abstract 摘录（多为 arXiv abs 或会议页面的一手摘要），「limitation」一栏只记录摘要/相关工作中作者自己承认或被他人指出的，未能读 limitation section 的标注为「未读」。
- 所有 arXiv ID 都是搜索结果中直接出现的；2024 年以前的经典锚点（FastV、SparseVLM、VisionZip、PruneVid、H2O、SnapKV、PyramidKV、LayerSkip、Lookahead）在相关工作中被反复点名，ID 为本人确认过的公开 ID。

---

## 1. 论文清单（按子方向）

TF = training-free。日期为 arXiv 首发（或会议）。

### 1.1 KV cache 压缩 / 驱逐 / 合并 / 量化

| 论文 | ID / 会议 | 日期 | 问题 | 方法一句话 | 关键数字 | TF | 自认 limitation / 备注 |
|---|---|---|---|---|---|---|---|
| H2O / SnapKV / PyramidKV（锚点） | 2306.14048 / 2404.14469 / 2406.02069 | 2023–24 | prefill 后 KV 过大 | 累计注意力 / 观察窗注意力 / 逐层金字塔预算驱逐 | — | 是 | 后续论文反复指出：query-aware、依赖 attention score、与 FlashAttention 不兼容、长解码任务未评测 |
| KIVI（锚点） | [2402.02750](https://arxiv.org/abs/2402.02750) ICML'24 | 2024-02 | KV 量化 | K 按 channel、V 按 token 的 2-bit 非对称量化 | 峰值显存降 2.6× | 是 | — |
| KVzip | [2505.23416](https://arxiv.org/abs/2505.23416) NeurIPS'25 oral | 2025-05 | query-aware 驱逐在多 query / prefix 复用场景失效 | 用「模型能否从 cache 重建原文」做 query-agnostic 打分 | cache 3–4×↓，FlashAttention 解码延迟≈2×↓；query-aware 方法在 90% 预算下即退化 | 是（需一次重建前向） | 打分需额外 prefill 级前向；未读 |
| Inference-Time Hyper-Scaling (DMS) | [2506.05345](https://arxiv.org/abs/2506.05345) NeurIPS'25 | 2025-06 | 推理模型生成受 KV 读带宽限制 | 1K 步训练的延迟驱逐稀疏化，省下的内存换更多采样/更长推理 | Qwen-R1-32B 等读带宽下 AIME24 +12.0 | 否（1K 步） | 作者称优于 training-free 稀疏注意力 → 对 TF 路线的直接挑战 |
| LAVa | [2509.09754](https://arxiv.org/abs/2509.09754) EMNLP'25 Findings | 2025-09 | 预算如何在层与头间分配 | 统一的跨头可比指标 + 动态层/头预算 | LongBench/NIAH/RULER/InfiniteBench | 是 | 发现：层预算对生成类任务关键、头预算对抽取类关键（任务依赖） |
| Expected Attention | [2510.00636](https://arxiv.org/abs/2510.00636) | 2025-10 | 未来 query 的注意力不可得，FlashAttention 不暴露分数 | 假设激活分布，闭式估计未来 query 对每个 KV 的期望注意力；附 KVPress 库(20+方法) | prefill 与 decode 两阶段均优于基线；120k 上下文省 ~15GB（二手） | 是 | 依赖激活分布假设；KVPress 非部署引擎 |
| The Pitfalls of KV Cache Compression | [2510.00231](https://arxiv.org/abs/2510.00231) ACL'26 | 2025-09 | 多指令提示下压缩效果未评测 | 用 IFEval 评测 StreamingLLM/SnapKV/TOVA/H2O/K-Norm；系统提示泄露案例 | 某些指令被更快驱逐→被完全忽略；受方法、指令顺序、驱逐偏置影响 | — (诊断) | 只给出「简单修改驱逐策略」的初步修复 |
| Rethinking KV Compression for Serving | [2503.24000](https://arxiv.org/abs/2503.24000) MLSys'25 | 2025-03 | 压缩的内存节省是否变成吞吐 | 系统级评测 | 压缩会**拉长输出**→端到端延迟反升；FlashAttention/PagedAttention 未对压缩优化 | — (诊断) | 未解释为什么变长 |
| Physics of KV Compression | [2603.01426](https://arxiv.org/abs/2603.01426) | 2026-03 | 压缩为何/何时失效 | 把压缩视为注意力路由扰动；合成多实体/多跳任务 | 所有模型在 ~90% 压缩处出现幻觉悬崖，与 Global Eviction Ratio 尖峰同步；LLaMA 与 Qwen 头一致性动态不同 | — (诊断) | 合成任务；未给在线控制器 |
| LU-KV | [2602.08585](https://arxiv.org/abs/2602.08585) | 2026-02 | 头级预算 | 头分「即时贡献」vs「长程效用」，凸包松弛+边际效用贪心，含离线 profiling | — | 需离线 profiling | 是否严格 TF 未确认 |
| MixedDimKV | [2603.20616](https://arxiv.org/abs/2603.20616) | 2026-03 | 只驱逐整 token 太粗 | 按 token 分配保留维度，-H 版加头重要性 | 6.25% cache ≈ full attention (LongBench) | 需 profiling | 对比 HeadKV |
| VQKV | [2603.16435](https://arxiv.org/html/2603.16435v1) | 2026-03 | KV 量化 | 向量量化 KV | LLaMA-3.1-8B 压缩 82.8%，保 98.6% LongBench | 是 | 仅 8B，未测小模型/端侧 |
| GRKV | [2605.31105](https://arxiv.org/abs/2605.31105) | 2026-05 | 先驱逐后合并造成注意力失衡 | 岭回归把被驱逐 token 信息分配到保留 token，直接最小化压缩后注意力输出差 | LongBench/RULER | 是 | — |
| SelKV | [2607.16213](https://arxiv.org/abs/2607.16213) | 2026-07 | 合并 vs 丢弃按 token 决定 | 基于 V 相似度的软 cosine 门 + 解码期 logit 补偿 softmax 失衡 | 25% cache LongBench | 是 | — |
| RDKV / MoE-nD / HqeKV / CAPKV | [2605.08317](https://arxiv.org/abs/2605.08317) / [2604.17695](https://arxiv.org/abs/2604.17695) / ACL'26 Findings / [2604.25975](https://arxiv.org/abs/2604.25975) | 2026 | 驱逐 × 量化联合 | 率失真比特分配 / 逐层路由驱逐率与位宽 / 逐元素选择压缩动作 / 容量感知驱逐+HQQ 4bit | — | 大多 TF | 2026 上半年**密集涌现**，说明「联合」已成红海 |
| kvtc / Transform coding 视角 | [2511.01815](https://arxiv.org/abs/2511.01815) / [2608.14191](https://arxiv.org/abs/2608.14191) | 2025-11 / 2026-08 | 跨轮存储的 KV 压缩 | 变换编码；指出现有量化忽略误差经注意力传播 | kvtc 20× 压缩保持推理精度 | 是 | — |
| Compactor | [2507.08143](https://arxiv.org/abs/2507.08143) | 2025-07 | query-aware 与 prefix caching 不兼容 | 近似 leverage score 的 query-agnostic 驱逐 | — | 是 | 指出「驱逐多少不伤质量」几乎无先前工作 |
| Hold Onto That Thought | NeurIPS'25 WS; [2512.12008](https://arxiv.org/abs/2512.12008) | 2025-12 | 压缩方法多为 prefill 设计，长解码推理未评测 | 系统评测 | 没有单一策略通吃，强依赖数据集类型 | — | — |
| R-KV / RPC / ThinKV / BeaconKV / Moment-KV | [2505.24133](https://arxiv.org/abs/2505.24133) / [2505.13866](https://arxiv.org/abs/2505.13866) / [2510.01290](https://arxiv.org/abs/2510.01290) / [2609.04971](https://arxiv.org/abs/2609.04971) ICML'26 / [2605.29873](https://arxiv.org/abs/2605.29873) | 2025-05 → 2026-09 | 推理模型解码期 KV 爆炸（R1-8B 32K tokens → 4.1GB KV） | 冗余反思去重 / 近期 query 窗口打分（QwQ-32B 1.6× 吞吐、AIME -1.2） / 思维感知量化+驱逐 / 「beacon query」预测会被重访的远距 KV（5.8× 内存、>4.3× 吞吐，Qwen3-14B AIME24 +31.7pt@1024 预算） / 动量式解码期压缩 | 是 | BeaconKV 指出：依赖近期 query 的方法在长推理中失效（存在 Thought Revisiting Tokens）；ThinKV 指出激进量化会**拉长生成**抵消节省 |
| ST-Lite | [2603.00188](https://arxiv.org/abs/2603.00188) | 2026-03 | 长程 GUI agent 截图+动作历史使 cache 暴涨 | 空间+轨迹感知打分 | — | 是 | 仅 GUI agent |
| UltraQuant | [2606.20474](https://arxiv.org/abs/2606.20474) | 2026-06 | 上下文重的 agent：长前缀被多轮短 turn 复用 | 4-bit KV 量化研究 | — | 未确认 | — |
| Don't Waste Bits / SparKV / Persistent Q4 KV | [2604.04722](https://arxiv.org/abs/2604.04722) / [2604.21231](https://arxiv.org/abs/2604.21231) / [2603.04428](https://arxiv.org/abs/2603.04428) | 2026 | 端侧 KV | 学习式逐 token 位宽控制器（**非 TF**）/ 开销感知 KV 加载 / llama.cpp Q4 持久 KV 用于多 agent | — | 混合 | 端侧小模型 (0.5–3B) 的 TF 压缩几乎无人系统评测 |

### 1.2 视觉 token 剪枝 / 合并（图像、高分辨率、视频）

| 论文 | ID / 会议 | 日期 | 问题 | 方法 | 数字 | TF | limitation / 备注 |
|---|---|---|---|---|---|---|---|
| FastV / SparseVLM / VisionZip / PruneVid（锚点） | 2403.06764 ECCV'24 / 2410.04417 ICML'25 / 2412.04467 CVPR'25 / 2412.16117 ACL'25 | 2024 | 视觉 token 冗余 | 层内注意力剪枝 / 文本引导稀疏化 / 编码器端主导 token + 上下文合并 / 视频时空合并 | — | 是 | 后续被指出：位置偏置、attention-sink、深层无效 |
| Are We Solving the Right Problem? | [2502.11501](https://arxiv.org/abs/2502.11501) | 2025-02 | 剪枝设计是否真的有用 | 诊断 | 随机/池化的空间均匀选择常胜复杂设计；应报 latency 而非 FLOPs | — | — |
| HoloV | [2510.02912](https://arxiv.org/abs/2510.02912) NeurIPS'25 | 2025-10 | attention-first 保留语义相似 token，高剪枝比崩溃 | 预算按空间 crop 分配 | LLaVA-1.5 剪 88.9% 保 95.8% | 是 | — |
| Attention debiasing | [2508.17807](https://arxiv.org/abs/2508.17807) | 2025-08 | VLM 注意力偏向后位 token/图像下部/padding | 去除 recency bias 与 sink，得到位置无关重要性 | — | 是 | — |
| D²Pruner | [2512.19443](https://arxiv.org/abs/2512.19443) AAAI'26 | 2025-12 | 重要性法的位置偏置 | 去偏重要性 + 结构多样性 | — | 是 | — |
| When Token Pruning is Worse than Random | [2512.07580](https://arxiv.org/abs/2512.07580) CVPR'26 | 2025-12 | TF 剪枝是否优于随机 | 「vanishing token information」：用删 token 后输出概率变化度量 | ≥~20 层后现有方法不优于随机；DivPrune+随机 50% 剪枝保 Qwen2.5-VL-7B 96.9%（v1）/ DART+随机 93.9%（CVPR 版） | — | 对整个子方向的根本性质疑 |
| VisionTrim | [2601.22674](https://arxiv.org/abs/2601.22674) ICLR'26 | 2026-01 | 统一图像+视频 | 全局-局部主导 token 选择 + 文本引导补全合并 | — | 是 | — |
| ForestPrune | [2603.22911](https://arxiv.org/abs/2603.22911) | 2026-03 | 视频 MLLM 高比例剪枝 | 跨帧语义/空间/时间约束的 token forest | — | 是 | — |
| Unified Spatiotemporal Token Compression | [2603.21957](https://arxiv.org/abs/2603.21957) | 2026-03 | 10% 保留下 32 帧仍剩数百 token | 单阶段时空联合压缩 | — | 是 | — |
| FLoC | ICLR'26 (mlanthology cho2026iclr-floc) | 2026 | 长视频 | facility-location + lazy greedy，query-agnostic、模型无关 | Video-MME/MLVU/LongVideoBench/EgoSchema 均超 SOTA | 是 | — |
| InfiniPot-V | [2506.15745](https://arxiv.org/abs/2506.15745) NeurIPS'25 | 2025-06 | 流式视频 KV 线性增长 | 硬内存上限 + 时间冗余(TaR) + 值范数(VaN) 排序 | 峰值显存 -94%，多轮对话保持精度 | 是 | 端侧目标但实验在 GPU；未读 |
| METok | [2506.02850](https://arxiv.org/abs/2506.02850) | 2025-06 | 长视频 | 多阶段事件级压缩 | — | 是 | — |
| ERASE / HAWK / SepPrune / Structured Redundancy | [2605.09982](https://arxiv.org/abs/2605.09982) / CVPR'26 / 2026-07 / [2607.23046](https://arxiv.org/abs/2607.23046) | 2026 | Qwen2.5-VL 高分辨率 | 两阶段剪枝（85% 剪枝保 89.5% vs 先前 78.1%）/ 头重要性感知（剪 80.2% 保 96.0%）/ 分隔符 token 作统一 query / 结构化冗余建模（报告 CDPruner/DivPrune/D²Pruner 选择开销） | — | 是 | 都未涉及多图交错输入 |
| TOPS | [2606.27161](https://arxiv.org/abs/2606.27161) | 2026-06 | 第一性原理重述剪枝 | 任务相关性+信息覆盖+语义多样性的「最优保留集」 | LLaVA-NeXT 剪 77.8% 保 ~100%；7 backbone 14 benchmark | 是 | 以图像为主 |
| IPCV | [2512.18747](https://arxiv.org/abs/2512.18747) | 2025-12 | 在 ViT 内压缩 | 邻居引导重建让被剪 token 参与注意力，进 LLM 前恢复 | 可与 LLM 侧剪枝叠加 | 是 | — |
| QAPruner / Joint Q+P | [2604.02816](https://arxiv.org/abs/2604.02816) / [2604.17320](https://arxiv.org/abs/2604.17320) | 2026-04 | 剪枝 × PTQ 交互 | 语义剪枝会删掉激活离群 token → W4A4 误差上升；混合敏感度打分 / FlatQuant 扩到视觉塔+投影器 W4A4+KV4 | 12.5% 保留下 +2.24%，超过未剪枝稠密量化 | 是 | 自称首个联合优化 |

### 1.3 多模态 KV cache

| 论文 | ID | 日期 | 要点 | TF |
|---|---|---|---|---|
| LOOK-M | [2406.18139](https://arxiv.org/abs/2406.18139) | 2024-06 | 剪视觉 KV 并把驱逐 token 合并进保留 token；被 MadaKV 批评为固定模态优先级 | 是 |
| MadaKV | [2506.15724](https://arxiv.org/abs/2506.15724) ACL'25 | 2025-06 | 头有模态偏好 → 按头模态自适应驱逐；解码延迟 1.3–1.5× | 是 |
| FlashCache | [2511.16786](https://arxiv.org/abs/2511.16786) CVPR'26 | 2025-11 | 自称首个 attention-score-free、FlashAttention 兼容的多模态 KV 压缩：频域低频能量主导，远离主能量的「Outlier KV」必须保留；1.69× 解码、-80% KV | 是 |
| RetentiveKV | ACL'26 Findings | 2026 | 视觉 token 的重要性可能**在解码后期才出现**，「重要性持续」假设对视觉不成立；熵打分 + 状态空间记忆可重激活；5.0× KV、1.5× 解码 | 未确认 |

### 1.4 投机 / 并行解码（无训练 draft）

| 论文 | ID / 会议 | 日期 | 要点 | TF |
|---|---|---|---|---|
| Lookahead Decoding / LayerSkip（锚点） | 2402.02057 / 2404.16710 | 2024 | n-gram 并行猜测 / 早退自投机（需训练） | 是 / 否 |
| Training-free alignment-augmented SD | [2505.13204](https://arxiv.org/abs/2505.13204) | 2025-05 | 用目标模型 prefill 分布生成更对齐的 draft + 灵活验证阈值 | 是 |
| SpecExtend | [2505.20776](https://arxiv.org/abs/2505.20776) ACL'26 Findings | 2025-05 | 长序列 SD 的 drop-in 增强：用目标注意力选择 draft 保留的上下文；16K 摘要 2.84×、长推理 3.86× | 是 |
| FLy (Loosely SD) | [2511.22972](https://arxiv.org/abs/2511.22972) | 2025-11 | 放松验证：熵门 + token 级延迟窗口接受语义等价变体，跨模型域无需重调 | 是 |
| SparseSpec (reasoning) | [2512.01278](https://arxiv.org/abs/2512.01278) MLSys'26 | 2025-12 | 大规模推理模型稀疏自投机，2.13× 吞吐 | 是 |
| Practical Investigation of Relaxed SD | [2607.08690](https://arxiv.org/abs/2607.08690) | 2026-07 | 基准：多数 relaxed 方法依赖 drafter 本身是好 LM，不适合轻量 MTP 头 | — |
| Progressive Tree Drafting | [2607.10661](https://arxiv.org/abs/2607.10661) COLM'26 | 2026-07 | 综述内生 TF 草拟（Lookahead、Self-Draft 等） | 是 |
| SparseSpec-L | [2607.27735](https://arxiv.org/abs/2607.27735) | 2026-07 | 目标模型用稀疏化、可召回的 KV 自草拟；复用验证期注意力统计做重要性信号；熵控制猜测长度；≤2.79×；猜测过长在接受率低于相对草拟成本时反而变慢 | 是 |
| DraftExpert | [2607.24434](https://arxiv.org/abs/2607.24434) | 2026-07 | 专家卸载 MoE 的自投机：瓶颈是「专家扩张」而非 draft 长度 | 是 |
| MMSpec + ViSkip | [2603.14989](https://arxiv.org/abs/2603.14989) | 2026-03 | 600 样本 6 类 10 种 SD 方法基准：文本 SD 在多模态退化（Lookahead 在 Qwen2.5-VL-7B 仅 1.07×，部分子任务更慢）；batch 越大视觉感知越重要；吞吐 speedup 不等于延迟；ViSkip 视觉相关性高时关闭投机（二手称 TF） | 是 |
| Fast-dLLM / Sparse-dLLM | [2505.22618](https://arxiv.org/abs/2505.22618) ICLR'26 / [2508.02558](https://arxiv.org/abs/2508.02558) | 2025 | dLLM 块级近似 KV cache + 置信阈值并行解码，≤27.6× 吞吐 / 动态 cache 驱逐再 10× | 是 |

### 1.5 PTQ / 旋转 / 无数据量化 / MLLM 量化

| 论文 | ID / 会议 | 日期 | 要点 | 校准需求 |
|---|---|---|---|---|
| QuaRot / SpinQuant（锚点） | [2404.00456](https://arxiv.org/abs/2404.00456) NeurIPS'24 / [2405.16406](https://arxiv.org/abs/2405.16406) ICLR'25 | 2024 | Hadamard 旋转去离群（6/8-bit 无校准无损，4-bit 需 GPTQ）/ 学习旋转（随机旋转间差距达 13 点） | 无 / 有 |
| SingleQuant | [2511.22316](https://arxiv.org/abs/2511.22316) | 2025-11 | 闭式确定性旋转替代梯度优化，W4A4 强且量化时间大降 | 低 |
| OptRot | [2512.24124](https://arxiv.org/abs/2512.24124) | 2025-12 | data-free：最小化旋转后权重四阶矩去权重离群；权重量化胜 SpinQuant/OSTQuant，但 W4A4 存在权重-激活折中 | 无 |
| SERQ | [2603.08185](https://arxiv.org/abs/2603.08185) | 2026-03 | 显著性感知低秩误差重建；附录列 SERQ 与 QuaRot 为仅有的 TF 选项，FlatQuant 最准但重优化 | 低 |
| ReSpinQuant | [2604.11080](https://arxiv.org/abs/2604.11080) | 2026-04 | 层级子空间残差旋转近似（非 TF） | 有 |
| Block Rotation for MXFP4 | [2511.04214](https://arxiv.org/abs/2511.04214) | 2025-11 | 块旋转适配 MXFP4 微缩格式 | 低 |
| Q-VLM / LUQ / SplitQ | [2410.08119](https://arxiv.org/abs/2410.08119) NeurIPS'24 / [2509.23729](https://arxiv.org/abs/2509.23729) / [2605.19929](https://arxiv.org/abs/2605.19929) | 2024–26 | 首个 LVLM W4A4 / 首个 MLLM <4bit 研究 / 文本与视觉激活分布异质→通道拆分，Qwen2.5-VL-7B W4A4 73.5 vs FP16 74.3 | 有 |

### 1.6 稀疏注意力 / prefill / 提示压缩

| 论文 | ID / 会议 | 日期 | 要点 | TF |
|---|---|---|---|---|
| MInference（锚点） | [2407.02490](https://arxiv.org/abs/2407.02490) | 2024-07 | A-shape / Vertical-Slash / Block-Sparse 三模式 | 是 |
| The Sparse Frontier | [2504.17768](https://arxiv.org/abs/2504.17768) ACL'26 Findings | 2025-04 | 6 方法、128K、稀疏 0.95、9 任务：大稀疏模型优于小稠密；prefill 中细粒度 per-query 选择不可行；**越长序列能容忍越高稀疏 → 固定预算在生产中次优** | 是 |
| TCA-Attention | [2512.09238](https://arxiv.org/abs/2512.09238) | 2025-12 | 离线头稀疏预算 + 在线 token 选择，128K 2.8×、KV -61% | 是 |
| BFLA / RBS-Attention | [2605.12193](https://arxiv.org/abs/2605.12193) / [2609.20971](https://arxiv.org/abs/2609.20971) | 2026 | 块过滤 Triton 映射 / 针对 centroid 选块的「mean dilution」加 rescue 分支 | 是 |
| LLMLingua / LongLLMLingua（锚点） | [2310.05736](https://arxiv.org/abs/2310.05736) / [2310.06839](https://arxiv.org/abs/2310.06839) | 2023 | 小模型困惑度丢 token，≤20× | 是 |
| Perception Compressor | NAACL'25 Findings | 2025 | 引导问题检索示例 + 双斜率比例分配 + 迭代 token 压缩 | 是 |
| Similarity→Structure | [2604.23277](https://arxiv.org/abs/2604.23277) | 2026-04 | 混合图先验的句级压缩，对标 LLMLingua-2 | 是 |

### 1.7 层跳过 / 早退 / MoE

| 论文 | ID / 会议 | 日期 | 要点 | TF |
|---|---|---|---|---|
| Diminishing Returns of Early-Exit | [2603.23701](https://arxiv.org/abs/2603.23701) | 2026-03 | 新一代模型层冗余更低，早退收益逐代下降；稠密 > MoE/SSM；>20B 与 base 模型潜力更高；提出可早退性指标与基准 | 诊断 |
| River-LLM / WAVE | ACL'26 long.1746 / ICML'26 | 2026 | token 级早退补 KV 缺口，1.71–2.16× / 离线校准退出窗口 + 代理 LM 头，Llama-2-7B ≤1.4× | 是 |
| ACE | [2609.05228](https://arxiv.org/abs/2609.05228) | 2026-09 | MoE 无校准 token 自适应专家跳过，双视角都判低贡献才跳；50% 跳过最优 | 是 |
| Dropping Experts, Recombining Neurons / COMPEL / STUN | EMNLP'25 F / ACL'26 F / ACL'25 | — | 路由统计剪专家再神经元级重组 / Fisher 层感知专家剪枝 / 结构+非结构剪枝 | 部分 |
| MoBiLE / FloE | [2510.12357](https://arxiv.org/abs/2510.12357) / [2505.05950](https://arxiv.org/abs/2505.05950) | 2025 | 消费级 GPU 卸载：不重要 token 专家数减半，1.60–1.72× / 隐藏专家 I/O，指出超低比特专家量化伤质量 | 是 |

### 1.8 推理模型 over-thinking（仅效率视角）

| 论文 | ID / 会议 | 要点 | TF |
|---|---|---|---|
| Dynamic Early Exit (DEER) | [2504.15895](https://arxiv.org/abs/2504.15895) ICLR'26 | 在推理转折点试答，置信即停；10 基准 11 模型 CoT 长度 -19.1%～-80.1%，准确率 +0.3～5.0 | 是 |
| Confidence-Coverage Gating | NeurIPS'25 WS | 换行处低置信连接词 + 与提示语义近即截断 | 是 |
| Activation Steering for CoT Compression | NeurIPS'25 WS | 50 对样本得 steering 向量，MATH500/GSM8K ≤-67.4% | 是(少样本) |
| Memory Augmentation for CoD | [2608.21265](https://arxiv.org/abs/2608.21265) | 历史轨迹构建可复用推理记忆作 prefill 脚手架 | 是 |

---

## 2. Landscape map：饱和 vs 起步

**已经做烂（红海，再做纯方法很难过审）**
1. 单图 LLaVA-1.5 上的 TF 视觉 token 剪枝（FastV→SparseVLM→VisionZip→HoloV→D²Pruner→TOPS…每月数篇；且 2512.07580/2502.11501 证明深层不如随机）。
2. prefill 阶段 KV 驱逐 + 层/头预算分配（SnapKV/PyramidKV/LAVa/LU-KV/MixedDimKV/HeadKV），LongBench/RULER 上已接近饱和。
3. KV 驱逐 × 量化联合（RDKV/MoE-nD/HqeKV/CAPKV/ThinKV/EVICPRESS，2026 上半年集中爆发）。
4. 旋转式 PTQ 的「又一个旋转」（SpinQuant/SingleQuant/OptRot/ReSpinQuant/ButterflyQuant）。
5. dLLM 的 KV cache + 并行解码（Fast-dLLM 系）。

**半成熟（有主线但尚有空位）**
- 推理模型解码期 KV 压缩（R-KV/RPC/ThinKV/BeaconKV/Moment-KV）：方法多，但「压缩导致输出变长」的机制未解释。
- 长视频 / 流式视频 token 压缩（FLoC、InfiniPot-V、ForestPrune）：query-agnostic 与硬内存上限刚起步。
- 多模态 KV 压缩（MadaKV/FlashCache/RetentiveKV）：FlashAttention 兼容与「视觉重要性迟发」刚被提出。
- TF 自投机（SparseSpec-L、DraftExpert）：把 KV 稀疏当 draft 的思路刚出现。

**刚起步（蓝海）**
- MLLM/VLM 上的 TF 投机解码（MMSpec 证明文本法失效，仅 ViSkip 一个门控）。
- 多轮 / agent / 工具调用场景的 TF 压缩（仅 ST-Lite GUI、UltraQuant 量化、Pitfalls 诊断）。
- 端侧小模型 (≤3B) 的 TF KV 压缩/量化评测（现有工作全在 7–14B）。
- 剪枝/压缩方法在量化模型上的迁移性（QAPruner 只做了 token 剪枝×PTQ，KV 驱逐×权重量化空白）。
- 诊断类：压缩「为什么」失败的物理（Physics 2603.01426、2512.07580 刚开头）。
- 多图交错 / 高分辨率多图输入的剪枝（2607.23046 等均未覆盖）。
- 早退在现代模型上的「可早退性」预测（2603.23701 刚提出指标）。

---

## 3. 五个 lens 的结构性 gap

### Lens A：Method-transfer
- **A1. query-agnostic「重建式」打分 → 视觉 token / 多模态 KV。** KVzip (2505.23416) 用「能否重建上下文」做 query-agnostic 打分并因此兼容 prefix 复用；Compactor (2507.08143) 指出 query-aware 驱逐与多 query prefix caching 根本不兼容。视觉侧所有主流剪枝 (FastV/SparseVLM/VisionZip/VisionTrim) 要么 query-aware 要么静态；RetentiveKV (ACL'26) 又指出视觉 token 重要性会「迟发」。→ 没有人做「一次压缩、多问题复用」的视觉 KV 打分。
- **A2. 期望注意力（闭式未来 query 分布）→ 模态条件化。** Expected Attention (2510.00636) 假设单一激活分布；MadaKV (2506.15724) 证明头对模态有不同偏好；FlashCache (2511.16786) 绕开注意力分数走频域。→ 「按模态/按头拟合 query 分布」的 FlashAttention 兼容多模态驱逐未被尝试。
- **A3. 稀疏 KV 自草拟 (SparseSpec-L 2607.27735) → VLM。** VLM 的 KV 绝大部分是视觉 token，而 MMSpec (2603.14989) 显示文本 SD 方法在 VLM 上接近失效 (Lookahead 1.07×)。视觉 token 剪枝已证明可在深层大幅删减而质量不变 → 天然的「草拟模型」，但无人把「剪枝后的自己」当 drafter 做无损验证。
- **A4. 放松验证 (FLy 2511.22972) → 内生 n-gram/lookahead draft。** 2607.08690 指出 relaxed 方法依赖 drafter 是好 LM；内生 draft（n-gram）不是 LM，于是被排除。但 2505.13204 用目标模型 prefill 分布作对齐信号，可作为 n-gram draft 的语义接受 oracle——未组合。

### Lens B：Contradiction
- **B1. 「压缩省内存」vs「压缩拉长输出、延迟反升」。** MLSys'25 2503.24000 与 ThinKV 2510.01290 都报告压缩/量化使生成变长，而几乎所有方法论文只报 token-per-second 或峰值显存。机制（丢失了「结论/停止」线索？重复循环？）无人解释，也无 TF 修复。
- **B2. 「attention-based 剪枝优于基线」vs「深层不如随机」。** 2512.07580 (CVPR'26) 与 2502.11501 对整个视觉剪枝子方向的实验结论构成直接矛盾（HoloV/TOPS/ERASE 等仍报告大幅领先）。差异可能来自剪枝层位置、是否报 latency、随机基线实现。需要控制变量的公平复现。
- **B3. 「固定预算」vs「长序列可承受更高稀疏」。** Sparse Frontier 2504.17768 指出固定预算在生产中次优，Physics 2603.01426 指出 ~90% 处有幻觉悬崖；但 SnapKV/PyramidKV/LAVa 等仍用固定 budget 比较。方法与评测设定互相矛盾。
- **B4. 早退「有效」(River-LLM 2.16×, WAVE 1.4×) vs 「收益逐代递减」(2603.23701)。** 前者仍在 Llama-2/早期模型上报结果；后者说 Qwen3/新一代模型层冗余更低。

### Lens C：Untested assumption
- **C1. 在 FP16 上设计/调参的 KV 驱逐打分可以直接用于 W4A4 旋转量化模型。** QuaRot/SpinQuant 的 Hadamard 旋转改变激活的分布形状（Expected Attention 的闭式假设、K-Norm 的范数启发、FlashCache 的频域能量都依赖该分布）。QAPruner 2604.02816 只在「视觉 token 剪枝 × PTQ」上证明了交互存在，KV 驱逐 × 权重量化完全未验证。
- **C2. 「token 重要性在解码期是持续的」。** BeaconKV (Thought Revisiting Tokens) 与 RetentiveKV（视觉迟发重要性）各自在文本推理和视觉上推翻了它，但主流 prefill 驱逐方法都隐含此假设；两者都没有量化「重要性回流」的频率与可预测性。
- **C3. 「随机/均匀剪枝只是弱基线」。** 2512.07580 说明深层 attention 分数不再携带信息；视频方向 (ForestPrune/FLoC/METok) 从未与「均匀时间采样 + 随机空间」做公平对比。
- **C4. 「推理模型的 CoT 压缩/早停对下游 KV 压缩中性」。** DEER 类早停和 R-KV 类解码期驱逐从未联合评测；早停改变生成长度分布，会改变驱逐预算的最优点。

### Lens D：Scaling-regime
- **D1. 端侧小模型 (0.5–3B)。** 所有 TF KV 压缩/量化结果在 7B–32B；2604.04722 转向训练控制器；小模型头更少、冗余更低，Physics 论文显示 LLaMA/Qwen 动力学不同——小模型的「幻觉悬崖」位置未知。
- **D2. agent / 工具调用 / 多轮。** Pitfalls 2510.00231 证明多指令提示下压缩会「选择性遗忘」指令；ST-Lite 仅 GUI；工具 schema（JSON 长前缀）+ 多轮观测的 TF 压缩无人做，且无对应基准（BFCL/τ-bench 未被用于压缩评测）。
- **D3. 多图交错 / 高分辨率多图。** 2026 的高分辨率剪枝 (ERASE/HAWK/SepPrune/2607.23046) 全是单图；跨图 token 预算分配未研究。
- **D4. 超长视频 + 多轮问答 + 硬内存上限。** InfiniPot-V 做了流式硬上限，FLoC 做了 query-agnostic 长视频，但「被压掉的帧在后续 turn 被问到」的召回（类 SparseSpec-L 的 recall / RetentiveKV 的重激活）在视频上无人做。
- **D5. MoE + 长上下文 + 压缩。** 2603.23701 指出 MoE 早退潜力更低；ACE 做专家跳过；但 KV 压缩在 MoE（Qwen3-MoE/DeepSeek-V3 系列 MLA）上的表现几乎空白——MLA 的压缩潜变量本身是否可再驱逐？

### Lens E：Diagnostic
- **E1. 压缩导致「输出变长」的机制。** 是否因驱逐了 EOS/结论锚点、破坏了重复检测、还是降低了置信度触发了更多 self-check？可用 Physics 的 Global Eviction Ratio 与 token 级 logit 追踪回答。
- **E2. 「重要性回流」的可预测性。** 多少比例的被驱逐 token 会在 N 步后被重新高注意？这与任务类型（推理 vs 抽取，LAVa 已暗示任务依赖）、模型家族（Physics：LLaMA 早一致晚分化 vs Qwen 晚收敛）如何相关？
- **E3. 随机剪枝的「信息地平线」是否随分辨率/帧数/模型代际移动？** 2512.07580 只在 LLaVA/Qwen2.5-VL 单图上测了层位置；视频与高分辨率可能更早到达地平线。
- **E4. 投机解码的 speedup 与延迟脱节 (MMSpec)、以及与 KV 压缩的交互。** 压缩 KV 会改变 target 分布→接受率；SparseSpec-L 用稀疏 KV 当 drafter 恰说明二者相关，但「压缩后的 target + 任意 drafter」的接受率退化无人量化。

---

## 4. 候选 idea（idea-creator schema）

### Idea 1：Why Compression Makes Models Talk Longer — 诊断并训练无关地修复 KV 压缩导致的生成膨胀
- **Method**：(1) 在 DeepSeek-R1-Distill-Qwen-7B / Qwen3-8B 上跑 SnapKV、R-KV、KIVI-2bit、Expected Attention，记录每个样本的生成长度分布、重复 n-gram 率、答案出现后继续生成的 token 数；(2) 用 token 级归因（删除哪些 KV 导致长度膨胀）定位是「结论锚点被驱逐」还是「自检循环」；(3) 提出 TF 修复：对「答案候选/结论标记」及其邻域 KV 加保护位，并用重复熵触发器配合 budget forcing 提前收尾；(4) 以端到端延迟（非 token/s）汇报。
- **Hypothesis**：压缩拉长输出主要源于早期计划/结论锚点被驱逐导致的重复自检，保护 <2% 的锚点 KV 可消除 >70% 的长度膨胀。
- **MVE**：模型 R1-Distill-Qwen-7B、Qwen3-8B；数据 AIME24/25、MATH500、LiveCodeBench 子集、LongBench 生成类；指标 准确率、平均长度、端到端延迟、峰值显存；单卡 A100-80G，约 40–60 GPU-h。
- **Closest work**：2503.24000（发现现象未解释）、ThinKV 2510.01290（量化拉长，提出思维感知但未归因）、BeaconKV 2609.04971（重访 token，但目标是精度非长度）。不同点：第一个把「长度膨胀」作为一等目标并给出因果归因 + TF 修复。
- **Novelty 7/10 | Risk MEDIUM | Contribution：diagnostic + method**
- **Reviewer objection**：现象是否在非推理模型上也存在；锚点保护可能等价于已有的「保留首尾 token」技巧。
- **dedup_key**：kv-compression-output-length-inflation-anchor-protection

### Idea 2：ReconVis — 一次压缩多问复用的 query-agnostic 视觉 KV 驱逐
- **Method**：(1) 图像/视频 prefill 后，用 MLLM 自身以「描述这张图/这段视频」为 pseudo-query 做一次重建前向，取重建注意力作为视觉 KV 的 query-agnostic 重要性（KVzip 思路迁移）；(2) 对低分视觉 KV 做 GRKV 式岭回归合并而非直接丢弃，保留「迟发重要性」通道；(3) 压缩后的 cache 作为 prefix 被同一图像的多个问题复用；(4) 对比 query-aware 剪枝在多问设置的退化。
- **Hypothesis**：在「同图多问」设置下，重建式 query-agnostic 打分在 20% 视觉 KV 预算下比 VisionZip/SparseVLM 等 query-aware 方法高 ≥5 点，并允许 prefix cache 命中。
- **MVE**：Qwen2.5-VL-7B、LLaVA-OneVision-7B；数据 MMBench/SEED（同图多问重组）、VisDial、Video-MME 多问子集；指标 准确率@预算、prefix 复用率、解码延迟；单卡 48–80G，约 30 GPU-h。
- **Closest work**：KVzip 2505.23416（文本）、RetentiveKV ACL'26（重激活但疑似需训练）、FlashCache 2511.16786（attention-free，但 query 无关性未针对多问评测）。
- **Novelty 7/10 | Risk MEDIUM | Contribution：method**
- **Reviewer objection**：重建前向增加一次 prefill 成本；多问基准是重组而非原生。
- **dedup_key**：query-agnostic-visual-kv-reconstruction-multiquery

### Idea 3：Depth-Switch Pruning — 根据「信息地平线」自动在注意力剪枝与均匀剪枝之间切换
- **Method**：(1) 无标签估计每个 MLLM 的「地平线层」：逐层测删 token 对输出分布的 KL 变化并找拐点（2512.07580 的度量，改为校准集 50 张图即可）；(2) 地平线前用去偏注意力（2508.17807）剪枝，地平线后改用空间/时间均匀 + 随机保留；(3) 在高分辨率与视频上重复，检验地平线是否前移；(4) 报告 latency 而非 FLOPs。
- **Hypothesis**：地平线层随输入 token 数增加而前移；混合策略在 ≥80% 剪枝时优于单一策略且几乎零额外开销。
- **MVE**：LLaVA-1.5-7B、Qwen2.5-VL-7B、LLaVA-Video-7B；数据 TextVQA/DocVQA（高分辨率）、Video-MME；单卡 24–48G，约 20 GPU-h。
- **Closest work**：2512.07580（诊断，未给出自适应方法）、2502.11501（主张均匀）、HoloV 2510.02912（空间 crop 预算，但全层统一）。
- **Novelty 5/10 | Risk LOW | Contribution：method + diagnostic（可作为 workshop/短文）**
- **Reviewer objection**：增量式；地平线可能与特定 benchmark 耦合。
- **dedup_key**：depth-adaptive-random-vs-attention-visual-pruning

### Idea 4：Rotation-Aware KV Eviction — 检验并修复 KV 驱逐打分在 W4A4 旋转量化模型上的失效
- **Method**：(1) 取 Llama-3.1-8B / Qwen2.5-7B 的 QuaRot/SpinQuant/OptRot W4A4(+KV4) 版本，叠加 SnapKV、K-Norm、Expected Attention、FlashCache 风格打分；(2) 测量打分排序与 FP16 版本的 Spearman 相关及最终精度差；(3) 若失效，提出在旋转空间重新估计 query 分布/范数统计（闭式、无需校准数据）的修正；(4) 给出「量化 × 驱逐」的组合推荐表。
- **Hypothesis**：Hadamard 旋转使 K 范数与注意力分布趋于各向同性，范数/分布类打分 (K-Norm、Expected Attention) 在 W4A4 下相关性显著下降，而观察窗类 (SnapKV) 相对稳健。
- **MVE**：上述模型；LongBench、RULER 32K、GSM8K；指标 精度、排序相关、显存；单卡 80G，约 25 GPU-h（量化模型可用公开权重）。
- **Closest work**：QAPruner 2604.02816（视觉 token 剪枝 × PTQ，证明交互存在）、2604.17320（FlatQuant 扩展到 VLM，无驱逐）、RDKV/HqeKV（KV 量化×驱逐，但权重为 FP16）。不同点：首次系统评测权重旋转量化与 KV 驱逐的交互。
- **Novelty 6/10 | Risk LOW | Contribution：diagnostic + fix**
- **Reviewer objection**：若交互很小，贡献变成「负结果」；需保证量化实现一致。
- **dedup_key**：kv-eviction-under-rotation-quantization

### Idea 5：PruneDraft — 用「剪掉视觉 token 的自己」做 VLM 的无训练无损自投机
- **Method**：(1) 草拟阶段用深层视觉 token 剪枝（仅保留 10–20%）+ 浅层早退的同一模型生成 k 个 token；(2) 验证阶段用全 KV 并行校验（标准 SD 无损）；(3) 用 ViSkip 式视觉相关性门控与 SparseSpec-L 式熵控制器自适应 k；(4) 在 MMSpec 六类任务上报 speedup 与 p99 延迟。
- **Hypothesis**：因视觉 token 占 VLM KV 的 >80% 且深层信息「消失」，剪枝版自草拟接受率 >0.8，整体 1.6–2.2× 加速，优于 Lookahead (1.07×) 且无需训练。
- **MVE**：Qwen2.5-VL-7B、LLaVA-1.5-7B；MMSpec 基准 + Video-MME 短答；指标 walltime speedup、接受长度、p99；单卡 48–80G，约 30 GPU-h。
- **Closest work**：SparseSpec-L 2607.27735（文本、稀疏 KV 草拟）、MMSpec/ViSkip 2603.14989（门控不草拟）、LayerSkip（需训练）。不同点：视觉 token 剪枝作为 draft 机制是新的，且利用了 2512.07580 的发现。
- **Novelty 7/10 | Risk MEDIUM | Contribution：method**
- **Reviewer objection**：草拟仍需跑全部层，加速上限受限；与 EAGLE 类训练 drafter 相比速度可能不占优。
- **dedup_key**：visual-token-pruned-self-speculative-decoding-vlm

### Idea 6：ToolKV — 面向工具调用 agent 的指令/schema 保护型 KV 压缩与基准
- **Method**：(1) 构建压缩评测：BFCL、τ-bench、多轮 SWE-style 轨迹，在 SnapKV/H2O/StreamingLLM/KVzip/Expected Attention 下测工具调用合法率、参数正确率、指令遵循；(2) 诊断哪些片段（system prompt、tool schema、历史 observation）最先被驱逐（Pitfalls 方法）；(3) 提出按「段类型」分配预算、schema 段 KV 冻结 + observation 段激进压缩的 TF 策略；(4) 测多轮下的 prefix 复用兼容性。
- **Hypothesis**：标准驱逐优先丢弃 tool schema 的 JSON 字段，使合法调用率在 50% 预算下下降 >15 点；段感知预算可几乎恢复。
- **MVE**：Qwen2.5-7B-Instruct、Llama-3.1-8B；单卡 48G，约 30 GPU-h。
- **Closest work**：Pitfalls 2510.00231（IFEval，非 agent）、ST-Lite 2603.00188（GUI 截图）、UltraQuant 2606.20474（量化）。不同点：首个工具调用 agent 的压缩基准 + 段感知策略。
- **Novelty 6/10 | Risk LOW | Contribution：benchmark + method**
- **Reviewer objection**：段感知预算是启发式；agent 轨迹长度可能还没到需要压缩的量级。
- **dedup_key**：tool-calling-agent-kv-compression-segment-aware-budget

### Idea 7：CliffGuard — 用在线头一致性/驱逐比例代理把每请求压缩率控制在幻觉悬崖之下
- **Method**：(1) 复现 Physics 2603.01426 的 90% 悬崖，在 RULER 多长度与真实任务上找悬崖位置与序列长度、模型家族的关系；(2) 设计无训练在线代理（Global Eviction Ratio、头间注意力一致性熵），在 prefill 结束时即可计算；(3) 根据代理动态选每请求压缩率（长序列更激进，Sparse Frontier 结论），而非固定 budget；(4) 与固定预算 SnapKV/PyramidKV/LAVa 在相同平均内存下比较。
- **Hypothesis**：悬崖位置是序列长度与模型家族的可预测函数；同等平均内存下自适应压缩率比固定预算提升 ≥3 点并消除尾部失败样本。
- **MVE**：Llama-3.1-8B、Qwen2.5-7B/14B；RULER 4K–128K、LongBench；单卡 80G，约 40 GPU-h。
- **Closest work**：Sparse Frontier 2504.17768（指出固定预算次优，未给控制器）、Physics 2603.01426（悬崖诊断，合成任务）、LAVa（动态分配但总预算固定）。
- **Novelty 6/10 | Risk MEDIUM | Contribution：method + diagnostic**
- **Reviewer objection**：代理指标可能只是序列长度的函数；悬崖可能是 benchmark 伪影。
- **dedup_key**：per-request-adaptive-kv-budget-hallucination-cliff

### Idea 8：Uniform-Is-All-You-Need? 视频 token 压缩的公平基线审计与时间地平线
- **Method**：(1) 用 ForestPrune、FLoC、METok、InfiniPot-V 的公开实现，在 Video-MME/MLVU/LongVideoBench 上加入「均匀帧采样 + 空间随机」与「均匀帧采样 + 深层随机」两条强基线；(2) 测「时间地平线」：删除某帧 token 对输出分布影响随层与帧数的衰减；(3) 分流式与离线两种设定、报 walltime；(4) 若基线接近 SOTA，提出零开销混合策略。
- **Hypothesis**：视频时间冗余远高于图像空间冗余，因此随机/均匀基线在视频上更接近甚至超过精心设计的方法。
- **MVE**：LLaVA-Video-7B、Qwen2.5-VL-7B；单卡 48–80G，约 25 GPU-h。
- **Closest work**：2512.07580（图像）、2502.11501（图像）；视频侧无对应审计。
- **Novelty 5/10 | Risk LOW | Contribution：diagnostic / negative result**
- **Reviewer objection**：负结果论文难发顶会；需覆盖足够多方法。
- **dedup_key**：video-token-pruning-random-baseline-audit

### Idea 9：MoE-KV — MLA/MoE 模型上 KV 驱逐与专家跳过的联合 TF 效率
- **Method**：(1) 在 Qwen3-30B-A3B / DeepSeek-V2-Lite（MLA）上评测 SnapKV/Expected Attention/KVzip：MLA 的压缩潜变量能否再驱逐、驱逐后精度如何；(2) 与 ACE 专家跳过叠加，看两者是否互补（KV 驱逐减少注意力带宽，专家跳过减少 FFN 计算）；(3) 分析被驱逐 token 与被跳过专家的相关性（是否同一批「低贡献」token）。
- **Hypothesis**：MLA 潜 KV 对驱逐更敏感（每个潜向量承载更多信息），需更保守预算；但 token 级驱逐与专家跳过决策高度相关，可共享一个打分。
- **MVE**：上述两模型（单卡 80G 可放下 A3B 级）；LongBench、GSM8K、RULER 16K；约 40 GPU-h。
- **Closest work**：ACE 2609.05228（专家跳过）、2603.23701（MoE 早退潜力低）、DraftExpert 2607.24434（MoE 自投机）。KV 驱逐在 MLA/MoE 上几乎无评测。
- **Novelty 6/10 | Risk MEDIUM | Contribution：diagnostic + method**
- **Reviewer objection**：MLA 已经是压缩，再压缩收益可能很小；需要自定义 kernel 才能体现速度。
- **dedup_key**：kv-eviction-expert-skipping-mla-moe

### Idea 10（备选）：Early-Stop × KV-Evict 的交互：推理模型的联合预算
- **Method**：联合运行 DEER 式早停与 R-KV/BeaconKV 式解码期驱逐，网格扫描早停阈值与 KV 预算，绘制精度-延迟-显存 Pareto 面；分析早停改变的长度分布如何移动最优预算。
- **Hypothesis**：早停使最优 KV 预算可再降 30–50%，两者组合优于任一单独。
- **MVE**：R1-Distill-Qwen-7B；AIME/MATH500；约 20 GPU-h。
- **Closest work**：DEER 2504.15895、R-KV、ThinKV。
- **Novelty 4/10 | Risk LOW | Contribution：empirical**
- **Reviewer objection**：纯组合。
- **dedup_key**：early-exit-x-kv-eviction-reasoning-pareto

---

## 5. 摘要（≤600 字）

**最热/最饱和**：(1) 单图 LLaVA 上的 training-free 视觉 token 剪枝——每月数篇，且 CVPR'26 的 2512.07580 与 2502.11501 已证明深层不如随机；(2) prefill 阶段 KV 驱逐及层/头预算分配；(3) 2026 上半年集中爆发的 KV 驱逐×量化联合（RDKV/MoE-nD/HqeKV/CAPKV）；(4) 「又一个旋转」式 PTQ；(5) dLLM 的 cache+并行解码。

**最有空间的 3 个 gap**：
1. **压缩导致输出变长（contradiction + diagnostic）**：MLSys'25 2503.24000 与 ThinKV 都观察到 KV 压缩/量化使生成膨胀、端到端延迟反升，但方法论文只报 token/s，机制与 TF 修复均空白。
2. **压缩在非 FP16 / 非文本 / 非单轮条件下的迁移性（untested assumption + scaling）**：驱逐打分在 W4A4 旋转量化模型上是否仍有效无人验证（QAPruner 仅证明了视觉剪枝×PTQ 有交互）；工具调用 agent、多图交错、端侧 ≤3B、MLA/MoE 上几乎没有 TF 压缩评测，Pitfalls (ACL'26) 已显示多指令会被「选择性遗忘」。
3. **VLM 的 training-free 投机解码（method-transfer）**：MMSpec 证明文本 SD 在 VLM 上失效（Lookahead 仅 1.07×），而 SparseSpec-L 刚证明「稀疏 KV 的自己」可做 drafter；视觉 token 剪枝天然提供了一个高接受率的无训练 drafter，尚无人组合。

**最看好的 3 个 idea**：
- Idea 1 *Why Compression Makes Models Talk Longer*（诊断 + 锚点保护的 TF 修复，单卡 40–60 GPU-h）。
- Idea 5 *PruneDraft*（视觉 token 剪枝版自投机，VLM 无损加速）。
- Idea 2 *ReconVis*（KVzip 重建式 query-agnostic 打分迁移到视觉 KV，兼容多问 prefix 复用）。

次优但低风险：Idea 4（旋转量化 × KV 驱逐审计）、Idea 6（ToolKV 基准）。

**限制**：本次 arXiv/HF/OpenReview 全部被代理阻断，仅凭搜索摘要核实；RetentiveKV、ViSkip 是否 training-free、LU-KV/MixedDimKV 的 profiling 需求等需在可访问环境二次确认。
