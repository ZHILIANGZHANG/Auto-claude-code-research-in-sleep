# 三条 Training-free Idea 与单卡验证方案（自包含版）

**用途**：交给执行 agent 全自动验证。每条 idea 先讲"是什么、为什么还能做"，再给"怎么验证、怎么判定"。凡标 **[决策规则]** 的阈值 agent 不得修改。
**硬件**：AutoDL 实例，vGPU-32GB × 1（算力未知，先实测）。
**总预算**：≤ 20 GPU-h。执行顺序：Idea B（PruneDraft）→ Idea C（ReconVis）→ Idea A（Eviction vs Quantization）。
**日期**：2026-10-18。

---

## 0. 公共环境与约束

### 0.1 机器与存储
- 系统盘小，数据盘几乎空。**所有模型、数据、结果放 `/root/autodl-tmp/`**，HF 缓存指向 `/root/autodl-tmp/hf`。
- 显存：7B/8B bf16 权重约 15–17 GB，留 ≥ 10 GB 给 KV cache 与激活。所有实验 batch ≤ 8，长生成 batch ≤ 4。

### 0.2 软件
```bash
source /etc/network_turbo 2>/dev/null || true      # AutoDL 学术加速（如有）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/{hf,data,results,repos}

pip install -U "transformers>=4.51" accelerate "torch>=2.4" "qwen-vl-utils[decord]" \
  datasets math-verify optimum-quanto sentencepiece pillow pandas scipy
```
- 统一 `attn_implementation="sdpa"`；需要注意力分数的阶段用 `"eager"`。vGPU 上 flash-attn 可能不可用，不要依赖。
- 模型（公开权重，bf16）：
  - Idea A：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
  - Idea B / C：`Qwen/Qwen2.5-VL-7B-Instruct`
- 禁止任何训练/微调。

### 0.3 输出格式
每个 idea 一个目录 `/root/autodl-tmp/results/idea{A|B|C}/`：
- `config.json`：模型、数据、方法、超参、实测吞吐、代码 commit。
- `per_sample.jsonl`：逐样本原始记录。
- `metrics.json`：汇总指标（键名见各节）。
- `summary.md`：≤ 1 页，含判定（GO / NARROW / KILL）与触发的规则编号。
- `logs/`：stdout/stderr，各阶段起止时间，峰值显存（`torch.cuda.max_memory_allocated`）。
全部跑完合并为 `/root/autodl-tmp/results/PILOT_SUMMARY.md`，并打包 `results.tar.gz`。

### 0.4 Step 0：吞吐实测（必做，≤ 15 min）
1. 加载 R1-Distill-Qwen-7B，sdpa，batch=4，`max_new_tokens=2048`，跑 4 条 MATH-500，记录**聚合 decode tok/s**（记为 T）与峰值显存。
2. 加载 Qwen2.5-VL-7B，`max_pixels=1280*28*28`，batch=4，4 张 COCO 图各生成 256 token，记录 tok/s 与 prefill 时延。
3. **[缩放规则]** 各 idea 样本数 = 基准样本数 × min(1, T/80)，向下取整到 10 的倍数，不低于最小样本数。

| Idea | 基准样本数 | 最小样本数 |
|---|---|---|
| A | 200 题 × 3 配置 | 120 题 |
| B | 180 样本 | 90 样本 |
| C | 150 图 / 600 问 | 80 图 / 320 问 |

### 0.5 agent 行为约束
1. 不得修改任何 **[决策规则]** 阈值；有异议写在 summary 备注，判定仍按原规则给。
2. 任何偏离（换模型、换数据、降样本）写进 `config.json` 与 summary。
3. 每阶段开始/结束追加一行到 `results/PROGRESS.log`。
4. OOM / 依赖失败按各节"失败处理"执行，最多重试 2 次；仍失败则跳到下一 idea，summary 标 **BLOCKED** 并给日志路径。
5. 不下载非公开数据；不上传结果到外部服务。

---

## Idea A：Eviction or Quantization — 谁让推理模型"话变多"

### A.1 Idea 是什么
**问题**。KV cache 压缩是推理模型（R1-Distill 系）省显存的常用手段，但压缩后模型往往生成更长的思维链，端到端延迟反而上升。

**已经被发表的部分**（不能再当发现卖）：
- MLSys'25 [2503.24000](https://arxiv.org/abs/2503.24000)：压缩拉长输出、延迟反升。
- ReCo [2608.04771](https://arxiv.org/abs/2608.04771)（2026-08）：R-KV 25% 压缩使 R1-Distill-Qwen-7B 在 MATH-500 上长度 +38.8%、R1-Distill-Llama-8B +79.0%；修复是训练式 process-reward 估计器。
- RLKV v3 [2510.08525](https://arxiv.org/abs/2510.08525)：R-KV "often results in longer reasoning steps"；并指出激进压缩下的"长度减少"多半因为只解出了简单题。
- VisionPulse [2605.31457](https://arxiv.org/abs/2605.31457)（ICML'26）：视觉 token 静态剪枝使思考型 VLM 输出 +18%～+108%，但全部发生在准确率崩 25–75% 的极端保留率下。

**矛盾**。ThinKV [2510.01290](https://arxiv.org/abs/2510.01290)（ICLR'26 Oral）全文明确写："Unlike quantization, eviction does not cause an increase in generation length"，而纯量化最高拉长 5.1×。这与 ReCo/RLKV 的"驱逐拉长"正面冲突。分歧可能来自驱逐策略（ThinKV 思维感知驱逐 vs R-KV 冗余去重）、模型族、压缩率或长度统计口径，无人裁决。

**没人做的部分（我们的 delta）**：
1. **裁决矛盾**：同一模型、同一题集、同一口径下分别扫驱逐与量化。
2. **把两种机制分开**：上述四篇都没区分"答错了所以啰嗦"与"证据被删了所以补偿"。做法是按最终正确性分层，并在"压缩前后都答对的题"上配对比较长度。
3. **免训练修复**（全量阶段）：保护极少量"结论锚点"KV + 重复熵触发收尾；对照 StreamingLLM 的 sink 保留与 R-KV 自带去重；ReCo 为训练式 baseline。

**假设**。驱逐与量化都会拉长输出但机制不同：驱逐丢"结论锚点"导致重复自检，量化抬高熵导致更多反思 token；ThinKV 的"驱逐不拉长"只在其特定驱逐策略与模型上成立。

**Novelty 自评** 6/10，Risk MEDIUM，贡献类型 diagnostic + method。核实后它的优先级低于 CL 方向的两条 idea，但实验最便宜、无论哪种结果都有东西写。

### A.2 Pilot 要回答的三个问题
(a) R-KV 25% 驱逐是否拉长（复现 ReCo 还是 ThinKV）；(b) 2-bit 量化是否拉长；(c) 变长只发生在答错的题上，还是答对的题也变长。

### A.3 数据与解码
- MATH-500（`HuggingFaceH4/MATH-500`），按 Level 1–5 分层随机取 200 题（每层 40，固定种子）。
- **greedy**，`max_new_tokens=8192`，R1 官方模板（`<think>` 开头），不加系统提示。所有配置 prompt 与 seed 完全相同。
- batch=4。

### A.4 配置
- **C0 Full**：标准 DynamicCache。
- **C1 Evict-RKV-25%**：R-KV 官方实现（`git clone https://github.com/Zefan-Cai/R-KV` 到 `/root/autodl-tmp/repos`），budget=25%（ReCo 设定），其余默认。若与当前 transformers 不兼容，按其 README 固定版本装独立 venv。
- **C2 Quant-2bit**：HF `QuantizedCache`（`cache_implementation="quantized"`, `backend="quanto"`, `nbits=2`, `residual_length=128`）。summary 中注明这是 KIVI 式 2-bit 的现成替代，非 KIVI 原实现。
- **C3（可选）Evict-SnapKV-25%**：时间允许时，用 KVPress 的 SnapKV 每 512 个解码 token 周期性驱逐到 25%，作为"非 R-KV 的驱逐"对照，用于判断分歧是否来自驱逐策略。

### A.5 逐样本指标
- `len_total`：生成 token 数。
- `pos_first_answer`：第一次出现 `\boxed{` 的 token 位置（无则 = len_total）。
- `tail_after_answer = len_total − pos_first_answer`。
- `n_reflect`：`wait` / `alternatively` / `hmm` / `let me double-check` 的出现次数（小写、词边界）。
- `rep4`：重复 4-gram 比例。
- `correct`：`math_verify` 比对最终 `\boxed{}` 与标准答案。
- `hit_max`：是否撞到 8192 上限。

### A.6 汇总指标（metrics.json）
对每个配置 C 相对 C0：
- `len_ratio_all[C]` = mean(len_total[C]) / mean(len_total[C0])。
- `len_ratio_paired_correct[C]`：只在 **C 与 C0 都答对** 的题上，逐题 len[C]/len[C0] 的几何均值。
- `len_ratio_paired_wrong[C]`：只在 C 答错的题上同上。
- `acc[C]`、`tail_ratio[C]`、`reflect_ratio[C]`、`rep4_delta[C]`、`hit_max_rate[C]`。
- 按 Level 分层的 `len_ratio_all`。

### A.7 [决策规则]
- **RA-REPLICATE**：`len_ratio_all[C1] ≥ 1.20` 视为复现 ReCo 的驱逐膨胀；`len_ratio_all[C2] ≥ 1.20` 视为复现 ThinKV 的量化膨胀。
- **RA-MECHANISM**：对复现了膨胀的配置，`len_ratio_paired_correct ≥ 1.15` → "缺证据补偿"；`len_ratio_paired_correct < 1.05` 且 `len_ratio_paired_wrong ≥ 1.20` → "答错啰嗦伪影"。
- **GO**：至少一个配置复现膨胀且判为"缺证据补偿"。→ 全量做机制 + 免训练修复。
- **NARROW**：复现膨胀但全部判为"伪影"。→ 诊断短文："压缩导致的长度膨胀主要是准确率下降的副产物"，顺带裁决 ThinKV vs ReCo。
- **KILL**：C1、C2 都 `len_ratio_all < 1.10`。→ 一页复现说明，不立项。
- 附加：C1 不膨胀而 C2 膨胀 → 支持 ThinKV；相反 → 支持 ReCo；都膨胀 → 分歧来自策略/模型族，summary 列出 C3 是否跑过。

### A.8 时间与保护
- 粗估 200 题 × 3 配置 × 平均 3K token ≈ 1.8M token；T=80 tok/s 时约 6.5 h，加压缩开销 8–10 h。放夜间。
- `hit_max_rate > 30%` → `max_new_tokens` 降到 6144，**所有配置重跑同一批题**。
- 每 20 题落盘一次 `per_sample.jsonl`，支持断点续跑。

---

## Idea B：PruneDraft — 不用独立 draft 模型的 VLM 自投机

### B.1 Idea 是什么
**问题**。文本投机解码在 VLM 上基本失效：MMSpec [2603.14989](https://arxiv.org/abs/2603.14989) 测得 Lookahead 在 Qwen2.5-VL-7B 上仅 1.07×。VLM 的 KV 里 >80% 是视觉 token，而 CVPR'26 [2512.07580](https://arxiv.org/abs/2512.07580) 证明深层视觉 token 信息"消失"、大幅删减不影响输出。

**已经被发表的部分**：
- SpecVLM [2508.16201](https://arxiv.org/abs/2508.16201)（EMNLP'25 main）：发现"draft 对视频 token 剪枝不敏感"，两阶段剪枝 draft 的视频 token（剪 90%），LLaVA-OV-72B 2.68×、Qwen2.5-VL-32B 2.11×。**但它需要一个独立的小 Vid-LLM 当 draft**，且只做视频。
- SparseSpec-L [2607.27735](https://arxiv.org/abs/2607.27735)：用稀疏 KV 的"自己"当 drafter（文本）。
- ViSkip（MMSpec 附带）：视觉相关性高时关闭投机的门控，不草拟。

**我们的 delta**：**零 draft 模型**——同一个 VLM，深层只保留 10–20% 视觉 token 作为 draft，全 KV 并行验证（标准 SD，无损）；面向图像与短视频；在 MMSpec 类任务上与 SpecVLM、Lookahead 同台，报 wall-time 与 p99。

**假设**。剪枝版自草拟的接受率 >0.8，整体 1.5–2.0×，无需任何 draft 模型。

**Novelty 自评** 5/10（核实后从 7 降下来，因为核心 insight 已被 SpecVLM 发表），Risk MEDIUM。审稿人会问"SpecVLM 的增量"，必须证明去掉 draft 模型后加速比不降且部署更简单。

### B.2 Pilot 要回答的问题
"剪掉 80–90% 视觉 token 的自己"逐 token 的 top-1 预测与全模型一致的比例（接受率代理）有多高？draft 前向比全前向省多少？理论加速比是否 ≥ 1.5×？

### B.3 数据（3 类 × 60 = 180，按 §0.4 缩放）
- **长描述**：COCO val2017 图 60 张，"Describe this image in detail."，`max_new_tokens=160`。
- **图表推理**：ChartQA test 60 题，加 "Think step by step, then give the final answer."，`max_new_tokens=256`。
- **文本密集**：TextVQA val 60 题，同样 CoT prompt，`max_new_tokens=192`。
固定种子取前 N 条，不筛选。

### B.4 方法与配置
- **Target**：全模型 greedy 生成序列 y（teacher-forcing 参考）。
- **Draft 配置**（FastV 式：第 K 层之后按该层"最后一个文本 token → 视觉 token"的注意力均值选 top-r% 视觉 token，其余从后续层移除）：
  - K ∈ {2, 8}，r ∈ {10%, 20%, 50%} → 6 配置。
  - 实现：在 `Qwen2_5_VLForConditionalGeneration` 第 K 层注册 forward hook（eager 注意力），构造 `keep_mask`；在第 K+1 层输入处按 mask 裁剪 hidden states、attention mask、position ids（M-RoPE position ids 保留原值，不重排）。prefill 时一次决定，解码期复用同一 mask。
- **teacher forcing 测一致率**：把 y 作为输入，在 draft 配置下一次前向，逐位置 argmax 与 y 比较。
- **时延**：draft 配置与全模型各测 20 步单步解码时延取中位数，c = t_draft / t_full。

### B.5 指标（metrics.json）
- `top1_agree[K][r][task]`：逐 token top-1 一致率。
- `exp_accept_len[K][r][task]`：从每个位置起连续一致 run length 的均值（γ=5 截断），即期望接受长度 E[τ]。
- `cost_ratio[K][r]`：c。
- `theory_speedup[K][r]` = (E[τ]+1) / (γ·c + 1)，γ=5。
- 峰值显存、各配置耗时。

### B.6 [决策规则]
- **RB-GO**：存在 (K, r≤20%) 使三类任务平均 `top1_agree ≥ 0.80` 且 `theory_speedup ≥ 1.5`。
- **RB-NARROW**：`top1_agree ∈ [0.60, 0.80)`，或只有 r=50% 达标 → 只值得做"中等剪枝 + 早退"的组合，不单独立项。
- **RB-KILL**：所有 (K, r≤20%) 平均 `top1_agree < 0.60`。
- 附加：ChartQA 一致率比 COCO 低 >0.15 → summary 标"任务依赖"（对应 MMSpec 的"视觉感知重的任务更难投机"）。

### B.7 失败处理
- eager 注意力 OOM → `max_pixels=768*28*28`，batch=1。
- hook 实现出错 → 退化为"prefill 后直接裁剪 past_key_values 的视觉位置"（等价于全层 KV 驱逐），summary 注明与 FastV 式的差异。

---

## Idea C：ReconVis — 同图多问时的 query-agnostic 视觉 KV 复用

### C.1 Idea 是什么
**问题**。主流视觉 token 剪枝（FastV、SparseVLM、VisionZip、VisionTrim）要么 query-aware、要么静态；query-aware 方法换一个问题就得重压，与 prefix cache 天然冲突。文本侧 KVzip [2505.23416](https://arxiv.org/abs/2505.23416)（NeurIPS'25）用"模型能否从 cache 重建原文"做 query-agnostic 打分，兼容多 query 复用；Compactor [2507.08143](https://arxiv.org/abs/2507.08143) 指出 query-aware 驱逐与 prefix caching 根本不兼容。

**已经被发表的部分**：
- StreamMem [2508.15717](https://arxiv.org/abs/2508.15717)：流式视频的 query-agnostic KV 记忆（通用 query token 打分），只做流式视频。
- VisCache [2608.24063](https://arxiv.org/abs/2608.24063)（2026-08）：training-free 视觉 KV 剪枝，但 prompt-aware。
- FlashCache [2511.16786](https://arxiv.org/abs/2511.16786)（CVPR'26）：attention-free 频域打分，无多问评测。
- RetentiveKV [2605.04075](https://arxiv.org/abs/2605.04075)（ACL'26 Findings）：指出视觉 token 重要性会"迟发"，但含可学习门控、非严格 training-free。

**我们的 delta**："同图/同视频多问 + prefix 复用"设定下的 query-agnostic 视觉 KV 打分（重建式：让模型自己"描述这张图"，用描述 token 对视觉 KV 的注意力累计打分）+ 低分 KV 合并而非丢弃（保留迟发重要性通道）；报告 prefix 命中率与多问退化曲线。

**假设**。同图多问下，20% 视觉 KV 预算时，重建式 query-agnostic 打分比 query-aware 复用高 ≥ 5 点，并允许 prefix 命中。

**Novelty 自评** 6/10，Risk MEDIUM。审稿人会说重建前向多一次 prefill、多问基准是重组而非原生——全量时补 VisDial / Video-MME 原生多问。

### C.2 Pilot 要回答的问题
用第一个问题压缩视觉 KV 后复用给后续问题，比"每问重压"退化多少（这是前提）？重建式打分能救回多少？

### C.3 数据
- **主**：VQAv2 val，挑每图 ≥ 4 问的图 150 张，每图 4 问 → 600 问。短答 prompt："Answer the question using a single word or phrase."。VQA 标准 soft accuracy。
- **副（可选）**：DocVQA val 50 份 × 3 问（高分辨率、问题差异大），ANLS。
- 问题按原序；Q1 = "已见问题"，Q2–Q4 = "未见问题"。

### C.4 方法（视觉 KV 预算 20%；文本 KV 不压）
统一走 HF `DynamicCache` 切片：prefill（系统 prompt + 图 + 打分用 query）→ 每层每头对视觉位置打分 → **逐层**选 top-20% 视觉位置 → 删除 cache 中其余视觉位置 → 每个问题的 token 接在压缩 cache 之后解码（即 prefix 复用）。
- **M-full**：不压缩（上界）。
- **M-random**：随机保留 20%（对照）。
- **M-qaware-reuse**：用 Q1 的 token 对视觉 KV 的注意力（最后 8 个 query token 均值，SnapKV 式）打分，压一次，Q1–Q4 全复用。
- **M-qaware-recompress**：每问各自用自己的 query 打分重压（无复用上界，代表现有 query-aware 方法的理想情况）。
- **M-recon（ours）**：pseudo-query = "Describe this image in detail."，prefill 后生成 64 个 token 的描述，用**生成的描述 token 对视觉 KV 的注意力累计**打分，压一次，Q1–Q4 复用。
- **M-recon+merge（ours，可选）**：同上，但被淘汰的视觉 KV 按 value 余弦相似度均值合并进最近的保留 token（GRKV/SelKV 最简版）。时间不够可跳过。

注意：打分阶段需要注意力分数 → 全程 `eager`，batch=1–2；删视觉位置后**不重排 position ids**（M-RoPE 已编码在 K 上）；问题 token 的 position 从原序列末尾继续。

### C.5 指标（metrics.json）
- `acc[method][seen|unseen]`：VQA soft accuracy，分 Q1 与 Q2–Q4。
- `gap_reuse = acc[qaware-recompress][unseen] − acc[qaware-reuse][unseen]`。
- `recover = (acc[recon][unseen] − acc[qaware-reuse][unseen]) / gap_reuse`。
- `prefill_time[method]`（含 pseudo-query 生成开销）、峰值显存。

### C.6 [决策规则]
- **RC-PRECHECK**：`gap_reuse < 2.0` 点 → **KILL**（前提不成立：同图多问下 query-aware 复用不退化）。
- **RC-GO**：`gap_reuse ≥ 2.0` 且 `recover ≥ 0.6` 且 `acc[recon][unseen] − acc[random][unseen] ≥ 2.0`。
- **RC-NARROW**：`gap_reuse ≥ 2.0` 但 `recover ∈ [0.3, 0.6)`，或 recon 只比 random 好 <2 点 → 退化真实存在但重建式打分不够，转为探索其他 query-agnostic 信号（K-norm / 频域能量）。
- **RC-KILL**：`recover < 0.3`。

### C.7 失败处理
- Qwen2.5-VL 的 cache 切片触发 shape 错误 → 改用 `llava-hf/llava-onevision-qwen2-7b-ov-hf`（视觉 token 位置固定），并记录。
- VQAv2 下载慢 → 用 `lmms-lab/VQAv2` 或 `HuggingFaceM4/VQAv2` 的 val 子集。

---

## 4. 执行顺序、里程碑与汇报

| 阶段 | 内容 | 产物 | 预计 |
|---|---|---|---|
| S0 | 环境 + 吞吐实测 + 缩放样本数 | `results/step0_throughput.json` | 0.5 h |
| S1 | Idea B 全部 6 配置 | `results/ideaB/*` + 判定 | 2–3 h |
| S2 | Idea C 五个方法（merge 可选） | `results/ideaC/*` + 判定 | 4–6 h |
| S3 | Idea A C0–C2（C3 可选） | `results/ideaA/*` + 判定 | 8–10 h |
| S4 | 合并 summary、打包 | `results/PILOT_SUMMARY.md`、`results.tar.gz` | 0.5 h |

**`PILOT_SUMMARY.md` 格式**：
```
| Idea | 判定 | 触发规则 | 关键数字 | 耗时 | 峰值显存 | 备注/偏离计划之处 |
```
之后每个 idea 附 1 段解读与"若 GO，下一步最小全量实验"。

---

## 5. 判定之后怎么走（给人看的）

| Idea | GO 之后的最小全量实验 | NARROW 之后 | KILL 之后 |
|---|---|---|---|
| A | 加 R1-Distill-Llama-8B、Qwen3-8B 与 AIME/LiveCodeBench；实现锚点保护 + 重复熵收尾；与 ReCo、StreamingLLM sink 保留、R-KV 去重同台；报端到端延迟。约 40–60 GPU-h | 写诊断短文（workshop / short paper），把 ThinKV vs ReCo 的裁决表作为主结果 | 一页复现说明，附到其他工作 |
| B | 在 MMSpec 六类任务 + Video-MME 短答上实现完整 SD 循环（draft γ=5 + 全 KV 验证），与 SpecVLM、Lookahead 比 wall-time 与 p99；加 ViSkip 式门控与熵控制 γ。约 30 GPU-h | 做"中等剪枝 + 早退"组合，或并入其他 VLM 效率工作 | 放弃 |
| C | 补 VisDial、Video-MME 原生多问；加岭回归合并；与 VisionZip/SparseVLM/StreamMem/FlashCache 同台；报 prefix 命中率与解码延迟。约 30 GPU-h | 换 query-agnostic 信号（K-norm、频域）做小型比较 | 放弃 |

---

## 附：相关论文索引（全部已用 arxiv.org/abs 核实）

| 简称 | arXiv | 与本文件的关系 |
|---|---|---|
| ReCo | 2608.04771 | Idea A 的训练式 baseline 与"驱逐拉长"证据 |
| ThinKV | 2510.01290 | Idea A 的反方证据（驱逐不拉长） |
| RLKV | 2510.08525 | Idea A 的"只解出简单题"混淆提醒 |
| VisionPulse | 2605.31457 | 视觉侧的长度膨胀证据（极端保留率） |
| MLSys'25 Rethinking KV Compression | 2503.24000 | 现象首报 |
| SpecVLM | 2508.16201 | Idea B 最近工作（需独立 draft） |
| SparseSpec-L | 2607.27735 | Idea B 的"自己当 drafter"文本先例 |
| MMSpec | 2603.14989 | Idea B 的基准与 ViSkip 门控 |
| When Token Pruning is Worse than Random | 2512.07580 | Idea B 的"深层视觉 token 可丢"依据 |
| KVzip | 2505.23416 | Idea C 的重建式打分来源 |
| Compactor | 2507.08143 | Idea C 的"query-aware 与 prefix caching 不兼容"依据 |
| StreamMem | 2508.15717 | Idea C 最近工作（流式） |
| VisCache | 2608.24063 | Idea C 最近工作（prompt-aware） |
| FlashCache | 2511.16786 | Idea C 的 attention-free 对照 |
| RetentiveKV | 2605.04075 | Idea C 的"迟发重要性"依据 |
