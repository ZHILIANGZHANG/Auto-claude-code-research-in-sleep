# Pilot Plan：Idea 1 / 6 / 7 在 AutoDL vGPU-32GB × 1 上的全自动初步验证

**目标**：用一张 32GB 显存的单卡，在 ≤ 20 GPU-h 内，对 `IDEA_REPORT.md`（v2）里的三条 idea 各给出一个 **GO / NARROW / KILL** 的判定，并产出可复用的测量脚本。本文件是写给执行 agent 的，字段尽量机械化；凡标 **[决策规则]** 的地方 agent 不得自行修改，只能按规则输出判定。

| Idea | 名称 | 一句话问题 | 预计 GPU-h | 执行顺序 |
|---|---|---|---|---|
| 7 | PruneDraft | "剪掉 80% 视觉 token 的自己"能否当 VLM 的无训练 drafter | 2–3 | 第 1 个（最短，先验证环境） |
| 6 | ReconVis | 同图多问时 query-aware 视觉 KV 压缩会退化多少，query-agnostic 重建式打分能否救回 | 4–6 | 第 2 个 |
| 1 | Eviction vs Quantization | KV 驱逐 / 量化谁让推理模型输出变长，是"缺证据补偿"还是"答错啰嗦"的伪影 | 8–10 | 第 3 个（最长，放夜间） |

---

## 0. 环境与硬约束

### 0.1 机器
- AutoDL 实例 `twk9fdwkpj-7c250eeb`，规格 **vGPU-32GB × 1**。vGPU 的算力未知（可能是 A100/4090 的切片），**第一步必须实测吞吐**（§0.4），再按实测吞吐缩放样本数。
- 系统盘小，数据盘几乎空：**所有模型、数据、结果放 `/root/autodl-tmp/`**，HF 缓存指向 `/root/autodl-tmp/hf`。
- 显存预算：7B/8B bf16 权重约 15–17 GB，留 ≥10 GB 给 KV cache 与激活。所有实验 **batch ≤ 8**，长生成实验 batch ≤ 4。

### 0.2 软件
```bash
# AutoDL 学术加速（如有）；HF 走镜像
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/{hf,data,results,repos}

pip install -U "transformers>=4.51" accelerate "torch>=2.4" qwen-vl-utils[decord] \
  datasets math-verify optimum-quanto sentencepiece pillow pandas scipy
# flash-attn 在 vGPU 上可能装不上/不可用：统一用 attn_implementation="sdpa"；
# 需要注意力分数的实验（Idea 6/7 的打分）用 "eager"。
```
- 模型（全部 bf16，公开权重）：
  - Idea 1：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
  - Idea 6/7：`Qwen/Qwen2.5-VL-7B-Instruct`
- 禁止：任何训练/微调；任何会改动 §2–§4 **[决策规则]** 阈值的行为。

### 0.3 通用输出格式
每个 idea 一个目录 `/root/autodl-tmp/results/idea{N}/`：
- `config.json`：模型、数据、方法、超参、实测吞吐、commit hash。
- `per_sample.jsonl`：逐样本原始记录（id、方法、输出文本、各指标）。
- `metrics.json`：汇总指标（按 §2–§4 规定的键名）。
- `summary.md`：≤ 1 页，含决策表（GO / NARROW / KILL）与触发的规则编号。
- `logs/`：stdout/stderr，含每个阶段的起止时间与峰值显存（`torch.cuda.max_memory_allocated`）。
全部跑完后把三个 `summary.md` 合并成 `/root/autodl-tmp/results/PILOT_SUMMARY.md`，并 `tar` 整个 results 目录。

### 0.4 Step 0：吞吐实测（必做，≤ 15 min）
1. 加载 R1-Distill-Qwen-7B bf16，sdpa，`max_new_tokens=2048`，batch=4，跑 4 条 MATH-500 题，记录 **decode tok/s（聚合）** 与峰值显存。
2. 加载 Qwen2.5-VL-7B，`max_pixels=1280*28*28`，batch=4，跑 4 张 COCO 图 "Describe this image in detail." 各 256 token，记录 tok/s 与 prefill 时延。
3. **[缩放规则]**：令实测聚合 decode 吞吐为 T tok/s。各 idea 的样本数按下表的"基准样本数 × min(1, T/80)" 缩放，向下取整到 10 的倍数，且不低于"最小样本数"。

| Idea | 基准样本数 | 最小样本数 |
|---|---|---|
| 1 | 200 题 × 3 配置 | 120 题 |
| 6 | 150 图 / 600 问 | 80 图 / 320 问 |
| 7 | 180 样本 | 90 样本 |

---

## 1. Idea 7 PruneDraft（先跑）

### 1.1 要回答的问题
不训练任何 draft 模型，把 **同一个 Qwen2.5-VL-7B 在第 K 层之后只保留 r% 视觉 token** 当作 draft，它逐 token 的 top-1 预测与全模型一致的比例（接受率代理）有多高？draft 前向比全前向省多少时间？两者合起来的理论加速比是否 ≥ 1.5×？

### 1.2 数据（3 类 × 60，共 180；按 §0.4 缩放）
- **长描述**：COCO val2017 图 60 张，prompt "Describe this image in detail."，`max_new_tokens=160`。
- **图表推理**：ChartQA test 60 题，prompt 加 "Think step by step, then give the final answer."，`max_new_tokens=256`。
- **文本密集**：TextVQA val 60 题，同样 CoT prompt，`max_new_tokens=192`。
数据用 `datasets` 下载到 `/root/autodl-tmp/data`；取前 N 条固定种子即可，不做筛选。

### 1.3 方法与配置
- **Target**：全模型，greedy，生成序列 y（作为 teacher-forcing 的参考）。
- **Draft 配置**（FastV 式：在第 K 层之后，按第 K 层文本→视觉注意力均值选 top-r% 视觉 token，其余从后续层的 KV/hidden 中移除）：
  - K ∈ {2, 8}，r ∈ {10%, 20%, 50%} → 6 个配置。
  - 实现：在 HF 的 `Qwen2_5_VLForConditionalGeneration` 上注册第 K 层的 forward hook：取该层注意力（eager）中最后一个文本 token 对视觉 token 的注意力均值作重要性，构造 `keep_mask`，用 hook 在第 K+1 层输入处按 mask 裁剪 hidden states、attention mask 与 position ids（M-RoPE 的 position ids 保留原值，不重排）。一次性剪枝（prefill 时决定），解码期对所有步复用同一 mask。
- **teacher forcing 测一致率**：对每个样本，把 target 的 y 作为输入，在 draft 配置下一次前向（prefill 整段），取每个位置的 argmax 与 y 比较。
- **附加**：draft 配置下单步解码时延 / 全模型单步时延（各测 20 步取中位数），记作 c = t_draft / t_full。

### 1.4 指标（写入 metrics.json）
- `top1_agree[K][r][task]`：逐 token top-1 一致率。
- `exp_accept_len[K][r][task]`：按标准 SD 的期望接受长度 E[τ]，用一致率序列估计：从每个位置开始连续一致的 run length 的均值（γ=5 截断）。
- `cost_ratio[K][r]`：c。
- `theory_speedup[K][r]`：(E[τ]+1) / (γ·c + 1)，γ=5。
- 峰值显存、每配置耗时。

### 1.5 [决策规则]
- **R7-GO**：存在某个 (K, r≤20%) 使三类任务平均 `top1_agree ≥ 0.80` 且 `theory_speedup ≥ 1.5`。
- **R7-NARROW**：`top1_agree ∈ [0.60, 0.80)` 或只有 r=50% 达标 → 只值得做"中等剪枝 + 早退"的组合，不单独立项。
- **R7-KILL**：所有 (K, r≤20%) 的平均 `top1_agree < 0.60`。
- 附加记录：若 ChartQA 的一致率显著低于 COCO（差 >0.15），在 summary 里标注"任务依赖"，对应 MMSpec 的"视觉感知重的任务更难投机"结论。

### 1.6 失败处理
- eager 注意力 OOM → 把 `max_pixels` 降到 `768*28*28`，batch=1。
- hook 实现出错 → 退化为"只在 prefill 后裁剪 past_key_values 的视觉位置"（等价于 K=全层的 KV 驱逐），并在 summary 注明与 FastV 式的差异。

---

## 2. Idea 6 ReconVis（第二跑）

### 2.1 要回答的问题
同一张图连续问多个问题时，用 **第一个问题** 做 query-aware 视觉 KV 压缩并把压缩后的 cache 复用给后续问题，会比"每问重压"退化多少？用模型自己的 **"描述这张图"** 当 pseudo-query 做 query-agnostic 打分（KVzip 思路迁移）能否把退化救回？

### 2.2 数据
- **主**：VQAv2 val，挑每张图 ≥ 4 个问题的图 150 张，每图取 4 问 → 600 问。短答 prompt："Answer the question using a single word or phrase."。评分用 VQA 标准 soft accuracy（10 个人工答案）。
- **副（可选，若时间允许）**：DocVQA val 50 份文档 × 3 问（高分辨率、问题间差异大）。ANLS 评分。
- 问题顺序固定为数据集原序；Q1 作为"已见问题"，Q2–Q4 为"未见问题"。

### 2.3 方法（视觉 KV 预算统一 20%；文本 KV 不压）
实现统一走 HF `DynamicCache` 的切片：先 prefill（系统 prompt + 图像 + 打分用 query），得到每层每头对视觉位置的打分，按 **逐层** 选 top-20% 视觉位置（各层独立），把 cache 中其余视觉位置删掉；随后对每个问题，把问题 token 接在压缩后的 cache 之后解码（这就是 prefix 复用）。
- **M-full**：不压缩（上界）。
- **M-random**：随机保留 20%（下界/对照）。
- **M-qaware-reuse**：用 Q1 的 token 对视觉 KV 的注意力（最后 8 个 query token 均值，SnapKV 式）打分，压一次，Q1–Q4 全部复用。
- **M-qaware-recompress**：每个问题各自用自己的 query 打分重压（无复用的上界，代表现有 query-aware 方法的理想情况）。
- **M-recon（ours）**：pseudo-query = "Describe this image in detail."，prefill 后让模型生成 64 个 token 的描述，用 **生成的描述 token 对视觉 KV 的注意力累计**（而不是 prompt token）作为打分，压一次，Q1–Q4 复用。这是 KVzip "能否重建上下文"的视觉版。
- **M-recon+merge（ours，可选）**：同上，但被淘汰的视觉 KV 不丢，而是按 value 余弦相似度合并进最近的保留 token（GRKV/SelKV 式最简版：均值合并，不做岭回归）。时间不够可跳过。

注意事项：
- 打分需要注意力分数 → 打分阶段 `attn_implementation="eager"`；解码阶段可换回 sdpa（同一模型两份权重会超显存，直接全程 eager，batch=1–2）。
- 去掉视觉位置后 **不要重排 position ids**；Qwen2.5-VL 的 M-RoPE 已在 K 上编码，切片即可。
- 问题 token 的 position 从原序列末尾继续。

### 2.4 指标
- `acc[method][seen|unseen]`：VQA soft accuracy，分 Q1（seen）与 Q2–Q4（unseen）。
- `gap_reuse = acc[qaware-recompress][unseen] − acc[qaware-reuse][unseen]`：复用造成的退化。
- `recover = (acc[recon][unseen] − acc[qaware-reuse][unseen]) / gap_reuse`：救回比例。
- `prefill_time[method]`（含 pseudo-query 生成的额外开销）、峰值显存。

### 2.5 [决策规则]
- **R6-PRECHECK**：`gap_reuse < 2.0` 个点 → **KILL**（"同图多问下 query-aware 复用会退化"这个前提不成立）。
- **R6-GO**：`gap_reuse ≥ 2.0` 且 `recover ≥ 0.6` 且 `acc[recon][unseen] − acc[random][unseen] ≥ 2.0`。
- **R6-NARROW**：`gap_reuse ≥ 2.0` 但 `recover ∈ [0.3, 0.6)`，或 recon 只比 random 好 <2 点 → 退化真实存在但重建式打分不够，改为探索其他 query-agnostic 信号（K-norm / 频域能量）的小型研究。
- **R6-KILL**：`recover < 0.3`。

### 2.6 失败处理
- 若 Qwen2.5-VL 的 cache 切片触发 shape 错误，改用 `LLaVA-OneVision-7B`（视觉 token 位置固定，更易切片），并记录。
- VQAv2 下载慢 → 用 `lmms-lab/VQAv2` 的 val 子集或 `HuggingFaceM4/VQAv2`。

---

## 3. Idea 1 Eviction vs Quantization（最后跑，夜间）

### 3.1 要回答的问题
在同一模型、同一题集、同一长度统计口径下：(a) KV **驱逐**（R-KV 25%）是否让输出变长（ReCo 说 +38.8%，ThinKV 说不变）；(b) KV **量化**（2-bit）是否让输出变长（ThinKV 说最高 5×）；(c) 变长是否只发生在答错的题上（"啰嗦伪影"）还是答对的题也变长（"缺证据补偿"）。

### 3.2 数据与解码
- MATH-500（`HuggingFaceH4/MATH-500`），按 Level 1–5 分层随机取 200 题（每层 40）。
- 解码：**greedy**，`max_new_tokens=8192`，R1 官方模板（`<think>` 开头），不加系统提示。所有配置用完全相同的 prompt 与 seed。
- batch=4（8K 上下文下 KV 占用 <1 GB/样本，安全）。

### 3.3 配置（3 个必跑 + 1 个可选）
- **C0 Full**：标准 DynamicCache。
- **C1 Evict-RKV-25%**：R-KV 官方实现（`git clone https://github.com/Zefan-Cai/R-KV` 到 `/root/autodl-tmp/repos`），budget = 25%（按 ReCo 设定），其余超参用仓库默认。若仓库与当前 transformers 版本不兼容，按其 README 固定版本装到独立 venv。
- **C2 Quant-2bit**：HF `QuantizedCache`（`cache_implementation="quantized"`, `backend="quanto"`, `nbits=2`, `residual_length=128`）。这是 KIVI 式 2-bit 的现成替代；在 summary 中注明不是 KIVI 原实现。
- **C3（可选）Evict-SnapKV-25%**：若时间允许，用 KVPress 的 SnapKV 在每 512 个解码 token 处周期性驱逐到 25%，作为"非 R-KV 的驱逐"对照，用于判断 ThinKV 与 ReCo 的分歧是否来自驱逐策略。

### 3.4 逐样本指标
- `len_total`：生成 token 数。
- `pos_first_answer`：第一次出现 `\boxed{` 的 token 位置（无则 = len_total）。
- `tail_after_answer = len_total − pos_first_answer`。
- `n_reflect`：`Wait` / `Alternatively` / `Hmm` / `Let me double-check` 的出现次数（小写匹配，词边界）。
- `rep4`：重复 4-gram 比例。
- `correct`：`math_verify` 对最终 `\boxed{}` 与标准答案的判定。
- `hit_max`：是否撞到 8192 上限。

### 3.5 汇总指标（metrics.json）
对每个配置 C 相对 C0：
- `len_ratio_all[C]` = mean(len_total[C]) / mean(len_total[C0])。
- `len_ratio_paired_correct[C]`：只在 **C 与 C0 都答对** 的题上，逐题 len[C]/len[C0] 的几何均值。
- `len_ratio_paired_wrong[C]`：只在 C 答错的题上同上。
- `acc[C]`、`tail_ratio[C]`、`reflect_ratio[C]`、`rep4_delta[C]`、`hit_max_rate[C]`。
- 按 Level 分层的 `len_ratio_all`。

### 3.6 [决策规则]（分别对驱逐 C1 与量化 C2 判定，再合成）
- **R1-REPLICATE**：`len_ratio_all[C1] ≥ 1.20` 视为复现 ReCo 的驱逐膨胀；`len_ratio_all[C2] ≥ 1.20` 视为复现 ThinKV 的量化膨胀。
- **R1-MECHANISM**：对复现了膨胀的配置，若 `len_ratio_paired_correct ≥ 1.15` → 判"缺证据补偿"（答对的题也变长）；若 `len_ratio_paired_correct < 1.05` 且 `len_ratio_paired_wrong ≥ 1.20` → 判"答错啰嗦伪影"。
- **GO**：至少一个配置复现膨胀 **且** 判为"缺证据补偿"。→ 全量做机制 + 免训练修复（锚点保护 + 重复熵收尾），ReCo 为训练式 baseline。
- **NARROW**：复现膨胀但全部判为"伪影"。→ 改写成诊断短文："压缩导致的长度膨胀主要是准确率下降的副产物"，顺带裁决 ThinKV vs ReCo。
- **KILL**：C1、C2 都 `len_ratio_all < 1.10`。→ 一页复现说明，不立项。
- 附加：若 C1 不膨胀而 C2 膨胀，直接支持 ThinKV；若相反，支持 ReCo；两者都膨胀则分歧来自策略/模型族，summary 中列出 C3 是否跑过。

### 3.7 时间估计与保护
- 粗估：200 题 × 3 配置 × 平均 3K token = 1.8M token；T=80 tok/s 时约 6.5 h，加 R-KV 与量化开销约 8–10 h。
- `hit_max_rate` 若 >30%，把 `max_new_tokens` 降到 6144 **并对所有配置重跑同一批题**（不能只重跑一个配置）。
- 每 20 题落盘一次 `per_sample.jsonl`，支持断点续跑。

---

## 4. 执行顺序、里程碑与汇报

| 阶段 | 内容 | 产物 | 预计 |
|---|---|---|---|
| S0 | 环境 + 吞吐实测 + 缩放样本数 | `results/step0_throughput.json` | 0.5 h |
| S1 | Idea 7 全部 6 配置 | `results/idea7/*` + 判定 | 2–3 h |
| S2 | Idea 6 五个方法（merge 可选） | `results/idea6/*` + 判定 | 4–6 h |
| S3 | Idea 1 C0–C2（C3 可选） | `results/idea1/*` + 判定 | 8–10 h |
| S4 | 合并 summary、打包 | `results/PILOT_SUMMARY.md`、`results.tar.gz` | 0.5 h |

**汇报格式**（`PILOT_SUMMARY.md`）：
```
| Idea | 判定 | 触发规则 | 关键数字 | 耗时 | 峰值显存 | 备注/偏离计划之处 |
```
之后附每个 idea 的 1 段解读与"若 GO，下一步最小全量实验"。

**agent 行为约束**
1. 不得修改 §1.5 / §2.5 / §3.6 的阈值；若认为规则不合理，在 summary 的"备注"里写理由，判定仍按原规则给。
2. 任何与计划的偏离（换模型、换数据、降样本数）必须写进 `config.json` 与 summary。
3. 每个阶段开始/结束时把一行状态追加到 `results/PROGRESS.log`（时间戳、阶段、状态）。
4. 中途 OOM / 依赖失败按各节"失败处理"执行，最多重试 2 次；仍失败则跳到下一 idea，并在 summary 标 **BLOCKED** 与错误日志路径。
5. 不下载任何非公开数据；不上传结果到外部服务。

---

## 5. 与报告的对应关系
- Idea 7 ↔ `IDEA_REPORT.md` §4 Idea 7（dedup_key `visual-token-pruned-self-speculative-decoding-vlm`）
- Idea 6 ↔ §4 Idea 6（`query-agnostic-visual-kv-reconstruction-multiquery`）
- Idea 1 ↔ §4 Idea 1（`kv-compression-output-length-inflation-anchor-protection`）
判定结果回填到 `IDEA_REPORT.md` 各条的 **Pilot result** 字段（原值 SKIPPED）。
