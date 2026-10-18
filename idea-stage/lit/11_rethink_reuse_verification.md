# ReThink-Reuse 可行性核验（一手资料 + 顶会论文）

日期：2026-10-18。核验人：Claude（单模型，无跨模型评审，无 GPU 实验）。

## 1. 问题是否真实存在 —— 是（一手资料确认）

| 来源 | 结论 |
|---|---|
| Qwen/Qwen3-8B 官方 `tokenizer_config.json` 的 chat template | 只给 `last_query_index` 之后的 assistant 消息渲染 `<think>…</think>`；更早轮次一律 `content.split('</think>')[-1]`，即删除 thinking |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 官方模板 | 同样 `content.split('</think>')[-1]` |
| openai/gpt-oss-20b 官方 `chat_template.jinja` | analysis（CoT）只在“后面还没有 final 消息”的工具调用中保留；出现 final 之后就丢弃 |
| SGLang 源码（commit 14ad64c，2026-10-18）`strip_thinking_cache`（#22373） | 可选项：请求结束后**只缓存 prompt 前缀**，thinking 和答案的 KV 全部释放 → 下一轮答案必然重算 |

未能核实：子 agent 引用的 vLLM issue #39321 / PR #39806 / SGLang PR #23315 的原文（本会话无 GitHub API 权限）；SGLang 代码中的 `strip_thinking_cache` 已足以证明问题和现有处理方式。

## 2. 但浪费有多大 —— 很小

关键推导（有前缀缓存时）：
- 第 t 轮的 prompt = 用户1 + 答案1 + … + 用户(t−1) + **答案(t−1)** + 用户t。
- 上一轮缓存里存的是 用户1 + … + 用户(t−1) + thinking(t−1) + 答案(t−1)。
- 两者从 thinking(t−1) 处分叉，所以**每轮只有“上一轮答案”这一段需要重算**，更早的所有轮次仍命中前缀缓存。

用 OpenThoughts-114k 前 100 条 R1 轨迹量级估算（字符数）：thinking 中位 13,139 字符（约 3.5k token），答案中位 1,532 字符（约 400 token），答案占 thinking+答案 的 10%（中位）。

| 口径 | ReThink-Reuse 能省下的比例（估算） |
|---|---|
| 用户感知延迟（一轮内要先解码几千个 thinking token） | 约 0.1%（省 ~400 token prefill ≈ 几十毫秒，对比几十秒的解码） |
| 批量服务的 GPU 时间（prefill 每 token 比 decode 便宜一个量级） | 约 1–3% |
| 纯 FLOPs 计数（prefill 与 decode 每 token FLOPs 相近） | 上限约 9% |
| 新一轮 TTFT（prefill 部分） | 可降数倍，但思考模型的首个可见回答 token 在 thinking 之后，TTFT 不是用户感知的瓶颈 |

工具调用循环（agent 场景下最长的轨迹）中 thinking 本来就保留，不受影响。

## 3. 方法层面的顶会先例 —— 方法不新

“把一段 KV 挪到新位置、RoPE 重新旋转、只重算偏差最大的少量 token”正是位置无关 KV 复用这条线的核心做法：

| arXiv | 论文 | 录用 |
|---|---|---|
| 2405.16444 | CacheBlend（选择性重算 KV 偏差最大的 token） | EuroSys 2025（据记忆，arXiv 页未写） |
| 2410.15332 | EPIC: Position-Independent Caching | ICML 2025（据记忆，arXiv 页未写） |
| 2502.05431 | APE: Adaptive Parallel Encoding | ICLR 2025 |
| 2409.15355 | Block-Attention | ICLR 2025 |
| 2311.04934 | Prompt Cache | MLSys 2024 |

子 agent 列的最接近工作都不是顶会：KVEraser 2606.17034（ICML 2026 **workshop** oral）、2606.17107（单作者，无录用）、RoPERelay 2610.21126 / IntentKV 2606.09916 / CommitKV 2608.07855（无录用信息）。

## 4. 结论

- **问题真实**：主流模型模板确实删 thinking，SGLang 的处理就是直接丢弃重算。
- **方法不新**：就是 CacheBlend / EPIC 这类位置无关 KV 复用在一个新场景的应用。
- **收益太小**：每轮只浪费上一轮答案的 prefill，延迟收益约 0.1%，GPU 时间约 1–3%。系统类顶会需要的是倍数级收益，撑不起一篇论文。
- 唯一有科学意义的子问题是：“看过 thinking 的答案 KV”拿来给下一轮用，质量会更好还是更差（模型训练时用的是删掉 thinking 的格式，存在训练/推理不一致）。这是一个小的分析点，适合作为 workshop 短文或 SGLang/vLLM 的工程 PR，不适合作为主线论文。

**判定：从 PROCEED（6/10）下调为不推荐作为主线（约 3/10）。**
