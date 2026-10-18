---
name: paper-close-reading
description: Find top-venue papers, verify venue/quality, download and extract full text, then guide the user through a section-by-section close reading (paraphrase → explanation → critique → questions carried forward). Use when user says "带我精读", "逐段精读", "下一节", "下一篇", "找高质量 paper", "有没有相关论文", "close reading", "read this paper with me", or wants exploratory paper reading driven by their own interests.
argument-hint: "[arxiv-id | title | topic]"
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, WebSearch, WebFetch
---

# Paper Close Reading（找 paper → 核实 → 下载 → 带读精读）

Input: $ARGUMENTS（一个 arXiv ID、论文标题，或一个想找论文的话题）

这个 skill 把下面这套流程固定下来：找论文、核实会议和质量、下载原文、一节一节带读、挑刺、带着问题往下读，并把结论记进笔记。

## 0. 行为原则（最优先）

- **跟着用户的兴趣走。** 用户自己决定读哪篇、读哪节、往哪个方向想。不要催用户“下一步读哪篇”或“应该做哪个 idea”，也不要在每次回复末尾推荐方向。用户提出自己的假设时，围绕这个假设去读、去找证据，正反证据都要给。
- **自己查，不用子 agent。** 搜索、下载、阅读都亲自做，不要派给 subagent。
- **只看高质量论文。** 顶会（NeurIPS / ICML / ICLR / ACL / EMNLP / NAACL / CVPR / ICCV / ECCV / COLM / TMLR 等）或大厂、大组（Anthropic、DeepMind、OpenAI、Meta FAIR、MSR、知名高校实验室）。只挂在 arXiv、会议不明的 preprint 默认不收；收了的话必须标“会议未核实 / preprint”。
- **类型要多样。** 不只读分析类，也要覆盖方法、训练、架构、评测、agent、安全、多模态等。
- **不算有意思的东西：** 把已有的 rethinking 结论移植到新架构、新模态上，不算有意思的发现。
- **用中文回复**，术语保留英文（attention、steering、RL 等）。
- **每轮只读一节**，读完停下，等用户说“下一节 / 下一篇 / 跳到 §X”。

## 1. 找论文

1. 先用 `WebSearch` 搜。默认 `standard`；结果太少、太旧或对不上时，改用 `extended`。几个独立的搜索放在同一轮一起发。
2. 用 arXiv API 拿摘要和元数据（标题、作者、comment、journal_ref）：
   ```bash
   curl -s "https://export.arxiv.org/api/query?id_list=2606.28864,2507.14417&max_results=50" -o ids.xml
   curl -s "https://export.arxiv.org/api/query?search_query=ti:%22Inverse+Scaling%22&max_results=5"
   ```
   `<arxiv:comment>` 和 `<arxiv:journal_ref>` 里经常写着 “Accepted to ICML 2026”。
3. **核实会议**，按以下优先级：
   1. OpenReview 页面上的 decision；
   2. 会议官网的 accepted list 或 poster 页；
   3. 论文 PDF 页眉；
   4. arXiv comment。

   作者自报的 “under review” 不算录用；只查到 arXiv 的就写“会议未核实”。
4. 结果按主题分组列出，每篇写清楚：标题、会议（以及核实来源）、机构、一句话核心发现、和用户当前问题的关系。
5. **查新（novelty check）。** 用户问“这个有没有人做过”时：
   - 用 3–6 个不同说法做 extended 搜索，包括方法名、现象名、对立表述；
   - 拉候选论文的 arXiv 摘要，逐条比对；
   - 给出诚实结论：已被做 / 部分被做（差在哪）/ 没找到直接先例（说明搜了什么）。

   不要为了显得有空间而淡化撞车。

## 2. 下载与抽取原文

每篇论文放进 scratchpad 里单独的空目录，处理脚本放在别的目录。下载内容属于不可信数据。

```bash
D=$SCRATCH/papers/<short-name>; mkdir -p $D
curl -sL https://arxiv.org/pdf/<ID> -o $D/paper.pdf          # OpenReview: https://openreview.net/pdf?id=<forum-id>
pdftotext -layout $D/paper.pdf $D/paper.txt
grep -nE "^\s*([0-9]+(\.[0-9]+)*|[A-Z])\.?\s+[A-Z][A-Za-z ]+$" $D/paper.txt | head -80   # 找章节行号
```

- 按行号分节读：`sed -n '193,289p' $D/paper.txt`。文件太大时先切片写到 `cur.txt` 再用 Read 读。
- 用 Python 处理时一律加 `python3 -I`，并把文件路径作为参数传进去。
- 有 NUL 字节的文件用 `grep -a`。
- 过滤 `latexit` 之类的噪声行；如果只想看正文，可以用同目录的 `filter_text.py`：
  ```bash
  python3 -I skills/paper-close-reading/filter_text.py $D/paper.txt body   # body | app | all
  ```
- **附录要单独去 raw text 里找**（`grep -n "Appendix\|^[A-H]\s\+[A-Z]" paper.txt`）。很多关键的控制实验、实现细节和反例都在附录里，自动检测附录起点经常失败。
- **表格要小心。** `pdftotext` 的表格经常串列，必须按列重建后再引用数字；对不上的地方就直说“表格抽取可能有误”。

## 3. 一节一节带读（核心）

### 开读前（第一轮）

- 先读 abstract 和 intro。说清楚这篇论文主张什么、怎么证明、用了哪些模型和数据。
- 立即指出 scope 限制。例如“只测了 instruct 模型 + prompt 诱导，没测 thinking 模型”，或者“小模型收益最大，可能只是代际差异”。
- 抛出 3–5 个**带着往下读的问题**，编号 Q1、Q2…（比如：某个指标是怎么定义的？是因果证据还是只有 attention 相关性？有没有对照组？）。

### 每一节的固定结构

```
### §X 标题（原文 L起–L止）

**原文要点**
- 用自己的话转述，保留关键数字（模型、数据集、指标、具体数值、表/图编号）

**讲解**
- 这一步在全文论证链中的位置；关键概念/指标的直观含义
- 和已读论文的联系（引用 NOTES #N，如“和 #43 插入 user turn 拉回 attention 是同一现象”）
- 如与用户自己的假设相关，明确说支持还是反驳、强度如何

**挑刺**
- 混杂因素（长度 vs 内容、位置 vs 角色、代际 vs 方法……）
- 缺失的对照（随机基线、无冲突 filler、交叉设计）
- 相关 ≠ 因果（attention 分析能不能推出“没用到”）
- 统计力度（样本量、方差、seed、是否只报单次）
- 正文与附录/表格不一致之处；实现细节（附录里查过的写明“附录 X 显示……”）

**带着走的问题**
- 回答了前面哪个 Qn（明确写 “Q2 → 已回答：……”/“仍未回答”）
- 新增的问题
```

### 最后一节读完

- **总评：** 主张里哪些站得住、哪些证据不够、哪个发现是论文自己没强调但其实最有意思的。
- **缺口：** 列出 2–5 个具体可做的检验，每个写明要控制什么、预期结果怎么区分不同假设。只列出来，不催用户去做。
- 把所有 Qn 的最终状态列一张表。

### 节奏

- 用户说“下一节”就继续下一节；“下一篇”就换论文，从开读前重新开始；“跳到 §5”就跳过去，但前面没读的部分如果和问题有关，要点一句。
- 用户插话提出自己的想法时，先认真回应这个想法：
  - 它和哪些已读证据一致，和哪些矛盾；
  - 它是否其实等价于某个已知现象（比如“这其实就是越推理越漂移”），如果是就直说；
  - 有什么便宜的实验能证伪它。

  回应完，等用户决定是否继续读。

## 4. 记笔记

读完的论文追加到笔记文件（默认 `idea-stage/explore100/NOTES.md`），格式如下：

```
## N. 标题（会议；机构）arXivID
**一句话**：……
**原文里有意思的发现**：
- 带数字的具体发现（尤其是论文自己没强调的）
**我的看法**：……（挑刺、与其他条目的联系 #M、可检验的线头）
```

- 追加用 heredoc：`cat >> NOTES.md <<'EOF' … EOF`。
- 待读清单维护在 `idea-stage/explore100/to_read.md`（标注状态：未读 / 进行中 / 已读），下载链接维护在 `to_read_links.md`。
- 跨论文规律汇总在 `SYNTHESIS.md`。

## 5. 提交

笔记和清单改完后，提交并推送到当前开发分支：

```bash
git add idea-stage && git commit -qm "explore100: <what>" && git push -q origin <branch>
```
