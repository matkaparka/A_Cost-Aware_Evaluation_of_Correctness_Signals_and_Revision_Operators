# 设计 v2 审阅

审阅日期：2026-11-03。审阅版本：`4f65a56adf1fb41ca67440187d969cb97644abc0`。

范围：`v2_review_request.md`、当前 `docs/research_design.md`、`CLAUDE.md`、逐条回应、STATUS、DECISIONS、`.gitignore`，并参照第一轮审阅。文中的设计行号与协议行号均指上述版本。审阅快照的文件 SHA256 和本次构造例子的计算见 `review_support/v2_design_checks.json`，脚本为 `review_support/v2_design_checks.py`。

**结论：v2 有实质进步，可以据此继续实现和做 dev pilot，但还不能直接冻结为确认性运行协议。** 我同意保留带门槛的 H3b/Exp 5，不要求为了缩小范围而强制删除它们。需要修正的是：H1d 的识别对象、Exp 0 的错误不变量、若干主检验定义，以及预算和结果同步的实际保证。工程可运行、统计有功效、理论可解释是不同的验收条件，不能用其中一个代替另外两个。

本次没有运行模型、租用云实例、访问账户凭证或修改方案。新增的计算全部是设计反例与算术，不是实验结果。

## 一 对请求中 11 项的直接回答

| 请求项 | 审阅判断 | 需要的处理 |
|---|---|---|
| 1 H1d / F3-match | 有意义，但目前只在 dev 上校准；不是测试集精确匹配，也不能识别抽象的“外部证据效应” | 见 M1；重新命名 estimand，并选择明确的匹配方案 |
| 2 有界观察者 / V-information | 主体表述合理，比 v1 明显准确 | 见后文 C1；不要把准确率或 probe AUROC 称为 V-information 的测量 |
| 3 Exp 0 | 隐藏随机量的思路正确，现有性质与通过标准不正确 | 见 M2；二元有效输出、两个不同答案、独立随机数及确定性管线检查 |
| 4 Exp 1B 和各基线 | 角色×标签是实质改进；B0/B1 有用；F1/F2 仍不等计算预算 | 见 M4；明确主数据、主反馈条件和 H1a 的方法对比含义 |
| 5 Exp 3 | 分开方向、prefill-only、dev 选设置均可保留 | 见 M7；固定标签位置、hook 点、控制构造和正式成功标准 |
| 6 Exp 5 | 2×2 与相同答案 mask 修复了主要混杂 | 见 M8；固定归一化算子、异常更新、评估集合与后端规则 |
| 7 Tier 2 保留 | 可以接受，理论连接不是充分理由，独立 dev 筛选也不自动带来功效 | 见 M5；门槛失败必须真能退出该模块而不阻塞整个项目 |
| 8 统计 | 配对、聚类、嵌套 masks 的方向正确 | 见 M4–M6；补齐复合假设、TOST 的 null、cross-fitting 的不确定性 |
| 9 边界相对性和状态表 | 基本接受 | 删除“必要条件效应”可能造成的误读，见 C1 |
| 10 两机操作 | 可行的总体架构，但目前尚不能承诺自动停费和无冲突同步 | 见 M9–M10；有几处可以确定的运行问题 |
| 11 约 35 GPU-h 的先验 | 作为待替换的先验可以保留，不能视为已核准成本 | 见 M11；调用清单、优先顺序、存储和 dtype 需对齐 |

## 二 必须在确认性运行前修改

以下项目不要求全部现在实现完，但须在 prereg 冻结及相关确认性模块启动前有明确规则和验收结果。M9 的停费能力必须在首次无人值守付费运行之前解决。

### M1 高优先级 H1d 的“匹配”和研究对象仍有两层歧义

位置：设计 103、120、208 行；DECISIONS 中关于 H1d 的解释。

**第一层：dev 上的匹配不等于 test 上的匹配。** 用 dev 估计 `s_R,s_W` 后独立随机生成 test verdict，能得到以 dev 估计值为目标的 verifier。即使来自同一分布，也有校准估计误差和随机 mask 误差；难度分布变化会再增加差异。它并不保证两个条件在 test 上具有相同的 realized confusion matrix。目前写成“same class-conditional accuracies”过强，尤其 H1d 是 Tier 1。

**第二层：即使精确匹配，测到的也主要是错误与可纠正性的关联。** 完全匹配 `s_R,s_W` 和标签分布，固定了 `(initial_correctness, verdict)` 的联合分布，但没有固定 verdict 与题目难度、模型内部状态、可修复性之间的关系。相同 wrapper 下，模型也未必知道 verdict 的生成来源。

构造例：100 道初始正确题、100 道初始错误题。两个 verifier 的 `s_R=s_W=0.8`。错误题中 50 道只要获得“incorrect”就可修好，另外 50 道无法修好；正确题在错误否定下被改坏。独立分配的外部判定在两类错误题各识别 40 道，最终正确率为 60%。

- 自评识别可修复错误 30 道、不可修复错误 50 道：最终正确率 55%，外部组高 5pp。
- 自评识别可修复错误 50 道、不可修复错误 30 道：最终正确率 65%，外部组低 5pp。

这不是对 H1d 的否定；它说明 H1d 是可检验的**错误分配/关联假设**，而不是 Claim 7 或“外部必然更有证据”的检验。v2 的 motivation 已部分指出这一点，应把标题、DECISIONS 和 H4a 的解释全部统一。

**建议二选一并冻结：**

1. 保留当前生成法，改名为“dev-calibrated external verifier”。报告 test 上 `s_R,s_W` 的差异及校准不确定性；不要把结果写成严格控制可靠性后的效果。
2. 若主张必须是严格匹配，则预注册一个 **按 dataset × 初始正确性分层置换 self-verdict** 的对照：保持 test 上实际判定计数，随机改变判定落在哪些题上，使用多个固定置换。这个算法可以在 prereg 后使用标签，由隔离的 evaluator 执行；不是按结果调参。但要诚实称为使用真值分层的 oracle randomized control，不是部署时可获得的普通 verifier。

建议主 estimand 命名为“在类条件判定可靠性受控时，保留或打散自评与题目状态的关联对纠错的影响”。H4a 沿用同一含义。参数更新与 context 的交互效果则应另设明确 contrast，不能凭“两边各自有显著差异”判断耦合差异。

### M2 高优先级 Exp 0 的三个硬性判断目前不成立

位置：设计 179–188、366、375 行。

1. **任意 policy 的期望准确率不一定为 0.5。** 输出无关值、拒答、无法解析的 policy 可以为 0。必须保证两个可能的 gold 不同，并约束 policy 始终输出其中一个，才有等号。若两个隐藏取值映射到相同最终答案，准确率还可能为 1。否则，独立隐藏位通常给出的是上界，而不是等号。
2. **`switch if incorrect → accuracy=p` 有条件。** 初始答案必须是两个有效候选之一，verdict 必须判定这一初始选择，switch 必须精确切到另一个。GSM8K 风格自由生成中算错、答第三个值或输出格式失败时，这个等式不适用。
3. **随机检验失败不等于程序错误。** n=200 时，一个常见的 95% 中心二项接受区间为正确 86–114 题；完全正确的独立随机管线仍有约 4.00% 的概率落在外面。反复改实现或换种子直到“通过”，会选择性筛除合法随机波动。F3(1.0) 下真实 LM 没达到 90%，也可能是不会用反馈或不会做题，不能一概判定 verdict 管线坏了。

`accuracy / p` 不是有辨识力的 utilization ratio：忽略反馈的模型在 p=0.5 时也会得到 1。对于已约束的二元任务，可以报告 `(accuracy−0.5)/(p−0.5)`，只在 p>0.5 使用；它仍需 CI，且只是该特定策略基准下的增益比例。

**修法：**先用程序化二元 agent 验证 verifier、switch、scorer 的真值表，将其作为硬不变量；真实 LM 的表现另作为行为校准报告。可为同一个可见 prompt 构造隐藏位 0/1 的成对世界，共享同一次无反馈输出，在两个世界评分：有效二元选择必然一对一错，这比抽取 200 个独立位后要求“接近 50%”更适合确定性泄漏检查。配对实例必须在同一 split，不能把某一世界的反馈留给另一个世界。

对请求中的具体疑问：**只要隐藏位确实独立公平，模型偏爱题面中的 X 或 Y 不会产生真正的预测优势。** 风险在于生成器没有保证独立性、答案映射不唯一、共享随机种子引入关联、隐藏状态或随机种子进入了可见 metadata。应分别使用 hidden-state、verifier、sampling 随机流，并检查全部渲染输入。

### M3 高优先级 “gold 字符串不得出现”会拒绝合法实验

位置：设计 211、368、375 行。

正确的 a0 本来就包含 gold；Exp 1 明确把 a0 放进上下文，Exp 5 也对正确 a0 的答案 span 做 NLL。数学题中的数字还可能自然出现在题干。Exp 0 更是刻意展示两个候选。因此，字面上的 `gold strings absent` 和 `gold never enters a TTT loss` 与方案不兼容。将这类情况统称偶然命中再人工处理，不适合作为所有实验必须通过的不变量。

**修法：**检查信息来源和依赖关系：`question`、`model_generated_a0` 是允许输入；dataset `gold_answer` 只在 evaluator/verifier 中可用。更新损失只能取 a0 的 span，不能用 evaluator 的 gold 去替换。用反事实标签测试：保持 q、a0 与随机种子不变，改变私有 gold 后，无反馈 prompt 与更新的 token targets 必须不变；反馈条件仅允许预定 verdict/pointer 变化。生成文本恰好等于 gold 是合法成功，不是泄漏。

### M4 高优先级 Tier 1/2 仍没有唯一的主检验和成功条件

位置：设计 85–87、101、109、113–114、120、231–251、350–357 行。

- **H3a 数据集冲突：**Tier 1 总则指定 D-nat，Exp 1B 又写 D-inj primary。需要指定唯一确认性数据集，其余作为 replication。
- **H3a 不是一个现成检验：**写了 role 与 label 两项双侧检验，另有 F0/F1/F3(1.0) 三种 feedback，却要求每个 hypothesis 恰好一个 primary test。要指定主 feedback，拆成 H3a-role/H3a-label 并纳入 family，或使用明确的联合检验并限定结论。role×label 交互与边际风险差的定义也应给出，不能把 logistic 系数自动读成准确率差。
- **H2b 是复合命题：**`F2−F0≤δ_w` 应靠单侧上置信限/非劣检验支持，不是点估计满足即可。`oracle W→R>0` 太弱：F0=50%、F2=50%、oracle=1% 仍可能满足文字条件。若结论是 oracle 能纠正 intrinsic 难以纠正的错误，应直接比较 oracle 与 intrinsic/F0。可以定义 intersection-union test，取必要子检验 p 值的最大值，再作 family 校正。
- **H3b 的额外要求只有描述，没有统计门槛：**主检验只比较 random control；generic control 和 R→W 只“报告”，即使 generic 同样好甚至更好，也可能宣布 H3b supported。必须把“超出通用修改倾向”写成必要 contrast，或将结论限定为超过随机扰动。
- **H4a 的 context contrast 与 H1d 重合：**若用相同题目和输出，不应当作第二份独立证据。明确 H4a 主检验只针对参数 arm，还是更新×反馈来源交互，context 数字链接到 H1d 即可。
- **H1a 的比较仍非等总计算量：**F2 多了单独 critique turn；共同的 revision `max_new_tokens` 不能匹配总预算。可保留为“两个纠错方法的整体效果比较”，但不能说识别了纯 critique 效果。若仍以分解机制为主张，增加同样两次调用/总预算的中性 scratchpad 或重新求解对照。正确地不回归调整实际 tokens，并不能替代实验设计上的预算对照。

这些不是要求现在确定所有超参数，而是要求 prereg 对每项命题有唯一的、能够执行的判定规则。

### M5 高优先级 有条件保留 Tier 2 合理，但门槛不能代替功效和退出路径

位置：设计 86、97、337–341、353–354、381 行；协议 127–135 行。

dev 上选设置、test 上检验，在隔离正确时是合理的；因此我接受保留 H3b 与 Exp 5。但目前只有 Tier 1 明确要求功效，H3b 的“一套 dev 选设置”及 H4a 的“oracle 在 dev 上赢过 B1”都不保证 test 能识别目标差异。H2b 的 60 题门槛也不是某个 margin 下功效足够的证明。

另外，设计允许 Exp 5 gate 失败后降级，协议 P2 却要求“Pass the Exp 5 feasibility gate”，P3 又要求所有 invariants 通过。应区分“可选模块不适用/关闭”与“已启用模块实现错误”；失败的可选模块不应阻止 Tier 1 完成。

**需补充：**

- 每个进入确认性 family 的 Tier 2 检验也有预定功效/精度要求；否则估计性或探索性报告。
- δ 先按科学上可忽略的效果确定，pilot 主要估计 nuisance variance 和资源。当前“choose n and δ so power≥0.8”会鼓励为预算增大 margin；冻结前这样做未必破坏检验形式，但会改变研究问题，必须有独立理由。
- 用多个合理效应/不一致率情景估计功效，不能只把经过调优的 pilot 最佳效果当成真实效应。
- 固定 family 和 eligibility 判定时点；保守简化方案是为未启用假设保留位置并设 p=1，而不是在观察 post-feedback test 结果后缩小 family。
- 为 supported/not supported/inconclusive/estimation-only 定义规则。没有完成有功效的 TOST，就不要从一个“看起来较窄”的事后 CI 自动升级成确认性等效。
- 统计不变量要按实际原假设模拟：**零效应是等效性检验的备择情形，TOST 的边界 null 在 ±δ**。非劣检验也应检验其边界。单测“零差异数据假阳性率 5%”不能统一验证所有检验。

### M6 H2a/H2b 的 cross-fitting 与 CI 必须采用一致的估计目标

位置：设计 108–109、270–274、347、352 行。

test 内 cross-fitting 并不自动构成泄漏：每题 out-of-fold、预先冻结 layer/C 是合理的。然而，不同题目的预测共用训练折，某题标签会影响其他折的预测。若 bootstrap 只重采样一次训练后固定的 OOF 分数，它通常只描述条件于这些拟合结果的波动，不自动涵盖重新训练 probe、重新划分折及形成 H2b 五分位的整个分析过程。普通 DeLong 也不是对此依赖的自动修复。

**选择其一：**

- 使用独立 probe-train/dev，冻结最终 probe 后只在 holdout 上评估；或
- 保留 cross-fitting，预注册整个 learner 的不确定性估计：在按 canonical ID 的外层重采样中重新执行训练/OOF 预测及相应的分组统计，保证同一原题的重复样本不跨折，并通过覆盖率模拟检查；或
- 明确只报告条件性 CI，把涉及整个 learner 的强推断降级。

这一点应在 pilot 中解决，并加入功效/算力预算，而不是到看见 test AUROC 后才选择误差条算法。

### M7 H3b 启用前明确因果可达性和 generic control

位置：设计 249、278–288 行。

分开 role/label direction 是改进，但以下实现选择会决定实验究竟有没有干预到拟议原因：

1. **source label 要位于被取 activation 的 solution tokens 之前。** 若标签写在之后，因果 LM 的这些 hidden states 不可能依赖它；所谓 label direction 将没有所需信号。2×2 渲染必须固定 label 的位置、对话说话者指代和被 reviewed span。
2. 明确“fixed label/role”是平均另一个因子的两个水平，还是只使用某一水平；分别记录 direction 的极性。当前 assistant−user、self−other 的符号与“toward external”的 α 符号需一致。
3. **F1−F0 的 revision-start state 不天然等于通用“多修改”方向。** F1 有 Wait prefill，两组可能比较了不同 token、不同位置。需要对齐共同 probe token，并验证该方向确实提高修改倾向；否则它只是另一个模板/词汇扰动控制。
4. 明确 hook 在 attention 前、block 输出还是其他位置，并验证它改变哪些后续 KV。特别是只修改最后一层已处理 solution tokens 的 block 输出、又不修改最后预测位置时，未必能影响后续生成；不能把这样的结构性零效应解释为否定机制。
5. 固定 random/generic controls 的 layer、span、实际扰动尺度、随机方向数量和种子；用相同 backend 比较。dev 成功标准及 test 判定按 M4 写清。

prefill-only 本身没有原则问题；它得到的是通过被改变的历史状态/KV 影响后续计算的效果。需要用诊断检查证明该路径确实发生。

### M8 H4a 启用前固定更新算子并隔离 held-out 评估

位置：设计 302–341 行。

2×2 和统一答案 span 可以保留。NLL 与 token-unlikelihood 是不同算法，它们在相同 mask、相同范数下也不是同一种“信息安装强度”；研究结论应限于这条具体更新规则。

必须补齐的定义：

- `Σ||ΔW_m||_F` 是按模块求和的范数，区别于全体权重的 `sqrt(Σ||ΔW_m||_F²)`。两者都可预先选择，但要命名清楚。写明是否使用单一全局比例 `c=τ/N(ΔW)` 保持模块比例，以及如何通过 LoRA 因子实现。
- 给出零范数、极小范数、NaN、没有可提取答案 span 的规则。强制缩放会把极小更新放大到 τ，不能仅凭固定范数保证稳定；异常样本不能在见到结果后静默删除。
- 冻结训练 token 序列、v 的位置或仅用于选择 loss、优化器状态、loss 数值稳定计算、模块 dtype。随机 LoRA 控制要记录实际合成矩阵范数。
- 固定 KL/logit probe set 与 spillover set 的 ID 和 split。用于调 τ 的集合属于 dev；最终未见评估集不能因每次更新都打分而变成事实上的调参集。若两个集合都来自公开基准，也须纳入 canonical-ID 隔离。
- HF 训练/评分和 vLLM 生成需做 adapter 加载与零更新对齐。精确 token 一致应在同 backend/config 下要求，跨 backend 使用预定容差和不一致率，避免数值差异被当成训练效果。

oracle parameter arm 在 dev 上没有胜出，可以作为保守的退出门槛；但只能说“未达到这项任务的可行性标准”，不能推断“更新无法携带信息”。胜出也不保证 H4a 对 self/ext-match 差异有功效。

### M9 付费无人值守前必须补上真正可执行的停费与累计预算规则

位置：协议 176–192 行；STATUS 中 API key 为 optional 的说明。

**目前的“双端硬限额”不成立。** 如果 VM 上没有 API key，watchdog 只写 STOP；若此时控制笔记本休眠、断网或 agent 不再运行，两端都不会执行暂停。再加上“先 sync 再 pause”没有超时，网络故障可以使停费无限等待。用两份说明文字并不等于两个独立、持续运行的监督进程。

[官方 CLI 文档](https://docs.jarvislabs.ai/cli)确认 pause 停止 compute 计费但保留 storage 计费，支持 `--yes --json`；resume 可能更换 ID。文档还说明 `jl status` 是账户视图，`jl list` 的 cost 是运行 session 或暂停 storage 的成本，不能直接当作本项目跨多次 resume 的累计费用。

**应在小额验证前就写明：**

- 至少一条不依赖聊天 agent 持续在线的停费路径能独立认证并执行；如果做不到，就不允许运行 unattended。凭证缺失时应在作业启动前失败，而不是跑到预算末尾再写 STOP。
- 监督器从 VM 开始计费时计时，覆盖下载、安装、idle、模型载入，不只是某个 GPU job 的时间。实例 ID 更新与监督器绑定必须原子完成。
- 使用有限时长 checkpoint/sync；到绝对截止时，暂停优先，不因为同步失败继续付费。pause 用非交互方式，重试后轮询供应商状态确认完成；测试控制机断连、同步挂起、任务失败等情景。
- 维护跨 session、跨 ID 的累计台账与成本来源，不能拿账户余额变化或当前 session cost 替代项目累计花费。
- 明确余量的语义。每次重新计算 `0.85 × remaining` 不会保留固定的项目总额 15%；随着多次 resume，允许的累计花费会逼近 cap。若要固定项目 reserve R，应使用 `cap−spent−storage_reserve−R`，并记录 session 的绝对截止，不在运行中混用累计 elapsed 与不断缩小的 remaining。
- storage 会在暂停后持续发生。`CLOUD_DESTROY=never` 与“包括全部未来 storage 的绝对项目 cap”不能同时无限期保证。应规定存储保留期和对应 reserve，以及到期后的人工责任或事先授权的处置范围。该审阅不要求自动删除数据。

本次未启动云资源，也没有要求用户增加授权；以上是协议下一次执行前应满足的条件。

### M10 结果同步和 Git 规则会遗漏材料并破坏复现

位置：`.gitignore:5`；协议 33–38、68–73 行。

**已实际复现：**

```text
git check-ignore -v --no-index results/example/token_ids.jsonl
.gitignore:5:*token*  results/example/token_ids.jsonl
```

`prompt_token_spans.json`、`reports/token_usage.json` 也被忽略。当前设计恰好要求保存并从 VM 同步这些材料，普通 git add/commit 会漏掉它们。应改成具体 secret 文件规则，另用 secret 检查；并对预期 artifact manifest 的每项验证是否已同步及 hash 一致。

另有两个运行冲突：

- [Git 官方文档](https://git-scm.com/docs/git-config#Documentation/git-config.txt-receivedenyCurrentBranch)说明 `updateInstead` 默认拒绝 working tree/index 不干净的 push。VM 正在写 tracked `runs.jsonl` 或 results 时，从控制机推代码就可能失败。两端编辑不同文件并不能保证无需同步协调。
- VM 的结果 commit 会改变 HEAD；若恢复时拿“当前 HEAD”与原 run 的 code commit 严格比较，纯输出提交也会使合法 shard 被判过期。反过来，在运行期间允许更新代码 checkout，又会产生同一 run 读到不同代码的风险。

**建议：**每个 run 锁定只读 code commit/worktree；结果目录或 results branch 独立。记录 `run.code_commit` 与当前 repository HEAD 的不同含义。若继续单分支方案，则必须在 block 边界停止写入、提交结果、pull/merge、push、校验 clean 状态再启动下一块，且运行恢复始终以该 run 的锁定代码为准。

### M11 实际运行顺序与预算优先级矛盾

位置：协议 138–145 行；设计 389–420 行。

协议先跑“完整 1.7B suite”，设计 cut order 却把 4B 主实验放第一，1.7B 完整 suite 根本不在优先表内。照协议执行可能在拿到 4B 主要结果之前，把钱花在 1.7B 的机制、thinking 之外的扩展和 streams 上。P4 中 1C 也先于 Exp 2，而优先表未明确其位置；P5 才做 8B，则不能保证优先表中 8B core 先于部分 4B 探索项。

**修法：**生成一个跨模型的唯一 block manifest，包含依赖、tier、优先级、预估花费、最小完整样本和停止规则。pilot 做 1.7B/4B，正式运行先完成 4B Tier 1，再依 manifest 做 gated Tier 2、replication 与探索项。不要仅在两份文字中写“顺序相同”。

预算停止也不能把已取得的非随机前缀当完整 test 样本。应预先随机化 item 顺序，按同题完整配对块输出，并区分完整分析、预注册的部分样本分析和纯描述性中止结果。

## 三 可以随后完善的项目及有条件接受的设计

### C1 V-information 和理论状态表总体可接受，仍需三处限缩

[Xu et al. 2020](https://arxiv.org/abs/2002.10689)确实提出计算可以增加受限预测类的 usable information。v2 用它解释“没有新的 Shannon 信息仍可提高表现”是合适的。它不是补救 v1 行为推论的定理，也不保证任意反馈都增加 usable information。

- 设计 31 行改成“隐藏状态使这个上界具有非平凡的预测含义”，而不是“strict form only applies”——DPI 并非只在隐藏状态任务成立。
- 除非定义预测类 V、目标分布和 log-loss 估计过程，否则只称为概念解释；本实验的 accuracy、AUROC 不等于正式的 V-information 测量。
- 设计 37 行的“necessary-condition effects”仍容易暗示检验了必要性。建议写“在固定参数接口下比较两种反馈利用方法”。1-bit verdict 可以称最小 self-assessment signal，不能仅因这个命名认为已满足理论的 self-model 条件。

边界相对性和“按最弱理论状态写作”应保留。缺失 skeleton 可以阻止引用其定理，但无需阻止一篇明确以操作定义和实验证据为主的论文写出保守 framework；否则项目会把可完成的经验论文卡在一个已声明不属于本轮的理论任务上。

### C2 角色×标签 2×2 能识别提示干预，但不恢复真实历史来源

assistant/other、user/self 两个交叉格可能与对话历史发生语义冲突。只要提示明确、所有格均报告，它仍是有用的因子设计；结论应是角色与来源标签的提示效应。D-inj 的 assistant 格是人工归属，v2 已准确说明。真实模型生成历史与人工 prefill 的区别需要额外实验，不要把这一层含义塞进 H3a。

B0 是有效保留基准；非 thinking 的 greedy B1 在相同 q、prompt、预算和 backend 下往往与 a0 相同，这不是失败。但“resample”容易误导，应称 fresh-context/no-update baseline。真正的重采样效果由 sampled robustness 提供。

### C3 持续流不是简单沿用 per-item reset 规则

位置：设计 329–335 行。因 H4c/H4d 已标为探索性，不阻塞 Tier 1；但运行前仍需规定：

- 每一步 a0 和 self-verdict 来自当前已更新模型，还是冻结模型预生成的记录。只有前者真正形成所描述的内生反馈流，后者更接近固定伪标签的在线训练。
- τ 限制的是本步增量还是总 adapter；两者的长期动力学不同。保留 rank 8、优化器状态、缩放后 Adam moments 的处理也要一致。
- context-memory 使用 self 还是 external verdict，应与参数流匹配；删除最旧记录、文本预算和每步可见信息要固定。
- 三条 order seeds 可以描述，不足以证明一般稳定性。held-out 多次观测不是新的独立轨迹。
- H4d 的另一次前向消除了上下文污染，但“预测将给什么答案”与“回答题目”仍是两种提示任务，只能报告一致性 proxy。

### C4 算力先验可保留，但当前清单不是完整测量

27 × 1.3 = **35.1 GPU-h**，按文档的 $1.49/h 为 **$52.30 compute**。若还要留项目总额 15% 余量，仅此就需 cap 约 **$61.53**，尚未包括 storage/其他成本。按所列价格，200GB storage 的月费用为 **$20**，且开发期间即使暂停也继续累计。[官方价格页](https://jarvislabs.ai/pricing)

这些是算术，不是建议现在购买该额度。实际 VM 类型/区域价格要从其报价确认，A100 单位工作成本优于 H100 也需要实际吞吐比较，不能仅由时价推断。

调用清单还有待校正：

- 未扩容时 D-nat 总池 1819 题，80% test 约 1455，不能把 1800 同时称为 test n。
- Exp 5 的两个 context 格可复用同一 `(q,a0,v)` 训练得到的 adapter，不必重复训练；“8 update cells”需要列出是哪些独立训练和哪些只是评估。
- 我读了方案所指 Qwen3-4B revision 的 [config](https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a72/config.json)：rank-8 q/k/v/o 的名义参数量为 5,898,240；bf16 约 11.80MB，fp32 约 23.59MB，另加 metadata。应实测 adapter 保存 dtype，不能把 12MB 当通用保证。
- forward scoring、generation prefill/decode、反向更新、HF/vLLM 加载不能共享一个 tokens/s；norm/KL probe、cross-fitting 的再拟合、失败重试也应计入。
- adapter 删除后只留 hash 无法从 hash 恢复；至少保留完整可重建输入、初始化种子、训练与最终 rescale 配置，重要确认性实例可保留 adapter 抽样。

### C5 其余实现边界

- verbal verdict 的六种大小写/前导空白变体需在**锁定 tokenizer**上验证单 token 性质、token ID 去重和 yes/no 集合不重叠。取 first token 是尚需证明的捷径，而不是通用的序列概率定义。
- P0 使用“dev item”，P1 才划分 dev/test。P0 先用非研究集 synthetic fixtures，或将 canonical split 提前；不要临时抽研究集样本后忘记排除。
- 统计 CI 的“所有均 item bootstrap”与 streams 以 trajectory 为单位的特例应在文档中显式覆盖。
- “training contamination 不影响 within-item comparison”过强：配对可控制题目身份，但训练暴露仍可能改变不同条件的反应及外推性。它不会自动令所有配对结论失效，也不能被宣称完全无关。
- 最小 related-work matrix 应在 prereg 前完成，以便发现关键对照或方法缺口；P7 再更新最终论文引用。仅凭限定检索日期，也不能预先保证那句新颖性声明必然成立。

## 四 建议本轮修订的验收清单

1. H1d/H4a 的命名、matching 生成法、test 校准诊断和解释范围统一。
2. Exp 0 改为有条件的二元性质；程序不变量与 LM 行为结果分开。
3. gold 检查改为来源/依赖检查，合法的正确 a0 可通过。
4. prereg 表为 H3a、H2b、H3b、H4a 给出唯一主检验、主数据和明确复合判定。
5. 所有启用的确认性 Tier 2 有功效/精度依据；未启用模块可以退出；TOST 在正确 null 边界做校准。
6. H2 的 OOF 训练、CI 与功效模拟使用一致的估计目标。
7. 若启用 Exp 3/5，M7/M8 的干预、更新和评估隔离规则已落实并通过独立 fixtures。
8. 停费与累计费用监督已在控制机失联、同步失败等情景验证；storage 保留期明确定义。
9. `.gitignore` 不遗漏必需结果；code commit 与 results commit 分离，恢复与同步无矛盾。
10. 唯一的 block manifest 优先完成 4B 主实验；预算不足时保留真实的完成状态与停止原因。

达到以上要求后，我会把 v2 的修订版判断为“可以进入确认性运行”。这仍是对设计与执行约束的验收，不代表实验会支持预定假设，也不代表整套动态回归理论已经得到验证。
