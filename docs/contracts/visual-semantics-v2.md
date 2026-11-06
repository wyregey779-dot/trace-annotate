# visual-semantics-v2 数据契约

状态:已实现并冻结。实现位于 `trace_annotate/workspace/semantic_schema.py`、`vision_schema.py`。
适用:annotation 的 `attributes.semantic`、`evidence_ids`、模型运行记录、生成来源与人工决定。
v1 继续作为历史契约被读取和校验;不要用 v1 的摘要规则重算 v2 文档,也不要把 v2 字段写回 v1。

## 1 版本与兼容

- 新写入的语义对象版本为 `visual-semantics-v2`;`visual-semantics-v1` 仍被接受,但只按 v1 字段集校验。
- 每条 annotation 的 `attributes` 只允许一个键 `semantic`,其值为 `{version, claims}`。
- 未知版本按未知未来版本拒绝,错误码 `UNKNOWN_SEMANTIC_VERSION`,不静默降级。
- `claims` 最多 32 条;每条 Claim 是完整对象,不接受省略字段的简写。

## 2 Claim 字段

v2 在 v1 的 11 个字段上增加一个必填键 `measurement`(可为 null):

| 字段 | 类型 | 约束 |
|---|---|---|
| `id` | string | `claim-<sha256前16位>`;同一 annotation 内唯一 |
| `claim_type` | enum | `observable` / `measured` / `interpretation` / `unspecified` |
| `statement` | string | 1..2000 字符,人可读 |
| `subject_ref` | string/null | 主体引用;无主体为 null |
| `time_window` | object/null | `{start:{seconds},end:{seconds}}`,有理数,左闭右开 |
| `evidence_ids` | array | 0..32 个已登记工件 id;必须属于 annotation 的 evidence_ids |
| `method` | object | `{id, version, parameters}` |
| `model_run_id` | string/null | 引用同一文档 `model_runs` 中的 id;规则方法为 null |
| `score` | number/null | 置信度或规则分数;不得承载测量值 |
| `score_kind` | enum | `similarity` / `softmax` / `rule_score` / `confidence` / `none` |
| `measurement` | object/null | 仅 `measured` 必填,其他类型必须为 null |
| `uncertainty_reason` | string/null | 低置信、遮挡、无主体等拒判原因 |

## 3 measurement

`measurement` 是独立对象,字段固定为:

```
{"value": 数值, "unit": "单位", "basis": "measured|declared", "analysis_window": time_window|null}
```

- `value`:有限数值,可为负;布尔值不算数值。
- `unit`:非空,最多 120 字符;无量纲写 `ratio`、`count` 或 `none`,不得留空。
- `basis`: `measured` 表示本轮真实算法/模型测得;`declared` 表示引用外部声明值。
- `analysis_window`:实际参与计算的窗口,与 claim 的 `time_window`(事件有效区间)分离。
- `score` 与 `measurement` 独立:测量值不能塞进 `score`;`score` 表示该方法对本次结果的置信度或规则分数。

## 4 四类语义

- `observable`:可直接从画面或音轨观察/听到的内容;模型输出仍是候选。
- `measured`:带单位或明确算法的测量;v2 必须有 measurement,否则 `INVALID_SEMANTIC_CLAIM`。
- `interpretation`:人的判断或推断;只能由 `actor_type=human` 写入,模型不得生成。
- `unspecified`:旧数据或无法归类内容;不得自动升级为事实。
- 一条 Claim 只能是一种类型;混合描述拆成多条。

## 5 事件时间、取证窗口与上下文窗口

- `time_window` 是事件有效区间,例如硬切两帧之间的区间、渐变的真实渐变区间。
- `measurement.analysis_window` 是计算窗口,例如 RMS 统计的 1 秒窗。
- claim 的 `method.parameters` 中的 scene/context window 是上下文窗口,不能替代事件区间。
- 三者在 UI、导出、证据包和评测中读取同一份 `time_window`,禁止各自解释。
- 帧身份统一使用 `frame_index` + `source_pts` + `proxy_pts`;时间统一左闭右开。

## 6 证据与来源

- 证据必须是不可变工件,annotation 的 `evidence_ids` 必须已登记;claim 的 `evidence_ids` 必须是其子集。
- `model_run_id` 必须指向同一文档声明的 `model_runs`,缺失或伪造按 `INVALID_SEMANTIC_CLAIM` 拒绝。
- 纯规则结果 `model_run_id` 可为 null,但 `method.id` / `method.version` / `method.parameters` 必须记录真实规则身份。
- 真实模型输出不得用空 `model_run_id` 逃避溯源。

## 7 摘要

v2 摘要与 v1 分离,防止同一内容在两个版本下产生相同摘要:

- Claim 摘要:`{"digest_version":"visual-semantics-v2","claim":<去掉 id 的 v2 投影>}`,投影包含 `measurement`。
- 语义摘要:`{"digest_version":"visual-semantics-v2","version":"visual-semantics-v2","claims":[...按 id 排序...]}`。
- 模型运行摘要与 annotation 审核绑定摘要沿用 v1 规则,不随 Claim 版本改变。
- 固定样例见 `visual-semantics-v2.examples.json`;投影定义见 `visual-semantics-v2.digest-projections.json`。

## 8 迁移

- v1 文档原样读取,不自动改写;需要 v2 时由显式迁移生成新文档并保留旧摘要与副本。
- v1 的 measured 把数值放在 `score`;迁移到 v2 时写入 `measurement.value/unit`,并把 `score` 重置为置信度或 null。
- 无法确定单位的历史 measured 保留 v1,不猜测单位。
- 迁移前后保留原始文档哈希与副本路径;新旧审核绑定不得混用。

## 9 失败码

| 码 | 场景 |
|---|---|
| `INVALID_SEMANTIC_CLAIM` | 字段缺失、未知键、类型非法、v2 measured 缺 measurement |
| `INVALID_CLAIM_TYPE` | claim_type 不在枚举 |
| `INVALID_SCORE_KIND` | 使用 accuracy 等未定义分数类型 |
| `INTERPRETATION_REQUIRES_HUMAN` | 模型写入 interpretation |
| `UNKNOWN_SEMANTIC_VERSION` | 版本不认识 |
| `UNSUPPORTED_PROFESSIONAL_EVIDENCE` | 证据未登记或越界 |
| `STALE_REVIEW_BINDING` | 内容已变而审核摘要仍旧 |

## 10 非目标

- 本契约不判断自然语言陈述是否真实,只保证类别、作者、证据、来源与审核可审计。
- 不定义情绪类别学、动作词表或景别词表;这些由各 extractor 冻结并记录版本。
- 不声称任何精度达标;精度只在独立留出集评测报告中出现。
