# visual-semantics-v1 数据契约

状态:待实现交付物的契约定义,已随本文件冻结;实现位于 `trace_annotate/workspace/semantic_schema.py`、`vision_schema.py`、`vision_sampling.py`。
适用:annotation 的 `attributes.semantic`、`evidence_ids`、模型运行记录、生成来源与人工决定。
原则:合约合法不等于语义正确;模型输出永远是候选;可观察、测量、解读三者不得混写。

## 1 版本与存储位置

- 语义版本:`visual-semantics-v1`。每条 annotation 的 `attributes` 只允许一个键 `semantic`,其值必须是 `{version, claims}`。
- `version` 必须等于 `visual-semantics-v1`;未知版本按未知未来版本拒绝,不静默降级。
- `claims` 是有界数组(最多 32 条),每条 Claim 是完整对象,不接受省略字段的简写。
- 大帧、帧矩阵、长曲线、完整模型输出不进入 annotation;它们作为工作区工件(CAS)登记,`evidence_ids` 只放已登记工件 id。

## 2 Claim

字段(全部必填,不允许额外键):

| 字段 | 类型 | 约束 |
|---|---|---|
| `id` | string | ASCII 标识符,`claim-<sha256前16位>` 形式;同一 annotation 内唯一 |
| `claim_type` | enum | `observable` / `measured` / `interpretation` / `unspecified` |
| `statement` | string | 1..2000 字符,人可读;不得为空 |
| `subject_ref` | string/null | 指向主体(annotation id 或 track);无主体为 null |
| `time_window` | object/null | `{start:{seconds}, end:{seconds}}`,有理数,左闭右开;整片声明为 null |
| `evidence_ids` | array | 0..32 个已登记工件 id;声明证据必须真实存在 |
| `method` | object | `{id, version, parameters}`,方法标识与版本不得为空 |
| `model_run_id` | string/null | 引用同一 `model_runs` 数组中的 id;规则方法可为 null |
| `score` | number/null | 有限数值;无分数为 null |
| `score_kind` | enum | `similarity` / `softmax` / `rule_score` / `confidence` / `none`;不得写 accuracy |
| `uncertainty_reason` | string/null | 低置信、遮挡、无主体等拒判原因 |

语义约束:

- `observable` 表示可直接从像素或音频听到/看到的内容,如"画面中出现笑容";模型给出的 observable 仍然是待确认候选。
- `measured` 表示带单位或有明确算法的测量,如"窗口 RMS -23.4 dBFS";必须给 `score_kind` 与方法参数。
- `interpretation` 表示人的判断或推断,如"人物显得紧张";只能由 `actor_type=human` 的作者写入,模型不得生成。
- `unspecified` 用于旧数据迁移与无法归类的内容;不得被当作已确认事实。
- 一条 Claim 只能是一种类型;混合描述必须拆成多条,否则留在 `unspecified` 待复核。

## 3 EvidenceRef

工件的登记事实在文档 `artifacts` 中(`id`、`relative_path=objects/<sha256>`、`sha256`、`size_bytes`)。annotation 的 `evidence_ids` 必须是这些 id 的子集。
Claims 的证据额外要求:

- 时间性证据(帧、帧矩阵、音频窗口)必须携带 `frame_index`、`source_pts`、`proxy_pts` 或 `audio_origin`;不能只写本地路径。
- 一个 evidence id 可被多条 Claim 引用;不得复制字节造第二份证据。
- 证据缺失时 Claim 只能 `unspecified` 或降低为待确认,不能凭方法名断言。

## 4 ModelRun

字段:

| 字段 | 约束 |
|---|---|
| `id` | `run-<sha256前16位>` |
| `provider` | `local` 或明确的外部服务名;外部服务默认关闭 |
| `model_name` / `model_version` | 非空 |
| `weights_sha256` | 64 位小写十六进制或 null(规则方法) |
| `license` | 非空字符串;未知写 `unknown`,不得编造 |
| `preprocessing` | 对象,记录尺寸、归一化、采样帧数等 |
| `parameters` | 对象,记录阈值与词表版本 |
| `input_sha256` / `output_sha256` | 输入媒体与结果工件的 SHA-256 |
| `code_version` | 代码版本或提交标识;未知写 `unknown` |
| `status` | `success` / `failed` / `not_run` |
| `failure_reason` | `failed`/`not_run` 必填,`success` 必须为 null |
| `score_kind` | 与 Claim 的枚举一致 |

状态语义:缺依赖或缺权重是 `not_run`;权重存在但损坏、加载失败或推理抛错是 `failed`;真实完成前向才是 `success`。`success` 可以包含低置信拒判,但必须另报有效覆盖率。

## 5 生成来源与人工决定

`generation_lineage`(不可变):

```
{"job_id","model_run_ids","input_sha256","result_sha256","parameters","created_at"}
```

- 人工接受后 lineage 不消失;它记录机器来源,不接受/拒绝决定覆盖它。
- 只把 `provenance.origin` 改成 `manual` 不算审核完成。

`review_decision`(独立保存):

```
{"actor_id","actor_type","decision","reviewed_at","content_sha256","previous_content_sha256","command_id","diff_summary"}
```

- `actor_type` ∈ `human` / `assistant` / `tool`;代理操作必须如实标注,不能冒充用户本人。
- `decision` ∈ `accepted` / `edited` / `rejected`;`edited` 需给出可审计的差异摘要。
- `content_sha256` 是决定生效后的内容摘要;任何后续内容修改都会使旧决定失效(旧摘要不再匹配)。
- 相同 `command_id` 的重复请求幂等;不同载荷使用同一 `command_id` 视为冲突。

## 6 审核与候选规则

- 模型产物: `provenance.origin=model_candidate`,`review_status=unreviewed`;禁止直接 `reviewed`。
- 模型候选可携带 `observable`、`measured`、`unspecified`;`interpretation` 一律由人写入。
- 接受候选必须走显式命令(accept/edit/reject),保留 lineage、证据与差异;不得只改 origin。
- 需要语义复核的混合描述保持 `unspecified` 并进入待确认队列。

## 7 摘要投影

`visual-semantics-v1.digest-projections.json` 列出参与摘要的字段与排除项,固定样例见同目录 `visual-semantics-v1.examples.json`。规则:

- Claim 摘要:`{"digest_version","claim"}` 中 claim 去掉 `id` 之外的任何自引用摘要字段;不加时间戳。
- 语义摘要:`{"digest_version","version","claims":[...按 id 排序...]}`。
- annotation 审核绑定摘要:`{"annotation": <去掉 review_status/review_binding 的 annotation>, "media_id","template_id","template_version"}`,与现有 store 规则一致。
- 模型运行摘要:`{"digest_version","run": <去掉 output_sha256 的 run>}` 用于"输入+参数+代码"身份;结果摘要单独用 output_sha256。
- 排除项:`content_sha256`、`review_binding`、任何 `created_at`/`reviewed_at` 之外的时钟值不参与内容摘要。

## 8 迁移

- 旧 annotation 无 `attributes.semantic` 时迁移为 `{"version":"visual-semantics-v1","claims":[]}`,并在迁移报告中标记 `claims=0`,不推断任何事实。
- 旧 `attributes` 非空但不符合本契约时拒绝导入并保留原文;不静默丢弃。
- 旧 `evidence_ids` 非空但引用缺失工件时拒绝,错误码 `UNSUPPORTED_PROFESSIONAL_EVIDENCE`。
- 迁移前后保留原始文档哈希与副本路径;新摘要不得与旧审核绑定混用。

## 9 失败码

| 码 | 场景 |
|---|---|
| `INVALID_SEMANTIC_CLAIM` | 字段缺失、未知键、类型非法 |
| `INVALID_CLAIM_TYPE` | claim_type 不在枚举 |
| `INVALID_SCORE_KIND` | 使用 accuracy 等未定义分数类型 |
| `INTERPRETATION_REQUIRES_HUMAN` | 模型写入 interpretation |
| `UNKNOWN_SEMANTIC_VERSION` | 版本不认识 |
| `UNSUPPORTED_PROFESSIONAL_EVIDENCE` | 证据未登记或越界 |
| `STALE_REVIEW_BINDING` | 内容已变而审核摘要仍旧 |

## 10 非目标

- 本契约不判断自然语言陈述是否真实,只保证类别、作者、证据、来源与审核可审计。
- 本契约不定义情绪类别学、动作词表或景别词表;这些由各 extractor 的词表文件冻结并记录版本。
- 本契约不声称任何精度达标;精度只在独立留出集评测报告中出现。
