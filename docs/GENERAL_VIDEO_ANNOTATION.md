# 通用视频片段标注

适用版本：`1.1.0.dev0` 开发版。已发布的 `v1.0.2` 不含本指南的新命令。适用任务包括普通视频的事件、动作、镜头片段、素材可用区间和人工字幕；不需要具身步骤、working/idle 或对错判定，也不使用模型自动识别。

## 安装与打开

在包含本次改动的源码目录执行：

```bash
python -m pip install -e ".[dev]"
trace-annotate --version
trace-annotate video-ui
```

也可以执行 `trace-annotate video-ui --print-path`，再手动打开输出的 HTML 路径。页面自带脚本与样式，通过标准文件选择器读取本地视频，不需要服务器或 API 密钥。浏览器必须支持视频本身的容器和编码；安装 OpenCV 不会改变浏览器的解码能力。

浏览器不能直接播放的格式，可以使用 `trace-annotate video-open input.mov -o new-project`。本机 FFmpeg/ffprobe 准备并核验兼容副本，再通过只读本机服务自动打开编辑器；原片和逐帧时间映射保留。已有项目使用 `trace-annotate video-ui --project new-project` 重开。HDR 预览需显式添加 `--tone-map-hdr`。操作与保存方式见[跨格式兼容指南](MEDIA_COMPATIBILITY.md)。

## 人工标注与保存

1. 打开本地视频，等待有限时长、可解码首帧及完整 SHA-256 校验完成。纯音频、浏览器只能播放音轨的文件会被拒绝。哈希按 1 MiB 分块在后台 Worker 中执行，显示进度，可以取消；通过前不会替换当前视频和标注。首帧通过只证明开头可解码，不能证明整条文件完整或内容正确。
2. 播放或拖动到起点和终点，用按钮或 `[`、`]` 记录时间；也可以直接填写秒数。步进是时间跳转，不保证逐帧定位。
3. 填写自由标签，例如“人物进入画面”“镜头移动”，以及文字描述、标签词和轨道。保存片段后可继续修改、删除或撤销。新片段默认未复核，人工确认后再改为已复核。
4. 同轨、跨轨事件都可以重叠。未标注区间只表示尚无记录，不会被推断为空闲或负样本。
5. 下载 JSON 作为完整主文件。下次先打开匹配视频，再导入此文件继续编辑。v2 文档核对完整内容哈希、大小和时长；同内容改名后可以明确确认重新关联，保留标注与复核。

文件名、大小和时长检查不能证明媒体内容相同。新文档和草稿绑定完整 SHA-256，即使另一个视频三项弱信息均相同，也不能接续已有 v2 标注。浏览器草稿只存标注、不存视频；恢复必须明确操作。浏览器清理数据、文件页存储限制或空间不足都可能导致草稿不可用，JSON 文件仍是持久保存方式。哈希组件需要 Web Worker；不支持时加载明确失败，当前项目保留，不退回整文件内存计算或弱身份匹配。

旧 v1 JSON 或旧文件名/大小草稿仍可读取。恢复前需明确确认已经核对当前视频并迁移到 v2，所有旧片段变为待复核。此举不能补证旧标注属于哪条原片；旧源文件和旧草稿不会被删除。取消迁移保留当前项目。新哈希键下的草稿损坏或版本不支持时会报错，不自动回退并使用另一个旧草稿。

修改表单后应先保存片段，再切换或导出。错误输入不会变成已保存片段。选择其他视频或导入其他项目时，应先下载当前 JSON。取消选择文件不应清空标注。

编辑以原生数据为准：只替换实际改动的字段。例如只改终点时，原有标签、轨道中的换行和描述中的 CRLF 会原样保留；未改动而直接保存也不会重写这些文本。主动编辑某个文字字段时，该字段采用浏览器控件中的新文字（换行通常为 LF）。标签和轨道支持多行，选择已有轨道时显示转义名称，避免把带换行的轨道误认成另一条轨道。

复核只对应当前内容。修改时间、类别、描述、标签集合或轨道后，旧的“已复核”自动取消；应核对修改后的内容，再明确勾选。撤销会恢复相应历史内容及其复核状态。草稿保存已保存片段、未保存表单及复核绑定；旧版草稿可恢复，但旧表单已改内容时需要重新复核。

如果播放器在编辑中解码失败，已保存记录和表单会保留。表单还有未保存修改时，JSON 备份会明确询问是否仅导出已保存片段；这份 JSON 不包含未保存表单，页面会继续提示，不能把它当作所有草稿都已保存。浏览器草稿失败时应保留页面并另行复制表单文字。

## 数据格式

秒是规范时间单位，区间采用 `[start, end)`。不假设固定 25 或 30 FPS。以下是仍受支持的 v1 示例，省略视频文件本身；绑定实际媒体后由页面或 `video-bind` 迁移到 v2：

```json
{
  "kind": "temporal",
  "schema_version": 1,
  "clip": "example.mp4",
  "duration": 12.0,
  "metadata": {"project": "人工事件标注"},
  "segments": [
    {
      "id": "event-1",
      "start": 1.0,
      "end": 4.5,
      "label": "人物进入画面",
      "text": "人物从左侧进入。",
      "tags": ["人物", "室内"],
      "track": "事件",
      "review_status": "unreviewed"
    }
  ]
}
```

- `kind` 固定为 `temporal`，`schema_version` 为整数 `1` 或 `2`。v1 支持已有文件，但不允许添加 `media_sha256` 冒充已绑定文档；v2 必须含 `media_size` 和 64 位小写十六进制 `media_sha256`。
- `clip` 是非空文件名，不接受目录分隔符、冒号、NUL、`.` 或 `..`。`media_size` 是不超过 JavaScript 安全整数范围的非负字节数，v1 可省略。
- `duration` 为有限正数。时间必须是实际数值，布尔值、字符串、NaN 和 Infinity 都无效；每段满足 `0 <= start < end <= duration`。所有数值（包括 metadata）绝对值不超过 `9007199254740991`，按 IEEE 754 双精度处理；不保留 JSON 原文的小数位写法。文本要求合法 Unicode，不接受孤立代理码元。
- 每段所有示例字段都必填。`id` 在文档内唯一；`label` 和 `track` 非空；`text` 可为空；`tags` 为非空字符串组成的数组，数组本身可以为空。
- `review_status` 仅允许 `unreviewed` 或 `reviewed`；这是人工复核状态，不是模型置信度或标签准确率。
- 可选 `metadata` 为 JSON 对象，用于项目扩展。未知根字段或片段字段会报错，不会静默丢掉；重复 JSON 键也会被拒绝。具身 working/idle JSON 不能直接导入此模式。

完整合成示例是 [`general-video.json`](../examples/annotations/general-video.json)：4 段、2 轨，包含两个同轨重叠事件和两条字幕。其内容为合成演示，不是实际视频观察或模型输出。

## 媒体绑定与时间反馈

```bash
trace-annotate video-hash example.mp4
trace-annotate video-bind annotations-v1.json --media example.mp4 --duration 12 -o annotations-v2.json
trace-annotate video-bind annotations-v2.json --media renamed.mp4 --duration 12 --allow-rename -o renamed-annotations.json
```

命令行的 `--duration` 是调用方提供的实际时长；命令不解码视频，也不能从任意文件哈希成功推断它是有效视频。绑定保留 ID、文字、tags、metadata 和原始秒数；v1 迁移只增加媒体身份、升级版本并取消旧复核。v1 必须同名；只有 v2 的完整哈希与大小匹配，才能用 `--allow-rename` 重定位。时长差超过 1 毫秒或任何终点越过媒体时长时拒绝绑定，不裁剪。Python 分块读取时检查普通文件属性前后变化；这不保证抵抗恶意并发改写并恢复文件时间戳的进程。

播放器将秒级播放位置与最近呈现帧 PTS 分别显示。呈现帧反馈使用浏览器的 [requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)，不可用时明确显示秒级模式。记起止点仍保存播放器秒数；此反馈不构成逐帧索引、前后帧导航或 VFR 精确验收。取得实际帧索引并验证解码/呈现映射后，才能声明准确帧定位。

浏览器哈希实现随软件离线打包，包含 MIT 许可的 js-sha256 0.11.1。精确上游版本、提交、摘要和许可证见随包的 `temporal/ui/MEDIA_HASH_LICENSE.txt`；运行时不会下载库或上传视频。

## 命令行校验与导出

```bash
trace-annotate video-validate examples/annotations/general-video.json
trace-annotate video-export examples/annotations/general-video.json --format json -o general-copy.json
trace-annotate video-export examples/annotations/general-video.json --format csv -o general-table.csv
trace-annotate video-export examples/annotations/general-video.json --format srt --track "字幕" -o general-subtitles.srt
trace-annotate video-export examples/annotations/general-video.json --format vtt --track "字幕" -o general-subtitles.vtt
```

所有新输出命令都拒绝已存在的目标，包括输入文件本身和符号链接。重新运行请使用新路径。命令先校验和转换，再创建结果；输入错误不会生成一个看似有效的空输出。

CLI 在目标同目录写完临时文件并刷新到磁盘后，通过硬链接原子发布完整结果。短写、磁盘错误或目标并发出现时会报错，不覆盖既有目标，也不把半写内容当作最终文件。输出文件系统必须支持硬链接（例如 NTFS）；FAT/exFAT 或部分网络盘不支持时会明确失败，请改用支持的本地目录。此机制不保证断电后的目录项持久化；强制结束进程或清理权限不足可能留下带点号前缀的临时文件。若已发布但临时清理失败，错误会明确说明结果已写入，应检查后使用新路径重试。

| 格式 | 用途 | 信息边界 |
| --- | --- | --- |
| 原生 JSON | 保存、恢复、继续编辑 | 保留全部受支持字段、ID、数值与 metadata |
| CSV | 人工表格复核 | 每行一个片段，含 ID、起止秒、标签、文字、JSON 编码 tags、轨道、复核状态；不保存全部视频/项目元数据，不提供 CSV 导入 |
| SRT / WebVTT | 字幕播放器或下游文本检查 | 只投影一个轨道的时间与文字，text 为空时使用 label；不保留 ID、tags、轨道、复核状态或 metadata |

CSV 带 UTF-8 BOM。可能被表格软件当作公式的文本以单引号前缀保护，包括以 `= + - @` 开头或危险制表/换行前缀的单元格。因此 CSV 也不是无损主文件。不得通过导出表格后删除 JSON 的方式保存项目。

页面在 CSV / SRT / WebVTT 下载前显示导出片段数、待复核数及丢失的信息；字幕还说明所选轨道。可以取消下载，标注不会因此改变。待复核为零只表示人工勾选状态，不代表通过了独立质量审计。

多轨字幕必须显式传 `--track`；字幕不会自动混轨。同轨重叠事件会保留重叠时间，但部分播放器不能同时显示全部重叠字幕。字幕时间按总毫秒舍入后拆分时分秒，最大误差约 0.5 毫秒；极短片段若舍入成零时长会报错。字幕文本中的空白段落、时间箭头及不支持的标记会被拒绝，不会被悄悄拼接成另一条字幕。需要这些原文时使用 JSON 或 CSV。

## 导入现有字幕

```bash
trace-annotate video-import input.srt --format srt --clip example.mp4 --duration 60 -o imported-srt.json
trace-annotate video-import input.vtt --format vtt --clip example.mp4 --duration 60 --track "字幕" -o imported-vtt.json
trace-annotate video-validate imported-srt.json
```

`--duration` 必须来自视频实际时长。导入生成新的片段 ID、`label="字幕"` 和 `unreviewed` 状态，不会假称恢复了原始标签或复核结果。页面的导入入口读取原生 JSON，字幕文件先通过此命令转换。

支持范围是带时间戳的纯文本字幕子集，保留连续多行文字。WebVTT 的样式、区域、位置设置、语音标记、内联时间及其他扩展不承诺支持；遇到不支持的结构会报错。它不是任意 SRT/WebVTT 的全兼容编辑器。跨字幕导出/再导入会改变 ID 等字段，并可能受到毫秒精度影响；无损恢复使用 JSON。

## 与具身流程及后续工作的关系

原有 `embodied-ui`、校验、idle、步骤对账、报告和导出继续使用原有数据规则，两种文档不自动互相转换。需要时可单独设计有字段映射和损失说明的迁移工具。

下一阶段的空间标注需要独立对象/轨迹结构，涵盖目标框、多边形、目标 ID、关键帧、遮挡/离场、插值、人工修订与标准导出；本轮片段编辑器未实现这些功能。之后才评估自动镜头切分、语音转录、Label Studio / CVAT 等格式适配及协作服务。任何上游算法接入都要经过来源许可、接口、依赖、真实媒体和失败路径验证。

## 复现浏览器检查

开发者可按[贡献指南](../CONTRIBUTING.md)运行 `python scripts/check_general_editor.py --output output/browser-check`。它使用 FFmpeg 合成视频和独立浏览器，验证真实页面的增改删、重叠、JSON 恢复、字幕下载回读、错误输入和响应式布局，并保存截图及 JSON 结果。该检查不访问用户已登录的浏览器，也不能替代原生文件选择器、其他浏览器、真实业务素材及外部标注平台的验收。

另运行 `python scripts/check_editor_integrity.py --output output/editor-integrity-check`，复现文字保真、复核失效、撤销和未保存表单恢复等回归；运行 `python scripts/check_media_identity.py --output output/media-identity-check`，验证同弱身份异内容、全文件哈希、旧文档迁移、改名重定位与取消加载。运行 `python scripts/check_media_admission.py --output output/media-admission-check`，验证纯音频、改名 MP4 的音频、无法解出画面的 ProRes+AAC、异常时间线和无效文件不会替换现有标注与未保存表单。运行 `python scripts/check_prepared_editor.py --output output/prepared-editor-check`，验证兼容转换、本机 Range 播放、标注与保存恢复。所有命令都要求新的输出目录；CI 会分别执行并保留验收产物。

## 文件兼容性和验收边界

“通用”指自由的时间片段类别。直接加载模式仍取决于浏览器；2026-10-23 的 Windows 本机测试使用 Chrome 155.0.8059.85 与独立 Chromium 153.0.8010.12，直接播放的 HEVC 10bit 支持存在差异，ProRes MOV、MPEG-4 Part 2 AVI、H.264 TS、FFV1 MKV 样本被拒绝。新增兼容入口可先转换这些格式，再用统一副本完成播放与标注。同扩展名可能包含不同编码，最终以实际解码、转换核验和编辑流程为准，不据文件后缀承诺所有组合。

页面使用从零起始且时长稳定的浏览器时间线；直接加载时发现非零播放器起点，或播放期间时长改变，会拒绝加载或暂停编辑。兼容入口将声音与画面的共同起点归零，并保留、验证原片到副本的逐帧映射；原片不变，新文件使用新的内容身份，旧 JSON 不能直接改哈希套用。

片段、事件、动作描述和手工字幕可以由用户填写。目标框、多边形、跨帧目标 ID、自动语音转写、逐字 OCR、模型自动理解和逐帧精确导航尚未包含在此编辑器中。播放流程通过不能代替语义复核。单个 JSON 的页面导入上限为 20 MiB；当前时间线全量绘制，历史使用文档快照。单次长片或大量片段测试不能构成任意规模的性能保证。
