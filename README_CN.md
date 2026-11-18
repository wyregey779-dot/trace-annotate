# TraceAnnotate

面向本地媒体复核与视频时序标注的 Python 工具集。

[![CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml/badge.svg)](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) · [中文说明](README_CN.md) · [快速上手](docs/QUICKSTART.md) · [版本发布](https://github.com/wyregey779-dot/trace-annotate/releases)

当前已发布预览版：[v1.1.0.dev1](https://github.com/wyregey779-dot/trace-annotate/releases/tag/v1.1.0.dev1)，包含多视频工作台、专业 JSON 与带标注视频导出。
最新稳定版仍为 [v1.0.2](https://github.com/wyregey779-dot/trace-annotate/releases/tag/v1.0.2)。

TraceAnnotate 提供本地媒体复核与视频时序标注。开发版新增通用片段模式，支持自由事件标签、文字描述、标签词、多轨道与重叠事件；具身流程继续提供 working/idle、步骤检查与对错判定。Python 库还提供媒体分批暂存、字段来源记录与资产复核辅助功能。

试验工作台新增[专业 JSON 交换与客观媒体分析](docs/PROFESSIONAL_JSON.md)：导出完整当前标注、精确帧锚和已有报告，历史文件需保留项目备份。后台分析明确区分播放代理与原片，结果确认保存后才进入标注修订；测量不会自动证明内容准确或完成声音听审。

尚未发布的 `1.1.0.dev2` 视觉辅助分支还增加有界且未经人工复核的对象框观测、同场景稀疏轨迹与几何关系候选、MediaPipe 21 点手部观测，以及可选的 16 帧 VideoMAE 窗口提示。应用候选时，专业 JSON 只携带确定性的限量截面；完整原始结果仍保存在耐久视觉任务和项目备份中。这些观测不能证明身份、接触、抓握、持有、状态变化、动作完成或模型准确率，也不属于已发布的 `v1.1.0.dev1` 预览版。

工作台还支持导出带标注的视频：保留完整画面，在上下左右添加黑底白字面板，自定义字号、字段和轨道。先预览精确帧，再下载验证后的视频及同一修订的 JSON。请按[工作台入口与操作顺序](docs/PROFESSIONAL_JSON.md#open-the-workspace)启动；原有普通视频和具身标注入口继续保留。

## 试用多视频工作台

`1.1.0.dev1` 预览版和 `main` 包含试验性工作台；稳定版 `v1.0.2` 不包含这一功能。请准备 Python 3.9 或更新版本，并将 FFmpeg 与 ffprobe 加入 `PATH`（参见[实测工具链与兼容边界](docs/MEDIA_COMPATIBILITY.md)），然后运行：

```bash
git clone https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境，Linux/macOS 使用 `source .venv/bin/activate`。在源码目录之外、支持硬链接的本地文件系统上选择新的项目目录，执行：

```bash
python -m pip install -e ".[dev]"
trace-annotate workspace init --project ../traceannotate-workspace
trace-annotate workspace serve --project ../traceannotate-workspace
```

打开命令输出的本地网址，选择视频，等待导入完成后，在各自任务中创建并保存标注。工作台支持选择解码帧、重叠多轨、专业 JSON 和可选的带标注视频导出；后续使用同一个项目目录继续工作。请保留原片和经验证的项目备份：JSON 导出不包含完整媒体及全部历史工件。详见[工作台指南](docs/WORKSPACE_DEVELOPMENT.md)和[专业 JSON 指南](docs/PROFESSIONAL_JSON.md)。

![TraceAnnotate 工作台中的实拍视频、片段编辑、相邻帧对照与多轨时间轴](docs/media/workspace-demo.png)

这是实际浏览器中的多轨标注示例，展示已保存的片段、视频预览和相邻帧对照。画面取自 Elene B 在 Wikimedia Commons 发布的 [《Patata tortilla》](https://commons.wikimedia.org/wiki/File:Patata_tortilla.webm)。截图将视频取帧与工作台标注合在同一画面，按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供。

原有 `video-ui` 和 `embodied-ui` 仍是独立流程，其文件和工作台任务不会自动互通；两者都没有框选和对象轨迹编辑界面。工作台的可选视觉提供器只生成待复核候选，不会证明标注语义正确。

## 普通视频片段标注（开发版）

新功能位于 `1.1.0.dev1` 预览版，稳定版 `v1.0.2` 尚不包含。请在当前源码目录安装 `python -m pip install -e ".[dev]"`，然后运行：

```bash
trace-annotate video-ui
trace-annotate video-open input.mov -o new-annotation-project
trace-annotate video-validate examples/annotations/general-video.json
trace-annotate video-export examples/annotations/general-video.json --format srt --track "字幕" -o output-subtitles.srt
```

选择本地视频后，可以创建片段、修改时间和标签、按轨道组织事件，并下载 JSON 供下次继续编辑。多个事件可以同时发生，不会被自动解释为工作段或判定对错。JSON 是完整主文件；CSV、SRT 和 WebVTT 是表格或字幕投影，不能完整保留所有字段。新命令拒绝覆盖已有输出。详见[通用视频标注指南](docs/GENERAL_VIDEO_ANNOTATION.md)，包含字幕导入、草稿恢复、格式限制与后续框选/跟踪计划。

新页面使用 v2 文档，把标注和草稿绑定到完整媒体 SHA-256；后台 Worker 本地分块计算，可以取消。同名、同大小、同时长但内容不同的视频会被拒绝。旧 v1 文档需确认迁移并重新复核；同内容改名后可重新关联。命令行提供 `video-hash` 与 `video-bind`。独立的 `video-ui` 页面分别显示呈现帧时间戳和秒级播放位置，该页面的准确逐帧导航仍待实现。

浏览器不能直接播放 ProRes、AVI、TS 或其他本地格式时，使用 `video-open 原片 -o 新项目目录`。它调用本机 FFmpeg/ffprobe 准备 H.264/AAC 兼容副本，逐帧核对时间对应、保留原片身份与音轨偏移，再自动打开编辑器。继续编辑使用 `video-ui --project 项目目录` 并导入上次下载的 JSON。HDR 预览需显式添加 `--tone-map-hdr`；损坏、加密或无法可靠解码的文件会明确失败。详见[跨格式兼容指南](docs/MEDIA_COMPATIBILITY.md)。

## 从一个完整示例开始

快速上手示例依次完成 idle 计算、记录校验、步骤对账、HTML 报告和 SRT 导出。使用合成标注数据，无需视频文件、模型权重或 API 密钥。

```bash
git clone --branch v1.0.2 --depth 1 https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境，Linux/macOS 使用 `source .venv/bin/activate`，然后执行：

```bash
python -m pip install .
python examples/run_quickstart.py --output output/quickstart
```

预期最后输出：

```text
Verified: 2 working segments, 3 idle intervals, 5 subtitle entries.
```

打开 `output/quickstart/qc-report.html` 查看报告。同一目录中还会生成处理后的 `assembly.json` 和 `assembly.srt`。脚本拒绝写入已存在的输出目录；再次运行时请换一个新路径。

![合成标注示例生成的质检报告](docs/images/quickstart-qc.jpg)

截图来自该示例生成的实际报告。标签是用于展示记录格式的合成数据，不是模型预测。[完整教程](docs/QUICKSTART.md)包含逐条命令、预期间隔和常见问题处理。

## 当前功能

| 领域 | 功能 | 状态与范围 |
| --- | --- | --- |
| 媒体准备 | 分批暂存、源/暂存目录边界检查、使用中锁定、文件大小与前 1 MiB MD5 校验 | 本地文件系统操作，有回归测试 |
| 资产复核 | 亮度/清晰度指标、dHash 聚类、可配置规则与拼图 | 辅助复核；标签仍需人工判断 |
| 来源记录 | 字段级 `fill` 来源、人工覆写、修订历史与回滚 | Python 库 |
| 通用片段 | 自由标签、文字、标签词、重叠多轨、复核状态、本地编辑器与 JSON/CSV/SRT/WebVTT | 开发版；尚无目标框和跟踪编辑界面 |
| 具身标注 | 浏览器编辑器、working/idle 补集、步骤对账、区间校验与采样一致性评估 | 独立命令与数据规则 |
| 报告与交换 | HTML 质检、SRT、Label Studio 配置/解析、CVAT 风格 XML、动作标签文本 | 有本地格式测试；外部软件及版本兼容性仍需往返验证 |
| 可选视觉/OCR | 运动、人脸、OCR 和场景辅助接口 | 需要相应运行时、可执行文件或本地模型 |
| 工作台视觉候选 | 通过耐久 `vision-jobs` 执行场景切分、关键帧、人脸/OCR/CLIP 分析和 VLM 描述草稿 | 保持 `model_candidate`/`unreviewed`；缺少权重时明确为 `not_run` |
| 对象与关系观测 | 有界 COCO-80 对象框、同场景稀疏关联摘要和保守几何关系候选 | 只读辅助并绑定不可变帧证据；不提供空间编辑 UI，也不证明身份连续、接触、持有或动作 |
| 手部观测 | 有界 MediaPipe Hand Landmarker 结果，保留完整 21 点归一化/像素/世界坐标 | 仅为待复核观测；不推断接触、抓握、持有状态、人物身份或动作 |
| 辅助动作窗口 | 固定 VideoMAE/SSV2 模型对 16 帧窗口给出有限的关闭/打开/拿起/放到表面提示 | 可选本地非商业权重；不绑定主体、对象和接收者，不证明精确动作边界或动作完成 |
| 可移植视觉投影 | 在专业 JSON 中保存确定性截面、来源哈希、总量/省略量、预算和不可变帧证据 | 参数最多 128 KiB、12 帧、24 个对象观测、12 个轨迹摘要、12 个关系候选、4 个手部观测；完整原始数据需保留项目备份 |
| CLIP 推理 | 通过 open_clip 进行真实零样本分类 | 由 `clip` 扩展启用；缺少权重时返回 `not_run`，不会伪造标签 |

## 安装发行版

使用新版工作台时，在虚拟环境中安装固定版本预览包：

```bash
python -m pip install "https://github.com/wyregey779-dot/trace-annotate/releases/download/v1.1.0.dev1/trace_annotate-1.1.0.dev1-py3-none-any.whl"
trace-annotate --version
```

版本输出应为 `1.1.0.dev1`。视频处理还需单独安装 FFmpeg/ffprobe；预览版存储和流程仍在演进，请保留原片与经过验证的项目备份。

使用 Python 3.9 或更新版本，核心依赖为 NumPy、Pillow 和 tqdm。不需要源码目录时，可在虚拟环境中安装固定版本源码包：

```bash
python -m pip install "https://github.com/wyregey779-dot/trace-annotate/archive/refs/tags/v1.0.2.zip"
trace-annotate --version
trace-annotate --help
```

核心时间轴处理在本地执行，无需 OpenAI API 密钥。安装及可选模型包可能需要网络；离线使用前请准备好依赖与权重。

参与开发时，在源码目录执行 `python -m pip install -e ".[dev]"`，具体流程见[贡献指南](CONTRIBUTING.md)。

## 标注和复核自己的视频

![使用合成操作视频的实际具身标注界面](docs/media/embodied-editor.png)

[具身标注界面、质检截图与 12 秒浏览器操作录屏](docs/DEMO_MEDIA.md)展示独立的具身流程，并非新版工作台截图。录屏使用合成素材与明确指定的测试标签，不代表真实视频语义准确率。

```bash
trace-annotate embodied-ui
```

选择浏览器支持的本地视频，标记区间，填写步骤编号并复核判定，然后导出 JSON。新建区间默认为 **无法判断**。选择其他视频会重置编辑器中的标注，请先导出当前工作。

使用命令行处理导出的记录：

```bash
trace-annotate embodied-validate clip.json
trace-annotate embodied-diff clip.json --nsteps 5
trace-annotate embodied-qc clip.json -o qc-report.html
trace-annotate embodied-export clip.json --format srt -o clip.srt
```

开发版增加统一质检入口：`trace-annotate embodied-pipeline clip.json --nsteps 5 --report output/new-run/report.html`。它会联动结构校验、预期步骤检查与可选双人比较，并通过退出码阻止未通过或待复核的批次自动放行。请提供真实媒体时长、独立确定的预期步骤数，并为每次报告选择新路径。[质检流水线指南](docs/QUALITY_PIPELINE.md)说明退出码、安全补齐、人工仲裁边界与 Windows 批处理。

需要重新计算 idle 时，执行 `trace-annotate embodied-idle clip.json --duration 60`，把时长改成视频真实秒数。默认打印生成的 JSON；只有准备覆盖文件时才添加 `--inplace`。保留真实总时长才能包含末尾空闲区间。

字段与标签约定见[标注 SOP](docs/EMBODIED_ANNOTATION_SOP.md)。`examples/` 内另外两个脚本分别用合成数据演示资产处理与时间轴一致性。资产示例使用模拟人脸/CLIP 字段，并会创建、删除 `examples/_demo_scratch`。

## 可选依赖

在源码目录中按实际需要安装：

| 命令或依赖 | 用途 |
| --- | --- |
| `python -m pip install -e ".[cv]"` | OpenCV 视频/运动处理与可选 HEIF 支持；YuNet 人脸检测还需要兼容的本地 ONNX 权重 |
| `python -m pip install -e ".[ocr]"` | RapidOCR 接口，模型与运行时取决于实际安装的包 |
| `python -m pip install -e ".[full]"` 并另行准备 MediaPipe 手部模型 | 可选 21 点手部观测；程序校验指定本地模型身份，缺少权重时明确为 `not_run` |
| 另行取得固定 VideoMAE/SSV2 检查点 | 可选 16 帧窗口提示；权重不随安装包提供，需配置为本地目录 |
| `PATH` 中的 `ffmpeg` 与 `ffprobe` | 视频处理外部程序；二者都必须可解析，pip 安装包不会暗中下载 |

`clip` 扩展安装 PyTorch/torchvision 与 open_clip，可执行真实零样本推理；首次构建可能把 OpenAI ViT-B-32 权重下载到用户缓存。`vlm` 扩展安装经过测试的 transformers 4.49.x、einops 与 timm，用于本地描述草稿。所有适配器遵循同一状态约定：缺少权重为 `not_run`，执行崩溃为 `failed`，只有真实前向推理才是 `success`；输出在人工升级前始终是 `model_candidate`/`unreviewed`。

VideoMAE 推理代码使用 Apache-2.0 的 `transformers`，但固定的 `MCG-NJU/videomae-small-finetuned-ssv2` 权重采用 CC-BY-NC-4.0。该权重不包含在 TraceAnnotate 的 MIT 安装包中，也不构成商业使用授权；仅在自身用途符合该许可证时另行获取和配置。项目代码、运行时、模型权重和输入数据可能分别受不同许可证约束。

## 验证与当前限制

[CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml) 构建发行包、在源码目录外检查安装后的命令行与内置资源、运行回归测试和示例。测试矩阵覆盖 Windows、Linux 与 Python 3.9、3.11、3.13。[版本说明](https://github.com/wyregey779-dot/trace-annotate/releases)记录各版本实际执行的检查。

- 项目处于早期阶段。测试与合成数据验证具体行为，尚无公开证据确立大规模采用、模型准确率或规模化性能。
- 暂存清理会删除暂存副本，应使用专用目录并保留源文件备份。复制校验覆盖文件大小与前 1 MiB，不是全文件校验，也不是硬件写保护，无法防御所有并发文件系统改动。
- 采样时间轴一致性不等于语义标签质量。复核规则与阈值需要人工解释。
- 可移植视觉声明只是耐久本地结果中有明确预算与损失说明的截面。结构合法不等于模型准确，也不代表对象、手部或动作已完整覆盖；需要省略数据时必须保留项目备份。
- 单机模型执行成功不能代替干净安装、支持版本矩阵或发布就绪证据；独立人工复核的准确率数据和外部工具往返仍未完成。
- 仓库中的 Label Studio “v2.1” 是本项目的标注约定，不是 Label Studio 官方产品版本。外部工具的兼容版本将随往返验证逐步记录。

## 文档与贡献

- [快速上手与预期输出](docs/QUICKSTART.md)
- [架构](docs/ARCHITECTURE.md)
- [标注 SOP](docs/EMBODIED_ANNOTATION_SOP.md)
- [资产 Schema 与字段来源](docs/PROVENANCE_SCHEMA.md)
- [视觉复核工作流](docs/FAST_AUDIT_WORKFLOW.md)
- [路线图与验收条件](docs/ROADMAP.md)
- [贡献指南](CONTRIBUTING.md) · [安全问题报告](SECURITY.md) · [更新记录](CHANGELOG.md)

通过 [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues) 报告可复现问题或提出具体改进，请附命令、项目/Python/系统版本、预期结果与最小合成数据。标注边界情况、外部格式往返检查、文档纠错都适合作为初次贡献。

## 许可证

[MIT](LICENSE)。外部模型权重、可选运行时和数据集可能使用各自的许可证。可选的固定 VideoMAE/SSV2 权重采用 CC-BY-NC-4.0，不随安装包提供，也不会继承本仓库的 MIT 许可证。
