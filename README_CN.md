# TraceAnnotate

面向本地媒体资产复核与具身智能视频时序标注的 Python 工具集。

[![CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml/badge.svg)](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | [中文说明](README_CN.md)

TraceAnnotate 将媒体暂存、字段来源记录和 working/idle 时间轴处理组合在一起。项目目前处于早期阶段：自动化测试与合成示例验证部分具体行为，尚无公开证据证明大规模采用或规模化性能。欢迎贡献与可复现的问题报告。

## 当前能力

| 领域 | 已有工具 |
| --- | --- |
| 媒体准备 | 分批暂存、源目录与暂存目录边界检查、文件大小与前 1 MiB MD5 校验、图像 dHash 与近重复聚类 |
| 资产复核 | 亮度与清晰度指标、可配置的人脸/实体/屏幕规则、联系表拼图与优先级复核队列 |
| 来源记录 | 字段级 `fill` 来源、人工覆写、修订历史与回滚 |
| 时序标注 | working/idle 补集、预期步骤对账、区间校验与采样时间轴一致性评估 |
| 复核与交换 | 本地浏览器时间轴编辑器、HTML 质检报告、Label Studio 配置与解析、SRT、CVAT 风格 XML 和动作标签文本导出 |

规则与质量阈值用于辅助复核，不能直接当作真实标签。仓库中的 Label Studio “v2.1” 指本项目的标注约定，并非 Label Studio 官方产品版本。导出适配器有本地测试，但尚未验证所有外部软件或训练流程的导入兼容性。

## 安装

使用 Python 3.9 或更新版本。当前验证版本以 [CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml) 为准。

```bash
git clone https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

激活虚拟环境：PowerShell 使用 `.venv\Scripts\Activate.ps1`，Linux/macOS 使用 `source .venv/bin/activate`。然后安装并验证：

```bash
python -m pip install -e ".[dev]"
trace-annotate --help
python -m pytest tests -q
```

核心依赖为 NumPy、Pillow 和 tqdm。按需安装可选组件：

| 组件或依赖 | 用途与限制 |
| --- | --- |
| `python -m pip install -e ".[cv]"` | OpenCV 视频/运动处理与可选 HEIF 图像支持；YuNet 人脸检测还需兼容的本地 ONNX 权重。 |
| `python -m pip install -e ".[ocr]"` | RapidOCR 接口，模型和运行时是否可用取决于实际安装的包。 |
| `python -m pip install -e ".[full]"` | 安装可选视觉/OCR 依赖，安装体积较大，不代表所有模型推理功能已完成。 |
| `PATH` 中的 `ffmpeg` 可执行文件 | FFmpeg 场景切分辅助函数需要此工具；pip 不会安装该可执行文件。 |

**CLIP 接口尚未完成。** 当前适配器没有完整的图像/文本推理实现。`clip` 扩展只安装 PyTorch/torchvision，没有安装 `cn_clip` 或提供模型权重。请勿将其输出当作模型预测，后续事项见[路线图](docs/ROADMAP.md)。

核心示例与时间轴计算在本地运行，无需 OpenAI API 密钥。安装及可选模型包可能访问网络；离线使用前应准备并验证依赖与权重。

## 运行示例

```bash
python examples/demo_asset_pipeline.py
python examples/demo_embodied_pipeline.py
```

资产示例创建合成图片，并使用模拟人脸/CLIP 字段演示数据流，不用于测量检测准确率。它会创建并删除 `examples/_demo_scratch`，请勿在该目录存放自己的文件。具身标注示例构造合成记录，展示 idle 区间、步骤顺序与一致性结果。

无需模型的最小步骤对账：

```python
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps

working = [
    {"start": 2.0, "end": 10.0, "step_number": 1},
    {"start": 15.0, "end": 20.0, "step_number": 3},
]
print(compute_idle_complement(working, total_duration=30.0))
# [{'start': 0.0, 'end': 2.0}, {'start': 10.0, 'end': 15.0},
#  {'start': 20.0, 'end': 30.0}]
result = diff_expected_steps(working, expected_step_count=3)
print(result["missing"])       # [2]
print(result["out_of_order"])  # []：没有步骤编号递减的相邻转换
```

## 命令行

以下命令使用你自己的标注文件；记录结构见[标注 SOP](docs/EMBODIED_ANNOTATION_SOP.md)。

```bash
# 校验标注记录
trace-annotate embodied-validate clip.json

# 计算 idle 补集，补齐至视频真实总时长
trace-annotate embodied-idle clip.json --duration 60

# 对账预期流程步骤
trace-annotate embodied-diff clip.json --nsteps 5

# 采样比较两位标注员的时间轴
trace-annotate embodied-kappa annotator_A.jsonl annotator_B.jsonl --fps 2

# 生成本地质检报告，或打开时间轴编辑器
trace-annotate embodied-qc clip.json -o qc_report.html
trace-annotate embodied-ui

# 导出本项目的 Label Studio 配置或字幕
trace-annotate embodied-xml -o label_studio_config.xml
trace-annotate embodied-export clip.json --format srt -o clip.srt

# 检查资产总表，或生成拼图
trace-annotate asset-verify annotations.jsonl --inventory inventory.jsonl
trace-annotate asset-sheet thumb1.jpg thumb2.jpg -o audit_sheet.jpg --cols 4
```

`embodied-idle --inplace` 会覆盖输入文件；首次使用可不加此参数，先查看标准输出。传入真实 `--duration`，才能保留最后一个 working 段之后的 idle。暂存清理会删除暂存副本，请使用专用暂存目录并保留源文件备份。复制校验仅检查大小与前 1 MiB，并非全文件加密完整性验证或硬件写保护。

## 文档与参与

- [架构](docs/ARCHITECTURE.md)
- [具身标注 SOP](docs/EMBODIED_ANNOTATION_SOP.md)
- [资产 Schema 与字段来源](docs/PROVENANCE_SCHEMA.md)
- [视觉复核工作流](docs/FAST_AUDIT_WORKFLOW.md)
- [已知限制与路线图](docs/ROADMAP.md)
- [贡献指南](CONTRIBUTING.md)

可通过 [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues) 报告问题和讨论改进。请附上命令、版本、操作系统与最小合成复现，不要上传私人视频或密钥。

## 许可证

[MIT](LICENSE)。外部模型权重、可选运行时和数据集可能使用各自的许可证。
