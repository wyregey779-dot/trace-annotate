<div align="center">

# TraceAnnotate ⚡

**面向多模态媒体资产与具身智能视频的高性能、冷存储安全、全链路来源可溯标注套件**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](pyproject.toml)
[![Tests: 13/13 Passing](https://img.shields.io/badge/Tests-13%2F13%20Passing-success.svg)](tests/)
[![Offline: 100%](https://img.shields.io/badge/Offline-Zero%20External%20Traffic-orange.svg)](#)

[English](README.md) | [中文说明](README_CN.md)

</div>

---

## 💡 项目背景与解决的痛点

市面现有的开源标注工具（如 CVAT、Label Studio、AnyLabeling、FiftyOne）大多聚焦在**“画框/打点等前端交互”**或**“重型大模型云端 API”**。

但面对**几十GB/数TB的实拍长视频脏数据**以及**腕部/第一人称具身智能动作视频标注**时，工程师常常遭遇以下致命工程痛点：
1. **外置硬盘与冷存储极度脆弱**：并发直接读写外置硬盘极易造成 I/O 堵塞、掉盘断连（`WinError 21`）甚至物理数据损坏。
2. **AI 标注黑盒化、不敢信**：传统平铺 JSON 无法分清哪些字段是模型猜的、哪些是规则推的、哪些是人工确认的。
3. **真实脏数据误报率高**：景区历史雕像被误当成人脸、街头招牌电话被误当手机屏、景区门票特写被误判为自然风景。
4. **视频复核成本极高**：逐一打开几千个高码率视频用肉眼看，时间成本无法承受。

**TraceAnnotate** 诞生于生产级高压数据工程实践，首创**“冷存储安全 Staging 协议 (R1-R5)”**、**“字段级全链路来源溯源 (Fill 字典)”**、**“实拍脏数据复合判据 (F1-F3)”**，并完整融合了**“具身智能时序切片与 7 类错误归因套件 (Specification v2.1)”**。

---

## 📊 与市面主流工具对比

| 核心特性 | CVAT / Label Studio | FiftyOne | **TraceAnnotate** (本项目) |
|---|:---:|:---:|:---:|
| **冷存储保护 Staging 协议 (R1-R5)** | ❌ (仅支持直接读写) | ❌ (无硬件防爆机制) | **✅ 批次暂存、源盘只读白名单、双校验即用即删** |
| **字段级来源溯源 (Provenance)** | ❌ (仅输出平铺值) | ⚠️ (仅支持字段标签) | **✅ 独创 `fill` 字典 (`auto_metric`, `human_confirmed` 等)** |
| **雕像/文物假人脸拆分判据 (F1)**| ❌ (误报为真人) | ❌ (无领域规则) | **✅ 结合语境拆解 `statue_face`，解除真人假阳性** |
| **手机屏 vs 街头店招 OCR 判据 (F3)** | ❌ | ❌ | **✅ 界面形态词 + 文本区域形态复合判据** |
| **相似素材快速聚类** | ⚠️ (需插件扩展) | ✅ (向量相似度) | **✅ 感知 dHash + 并查集 (DSU) 高速聚类** |
| **4x4 快速拼板视觉审计** | ❌ (单图/单视频) | ⚠️ (Web 网格) | **✅ 秒级生成 4x4 拼板图 (`sheet_generator`)** |
| **具身智能动作时序切片** | ⚠️ (需手工拉框) | ❌ | **✅ Working/Idle 自动补集、步骤对账、7类错误归因** |
| **标注员逐秒 IoU Kappa 一致性 ($\ge 0.80$)**| ❌ | ❌ | **✅ 内置逐秒 tIoU 时间轴 Cohen's Kappa 测评** |
| **100% 纯本地离线 / 零外网流量** | ⚠️ (依赖容器) | ⚠️ | **✅ 零依赖离线执行，杜绝数据泄露** |

---

## 🏗️ 整体架构设计

```
                    ┌──────────────────────────────────────────────┐
                    │       Cold Storage (只读源盘 / H:\ / 移动硬盘) │
                    └──────────────────────┬───────────────────────┘
                                           │ Staging (R1-R5 安全暂存)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Core Pipeline Engine                                 │
│                                                                                  │
│  [1. 物理技术底账]           [2. 离线轻量推理]           [3. 复合判据与共识]       │
│  - size & head1mb_md5        - 图像质量 (Laplacian CV5)  - F1: 雕像假人脸拆分      │
│  - 感知 dHash 指纹           - YuNet 人脸检测器          - F2: 实体候选/确信分档   │
│  - 并查集 (DSU) 相似聚类     - RapidOCR 招牌提取         - F3: 手机屏/店招 OCR     │
│                              - 中文 CLIP zero-shot       - 场景 0.35 全帧切分      │
│                                                                                  │
│  [4. 全链路来源溯源 (Provenance)]                                                │
│  fill: { auto_metric | auto_rule | auto_path | pending_review | human_confirmed }│
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Human-in-the-Loop 快速视觉审计                           │
│  - 4x4 拼板核验板 (Contact Sheet) - P1-P4 优先级复核队列 - 追加式覆写 (human_overrides)│
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         具身智能时序标注与质检引擎                               │
│  - 整段诊断 (正常/佩戴异常/描述不符) - 手部操作 Working 段边界判定                 │
│  - 段中段 Hole (橙色)             - 自动化 Idle 补集时间轴计算                   │
│  - 预期步骤对账 diff S=[1..N]     - 7 种错误子类归因与证据模板                   │
│  - 逐秒 tIoU 统计与 Kappa 达标评估 - Label Studio v2.1 规范 XML / JSON 互转      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速上手

### 1. 安装

```bash
git clone https://github.com/your-username/trace-annotate.git
cd trace-annotate

# 安装轻量核心包
pip install -e .

# 安装完整多模态套件（含 OpenCV、RapidOCR、PyTorch）
pip install -e ".[full]"
```

### 2. 运行开箱即用示例

```bash
# 1. 运行媒体资产管线全流程演示（冷存储暂存、dHash聚类、复合判据、21键质检）
python examples/demo_asset_pipeline.py

# 2. 运行具身智能视频标注演示（Working/Idle切片、步骤对账、7类错误诊断、Kappa评估）
python examples/demo_embodied_pipeline.py
```

### 3. CLI 命令行工具速查

```bash
# 对标注总表执行 21 键终检回归
trace-annotate asset-verify my_annotations.jsonl --inventory my_inventory.jsonl

# 快速生成 4x4 视觉核验拼板图
trace-annotate asset-sheet thumbs/*.jpg -o audit_sheet.jpg --cols 4

# 具身智能：根据 working 段自动补全 timeline idle 补集
trace-annotate embodied-idle clip_001.json --inplace

# 具身智能：对账预期流程 S = [1..N]（秒出遗漏、乱序、重复步骤）
trace-annotate embodied-diff clip_001.json --nsteps 5

# 具身智能：双人背靠背试标计算时间轴 Cohen's Kappa（对标 EgoOops 0.80 门槛）
trace-annotate embodied-kappa annotator_A.jsonl annotator_B.jsonl --fps 2.0

# 具身智能：导出官方 Label Studio v2.1 界面配置 XML
trace-annotate embodied-xml -o label_studio_config.xml
```

---

## 🧠 Python 核心代码演示

### 示例 1：冷存储安全暂存与相似素材聚类

```python
from trace_annotate.core.staging import StagingManager
from trace_annotate.core.hasher import compute_dhash
from trace_annotate.core.cluster import cluster_by_dhash

# 遵守 R1-R5 铁律：源盘只读、4个一组分批暂存、双校验、处理完立即释放本地空间
staging = StagingManager(source_root="E:/RawFootage", staging_root="./staging")

hashes = {}
for batch in staging.stage_batch(["clip1.mp4", "clip2.mp4", "img3.jpg"]):
    for rel_path, staged_path in batch.items():
        hashes[rel_path] = compute_dhash(staged_path)
    # 循环体结束后，暂存副本立即被自动物理删除！

# 使用并查集按汉明距离 <= 10 进行聚类
item_to_cluster, clusters = cluster_by_dhash(hashes, hamming_threshold=10)
print(f"聚类发现 {len(clusters)} 个独立相似素材簇。")
```

### 示例 2：具身智能时序标注与步骤对账

```python
from trace_annotate.embodied.schema import EmbodiedAnnotationDoc, WorkingSegment, Verdict
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps

doc = EmbodiedAnnotationDoc(
    clip="wrist_assembly_01.mp4",
    text_description="1. 拾取左侧底板; 2. 安装支撑轴; 3. 拧紧固定螺母",
    working=[
        WorkingSegment(start=2.0, end=15.0, step_number=1, verdict=Verdict.CORRECT.value),
        WorkingSegment(start=18.0, end=30.0, step_number=3, verdict=Verdict.WRONG.value) # 第3步在第2步前执行！
    ]
)

# 1. 自动计算全时间轴 idle 补集
idle_intervals = compute_idle_complement(
    [{"start": w.start, "end": w.end} for w in doc.working],
    total_duration=45.0
)

# 2. 预期步骤自动化对账 diff
diff = diff_expected_steps([{"step_number": w.step_number} for w in doc.working], expected_step_count=3)
print(diff["missing"])       # 遗漏步骤: [2]
print(diff["out_of_order"])  # 乱序步骤: [{'previous_step': 1, 'current_step': 3}]
```

---

## 🔬 21 键终检完整性检验体系

系统内置自动化终检引擎，针对全库主表进行 21 项量化指标硬性检验：

- `row_count_match`: 标注行数与底账 inventory 严格一致。
- `asset_id_missing == 0`: 零未命名主键资产。
- `duplicate_ids == 0`: 严格无重复主键。
- `key_missing == {}`: 46 个 Schema 键在全量行位无一缺失。
- `phash_missing == 0`: 感知指纹覆盖率 100%。
- `video_windows_null == 0`: 视频资产可用窗口必须有落值。
- `face_missing == 0`, `ocr_missing == 0`, `clip_missing == 0`。
- `fill_missing == 0`: 全量字段 100% 具备来源归因审计码。

---

## 📖 详细文档

- [系统核心架构设计](docs/ARCHITECTURE.md)
- [具身智能视频标注 SOP 规范 v2.1](docs/EMBODIED_ANNOTATION_SOP.md)
- [46 键主表 Schema 与 Fill 溯源字典规范](docs/PROVENANCE_SCHEMA.md)
- [4x4 拼板目检与追加式覆盖工作流](docs/FAST_AUDIT_WORKFLOW.md)

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源发布。
