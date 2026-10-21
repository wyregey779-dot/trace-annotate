# 46-Key Multi-Modal Asset Schema & Provenance Guide

TraceAnnotate standardizes video and image metadata into 46 deterministic keys. Every row contains a `fill` sub-dictionary recording the generation source of each field.

---

## 1. Schema Definition

| Group | Field Name | Type | Description | Sample Values |
|---|---|---|---|---|
| **Identity & Hashing** | `asset_id` | str | Unique asset ID | `"AST_0001"` |
| | `rel_path` | str | Relative path from root | `"scenic/lake_01.mp4"` |
| | `kind` | str | Asset media category | `"image"`, `"video"` |
| | `ext` | str | File extension | `".mp4"`, `".mov"`, `".jpg"` |
| | `size` | int | Size in bytes | `10485760` |
| | `head1mb_md5` | str | MD5 of first 1MB | `"d41d8cd98f00b204e9800998ecf8427e"` |
| | `full_md5` | str/null | Full file MD5 | `"..."` (video only) |
| **Technical Metrics** | `width`, `height` | int | Frame dimensions | `1920`, `1080` |
| | `aspect_ratio` | float | Width / Height | `1.7778` |
| | `duration` | float/null | Video duration in seconds | `15.5` |
| | `fps` | float/null | Frame rate | `30.0` |
| | `video_codec` | str/null | Video stream codec | `"h264"`, `"hevc"` |
| | `audio_codec` | str/null | Audio stream codec | `"aac"` |
| **Visual Quality** | `brightness` | float | Mean luminance (0..255) | `124.5` |
| | `sharpness` | float | Laplacian variance | `45.2` |
| | `is_bad_frame` | bool | Too dark, blown, or blurry | `false` |
| **Fingerprints & Clustering** | `dhash` | str | 16-hex perceptual hash | `"0f1e2d3c4b5a6978"` |
| | `similar_cluster_id`| str | DSU Near-dup cluster ID | `"cluster_0012"` |
| **Semantics & Entities** | `primary_entity` | str | Primary subject/location | `"岳麓书院"` |
| | `entity_status` | str | Ground truth confidence | `"confirmed"`, `"candidate"` |
| | `tags` | list[str]| Semantic keywords | `["室外", "建筑", "秋景"]` |
| | `indoor_outdoor` | str | Environment classification | `"室内"`, `"室外"` |
| | `time_of_day` | str | Lighting condition | `"白天"`, `"夜"` |
| | `weather` | str | Weather condition | `"晴朗"`, `"雨"` |
| | `screen_type` | str | On-screen or storefront | `"无屏"`, `"店招"`, `"手机屏"` |
| **Faces & People** | `face_count` | int | Detected face count | `1` |
| | `face_status` | str | Person context | `"无人物"`, `"单人特写"` |
| | `crowd_level` | str | Density classification | `"无人"`, `"单人"`, `"人群密集"` |
| | `face_subject_kind` | str | Face disambiguation (F1) | `"real_person"`, `"statue_face"` |
| | `face_review` | bool | Flag for human visual audit | `false` |
| **OCR & Text** | `ocr_text` | str | Raw concatenated OCR text | `"长沙南站 出发层"` |
| | `ocr_readable` | str | High-confidence signs | `"长沙南站"` |
| | `ocr_clean` | str | Symbol-stripped keywords | `"长沙南站出发层"` |
| **Video Windows** | `usable_windows` | list | High-quality shot intervals | `[[0.0, 5.2], [8.0, 14.1]]` |
| | `best_frame_ts` | float | Representative frame time | `2.5` |
| | `scene_sub_segments`| list | Calibrated scene shot cuts | `[{"start":0.0, "end":4.5}]` |
| **Audit & Provenance** | `usage_blacklist` | bool | Forbidden asset flag | `false` |
| | `source_type` | str | Origin classification | `"raw_footage"`, `"film_slice"` |
| | `fill` | dict | Field-level provenance | `{"face_count": "auto_metric"}` |
