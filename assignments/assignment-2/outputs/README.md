# video_detections_50_epochs.parquet

This file is the detection corpus for Assignment 2: image-to-video semantic retrieval.

- File: `video_detections_50_epochs.parquet`
- Rows: `1100` (one row per detection)
- Source video: `YcvECxtXoxQ`
- Frame sampling interval: `5` seconds
- Detector: YOLO segmentation model fine-tuned for `50` epochs

## Schema

| Column | Type | Description |
|---|---|---|
| `video_id` | string | YouTube video id for the corpus video. |
| `frame_index` | int | Frame number from sampled frames (`frame_0001.jpg`, etc.). |
| `timestamp_sec` | int | Time offset in seconds for the sampled frame. |
| `class_id` | int | Numeric class id predicted by the detector. |
| `class_label` | string | Semantic part label (for retrieval matching). |
| `confidence_score` | float | Detector confidence for this detection. |
| `bounding_box` | list[float] | `[x_min, y_min, x_max, y_max]` coordinates in pixels. |
| `x_min` | float | Left x-coordinate of bounding box. |
| `y_min` | float | Top y-coordinate of bounding box. |
| `x_max` | float | Right x-coordinate of bounding box. |
| `y_max` | float | Bottom y-coordinate of bounding box. |
| `frame_file` | string | Frame filename in `frames/` (e.g., `frame_0123.jpg`). |
| `detector_name` | string | Weights filename used for inference. |

## How It Was Generated

1. Fine-tune model with `part_a_finetune_seg_model.py` (50 epochs in this run).
2. Run `part_b_retrieval_corpus.py` over all sampled frames.
3. Save one row per detection into this parquet file.

## Notes

- Retrieval in Part C uses `class_label` overlap between query detections and this corpus.
- Temporal grouping is done using `timestamp_sec` with a 5-second gap rule.
