# AVTrack Annotation-to-Video Frame Mapping

Download the original high-frame-rate videos from [`Videos_high_fps.zip`](https://huggingface.co/datasets/FudanCVL/AVTrack/blob/main/Videos_high_fps.zip) in the official Hugging Face dataset repository:

```bash
hf download FudanCVL/AVTrack Videos_high_fps.zip --repo-type dataset --revision main --local-dir .
unzip Videos_high_fps.zip -d avtrack_data
```

The archive contains 871 MP4 clips under `Videos/`. Paths below are relative to the extraction directory (`avtrack_data/` in this example). Annotations are distributed separately in `AVTrack.zip`; this directory provides the frame alignment table and the MP4 paths in `Annotations/avtrack_meta.json`.

Use `frame_alignment.csv` to find the MP4 frame that corresponds to an image annotated in the original AVTrack release. The table covers 47,202 labeled frames across 871 clips.

1. Find the row using the clip name (`clip_id`) and annotated image filename (`reference_filename`).
2. Look up the same `clip_id` in `Annotations/avtrack_meta.json` to find the MP4 path.
3. Use `mp4_timestamp_seconds` to locate the matching image by time in the MP4. The `merged_mask` field gives the annotation PNG path.

**Example**. For `BulletsFly_A_11`, `frame_00001.jpg` corresponds to **1.033333333 seconds** in `Videos/BulletsFly_A/11_merged.mp4`. The annotation PNG is `Annotations/Instance_Masks_merged/BulletsFly_A_11/frame_00001.png`.
