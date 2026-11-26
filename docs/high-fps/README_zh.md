# AVTrack 标注帧与视频帧对照

从官方 Hugging Face 数据集仓库下载原始高帧率视频 [`Videos_high_fps.zip`](https://huggingface.co/datasets/FudanCVL/AVTrack/blob/main/Videos_high_fps.zip)：

```bash
hf download FudanCVL/AVTrack Videos_high_fps.zip --repo-type dataset --revision main --local-dir .
unzip Videos_high_fps.zip -d avtrack_data
```

压缩包包含871 段 MP4。下文的视频路径相对于解压目录（本例为 `avtrack_data/`）。标注单独提供在 `AVTrack.zip` 中；本目录提供帧对照表，以及包含 MP4 路径的 `Annotations/avtrack_meta.json`。

用 `frame_alignment.csv` 查 AVTrack 原版的某张标注图对应 MP4 里的哪一帧。表里有 871 段视频、47,202 个标注帧。

1. 用视频名（`clip_id`）和标注图文件名（`reference_filename`）在表里找到对应的一行。
2. 在 `Annotations/avtrack_meta.json` 中用同一个 `clip_id` 查找 MP4 路径。
3. 用这一行的 `mp4_timestamp_seconds` 在 MP4 里定位对应画面，单位是秒。`merged_mask` 是对应标注 PNG 的路径。

**例子**：`BulletsFly_A_11` 的 `frame_00001.jpg` 对应 `Videos/BulletsFly_A/11_merged.mp4` 中 **1.033333333 秒**的画面。这一行的标注 PNG 是 `Annotations/Instance_Masks_merged/BulletsFly_A_11/frame_00001.png`。
