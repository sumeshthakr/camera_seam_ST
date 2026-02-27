# Baseball Orientation Detection (Student Project)

This project detects a baseball in high-speed monocular video, extracts seam pixels, and estimates ball orientation from seam geometry.

## What the pipeline does

For each frame:
1. Undistort frame using camera calibration (`camera.py`)
2. Detect ball with YOLOv8 + short-term tracker (`detector.py`)
3. Crop ball ROI and extract seam pixels (`seam_pipeline.py`)
4. Estimate orientation using ellipse/PCA on seam pixels
5. Visualize:
   - bounding box
   - seam pixels
   - seam orientation axis (computed from seam angle)
   - tilt indicator (computed from seam tilt)

## Main files

- `main.py` - CLI entry point for processing video
- `seam_pipeline.py` - seam extraction + orientation estimation + visualization
- `detector.py` - YOLO detector and tracker
- `camera.py` - camera params loader + undistortion
- `orientation.py` - rotation conversions (matrix -> quaternion/euler)
- `verify.py` - verification checks
- `test_all.py` - pytest tests
- `extract_frames.py` - generate best-frame images + metrics

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run orientation video output

```bash
python3 main.py spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4 \
  --visualize --output outputs/results

python3 main.py spin_dataset/raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4 \
  --visualize --output outputs/results
```

Generated videos:
- `outputs/results/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685_output.mp4`
- `outputs/results/raw_spin_video_695d9b0a4899846853793e7d_1767742221_output.mp4`

## Generate frame artifacts

```bash
python3 extract_frames.py
```

Outputs:
- `docs/frames/video1_seam_best.jpg`
- `docs/frames/video2_seam_best.jpg`
- `docs/frames/metrics.json`

## Verify and test

```bash
python3 verify.py --quick
python3 verify.py
pytest test_all.py -v
```

## Notes

- Orientation is estimated only on frames with enough seam pixels.
- Seam extraction uses edge + color + texture cues, so it still works when seams are not strongly red.
- The displayed orientation axis is calculated from `seam_angle_deg`.
- The tilt indicator is calculated from `seam_tilt_deg`.
