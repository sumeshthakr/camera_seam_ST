# AI Collaboration Log

**Tool Used:** GitHub Copilot (Claude) in VS Code  
**Project:** Baseball Orientation Detection Pipeline

## Prompt History

### Prompt: Pipeline structure
> "Design a simple modular pipeline for baseball orientation from monocular video."

**Outcome:** Split implementation into:
- `camera.py` for calibration + undistortion
- `detector.py` for YOLO detection + short tracking bridge
- `seam_pipeline.py` for seam extraction + orientation estimation
- `orientation.py` for matrix-to-quaternion/euler conversion

### Prompt: Seam extraction robust to lighting
> "Extract seam pixels from cropped ball ROIs. Seams may look red, dull, or dark depending on frame."

**Outcome:** Implemented multi-cue seam extraction in `detect_seams()`:
- Canny edges on enhanced ROI
- warm/red color cue in HSV
- chroma deviation cue in LAB
- dark-stitch cue via black-hat morphology
- circular mask + blob filtering

### Prompt: Orientation from seam direction (per-frame)
> "Use seam direction to estimate orientation per frame without optical flow."

**Outcome:** Implemented `estimate_orientation_from_seams()`:
- Fit ellipse / PCA on seam pixels
- Seam angle from principal direction
- Seam tilt from minor/major axis ratio
- Build rotation matrix and derived quaternion/euler

### Prompt: Keep presentation safe and simple
> "Show orientation clearly in video, but keep visualization easy to explain."

**Outcome:** Visualization shows:
- seam pixels
- seam orientation axis (from `seam_angle_deg`)
- tilt indicator (from `seam_tilt_deg`)
- numeric seam angle/tilt values

### Prompt: Verification and final checks
> "Add checks for rotation math, seam extraction behavior, and frame-to-frame physical consistency."

**Outcome:** Verified with:
- `test_all.py` unit tests
- `verify.py` checks (rotation validity, synthetic seam behavior, consecutive-frame consistency, detection/geometry summary)

## Critical Review

**AI suggestion that would fail:**  
Direct PnP with arbitrary 2D seam points matched to synthetic 3D seam points.

**Why it fails:**  
Seam points are not uniquely identifiable, so correspondences are ambiguous. PnP can output unstable orientations from incorrect matches.

**Chosen approach:**  
Use seam-distribution geometry (ellipse/PCA) per frame. It is simpler, explainable, and avoids fragile correspondence assumptions.

## Optimization Notes

1. ROI-first processing reduced noise and compute by avoiding full-frame seam search.  
   - Measured seam-extraction runtime on this repo (same function, same frame source):
   - Full frame (1200x1700): ~72.6 ms/frame
   - Ball ROI (180x180): ~1.26 ms/frame
   - Effective speedup: ~57.8x for seam extraction stage.
2. Adaptive thresholds for small ROIs improved seam pickup on distant frames.  
3. Connected-component filtering removed tiny seam-noise blobs.  
4. Added orientation jump guard in seam pipeline to suppress physically implausible one-frame spikes.

## Final Note

AI was used for code drafting, alternative approaches, and test scaffolding.  
Final decisions and cleanup were manually reviewed against assignment scope.
