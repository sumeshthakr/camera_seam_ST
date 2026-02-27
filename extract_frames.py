#!/usr/bin/env python3
"""Extract best detection frames from the seam pipeline for README documentation.

Processes spin_dataset videos with the seam-based approach and saves:
  - Best seam detection frames (highest seam pixel count + ball detected)
  - Per-video JSON metrics

Output is saved to docs/frames/ for use in the README.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from camera import load_camera_params
from seam_pipeline import SeamPipeline


OUTPUT_DIR = Path("docs/frames")
CAMERA_JSON = "config/camera.json"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.25


def score_seam_frame(result):
    """Score a seam-pipeline frame — higher = better visual for README."""
    if not result["ball_detected"]:
        return -1
    score = result["num_seam_pixels"]
    if result["orientation"] is not None:
        score += 500
    return score


def annotate_seam_frame(frame, result, frame_idx, video_label):
    """Draw rich seam-pipeline annotations on a frame for README use."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header bar
    cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 60), -1)
    cv2.putText(vis, f"SEAM-BASED PIPELINE  |  {video_label}  |  Frame {frame_idx}",
                (10, 28), font, 0.65, (0, 200, 255), 2)

    if not result["ball_detected"]:
        return vis

    x1, y1, x2, y2 = result["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rad = max((x2 - x1), (y2 - y1)) // 2

    # Ball circle overlay
    color = (0, 255, 0) if not result.get("tracking") else (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.circle(vis, (cx, cy), rad, color, 1)
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

    # Seam pixels (red dots)
    if result.get("seam_pixels") is not None and len(result["seam_pixels"]) > 0:
        for px, py in result["seam_pixels"]:
            gx, gy = int(px + x1), int(py + y1)
            if 0 <= gx < w and 0 <= gy < h:
                cv2.circle(vis, (gx, gy), 2, (0, 0, 255), -1)

    # Info panel
    y_txt = 60
    info_lines = [
        (f"Seam pixels: {result['num_seam_pixels']}", (100, 180, 255)),
    ]
    if result.get("confidence"):
        info_lines.append((f"YOLO conf: {result['confidence']:.2f}", (180, 255, 180)))
    if result["orientation"]:
        e = np.degrees(result["orientation"]["euler_angles"])
        info_lines.append((f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg",
                           (255, 200, 100)))
        q = result["orientation"]["quaternion"]
        info_lines.append((f"Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
                           (255, 200, 100)))

    for text, col in info_lines:
        cv2.putText(vis, text, (10, y_txt), font, 0.55, col, 1)
        y_txt += 22

    return vis


def process_video(video_path, label, K, dist):
    """Process one video, return best frames and metrics."""
    print(f"\n  [{label}] Initializing pipeline...")
    seam_pipe = SeamPipeline(K, dist, confidence=CONFIDENCE, model_path=MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [{label}] {total_frames} frames @ {fps:.1f} fps")

    # Best-frame tracking
    best_seam = {"score": -1, "frame": None, "result": None, "idx": 0}

    # Metrics
    seam_detections = seam_orientations = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        sr = seam_pipe.process_frame(frame.copy(), timestamp)

        # Accumulate metrics
        if sr["ball_detected"]:
            seam_detections += 1
        if sr["orientation"] is not None:
            seam_orientations += 1

        # Track best frames
        ss = score_seam_frame(sr)
        if ss > best_seam["score"]:
            best_seam = {"score": ss, "frame": frame.copy(),
                         "result": sr, "idx": frame_idx}

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"    Frame {frame_idx}/{total_frames}")

    cap.release()

    metrics = {
        "label": label,
        "total_frames": frame_idx,
        "fps": fps,
        "seam": {
            "detection_count": seam_detections,
            "detection_rate_pct": 100.0 * seam_detections / max(frame_idx, 1),
            "orientation_count": seam_orientations,
            "orientation_rate_pct": 100.0 * seam_orientations / max(frame_idx, 1),
        },
    }

    print(f"  [{label}] Seam: {seam_detections}/{frame_idx} detected, "
          f"{seam_orientations} orientation estimates")

    return {
        "best_seam": best_seam,
        "metrics": metrics,
        "total_frames": frame_idx,
        "fps": fps,
    }


def save_frames(data, label, output_dir):
    """Annotate and save best frames to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Best seam frame
    bs = data["best_seam"]
    if bs["frame"] is not None:
        ann = annotate_seam_frame(bs["frame"], bs["result"], bs["idx"], label)
        path = output_dir / f"{label}_seam_best.jpg"
        # Scale down for README (keep ≤ 1200px wide)
        ann = _scale_down(ann, max_w=1200)
        cv2.imwrite(str(path), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["seam_best"] = str(path)
        print(f"  Saved: {path}")

    return saved


def _scale_down(img, max_w=1200):
    """Scale image down if wider than max_w, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract best detection frames from the seam pipeline for documentation.")
    parser.add_argument(
        "--videos", nargs="*",
        default=[
            "spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4",
            "spin_dataset/raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4",
        ],
        help="Input video paths (default: both spin_dataset videos)",
    )
    parser.add_argument("--output", default="docs/frames",
                        help="Output directory for frames (default: docs/frames)")
    parser.add_argument("--camera", default=CAMERA_JSON,
                        help="Camera JSON path (default: config/camera.json)")
    args = parser.parse_args()

    out_dir = Path(args.output)

    print("=" * 60)
    print("Baseball Frame Extractor — Best Detection Frames")
    print("=" * 60)

    K, dist, _ = load_camera_params(args.camera)

    # Build (path, label) pairs; auto-label as video1, video2, …
    videos = [(p, f"video{i+1}") for i, p in enumerate(args.videos)]

    all_metrics = {}
    for vpath, label in videos:
        if not os.path.exists(vpath):
            print(f"  Skipping {label}: file not found ({vpath})")
            continue

        print(f"\nProcessing {label}: {vpath}")
        data = process_video(vpath, label, K, dist)
        if data is None:
            continue

        saved = save_frames(data, label, out_dir)
        all_metrics[label] = data["metrics"]
        all_metrics[label]["saved_frames"] = saved

    # Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Video 1':>18} {'Video 2':>18}")
    print("-" * 70)
    if all_metrics:
        v1 = all_metrics.get("video1", {})
        v2 = all_metrics.get("video2", {})
        rows = [
            ("Frames", "total_frames", None),
            ("Seam Detection %", "seam.detection_rate_pct", ".1f"),
            ("Seam Orientation %", "seam.orientation_rate_pct", ".1f"),
        ]
        for name, key, fmt in rows:
            def _get(d, k):
                parts = k.split(".")
                val = d
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        return None
                return val

            v1v = _get(v1, key)
            v2v = _get(v2, key)
            f1 = f"{v1v:{fmt}}" if v1v is not None and fmt else (str(v1v) if v1v is not None else "N/A")
            f2 = f"{v2v:{fmt}}" if v2v is not None and fmt else (str(v2v) if v2v is not None else "N/A")
            print(f"  {name:<28} {f1:>18} {f2:>18}")

    print("\nDone. Frames in:", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
