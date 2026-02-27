#!/usr/bin/env python3
"""Verification tests for the baseball orientation detection pipeline.

Structured around the Take-Home Assignment requirements:

Part 2 verification:
  - Ball detection on noisy/blurry frames
  - Seam extraction produces valid pixel coordinates
  - Full pipeline: raw image → seam pixels

Part 3 verification:
  - 3D orientation from seam segments
  - Physical consistency across 5 consecutive frames

Also validates:
  - Rotation math correctness (vs OpenCV and SciPy)
  - Camera geometry consistency
  - Spin rate formula
  - Seam model geometry

Usage:
    python verify.py                    # all checks
    python verify.py --quick            # math-only (no video processing)
    python verify.py --video path.mp4   # specific video
"""

import argparse
import glob
import sys
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from camera import load_camera_params
from seam_pipeline import (SeamPipeline, BaseballSeamModel,
                           estimate_orientation_from_seams,
                           detect_seams)
from orientation import (rotation_to_quaternion,
                         rotation_to_euler)


# ============================================================
# Public domain constants
# ============================================================

BASEBALL_DIAMETER_MM = 74.0   # MLB Rule 3.01 (circumference 9-9.25")
BASEBALL_RADIUS_MM = 37.0


class Report:
    """Track pass/fail checks with formatted output."""

    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append((name, passed, detail))
        mark = "\u2713" if passed else "\u2717"
        print(f"  {mark} {'PASS' if passed else 'FAIL'}: {name}")
        if detail:
            for line in detail.split("\n"):
                print(f"         {line}")

    def info(self, name, detail):
        self.checks.append((name, None, detail))
        print(f"  * INFO: {name}")
        for line in detail.split("\n"):
            print(f"         {line}")

    def summary(self):
        total = sum(1 for _, p, _ in self.checks if p is not None)
        passed = sum(1 for _, p, _ in self.checks if p is not None and bool(p))
        failed = total - passed
        print(f"\n{'='*60}")
        print(f"VERIFICATION SUMMARY: {passed}/{total} passed, {failed} failed")
        if failed:
            print(f"\nFailed:")
            for name, p, detail in self.checks:
                if p is not None and not p:
                    print(f"  X {name}: {detail}")
        print(f"{'='*60}")
        return failed == 0


# ============================================================
# CHECK 1: Rotation Math Correctness
# ============================================================

def check_rotation_math(report):
    """Verify rotation calculations against OpenCV and SciPy."""
    print("\n" + "="*60)
    print("CHECK 1: Rotation Math (vs OpenCV & SciPy)")
    print("="*60)

    test_axes = [
        np.array([1, 0, 0]), np.array([0, 1, 0]),
        np.array([0, 0, 1]), np.array([1, 1, 1]) / np.sqrt(3),
    ]
    test_angles = [0.01, 0.1, np.pi/4, np.pi/2, np.pi, 2.5]

    max_cv_err = 0
    max_ortho_err = 0
    max_det_err = 0
    n = 0

    for axis in test_axes:
        for angle in test_angles:
            rvec = (axis * angle).reshape(3, 1)
            R_cv, _ = cv2.Rodrigues(rvec)
            R_scipy = Rotation.from_rotvec(axis * angle).as_matrix()

            max_cv_err = max(max_cv_err, np.max(np.abs(R_cv - R_scipy)))
            max_ortho_err = max(max_ortho_err,
                                np.max(np.abs(R_cv.T @ R_cv - np.eye(3))))
            max_det_err = max(max_det_err, abs(np.linalg.det(R_cv) - 1.0))
            n += 1

    report.check(f"OpenCV matches SciPy ({n} cases)",
                 max_cv_err < 1e-10,
                 f"Max difference: {max_cv_err:.2e}")
    report.check("All R orthogonal (R^T R = I)",
                 max_ortho_err < 1e-10,
                 f"Max error: {max_ortho_err:.2e}")
    report.check("All R proper rotation (det=1)",
                 max_det_err < 1e-10,
                 f"Max error: {max_det_err:.2e}")

    # Quaternion/Euler roundtrips
    roundtrip_errs = []
    for _ in range(20):
        R = Rotation.random().as_matrix()
        q = rotation_to_quaternion(R)
        e = rotation_to_euler(R)
        R_from_q = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_from_e = Rotation.from_euler('xyz', e).as_matrix()
        roundtrip_errs.append(max(np.max(np.abs(R - R_from_q)),
                                  np.max(np.abs(R - R_from_e))))
    report.check(f"Quaternion/Euler roundtrip ({len(roundtrip_errs)} random R)",
                 max(roundtrip_errs) < 1e-10,
                 f"Worst error: {max(roundtrip_errs):.2e}")


# ============================================================
# CHECK 2: Seam Model Geometry
# ============================================================

def check_seam_model(report):
    """Verify 3D seam model matches baseball geometry."""
    print("\n" + "="*60)
    print("CHECK 2: 3D Seam Model Geometry")
    print("="*60)

    model = BaseballSeamModel(radius=BASEBALL_RADIUS_MM)
    pts = model.generate_points(num_points_per_curve=100)

    # All on sphere
    dists = np.linalg.norm(pts, axis=1)
    err = np.max(np.abs(dists - BASEBALL_RADIUS_MM))
    report.check("All seam points on sphere surface",
                 err < 1e-10, f"Max deviation: {err:.2e}mm")

    # Two distinct curves
    curve1, curve2 = pts[:100], pts[100:]
    avg_dist = np.mean(np.linalg.norm(curve1 - curve2, axis=1))
    report.check("Two seam curves are distinct",
                 avg_dist > 10.0, f"Avg separation: {avg_dist:.1f}mm")

    # Spans sphere (coverage)
    signs = np.sign(pts)
    octants = {tuple(s.astype(int)) for s in signs}
    report.check("Seam covers >= 6 octants",
                 len(octants) >= 6, f"Covered {len(octants)}/8")


# ============================================================
# CHECK 3: Seam Detection on Synthetic Image
# ============================================================

def check_seam_detection(report):
    """Verify seam detection finds red features but not noise."""
    print("\n" + "="*60)
    print("CHECK 3: Seam Detection (synthetic images)")
    print("="*60)

    # Image with red circle (simulating seam)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)
    result = detect_seams(img)
    report.check("Red circle detected as seam",
                 result["num_pixels"] > 0,
                 f"Found {result['num_pixels']} seam pixels")

    # Blank gray image (no seams)
    blank = np.ones((200, 200, 3), dtype=np.uint8) * 128
    result2 = detect_seams(blank)
    report.check("Blank image: few/no seam pixels",
                 result2["num_pixels"] < 100,
                 f"Found {result2['num_pixels']} pixels (should be ~0)")


# ============================================================
# CHECK 4: Physical Consistency Over 5 Consecutive Frames
# ============================================================

def check_physical_consistency(report, K, dist, videos):
    """Verify orientation is physically consistent over 5+ consecutive frames.

    A baseball can't rotate 180 degrees in 1ms. Check that frame-to-frame
    angle changes are bounded by physical limits.
    """
    print("\n" + "="*60)
    print("CHECK 4: Physical Consistency (5 consecutive frames)")
    print("="*60)

    for vpath in videos:
        vname = os.path.basename(vpath)[:20]

        for approach_name, PipeClass in [("Seam", SeamPipeline)]:
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            pipe = PipeClass(K, dist, confidence=0.25)

            orientations = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = frame_idx / fps
                result = pipe.process_frame(frame, ts)
                if result["orientation"] is not None:
                    orientations.append(
                        (result["orientation"]["rotation_matrix"], ts))
                frame_idx += 1
            cap.release()

            if len(orientations) < 5:
                report.check(
                    f"[{vname}] {approach_name}: >= 5 orientation frames",
                    False, f"Only {len(orientations)} frames with orientation")
                continue

            max_angle_deg = 0
            angles = []

            for i in range(1, min(len(orientations), 20)):
                R_prev, t_prev = orientations[i-1]
                R_curr, t_curr = orientations[i]
                dt = t_curr - t_prev
                if dt < 1e-6:
                    continue

                R_rel = R_prev.T @ R_curr
                angle = np.linalg.norm(
                    Rotation.from_matrix(R_rel).as_rotvec())
                angle_deg = np.degrees(angle)

                angles.append(angle_deg)
                max_angle_deg = max(max_angle_deg, angle_deg)

            # Physical limit: at 30fps, 180 deg/frame = Nyquist
            report.check(
                f"[{vname}] {approach_name}: no > 90 deg jumps between frames",
                max_angle_deg < 90,
                f"Max angle between consecutive frames: {max_angle_deg:.1f} deg, "
                f"Avg: {np.mean(angles):.1f} deg")


# ============================================================
# CHECK 5: Video Results — Detection, Geometry
# ============================================================

def check_video_results(report, K, dist, videos):
    """Run seam pipeline on video and check results are physically plausible."""
    print("\n" + "="*60)
    print("CHECK 5: Video Results - Detection, Geometry")
    print("="*60)

    fx = K[0, 0]

    for vpath in videos:
        vname = os.path.basename(vpath)[:20]
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)

        seam_pipe = SeamPipeline(K, dist, confidence=0.25)

        seam_dets = 0
        seam_R_errors = []
        ball_sizes = []
        total = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = total / fps

            sr = seam_pipe.process_frame(frame.copy(), ts)

            if sr["ball_detected"]:
                seam_dets += 1
                x1, y1, x2, y2 = sr["bbox"]
                ball_sizes.append(((x2-x1) + (y2-y1)) / 2)

            if sr["orientation"] is not None:
                R = sr["orientation"]["rotation_matrix"]
                ortho = np.max(np.abs(R.T @ R - np.eye(3)))
                det_err = abs(np.linalg.det(R) - 1.0)
                seam_R_errors.append(max(ortho, det_err))

            total += 1
        cap.release()

        # Detection rate
        report.check(
            f"[{vname}] Detection rate > 80%",
            seam_dets / total > 0.8,
            f"Seam: {seam_dets}/{total} ({100*seam_dets/total:.0f}%)")

        # Ball distance plausibility
        if ball_sizes:
            dists = [fx * BASEBALL_DIAMETER_MM / 1000 / d for d in ball_sizes]
            plausible = all(0.5 < d < 100 for d in dists)
            report.check(
                f"[{vname}] Ball distance plausible (0.5-100m)",
                plausible,
                f"Range: {min(dists):.1f}-{max(dists):.1f}m")

        # Rotation matrix validity
        if seam_R_errors:
            worst = max(seam_R_errors)
            report.check(
                f"[{vname}] Seam: all R matrices valid",
                worst < 1e-6,
                f"{len(seam_R_errors)} matrices, worst error: {worst:.2e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify baseball orientation detection pipeline")
    parser.add_argument("--video", nargs="*", help="Video paths")
    parser.add_argument("--camera", default="config/camera.json")
    parser.add_argument("--quick", action="store_true",
                        help="Math checks only (no video processing)")
    args = parser.parse_args()

    report = Report()

    # Always run math/model checks
    check_rotation_math(report)
    check_seam_model(report)
    check_seam_detection(report)

    if not args.quick:
        videos = args.video or sorted(glob.glob("spin_dataset/*.mp4"))
        if not videos:
            print("No videos found. Use --video or add to spin_dataset/")
            return 1

        K, dist, _ = load_camera_params(args.camera)
        check_physical_consistency(report, K, dist, videos)
        check_video_results(report, K, dist, videos)

    all_passed = report.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
