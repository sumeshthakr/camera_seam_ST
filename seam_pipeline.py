"""Seam-based baseball orientation detection pipeline.

APPROACH:
    Detect the baseball's red stitching (seams) in 2D, then estimate the
    ball's 3D orientation from the seam pixel distribution using PCA and
    ellipse fitting.

PIPELINE (per frame):
    1. Undistort image (correct lens distortion)
    2. Detect baseball with YOLO
    3. Extract ROI (region of interest) around the ball
    4. Detect red seam pixels using edge detection + color filtering
    5. Estimate orientation from seam pixel distribution (PCA + ellipse)
    6. Convert to rotation matrix / quaternion / Euler angles

KEY INSIGHT:
    A baseball's seam pattern, when projected onto the image plane, forms a
    characteristic distribution. The principal direction and spread of seam
    pixels tell us the ball's orientation — the seam direction gives the
    in-plane rotation, and the spread ratio gives the out-of-plane tilt.
"""

import os
import numpy as np
import cv2

from camera import undistort
from detector import BallDetector, BallTracker
from orientation import rotation_to_quaternion, rotation_to_euler


# ============================================================
# Seam Detection
# ============================================================

def detect_seams(roi, canny_low=30, canny_high=100):
    """Detect baseball seam pixels in a ball ROI.

    Baseball seams are red stitching on a white ball. We detect them by:
        1. Creating a circular mask to exclude ball boundary edges
        2. Boosting color saturation (seams can be pale in video)
        3. Running Canny edge detection to find all edges
        4. Filtering edges to keep only those in red color regions
        5. Cleaning up with morphological operations

    The circular mask (inner 85% of ROI) is critical — without it, the
    ball's outer edge gets detected as "seam" and corrupts orientation.

    Args:
        roi:        Cropped ball image (H, W, 3) BGR format
        canny_low:  Lower Canny edge threshold
        canny_high: Upper Canny edge threshold

    Returns:
        dict with:
            seam_pixels: Nx2 array of (x, y) coordinates
            num_pixels:  Count of detected seam pixels
            edges:       Binary edge mask
    """
    h, w = roi.shape[:2]

    if h < 20 or w < 20:
        return {
            "seam_pixels": np.zeros((0, 2), dtype=int),
            "num_pixels": 0,
            "edges": np.zeros((h, w), dtype=np.uint8)
        }

    # Use more sensitive thresholds for small ROIs
    if h < 60 or w < 60:
        canny_low = max(20, canny_low - 10)
        canny_high = max(45, canny_high - 20)
        sat_low, val_low = 12, 35
    else:
        sat_low, val_low = 22, 45

    # --- Step 1: Circular mask to exclude ball boundary edges ---
    # Only look at the inner 85% of the ball to avoid edge artifacts
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, int(radius * 0.85), 255, -1)

    # --- Step 2: Contrast + saturation boost ---
    # CLAHE helps when seams are dim or lighting is uneven.
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    boosted = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2BGR)
    boosted = cv2.addWeighted(boosted, 0.6, enhanced, 0.4, 0)

    # --- Step 3: Canny edge detection ---
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    ksize = min(5, max(3, (h + w) // 20))
    if ksize % 2 == 0:
        ksize += 1  # Gaussian kernel must be odd
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    med = np.median(blurred)
    lo = int(max(10, 0.66 * med, canny_low))
    hi = int(max(lo + 10, min(255, 1.33 * med, canny_high + 25)))
    edges = cv2.Canny(blurred, lo, hi)

    # Apply circular mask — exclude ball boundary
    edges = edges & circle_mask

    # --- Step 4: Multi-cue seam mask ---
    # Cue A: red/warm seam color when visible.
    hsv_boosted = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv_boosted,
                          np.array([0, sat_low, val_low]),
                          np.array([25, 255, 255]))
    red_high = cv2.inRange(hsv_boosted,
                           np.array([155, sat_low, val_low]),
                           np.array([180, 255, 255]))
    warm_mask = red_low | red_high
    warm_mask = cv2.morphologyEx(
        warm_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Cue B: chromatic seam contrast in LAB (works when seam is not clearly red).
    # Neutral grayscale has a ~= 128; seams often shift away from it.
    a_dev = cv2.absdiff(a_ch, np.full_like(a_ch, 128))
    chroma_mask = cv2.inRange(a_dev, 8, 255)

    # Cue C: dark, thin stitch lines from black-hat filtering.
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    bh_thresh = int(max(8, np.percentile(blackhat, 80)))
    dark_stitch_mask = cv2.inRange(blackhat, bh_thresh, 255)

    cue_mask = warm_mask | chroma_mask | dark_stitch_mask
    cue_mask = cue_mask & circle_mask
    combined = edges & cue_mask

    # If too sparse, fall back to strongest texture cues only.
    min_seam_pixels = max(20, int(h * w * 0.003))
    if cv2.countNonZero(combined) < min_seam_pixels:
        fallback_cue = (dark_stitch_mask | chroma_mask) & circle_mask
        combined = edges & fallback_cue

    # --- Step 5: Dilate to connect nearby edge fragments ---
    combined = cv2.dilate(combined,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Re-apply circular mask after dilation
    combined = combined & circle_mask

    # Remove tiny isolated blobs (noise).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    filtered = np.zeros_like(combined)
    min_blob = max(2, int(0.0004 * h * w))
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_blob:
            filtered[labels == label] = 255
    combined = filtered

    # Extract (x, y) pixel coordinates
    yx = np.column_stack(np.where(combined > 0))
    seam_pixels = yx[:, [1, 0]] if len(yx) > 0 else np.zeros((0, 2), dtype=int)

    return {"seam_pixels": seam_pixels, "num_pixels": len(seam_pixels), "edges": combined}


# ============================================================
# Orientation Estimation from Seam Pixels
# ============================================================

def estimate_orientation_from_seams(seam_pixels, roi_shape):
    """Estimate ball orientation from seam pixel distribution.

    Uses PCA and ellipse fitting on the detected seam pixels to determine
    the ball's orientation. This is simpler and more reliable than PnP
    with approximate 2D-3D correspondences.

    How it works:
        1. Center seam pixels relative to ROI center
        2. Fit an ellipse to the seam pixel distribution
        3. The ellipse angle gives the in-plane seam direction
        4. The axis ratio (minor/major) gives the out-of-plane tilt
        5. Construct a rotation matrix from these two angles

    The seam direction tells us which way the seam runs across the ball
    (like a compass heading). The tilt tells us how much the seam plane
    is tilted toward or away from the camera.

    NOTE: This recovers 2 of 3 rotation degrees of freedom. The third
    (spin around the seam plane normal) requires frame-to-frame tracking.

    Args:
        seam_pixels: Nx2 array of (x, y) seam pixel coordinates in ROI
        roi_shape:   (height, width) of the ROI

    Returns:
        dict with rotation_matrix, seam_angle_deg, seam_tilt_deg — or None
    """
    # Minimum 6 points: fitEllipse needs ≥5, plus 1 extra for robustness
    if len(seam_pixels) < 6:
        return None

    roi_h, roi_w = roi_shape[:2]
    roi_center = np.array([roi_w / 2.0, roi_h / 2.0])

    # --- Method 1: Ellipse fitting (preferred, needs ≥ 5 points) ---
    points = seam_pixels.astype(np.float32).reshape(-1, 1, 2)
    try:
        ellipse = cv2.fitEllipse(points)
        (cx, cy), (axis1, axis2), angle = ellipse

        # Ensure major >= minor
        if axis1 > axis2:
            major_len, minor_len = axis1, axis2
        else:
            major_len, minor_len = axis2, axis1
            angle += 90

        # Normalize angle to [0, 180)
        angle = angle % 180

    except cv2.error:
        # Fallback: use PCA if ellipse fitting fails
        centered = seam_pixels.astype(np.float64) - roi_center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        angle = angle % 180
        if eigenvalues[0] > 0:
            major_len = np.sqrt(eigenvalues[0])
            minor_len = np.sqrt(max(eigenvalues[1], 1e-10))
        else:
            return None

    # --- Compute orientation angles ---
    # Seam angle: direction the seam runs in the image plane
    seam_angle_rad = np.radians(angle)

    # Seam tilt: how much the seam plane is tilted from the camera
    # Axis ratio close to 1 → seam plane is face-on (small tilt)
    # Axis ratio close to 0 → seam plane is edge-on (large tilt)
    if major_len > 0:
        axis_ratio = min(minor_len / major_len, 1.0)
        tilt_rad = np.arccos(np.clip(axis_ratio, 0.0, 1.0))
    else:
        tilt_rad = 0.0

    # --- Construct rotation matrix ---
    # R = Rz(seam_angle) @ Rx(tilt), where:
    #   Rz = [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]
    #   Rx = [[1, 0, 0], [0, ct, -st], [0, st, ct]]
    # Multiplying these gives the matrix below.
    ca, sa = np.cos(seam_angle_rad), np.sin(seam_angle_rad)
    ct, st = np.cos(tilt_rad), np.sin(tilt_rad)

    R = np.array([
        [ca, -sa * ct,  sa * st],
        [sa,  ca * ct, -ca * st],
        [0,   st,       ct]
    ])

    return {
        "success": True,
        "rotation_matrix": R,
        "seam_angle_deg": angle,
        "seam_tilt_deg": np.degrees(tilt_rad),
    }


# ============================================================
# 3D Baseball Seam Model (reference geometry)
# ============================================================

class BaseballSeamModel:
    """Parametric 3D model of baseball seam geometry.

    A real baseball has two continuous seam curves that spiral around the
    sphere. Each curve makes approximately 2.5 revolutions with a
    sinusoidally varying inclination angle.

    We generate 3D points along these curves. These serve as the "known 3D
    positions" needed for PnP solving — we match detected 2D seam pixels
    against these 3D model points.

    The seam is parameterized in spherical coordinates:
        phi(t)   = 2.5 * t + phase    (azimuthal — spirals around)
        theta(t) = π/2 + 0.4*sin(2.5*t) (polar — wobbles around equator)
    """

    def __init__(self, radius=37.0):
        """
        Args:
            radius: Ball radius in mm (standard baseball ≈ 37mm radius)
        """
        self.radius = radius

    def generate_points(self, num_points_per_curve=100):
        """Generate 3D points along both seam curves.

        Returns:
            (2*N, 3) numpy array of 3D points on the sphere surface
        """
        t = np.linspace(0, 2 * np.pi, num_points_per_curve)
        curve1 = self._make_curve(t, phase=0)
        curve2 = self._make_curve(t, phase=np.pi)  # Second curve offset by 180°
        return np.vstack([curve1, curve2])

    def _make_curve(self, t, phase=0):
        """Generate one seam curve.

        Args:
            t:     Parameter array (0 to 2π)
            phase: Phase offset (0 for curve 1, π for curve 2)

        Returns:
            (N, 3) array of 3D points
        """
        phi = t * 2.5 + phase                     # Azimuthal angle (spiral)
        theta = np.pi / 2 + 0.4 * np.sin(2.5 * t)  # Polar angle (wobble)

        # Spherical → Cartesian
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        return np.column_stack([x, y, z])


# ============================================================
# Complete Seam Pipeline
# ============================================================

class SeamPipeline:
    """Complete seam-based orientation detection pipeline.

    Combines all components: detection, seam finding, and orientation
    estimation into a single easy-to-use class.

    Usage:
        from camera import load_camera_params
        K, dist, _ = load_camera_params("config/camera.json")
        pipeline = SeamPipeline(K, dist)
        result = pipeline.process_frame(frame, timestamp=0.0)
    """

    def __init__(self, camera_matrix, dist_coeffs, ball_radius_mm=37.0,
                 confidence=0.5, model_path="yolov8n.pt",
                 max_orientation_step_deg=89.0):
        """
        Args:
            camera_matrix:  3x3 camera intrinsic matrix
            dist_coeffs:    Distortion coefficients
            ball_radius_mm: Baseball radius in mm (≈37mm)
            confidence:     YOLO detection confidence threshold
            model_path:     Path to YOLO model weights
            max_orientation_step_deg: Reject one-frame orientation jumps larger
                                      than this threshold (degrees)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ball_radius_mm = ball_radius_mm
        self.max_orientation_step_deg = max_orientation_step_deg

        # Initialize components
        self.detector = BallDetector(model_path, confidence)
        self.tracker = BallTracker(self.detector)
        self.frame_count = 0
        self._last_orientation = None

    def process_frame(self, image, timestamp=None):
        """Process a single video frame.

        Args:
            image:     BGR image (H, W, 3)
            timestamp: Frame time in seconds (unused, kept for API compatibility)

        Returns:
            dict with:
                ball_detected:  bool
                bbox:           (x1, y1, x2, y2) or None
                confidence:     float or None
                orientation:    dict with rotation_matrix, quaternion, euler_angles — or None
                frame_number:   int
                timestamp:      float or None
                seam_pixels:    Nx2 array or None
                num_seam_pixels: int
        """
        self.frame_count += 1

        result = {
            "ball_detected": False, "bbox": None, "confidence": None,
            "orientation": None, "spin_rate": None, "spin_axis": None,
            "frame_number": self.frame_count, "timestamp": timestamp,
            "seam_pixels": None, "num_seam_pixels": 0
        }

        # Step 1: Undistort
        try:
            image = undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception:
            pass  # Use original if undistortion fails

        # Step 2: Detect and track ball
        track = self.tracker.track(image)
        if not track["detected"]:
            self._last_orientation = None
            return result
        clamped_bbox = self._clamp_bbox(track["bbox"], image.shape[1], image.shape[0])
        if clamped_bbox is None:
            self._last_orientation = None
            return result

        result["ball_detected"] = True
        result["bbox"] = clamped_bbox
        result["confidence"] = track["confidence"]
        result["tracking"] = track["tracking"]

        # Step 3: Extract ROI
        x1, y1, x2, y2 = clamped_bbox
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return result

        # Step 4: Detect seam pixels
        seam = detect_seams(roi)
        result["seam_pixels"] = seam["seam_pixels"]
        result["num_seam_pixels"] = seam["num_pixels"]

        # Need minimum seam pixels for reliable results
        min_pixels = max(6, int(roi.shape[0] * roi.shape[1] * 0.003))
        if seam["num_pixels"] < min_pixels:
            return result

        # Step 5: Estimate orientation from seam distribution
        ori = estimate_orientation_from_seams(seam["seam_pixels"], roi.shape)
        if ori is not None and ori["success"]:
            R = ori["rotation_matrix"]
            orientation = {
                "rotation_matrix": R,
                "quaternion": rotation_to_quaternion(R),
                "euler_angles": rotation_to_euler(R),
                "seam_angle_deg": ori["seam_angle_deg"],
                "seam_tilt_deg": ori["seam_tilt_deg"],
            }
            result["orientation"] = self._stabilize_orientation(orientation)

        return result

    def reset(self):
        """Reset all pipeline state."""
        self.frame_count = 0
        self.tracker.reset()
        self._last_orientation = None

    def process_video(self, video_path, output_path=None, visualize=False):
        """Process an entire video file.

        Args:
            video_path:  Path to input video
            output_path: Path to save annotated output video (optional)
            visualize:   If True, write visualization overlay to output_path

        Returns:
            dict with total_frames, fps, detections
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path and visualize:
            writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else None
            result = self.process_frame(frame, timestamp)
            results.append(result)

            if visualize and writer:
                writer.write(self._visualize(frame, result, frame_idx))

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "detections": results,
            "average_spin_rate": None
        }

    def _visualize(self, frame, result, frame_idx):
        """Draw detection and orientation results on the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25

        cv2.putText(vis, f"Frame: {frame_idx} | Method: Seam",
                    (10, y), font, 0.6, (255, 255, 255), 2)
        y += 25

        if not result["ball_detected"]:
            cv2.putText(vis, "NO BALL DETECTED",
                        (10, y), font, 0.6, (0, 0, 255), 2)
            return vis

        x1, y1b, x2, y2b = result["bbox"]
        cx, cy = (x1 + x2) // 2, (y1b + y2b) // 2

        color = (0, 255, 0) if not result.get("tracking") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1b), (x2, y2b), color, 2)
        cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

        # Seam pixels (red dots)
        if result.get("seam_pixels") is not None and len(result["seam_pixels"]) > 0:
            for px, py in result["seam_pixels"]:
                gx, gy = int(px + x1), int(py + y1b)
                if 0 <= gx < w and 0 <= gy < h:
                    cv2.circle(vis, (gx, gy), 1, (0, 0, 255), -1)
            cv2.putText(vis, f"Seam pixels: {result['num_seam_pixels']}",
                        (10, y), font, 0.5, (0, 0, 255), 1)
            y += 20

        # Orientation info
        if result["orientation"]:
            sa = result["orientation"].get("seam_angle_deg", 0)
            st = result["orientation"].get("seam_tilt_deg", 0)

            self._draw_orientation_axis(vis, (cx, cy), (x1, y1b, x2, y2b), sa, st)
            cv2.putText(vis,
                        f"Seam angle: {sa:.1f} deg",
                        (10, y), font, 0.4, (200, 255, 200), 1)
            y += 18
            cv2.putText(vis,
                        f"Seam tilt: {st:.1f} deg",
                        (10, y), font, 0.4, (200, 255, 200), 1)
            y += 18

        if result["confidence"] is not None:
            cv2.putText(vis, f"Conf: {result['confidence']:.2f}",
                        (10, y), font, 0.5, (0, 255, 0), 1)

        return vis

    @staticmethod
    def _clamp_bbox(bbox, img_w, img_h):
        """Clip bbox to image bounds and reject invalid boxes."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _stabilize_orientation(self, orientation):
        """Hold previous orientation if one-frame jump is physically implausible."""
        if self._last_orientation is None:
            self._last_orientation = orientation
            return orientation

        prev_R = self._last_orientation["rotation_matrix"]
        curr_R = orientation["rotation_matrix"]
        rel = prev_R.T @ curr_R
        cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))

        if angle_deg > self.max_orientation_step_deg:
            return self._last_orientation

        self._last_orientation = orientation
        return orientation

    @staticmethod
    def _draw_orientation_axis(vis, center, bbox, seam_angle_deg, seam_tilt_deg):
        """Draw a simple seam-orientation axis and tilt indicator."""
        cx, cy = center
        x1, y1, x2, y2 = bbox

        radius = max(8, int(0.42 * min(x2 - x1, y2 - y1)))
        angle = np.radians(seam_angle_deg)
        dx = int(radius * np.cos(angle))
        dy = int(radius * np.sin(angle))

        # Seam axis from seam-angle calculation (bidirectional to avoid 180-degree ambiguity).
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        cv2.line(vis, p1, p2, (0, 255, 255), 3)
        cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)

        # Perpendicular tick length encodes tilt magnitude.
        tilt_scale = np.clip(seam_tilt_deg / 90.0, 0.0, 1.0)
        tilt_len = int(radius * tilt_scale)
        if tilt_len > 1:
            nx = int(tilt_len * np.cos(angle + np.pi / 2.0))
            ny = int(tilt_len * np.sin(angle + np.pi / 2.0))
            cv2.line(vis, (cx - nx, cy - ny), (cx + nx, cy + ny), (255, 0, 255), 2)
