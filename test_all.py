"""Consolidated tests for the baseball orientation detection pipeline.

Tests all components: camera, detector, seam detection, seam model,
orientation estimation, and the seam pipeline.

Run with:  pytest test_all.py -v
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch

from camera import load_camera_params, undistort
from detector import BallDetector, BallTracker
from orientation import (rotation_to_quaternion,
                         rotation_to_euler)
from seam_pipeline import (detect_seams, BaseballSeamModel,
                           estimate_orientation_from_seams,
                           SeamPipeline)

CAMERA_CONFIG = "config/camera.json"


# ============================================================
# Camera Tests
# ============================================================

class TestCamera:
    def test_load_camera_params(self):
        K, dist, img_shape = load_camera_params(CAMERA_CONFIG)
        assert K.shape == (3, 3)
        assert dist.shape == (1, 5)
        assert img_shape == (1700, 1200, 3)
        assert K[0, 0] == pytest.approx(10248.145, rel=0.01)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_camera_params("nonexistent.json")

    def test_undistort_preserves_shape(self):
        K, dist, _ = load_camera_params(CAMERA_CONFIG)
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        result = undistort(img, K, dist)
        assert result.shape == img.shape


# ============================================================
# Ball Detector Tests
# ============================================================

class TestBallDetector:
    def test_init(self):
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.5)
        assert detector is not None
        assert detector.confidence == 0.5

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            BallDetector(confidence=1.5)

    def test_detect_returns_structure(self):
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.25)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result

    def test_detect_invalid_input(self):
        detector = BallDetector()
        with pytest.raises(TypeError):
            detector.detect("not an image")
        with pytest.raises(ValueError):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))  # 2D

    def test_detect_with_ball(self):
        """Detection on a synthetic image with a gray circle."""
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.25)
        img = np.zeros((1200, 1700, 3), dtype=np.uint8)
        cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)
        result = detector.detect(img)
        # May or may not detect — just verify structure
        if result["detected"]:
            x1, y1, x2, y2 = result["bbox"]
            assert x2 > x1 and y2 > y1


# ============================================================
# Ball Tracker Tests
# ============================================================

class TestBallTracker:
    def test_init(self):
        detector = BallDetector()
        tracker = BallTracker(detector)
        assert tracker.bbox is None
        assert tracker.lost_frames == 0

    def test_reset(self):
        detector = BallDetector()
        tracker = BallTracker(detector)
        tracker.bbox = (10, 20, 30, 40)
        tracker.lost_frames = 3
        tracker.reset()
        assert tracker.bbox is None
        assert tracker.lost_frames == 0

    def test_track_returns_structure(self):
        detector = BallDetector(confidence=0.25)
        tracker = BallTracker(detector)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = tracker.track(img)
        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result
        assert "tracking" in result

    def test_velocity_prediction(self):
        """When ball is lost, tracker should predict using velocity."""
        detector = BallDetector(confidence=0.25)
        tracker = BallTracker(detector, max_lost_frames=5)

        # Manually set state as if we had a detection
        tracker.bbox = (100, 100, 200, 200)
        tracker.confidence = 0.8
        tracker.velocity = (10, 5)

        # Mock detector returning no detection
        with patch.object(tracker.detector, 'detect',
                          return_value={"detected": False, "bbox": None,
                                        "confidence": None}):
            result = tracker.track(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result["detected"] is True
        assert result["tracking"] is True
        # Position should have moved by velocity
        assert result["bbox"][0] == 110  # 100 + 10
        assert result["bbox"][1] == 105  # 100 + 5


# ============================================================
# Seam Detection Tests
# ============================================================

class TestSeamDetection:
    def test_detect_seams_structure(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)  # Red circle
        result = detect_seams(img)
        assert "edges" in result
        assert "seam_pixels" in result
        assert "num_pixels" in result
        assert result["edges"].shape == img.shape[:2]

    def test_red_circle_detects_pixels(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)
        result = detect_seams(img)
        assert result["num_pixels"] > 0
        if len(result["seam_pixels"]) > 0:
            assert result["seam_pixels"].shape[1] == 2

    def test_blank_image_few_pixels(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = detect_seams(img)
        # Blank gray image should have very few seam pixels
        assert result["num_pixels"] < 100


# ============================================================
# Seam Model Tests
# ============================================================

class TestSeamModel:
    def test_generate_points_shape(self):
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points(num_points_per_curve=50)
        assert points.shape == (100, 3)  # 2 curves × 50 points

    def test_points_on_sphere(self):
        """All seam points should be approximately at the specified radius."""
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points()
        distances = np.linalg.norm(points, axis=1)
        assert np.allclose(distances, 37.0, atol=1.0)

    def test_two_curves_separated(self):
        """The two seam curves should not be identical."""
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points(num_points_per_curve=50)
        curve1 = points[:50]
        curve2 = points[50:]
        assert not np.allclose(curve1, curve2, atol=1.0)


# ============================================================
# Orientation Estimation Tests
# ============================================================

class TestOrientationEstimation:
    def test_returns_none_for_few_points(self):
        """Need at least 6 points for ellipse fitting."""
        pts = np.array([[10, 10], [20, 20], [30, 30]])
        result = estimate_orientation_from_seams(pts, (200, 200))
        assert result is None

    def test_returns_valid_rotation(self):
        """Rotation matrix should be proper (det=1, R^T R = I)."""
        # Create seam pixels in an elongated elliptical pattern
        angles = np.linspace(0, 2 * np.pi, 50)
        pts = np.column_stack([
            100 + 40 * np.cos(angles),  # x
            100 + 20 * np.sin(angles),  # y (half the spread → tilted)
        ]).astype(np.float32)
        result = estimate_orientation_from_seams(pts, (200, 200))
        assert result is not None
        assert result["success"] is True
        R = result["rotation_matrix"]
        assert R.shape == (3, 3)
        # Check orthogonality: R^T R = I
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        # Check proper rotation: det(R) = 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_angle_changes_with_direction(self):
        """Seam angle should change when seam direction changes."""
        # Horizontal seam
        pts_h = np.column_stack([
            np.linspace(20, 180, 50),
            100 + np.random.randn(50) * 3,
        ]).astype(np.float32)
        result_h = estimate_orientation_from_seams(pts_h, (200, 200))

        # Vertical seam
        pts_v = np.column_stack([
            100 + np.random.randn(50) * 3,
            np.linspace(20, 180, 50),
        ]).astype(np.float32)
        result_v = estimate_orientation_from_seams(pts_v, (200, 200))

        assert result_h is not None and result_v is not None
        # Angles should be different (roughly 90° apart)
        angle_diff = abs(result_h["seam_angle_deg"] - result_v["seam_angle_deg"])
        assert 45 < angle_diff < 135, f"Angle difference: {angle_diff}"

    def test_tilt_from_axis_ratio(self):
        """Circular spread → low tilt, elongated → high tilt."""
        # Circular distribution (face-on seam)
        angles = np.linspace(0, 2 * np.pi, 50)
        pts_circle = np.column_stack([
            100 + 40 * np.cos(angles),
            100 + 40 * np.sin(angles),
        ]).astype(np.float32)
        result_c = estimate_orientation_from_seams(pts_circle, (200, 200))

        # Elongated distribution (tilted seam)
        pts_elongated = np.column_stack([
            100 + 40 * np.cos(angles),
            100 + 10 * np.sin(angles),
        ]).astype(np.float32)
        result_e = estimate_orientation_from_seams(pts_elongated, (200, 200))

        assert result_c is not None and result_e is not None
        # Elongated should have higher tilt than circular
        assert result_e["seam_tilt_deg"] > result_c["seam_tilt_deg"]


# ============================================================
# Rotation Conversion Tests
# ============================================================

class TestConversions:
    def test_quaternion_identity(self):
        q = rotation_to_quaternion(np.eye(3))
        # Identity rotation → quaternion [1, 0, 0, 0]
        assert q[0] == pytest.approx(1.0, abs=0.01)
        assert np.allclose(q[1:], 0, atol=0.01)

    def test_euler_identity(self):
        euler = rotation_to_euler(np.eye(3))
        assert np.allclose(euler, 0, atol=0.01)

    def test_quaternion_90deg_z(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('z', np.pi / 2).as_matrix()
        q = rotation_to_quaternion(R)
        # 90° about Z → q ≈ [cos(45°), 0, 0, sin(45°)]
        assert q[0] == pytest.approx(np.cos(np.pi / 4), abs=0.01)
        assert abs(q[3]) == pytest.approx(np.sin(np.pi / 4), abs=0.01)


# ============================================================
# Coordinate Transform Tests
# ============================================================

class TestCoordinateTransform:
    def test_ball_local_to_camera_reference_roundtrip(self):
        """Validate X_cam = R * X_ball + t and inverse recovery."""
        from scipy.spatial.transform import Rotation

        # Ball-local point (mm)
        x_ball = np.array([12.0, -8.0, 25.0])

        # Known rigid transform Ball->Camera
        R = Rotation.from_euler('xyz', [0.3, -0.2, 0.5]).as_matrix()
        t = np.array([120.0, -35.0, 780.0])  # mm

        # Forward transform
        x_cam = R @ x_ball + t

        # Inverse transform
        x_ball_recovered = R.T @ (x_cam - t)

        assert np.allclose(x_ball_recovered, x_ball, atol=1e-10)
        # Also verify expected rigid-rotation properties
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


# ============================================================
# Seam Pipeline Tests
# ============================================================

class TestSeamPipeline:
    @pytest.fixture
    def pipeline(self):
        K, dist, _ = load_camera_params(CAMERA_CONFIG)
        return SeamPipeline(K, dist)

    def test_init(self, pipeline):
        assert pipeline is not None
        assert pipeline.frame_count == 0

    def test_process_frame_structure(self, pipeline):
        img = np.ones((1200, 1700, 3), dtype=np.uint8) * 255
        cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)
        cv2.circle(img, (850, 600), 100, (0, 0, 200), 2)

        result = pipeline.process_frame(img, timestamp=0.0)
        assert "ball_detected" in result
        assert "orientation" in result
        assert "frame_number" in result
        assert result["frame_number"] == 1

    def test_process_video_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.process_video("/nonexistent/video.mp4")

    def test_reset(self, pipeline):
        pipeline.frame_count = 10
        pipeline.reset()
        assert pipeline.frame_count == 0
