"""Baseball detection and tracking using YOLOv8.

BallDetector: Finds baseballs in single frames using a pretrained YOLO model.
BallTracker:  Tracks the ball across frames with velocity-based prediction
              to handle brief detection failures.

The COCO dataset (which YOLOv8 is trained on) has a 'sports ball' class (index 32)
that we use to detect baseballs.
"""

import numpy as np
from ultralytics import YOLO


class BallDetector:
    """Detect baseballs in images using YOLOv8.

    Uses the COCO-pretrained model. The 'sports ball' class (index 32)
    captures baseballs, tennis balls, etc.
    """

    SPORTS_BALL_CLASS = 32  # COCO class index for sports ball

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Args:
            model_path: Path to YOLO model weights (e.g., yolov8n.pt, yolov8s.pt)
            confidence: Minimum detection confidence threshold (0-1)
        """
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        """Detect the most confident baseball in an image.

        Args:
            image: BGR image as numpy array (H, W, 3)

        Returns:
            dict with:
                detected:   bool — was a ball found?
                bbox:       (x1, y1, x2, y2) pixel coordinates, or None
                confidence: float detection confidence, or None
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
        if image.ndim != 3:
            raise ValueError(f"image must be 3D (H,W,C), got {image.ndim}D")

        results = self.model(image, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"detected": False, "bbox": None, "confidence": None}

        # Find the highest-confidence sports ball detection
        best_box, best_conf = None, 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == self.SPORTS_BALL_CLASS and conf > self.confidence and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            return {
                "detected": True,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": best_conf
            }

        return {"detected": False, "bbox": None, "confidence": None}


class BallTracker:
    """Track a baseball across video frames.

    Combines YOLO detection with simple velocity-based prediction
    to maintain tracking when detection briefly fails.

    How it works:
        1. Each frame, run YOLO detection
        2. If detected → update position and estimate velocity
        3. If NOT detected → predict next position using velocity
        4. After too many consecutive misses → give up (reset)

    The velocity is smoothed with an exponential moving average
    to avoid jerky predictions.
    """

    def __init__(self, detector, max_lost_frames=5):
        """
        Args:
            detector:        BallDetector instance
            max_lost_frames: How many frames to predict before giving up
        """
        self.detector = detector
        self.max_lost_frames = max_lost_frames
        self.reset()

    def reset(self):
        """Clear all tracking state."""
        self.bbox = None           # Current bounding box (x1, y1, x2, y2)
        self.confidence = 0.0      # Current detection confidence
        self.lost_frames = 0       # Consecutive frames without detection
        self.velocity = (0, 0)     # Estimated (vx, vy) in pixels/frame

    def track(self, frame):
        """Track the ball in a new frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            dict with:
                detected:   bool — True if ball found or predicted
                bbox:       (x1, y1, x2, y2) or None
                confidence: float (decays during prediction)
                tracking:   bool — True if position is PREDICTED, False if freshly DETECTED
        """
        detection = self.detector.detect(frame)

        if detection["detected"]:
            new_bbox = detection["bbox"]

            # Update velocity from previous position
            if self.bbox is not None:
                vx = new_bbox[0] - self.bbox[0]
                vy = new_bbox[1] - self.bbox[1]
                # Exponential moving average for smooth velocity
                self.velocity = (0.7 * self.velocity[0] + 0.3 * vx,
                                 0.7 * self.velocity[1] + 0.3 * vy)

            self.bbox = new_bbox
            self.confidence = detection["confidence"]
            self.lost_frames = 0

            return {"detected": True, "bbox": self.bbox,
                    "confidence": self.confidence, "tracking": False}

        # No detection — predict using velocity if we have a recent track
        if self.bbox is not None and self.lost_frames < self.max_lost_frames:
            x1, y1, x2, y2 = self.bbox
            vx, vy = self.velocity
            w, h = x2 - x1, y2 - y1

            self.bbox = (int(x1 + vx), int(y1 + vy),
                         int(x1 + vx + w), int(y1 + vy + h))
            self.lost_frames += 1
            self.confidence *= 0.9  # Decay confidence each predicted frame

            return {"detected": True, "bbox": self.bbox,
                    "confidence": self.confidence, "tracking": True}

        # Lost the ball completely
        self.reset()
        return {"detected": False, "bbox": None, "confidence": None, "tracking": False}
