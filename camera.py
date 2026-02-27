"""Camera utilities — load calibration parameters and undistort images.

Camera calibration corrects for lens distortion (barrel/pincushion)
using intrinsic parameters (focal length, principal point) and
distortion coefficients measured during calibration.
"""

import json
import numpy as np
import cv2


def load_camera_params(path):
    """Load camera intrinsic parameters from a JSON file.

    The JSON file should contain:
        camera_matrix: 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
        dist_coeffs:   5 distortion coefficients [k1, k2, p1, p2, k3]
        img_shape:     image dimensions [H, W, C]

    Args:
        path: Path to camera JSON file

    Returns:
        K:         3x3 numpy camera matrix
        dist:      Distortion coefficients array
        img_shape: Tuple (H, W, C)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Validate required keys
    for key in ["camera_matrix", "dist_coeffs", "img_shape"]:
        if key not in data:
            raise ValueError(f"Missing '{key}' in camera parameters file")

    # Validate camera_matrix is 3x3
    cm = data["camera_matrix"]
    if not isinstance(cm, list) or len(cm) != 3 or any(len(row) != 3 for row in cm):
        raise ValueError("camera_matrix must be a 3x3 array")

    # Validate dist_coeffs has 5 values (may be nested [[...]] or flat [...])
    dc = data["dist_coeffs"]
    if isinstance(dc, list) and len(dc) == 1 and isinstance(dc[0], list):
        if len(dc[0]) != 5:
            raise ValueError("dist_coeffs must contain exactly 5 coefficients")
    elif not isinstance(dc, list) or len(dc) != 5:
        raise ValueError("dist_coeffs must contain exactly 5 coefficients")

    # Validate img_shape
    if not isinstance(data["img_shape"], (list, tuple)) or len(data["img_shape"]) != 3:
        raise ValueError("img_shape must have 3 elements (H, W, C)")

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    img_shape = tuple(data["img_shape"])

    return K, dist, img_shape


def undistort(image, camera_matrix, dist_coeffs):
    """Remove lens distortion from an image.

    Uses the camera intrinsic matrix and distortion coefficients
    to map each pixel from the distorted image to its corrected position.

    Args:
        image:         Input image (H, W, 3) BGR
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs:   Distortion coefficients

    Returns:
        Undistorted image (same shape)
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs)
