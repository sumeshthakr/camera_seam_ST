"""Rotation format conversion utilities.

Provides helpers to convert a 3x3 rotation matrix to other common formats:
  - Quaternion [w, x, y, z]
  - Euler angles [roll, pitch, yaw] in radians
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    We use the scalar-first convention: q = w + xi + yj + zk
    where w is the real part.

    Args:
        R: 3x3 rotation matrix

    Returns:
        numpy array [w, x, y, z]
    """
    q = Rotation.from_matrix(R).as_quat()  # scipy gives [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # reorder to [w, x, y, z]


def rotation_to_euler(R):
    """Convert a 3x3 rotation matrix to Euler angles [roll, pitch, yaw].

    Uses the XYZ convention (intrinsic rotations).

    Args:
        R: 3x3 rotation matrix

    Returns:
        numpy array [roll, pitch, yaw] in radians
    """
    return Rotation.from_matrix(R).as_euler('xyz')
