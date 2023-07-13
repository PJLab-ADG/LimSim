import numpy as np
from typing import List

def separate_axis_theorem(vertices_a: np.ndarray,
                          vertices_b: np.ndarray) -> bool:
    """Check if two polygon overlaps, polygons are represented by its vertices

    Args:
        vertices1 (np.ndarray): of shape (n, 2), vertices of polygon a
        vertices2 (np.ndarray): of shape (m, 2), vertices of polygon b

    Returns:
        bool: True if two polygons overlap

    Reference:
        https://github.com/Chanuk-Yang/Deep_Continuous_Fusion_for_Multi-Sensor_3D_Object_Detection

        http://programmerart.weebly.com/separating-axis-theorem.html
    """
    # find all possible axes
    n, m = vertices_a.shape[0], vertices_b.shape[0]
    edges_a: List[np.ndarray] = [
        vertices_a[(i + 1) % n] - vertices_a[i] for i in range(n)
    ]
    edges_b: List[np.ndarray] = [
        vertices_a[(i + 1) % m] - vertices_a[i] for i in range(m)
    ]
    edges: List[np.ndarray] = edges_a + edges_b
    axes: List[np.ndarray] = [np.array([edge[1], -edge[0]]) for edge in edges]
    # normalize
    axes: List[np.ndarray] = [
        axis / np.linalg.norm(axis, ord=2) for axis in axes
    ]

    # check for collision with separate axis theorem
    for axis in axes:
        projection_a: np.ndarray = np.array(
            [np.dot(vertex, axis) for vertex in vertices_a])
        projection_b: np.ndarray = np.array(
            [np.dot(vertex, axis) for vertex in vertices_b])

        # if two line segments do not overlaps, then two polygons do not overlap
        if np.min(projection_a) > np.max(projection_b) or \
           np.min(projection_b) > np.max(projection_a):
            return False

    return True