from typing import Union
import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """compute the angle between two vectors, i.e.
        theta = arccos(<v1, v2> / (||v1||_2 * ||v2||_2))

    Args:
        v1 (np.ndarray): first vector
        v2 (np.ndarray): second vector of same length as first vector

    Returns:
        float: angle in radians, between [0, pi]
    """
    if np.all(v1 <= 1e-8) or np.all(v2 <= 1e-8):
        return 0.0
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def project(v1: np.ndarray, v2: np.ndarray) -> float:
    """compute the projection of v1 to v2

    Args:
        v1 (np.ndarray): vector to be projected
        v2 (np.ndarray): projection vector

    Returns:
        float: projection
    """
    return np.dot(v1, v2) / np.linalg.norm(v2, 2)


def normalize(yaw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """normalize an angle such that the angle lies in [0, pi)

    Args:
        yaw (Union[float, np.ndarray]): angle in radians

    Returns:
        Union[float, np.ndarray]: angle in [0, pi)
    """
    yaw = np.fmod(yaw, np.pi * 2) + np.pi * 2
    yaw = np.fmod(yaw, np.pi)

    return yaw
