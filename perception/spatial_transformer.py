"""
Spatial transformer providing raycasting vision inputs using Numba JIT compilation.
"""

import math
from typing import Any, Tuple
import numpy as np
from numpy.typing import NDArray
from numba import njit

import config


@njit(fastmath=True, nogil=True)
def cast_ray_numba(
    pos_x: float,
    pos_y: float,
    angle_rad: float,
    max_dist: float,
    wall_grid: NDArray[np.bool_],
    grid_width: int,
    grid_height: int
) -> float:
    """
    Numba-compiled Digital Differential Analysis (DDA) raycasting kernel.
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    step_size = 0.05
    dist = 0.0

    while dist < max_dist:
        check_x = int(math.floor(pos_x + cos_a * dist))
        check_y = int(math.floor(pos_y + sin_a * dist))

        if check_x < 0 or check_x >= grid_width or check_y < 0 or check_y >= grid_height:
            return dist

        if wall_grid[check_y, check_x]:
            return dist

        dist += step_size

    return max_dist


class SpatialTransformer:
    """
    Processes agent spatial environment inputs into neural network features.
    """

    def __init__(self) -> None:
        self.num_rays: int = config.VISION_RAYS
        self.arc_angle_rad: float = math.radians(config.VISION_ARC_ANGLE)
        self.max_dist: float = config.VISION_MAX_DIST
        self.include_compass: bool = config.INCLUDE_COMPASS

        if self.num_rays > 1:
            self.ray_angles = np.linspace(
                -self.arc_angle_rad / 2.0,
                self.arc_angle_rad / 2.0,
                self.num_rays,
                dtype=np.float64
            )
        else:
            self.ray_angles = np.array([0.0], dtype=np.float64)

    def generate_random_heading(
        self, map_data: Any = None, pos: Tuple[int, int] = None
    ) -> float:
        """
        Generates a uniform random spawn heading in radians [0, 2π).
        """
        return float(np.random.uniform(0.0, 2.0 * math.pi))

    def compile_feature_vector(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        health: float,
        wall_grid: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        Calculates ray distances and compiles sensor input features into a float array.
        """
        grid_height, grid_width = wall_grid.shape
        ray_distances = np.zeros(self.num_rays, dtype=np.float64)

        for i in range(self.num_rays):
            angle = heading + self.ray_angles[i]
            d = cast_ray_numba(x, y, angle, self.max_dist, wall_grid, grid_width, grid_height)
            ray_distances[i] = d / self.max_dist  # Normalized [0, 1]

        kinematics_features = np.array([speed, health], dtype=np.float64)
        return np.concatenate((ray_distances, kinematics_features))