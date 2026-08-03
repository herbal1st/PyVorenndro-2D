"""
Vectorized 2D DDA raycaster for wall proximity sensing.
"""

import math
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

import config
from core.map_data import MapData


class VisionArcSampler:
    """
    Casts probe rays measuring wall tile proximity across visual arc.
    """

    def __init__(
        self,
        num_rays: int = config.VISION_RAYS,
        arc_angle_deg: float = config.VISION_ARC_ANGLE,
        max_dist: float = config.VISION_MAX_DIST
    ) -> None:
        """
        Initializes relative ray angles across the visual arc.
        """
        self.num_rays: int = num_rays
        self.max_dist: float = max_dist

        half_arc: float = math.radians(arc_angle_deg / 2.0)
        if num_rays > 1:
            step: float = (2.0 * half_arc) / float(num_rays - 1)
            self.relative_angles: List[float] = [
                -half_arc + (i * step) for i in range(num_rays)
            ]
        else:
            self.relative_angles = [0.0]

    def sample_vision_channels(
        self,
        origin_x: float,
        origin_y: float,
        heading_rad: float,
        map_data: MapData
    ) -> NDArray[np.float32]:
        """
        Casts probe rays and returns a (VISION_RAYS,) wall proximity array.
        """
        channels: NDArray[np.float32] = np.zeros(
            self.num_rays, dtype=np.float32
        )

        for i, rel_angle in enumerate(self.relative_angles):
            ray_angle: float = heading_rad + rel_angle
            wall_prox, _ = self._cast_single_ray(
                origin_x, origin_y, ray_angle, map_data
            )
            channels[i] = wall_prox

        return channels

    def _cast_single_ray(
        self,
        ox: float,
        oy: float,
        angle_rad: float,
        map_data: MapData
    ) -> Tuple[float, float]:
        """
        Marches a single ray forward up to max_dist tiles.
        """
        dir_x: float = math.cos(angle_rad)
        dir_y: float = math.sin(angle_rad)
        step_size: float = 0.1
        num_steps: int = int(self.max_dist / step_size)

        wall_proximity: float = 0.0

        for step in range(1, num_steps + 1):
            curr_dist: float = step * step_size
            curr_x: int = int(math.floor(ox + (dir_x * curr_dist)))
            curr_y: int = int(math.floor(oy + (dir_y * curr_dist)))

            if map_data.is_wall(curr_x, curr_y):
                wall_proximity = 1.0 - (curr_dist / self.max_dist)
                break

        return max(0.0, wall_proximity), 0.0
