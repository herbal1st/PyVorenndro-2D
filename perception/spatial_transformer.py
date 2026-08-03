"""
Spatial feature compiler and candidate spawn heading randomizer.
"""

import random
import math
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from perception.vision_arc import VisionArcSampler


class SpatialTransformer:
    """
    Compiles sensory observations into normalized flat neural vectors.
    """

    def __init__(self) -> None:
        """
        Initializes internal vision arc sampler.
        """
        self.sampler: VisionArcSampler = VisionArcSampler()

    @staticmethod
    def generate_random_heading(
        map_data: Optional[MapData] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        max_attempts: int = 50
    ) -> float:
        """
        Returns heading angle in radians, avoiding immediate wall face.
        """
        if map_data is None or start_pos is None:
            return random.uniform(0.0, 2.0 * math.pi)

        start_x, start_y = start_pos
        center_x: float = float(start_x) + 0.5
        center_y: float = float(start_y) + 0.5

        for _ in range(max_attempts):
            heading: float = random.uniform(0.0, 2.0 * math.pi)
            probe_x: int = int(math.floor(center_x + math.cos(heading)))
            probe_y: int = int(math.floor(center_y + math.sin(heading)))

            if map_data.is_walkable(probe_x, probe_y):
                return heading

        return random.uniform(0.0, 2.0 * math.pi)

    def compile_feature_vector(
        self,
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        current_speed: float,
        health_ratio: float,
        map_data: MapData
    ) -> NDArray[np.float32]:
        """
        Samples wall rays and stereo target compass into 11 channels.
        """
        wall_channels: NDArray[np.float32] = (
            self.sampler.sample_vision_channels(
                candidate_x, candidate_y, heading_rad, map_data
            )
        )

        tg_left, tg_right = self._compute_stereo_compass(
            candidate_x, candidate_y, heading_rad, map_data.exit_pos
        )

        state_features: NDArray[np.float32] = np.array(
            [tg_left, tg_right, current_speed, health_ratio],
            dtype=np.float32
        )

        return np.concatenate([wall_channels, state_features])

    def compile_feature_batch(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        headings: NDArray[np.float64],
        speeds: NDArray[np.float64],
        healths: NDArray[np.float64],
        map_data: MapData,
        wall_grid: Optional[np.ndarray] = None
    ) -> NDArray[np.float32]:
        """
        Vectorized feature compiler for (N,) candidate states.
        Returns a (N, channels) float32 feature matrix.
        """
        wall_channels: NDArray[np.float32] = (
            self.sampler.sample_vision_channels_batch(
                xs, ys, headings, map_data, wall_grid
            )
        )

        tg_left, tg_right = self._compute_stereo_compass_batch(
            xs, ys, headings, map_data.exit_pos
        )

        state_features: NDArray[np.float32] = np.stack(
            [tg_left, tg_right, speeds, healths], axis=1
        ).astype(np.float32)

        return np.concatenate([wall_channels, state_features], axis=1)

    def _compute_stereo_compass_batch(
        self,
        cxs: NDArray[np.float64],
        cys: NDArray[np.float64],
        headings: NDArray[np.float64],
        exit_pos: Tuple[int, int]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Vectorized stereo binocular target compass for (N,) headings.
        """
        ex: float = float(exit_pos[0]) + 0.5
        ey: float = float(exit_pos[1]) + 0.5

        dx: NDArray[np.float64] = ex - cxs
        dy: NDArray[np.float64] = ey - cys

        target_angle: NDArray[np.float64] = np.arctan2(dy, dx)
        angle_delta: NDArray[np.float64] = (
            target_angle - headings
        ) % (2.0 * math.pi)
        angle_delta = np.where(
            angle_delta > math.pi, angle_delta - 2.0 * math.pi, angle_delta
        )

        tg_left: NDArray[np.float64] = np.where(
            (angle_delta >= -math.pi) & (angle_delta <= 0.0),
            1.0 - (np.abs(angle_delta) / math.pi),
            0.0,
        )
        tg_right: NDArray[np.float64] = np.where(
            (angle_delta >= 0.0) & (angle_delta <= math.pi),
            1.0 - (angle_delta / math.pi),
            0.0,
        )

        return (
            np.clip(tg_left, 0.0, 1.0),
            np.clip(tg_right, 0.0, 1.0),
        )

    def _compute_stereo_compass(
        self,
        cx: float,
        cy: float,
        heading: float,
        exit_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Computes TG-L and TG-R stereo binocular target compass channels.
        """
        ex: float = float(exit_pos[0]) + 0.5
        ey: float = float(exit_pos[1]) + 0.5

        dx: float = ex - cx
        dy: float = ey - cy

        target_angle: float = math.atan2(dy, dx)
        angle_delta: float = (target_angle - heading) % (2.0 * math.pi)

        if angle_delta > math.pi:
            angle_delta -= 2.0 * math.pi

        tg_left: float = 0.0
        tg_right: float = 0.0

        if -math.pi <= angle_delta <= 0.0:
            tg_left = 1.0 - (abs(angle_delta) / math.pi)
        if 0.0 <= angle_delta <= math.pi:
            tg_right = 1.0 - (angle_delta / math.pi)

        return max(0.0, min(1.0, tg_left)), max(0.0, min(1.0, tg_right))
