"""
Spatial feature compiler with Line-of-Sight gating, memory, and penalties.
"""

from collections import deque
import math
import random
from typing import Tuple, Optional, Dict, Any, Deque
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from perception.vision_arc import VisionArcSampler


class SpatialTransformer:
    """
    Compiles sensory observations into normalized flat neural vectors.
    """

    def __init__(
        self,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes internal vision arc sampler with profile configuration.
        """
        self.sampler: VisionArcSampler = VisionArcSampler(
            profile_config=profile_config
        )

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
        map_data: MapData,
        current_dist: Optional[float] = None,
        last_hit: bool = False,
        is_idle: bool = False,
        is_spinning: bool = False,
        profile_config: Optional[Dict[str, Any]] = None,
        history_buffer: Optional[Deque[NDArray[np.float32]]] = None
    ) -> NDArray[np.float32]:
        """
        Samples wall rays, goal sensors, and dedicated penalty channels.
        """
        sensory: Dict[str, Any] = (
            profile_config.get("sensory", {}) if profile_config else {}
        )
        topology: Dict[str, Any] = (
            profile_config.get("topology", {}) if profile_config else {}
        )
        kinematics: Dict[str, Any] = (
            profile_config.get("kinematics", {}) if profile_config else {}
        )

        include_compass: bool = bool(sensory.get("include_compass", False))
        include_bfs: bool = bool(sensory.get("include_bfs_sensor", False))
        enable_los: bool = bool(sensory.get("enable_los_gating", False))
        max_range: float = float(sensory.get("goal_sensor_max_range", 0.0))
        memory_frames: int = max(1, int(topology.get("memory_frames", 1)))
        max_speed: float = float(kinematics.get("move_speed", 0.1))

        norm_speed: float = min(
            1.0, max(0.0, current_speed / max(1e-6, max_speed))
        )

        wall_channels: NDArray[np.float32] = (
            self.sampler.sample_vision_channels(
                candidate_x, candidate_y, heading_rad, map_data
            )
        )

        ex: float = float(map_data.exit_pos[0]) + 0.5
        ey: float = float(map_data.exit_pos[1]) + 0.5
        dx: float = ex - candidate_x
        dy: float = ey - candidate_y
        dist_to_exit: float = math.sqrt((dx * dx) + (dy * dy))

        los_ok: bool = True
        if enable_los:
            los_ok = self.sampler.check_line_of_sight(
                candidate_x, candidate_y, ex, ey, map_data
            )

        range_factor: float = 1.0
        if max_range > 0.0:
            if dist_to_exit > max_range:
                range_factor = 0.0
            else:
                range_factor = 1.0 - (dist_to_exit / max_range)

        if not los_ok:
            range_factor = 0.0

        tg_left, tg_right = self._compute_stereo_compass(
            candidate_x, candidate_y, heading_rad, map_data.exit_pos
        )
        tg_left *= range_factor
        tg_right *= range_factor

        if include_compass:
            state_features: NDArray[np.float32] = np.array(
                [tg_left, tg_right, norm_speed, health_ratio],
                dtype=np.float32
            )
        else:
            state_features = np.array(
                [norm_speed, health_ratio], dtype=np.float32
            )

        if include_bfs:
            max_span: float = float(map_data.height + map_data.width)
            if current_dist is None or not los_ok or range_factor <= 0.0:
                dist_val: float = 0.0
            else:
                grad: float = max(
                    0.0, 1.0 - (float(current_dist) / max_span)
                )
                dist_val = grad * range_factor

            dist_channel: NDArray[np.float32] = np.array(
                [dist_val], dtype=np.float32
            )
            state_features = np.concatenate([state_features, dist_channel])

        hit_val: float = 1.0 if last_hit else 0.0
        idle_val: float = 1.0 if is_idle else 0.0
        spin_val: float = 1.0 if is_spinning else 0.0

        penalty_features: NDArray[np.float32] = np.array(
            [hit_val, idle_val, spin_val], dtype=np.float32
        )
        state_features = np.concatenate([state_features, penalty_features])

        single_frame: NDArray[np.float32] = np.concatenate(
            [wall_channels, state_features]
        )

        if history_buffer is not None and memory_frames > 1:
            history_buffer.append(single_frame)
            while len(history_buffer) < memory_frames:
                history_buffer.appendleft(single_frame)
            return np.concatenate(list(history_buffer))

        return single_frame

    def compile_feature_batch(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        headings: NDArray[np.float64],
        speeds: NDArray[np.float64],
        healths: NDArray[np.float64],
        map_data: Optional[MapData] = None,
        wall_grid: Optional[NDArray[np.bool_]] = None,
        wall_grids: Optional[NDArray[np.bool_]] = None,
        exit_positions: Optional[NDArray[np.int64]] = None,
        current_dists: Optional[NDArray[np.float64]] = None,
        last_hits: Optional[NDArray[np.bool_]] = None,
        is_idles: Optional[NDArray[np.bool_]] = None,
        is_spinnings: Optional[NDArray[np.bool_]] = None,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float32]:
        """
        Vectorized feature compiler for (N,) candidate states.
        """
        sensory: Dict[str, Any] = (
            profile_config.get("sensory", {}) if profile_config else {}
        )
        kinematics: Dict[str, Any] = (
            profile_config.get("kinematics", {}) if profile_config else {}
        )
        include_compass: bool = bool(sensory.get("include_compass", False))
        include_bfs: bool = bool(sensory.get("include_bfs_sensor", False))
        max_speed: float = float(kinematics.get("move_speed", 0.1))

        norm_speeds: NDArray[np.float64] = np.clip(
            speeds / max(1e-6, max_speed), 0.0, 1.0
        )

        if wall_grids is not None:
            wall_channels: NDArray[np.float32] = (
                self.sampler.sample_vision_channels_batch(
                    xs, ys, headings, wall_grids=wall_grids
                )
            )
        else:
            wall_channels = self.sampler.sample_vision_channels_batch(
                xs, ys, headings, map_data, wall_grid
            )

        if exit_positions is not None:
            tg_left, tg_right = self._compute_stereo_compass_batch(
                xs, ys, headings, exit_positions
            )
        elif map_data is not None:
            tg_left, tg_right = self._compute_stereo_compass_batch(
                xs, ys, headings, map_data.exit_pos
            )
        else:
            n_cands: int = len(xs)
            tg_left = np.zeros(n_cands, dtype=np.float64)
            tg_right = np.zeros(n_cands, dtype=np.float64)

        if include_compass:
            state_features: NDArray[np.float32] = np.stack(
                [tg_left, tg_right, norm_speeds, healths], axis=1
            ).astype(np.float32)
        else:
            state_features = np.stack(
                [norm_speeds, healths], axis=1
            ).astype(np.float32)

        if include_bfs:
            if current_dists is not None and wall_grids is not None:
                max_span: float = float(
                    wall_grids.shape[1] + wall_grids.shape[2]
                )
                dist_channel: NDArray[np.float32] = np.clip(
                    1.0 - (
                        np.asarray(
                            current_dists, dtype=np.float64
                        ) / max_span
                    ),
                    0.0,
                    1.0
                ).astype(np.float32).reshape(-1, 1)
            else:
                dist_channel = np.zeros(
                    (len(xs), 1), dtype=np.float32
                )

            state_features = np.concatenate(
                [state_features, dist_channel], axis=1
            )

        n_cands = len(xs)
        hit_arr: NDArray[np.float32] = (
            last_hits.astype(np.float32) if last_hits is not None
            else np.zeros(n_cands, dtype=np.float32)
        )
        idle_arr: NDArray[np.float32] = (
            is_idles.astype(np.float32) if is_idles is not None
            else np.zeros(n_cands, dtype=np.float32)
        )
        spin_arr: NDArray[np.float32] = (
            is_spinnings.astype(np.float32) if is_spinnings is not None
            else np.zeros(n_cands, dtype=np.float32)
        )

        penalty_batch: NDArray[np.float32] = np.stack(
            [hit_arr, idle_arr, spin_arr], axis=1
        ).astype(np.float32)

        state_features = np.concatenate(
            [state_features, penalty_batch], axis=1
        )

        return np.concatenate([wall_channels, state_features], axis=1)

    def _compute_stereo_compass(
        self,
        cx: float,
        cy: float,
        heading: float,
        exit_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Computes TG-L and TG-R stereo target compass channels.
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

    def _compute_stereo_compass_batch(
        self,
        cxs: NDArray[np.float64],
        cys: NDArray[np.float64],
        headings: NDArray[np.float64],
        exit_pos: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Vectorized stereo target compass for (N,) candidate headings.
        """
        if isinstance(exit_pos, tuple):
            ex: NDArray[np.float64] = np.full_like(
                cxs, float(exit_pos[0]) + 0.5
            )
            ey: NDArray[np.float64] = np.full_like(
                cys, float(exit_pos[1]) + 0.5
            )
        else:
            ep: NDArray[np.float64] = np.asarray(exit_pos, dtype=np.float64)
            ex = ep[:, 0] + 0.5
            ey = ep[:, 1] + 0.5

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
