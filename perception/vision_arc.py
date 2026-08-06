"""
Grid-aligned DDA raycaster for wall proximity sensing.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData


class VisionArcSampler:
    """
    Casts probe rays measuring wall tile proximity across visual arc.
    """

    def __init__(
        self,
        num_rays: int = 9,
        arc_angle_deg: float = 120.0,
        max_dist: float = 5.0,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes relative ray angles across the visual arc.
        """
        if profile_config is not None:
            sens: Dict[str, Any] = profile_config.get("sensory", {})
            num_rays = int(sens.get("vision_rays", num_rays))
            arc_angle_deg = float(sens.get("vision_arc_angle", arc_angle_deg))

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

    def sample_vision_channels_batch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        map_data: MapData = None,
        wall_grid: np.ndarray = None,
        wall_grids: np.ndarray = None
    ) -> NDArray[np.float32]:
        """
        Vectorized Amanatides-Woo DDA across (N,) origins and headings.
        """
        n_cands: int = int(xs.shape[0])
        if n_cands == 0:
            return np.zeros((0, self.num_rays), dtype=np.float32)

        multi_grid: bool = wall_grids is not None

        if multi_grid:
            h, w = wall_grids.shape[1], wall_grids.shape[2]
        else:
            if wall_grid is None:
                wall_grid = map_data.build_wall_grid()
            h, w = wall_grid.shape

        offsets: np.ndarray = np.asarray(
            self.relative_angles, dtype=np.float64
        )
        angles: np.ndarray = headings[:, None] + offsets[None, :]
        dir_x: np.ndarray = np.cos(angles).ravel()
        dir_y: np.ndarray = np.sin(angles).ravel()

        ox: np.ndarray = np.repeat(xs, self.num_rays)
        oy: np.ndarray = np.repeat(ys, self.num_rays)

        run_idx: np.ndarray = np.repeat(
            np.arange(n_cands), self.num_rays
        )

        tile_x: np.ndarray = np.floor(ox).astype(np.int64)
        tile_y: np.ndarray = np.floor(oy).astype(np.int64)

        n_rays: int = int(ox.shape[0])
        prox: np.ndarray = np.zeros(n_rays, dtype=np.float64)
        done: np.ndarray = np.zeros(n_rays, dtype=np.bool_)

        inb: np.ndarray = (
            (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
        )
        txc: np.ndarray = np.clip(tile_x, 0, w - 1)
        tyc: np.ndarray = np.clip(tile_y, 0, h - 1)
        if multi_grid:
            is_wall_at: np.ndarray = wall_grids[run_idx, tyc, txc]
        else:
            is_wall_at = wall_grid[tyc, txc]
        inside_wall: np.ndarray = inb & is_wall_at
        prox[inside_wall] = 1.0
        done[inside_wall] = True

        step_x: np.ndarray = np.where(
            dir_x > 0.0, 1, np.where(dir_x < 0.0, -1, 0)
        )
        step_y: np.ndarray = np.where(
            dir_y > 0.0, 1, np.where(dir_y < 0.0, -1, 0)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            t_delta_x: np.ndarray = np.where(
                np.abs(dir_x) > 1e-9, np.abs(1.0 / dir_x), np.inf
            )
            t_delta_y: np.ndarray = np.where(
                np.abs(dir_y) > 1e-9, np.abs(1.0 / dir_y), np.inf
            )
            t_max_x: np.ndarray = np.where(
                step_x > 0,
                (tile_x + 1.0 - ox) / dir_x,
                np.where(step_x < 0, (tile_x - ox) / dir_x, np.inf),
            )
            t_max_y: np.ndarray = np.where(
                step_y > 0,
                (tile_y + 1.0 - oy) / dir_y,
                np.where(step_y < 0, (tile_y - oy) / dir_y, np.inf),
            )

        max_t: float = self.max_dist
        max_iter: int = int(2.0 * max_t) + 4

        for _ in range(max_iter):
            active: np.ndarray = ~done
            if not active.any():
                break

            move_x: np.ndarray = t_max_x < t_max_y
            hit_t: np.ndarray = np.where(move_x, t_max_x, t_max_y)
            tile_x = np.where(move_x, tile_x + step_x, tile_x)
            tile_y = np.where(move_x, tile_y, tile_y + step_y)
            t_max_x = np.where(move_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = np.where(move_x, t_max_y, t_max_y + t_delta_y)

            no_hit: np.ndarray = hit_t > max_t
            prox = np.where(active & no_hit, 0.0, prox)
            done |= active & no_hit

            inb = (
                (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
            )
            txc = np.clip(tile_x, 0, w - 1)
            tyc = np.clip(tile_y, 0, h - 1)
            if multi_grid:
                is_wall_at = wall_grids[run_idx, tyc, txc]
            else:
                is_wall_at = wall_grid[tyc, txc]
            wall_hit: np.ndarray = inb & is_wall_at
            blocked: np.ndarray = active & ~no_hit & (wall_hit | ~inb)
            prox = np.where(
                blocked,
                np.clip(1.0 - (hit_t / max_t), 0.0, 1.0),
                prox,
            )
            done |= blocked

        prox[~done] = 0.0
        return prox.reshape(n_cands, self.num_rays).astype(np.float32)

    def _cast_single_ray(
        self,
        ox: float,
        oy: float,
        angle_rad: float,
        map_data: MapData
    ) -> Tuple[float, float]:
        """
        Marches a single ray across grid cells via Amanatides-Woo DDA.
        """
        dir_x: float = math.cos(angle_rad)
        dir_y: float = math.sin(angle_rad)

        tile_x: int = int(math.floor(ox))
        tile_y: int = int(math.floor(oy))

        if map_data.is_wall(tile_x, tile_y):
            return 1.0, 0.0

        if abs(dir_x) < 1e-9:
            step_x: int = 0
            t_max_x: float = float("inf")
            t_delta_x: float = float("inf")
        else:
            step_x = 1 if dir_x > 0.0 else -1
            t_max_x = (
                (tile_x + 1.0 - ox) / dir_x if step_x > 0
                else (tile_x - ox) / dir_x
            )
            t_delta_x = abs(1.0 / dir_x)

        if abs(dir_y) < 1e-9:
            step_y: int = 0
            t_max_y: float = float("inf")
            t_delta_y: float = float("inf")
        else:
            step_y = 1 if dir_y > 0.0 else -1
            t_max_y = (
                (tile_y + 1.0 - oy) / dir_y if step_y > 0
                else (tile_y - oy) / dir_y
            )
            t_delta_y = abs(1.0 / dir_y)

        while True:
            if t_max_x < t_max_y:
                hit_t: float = t_max_x
                tile_x += step_x
                t_max_x += t_delta_x
            else:
                hit_t = t_max_y
                tile_y += step_y
                t_max_y += t_delta_y

            if hit_t > self.max_dist:
                return 0.0, 0.0

            if map_data.is_wall(tile_x, tile_y):
                return max(0.0, 1.0 - (hit_t / self.max_dist)), 0.0
