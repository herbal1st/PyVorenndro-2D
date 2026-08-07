"""
Grid-aligned DDA raycaster for wall proximity sensing and line-of-sight.
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
        Casts probe rays and returns a (num_rays,) wall proximity array.
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
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        headings: NDArray[np.float64],
        map_data: Optional[MapData] = None,
        wall_grid: Optional[NDArray[np.bool_]] = None,
        wall_grids: Optional[NDArray[np.bool_]] = None
    ) -> NDArray[np.float32]:
        """
        Vectorized Amanatides-Woo DDA across (N,) origins and headings.
        """
        n_cands: int = int(xs.shape[0])
        if n_cands == 0:
            return np.zeros((0, self.num_rays), dtype=np.float32)

        multi_grid: bool = wall_grids is not None

        if multi_grid and wall_grids is not None:
            h, w = wall_grids.shape[1], wall_grids.shape[2]
        else:
            if wall_grid is None and map_data is not None:
                wall_grid = map_data.build_wall_grid()
            if wall_grid is None:
                return np.zeros((n_cands, self.num_rays), dtype=np.float32)
            h, w = wall_grid.shape

        offsets: NDArray[np.float64] = np.asarray(
            self.relative_angles, dtype=np.float64
        )
        angles: NDArray[np.float64] = headings[:, None] + offsets[None, :]
        dir_x: NDArray[np.float64] = np.cos(angles).ravel()
        dir_y: NDArray[np.float64] = np.sin(angles).ravel()

        ox: NDArray[np.float64] = np.repeat(xs, self.num_rays)
        oy: NDArray[np.float64] = np.repeat(ys, self.num_rays)

        run_idx: NDArray[np.int64] = np.repeat(
            np.arange(n_cands), self.num_rays
        )

        tile_x: NDArray[np.int64] = np.floor(ox).astype(np.int64)
        tile_y: NDArray[np.int64] = np.floor(oy).astype(np.int64)

        n_rays: int = int(ox.shape[0])
        prox: NDArray[np.float64] = np.zeros(n_rays, dtype=np.float64)
        done: NDArray[np.bool_] = np.zeros(n_rays, dtype=np.bool_)

        inb: NDArray[np.bool_] = (
            (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
        )
        txc: NDArray[np.int64] = np.clip(tile_x, 0, w - 1)
        tyc: NDArray[np.int64] = np.clip(tile_y, 0, h - 1)
        if multi_grid and wall_grids is not None:
            is_wall_at: NDArray[np.bool_] = wall_grids[run_idx, tyc, txc]
        elif wall_grid is not None:
            is_wall_at = wall_grid[tyc, txc]
        else:
            is_wall_at = np.zeros(n_rays, dtype=np.bool_)

        inside_wall: NDArray[np.bool_] = inb & is_wall_at
        prox[inside_wall] = 1.0
        done[inside_wall] = True

        step_x: NDArray[np.int64] = np.where(
            dir_x > 0.0, 1, np.where(dir_x < 0.0, -1, 0)
        )
        step_y: NDArray[np.int64] = np.where(
            dir_y > 0.0, 1, np.where(dir_y < 0.0, -1, 0)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            t_delta_x: NDArray[np.float64] = np.where(
                np.abs(dir_x) > 1e-9, np.abs(1.0 / dir_x), np.inf
            )
            t_delta_y: NDArray[np.float64] = np.where(
                np.abs(dir_y) > 1e-9, np.abs(1.0 / dir_y), np.inf
            )
            t_max_x: NDArray[np.float64] = np.where(
                step_x > 0,
                (tile_x + 1.0 - ox) / dir_x,
                np.where(step_x < 0, (tile_x - ox) / dir_x, np.inf),
            )
            t_max_y: NDArray[np.float64] = np.where(
                step_y > 0,
                (tile_y + 1.0 - oy) / dir_y,
                np.where(step_y < 0, (tile_y - oy) / dir_y, np.inf),
            )

        max_t: float = self.max_dist
        max_iter: int = int(2.0 * max_t) + 4

        for _ in range(max_iter):
            active: NDArray[np.bool_] = ~done
            if not bool(active.any()):
                break

            move_x: NDArray[np.bool_] = t_max_x < t_max_y
            hit_t: NDArray[np.float64] = np.where(move_x, t_max_x, t_max_y)
            tile_x = np.where(move_x, tile_x + step_x, tile_x)
            tile_y = np.where(move_x, tile_y, tile_y + step_y)
            t_max_x = np.where(move_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = np.where(move_x, t_max_y, t_max_y + t_delta_y)

            no_hit: NDArray[np.bool_] = hit_t > max_t
            prox = np.where(active & no_hit, 0.0, prox)
            done |= active & no_hit

            inb = (
                (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
            )
            txc = np.clip(tile_x, 0, w - 1)
            tyc = np.clip(tile_y, 0, h - 1)
            if multi_grid and wall_grids is not None:
                is_wall_at = wall_grids[run_idx, tyc, txc]
            elif wall_grid is not None:
                is_wall_at = wall_grid[tyc, txc]
            wall_hit: NDArray[np.bool_] = inb & is_wall_at
            blocked: NDArray[np.bool_] = active & ~no_hit & (wall_hit | ~inb)
            prox = np.where(
                blocked,
                np.clip(1.0 - (hit_t / max_t), 0.0, 1.0),
                prox,
            )
            done |= blocked

        prox[~done] = 0.0
        return prox.reshape(n_cands, self.num_rays).astype(np.float32)

    def check_line_of_sight(
        self,
        ox: float,
        oy: float,
        tx: float,
        ty: float,
        map_data: MapData
    ) -> bool:
        """
        Checks if a direct raycast from origin to target hits any wall.
        """
        dx: float = tx - ox
        dy: float = ty - oy
        dist: float = math.sqrt((dx * dx) + (dy * dy))
        if dist < 1e-6:
            return True

        angle_rad: float = math.atan2(dy, dx)
        wall_prox, _ = self._cast_single_ray(
            ox, oy, angle_rad, map_data, max_override=dist
        )
        return wall_prox <= 0.0

    def _cast_single_ray(
        self,
        ox: float,
        oy: float,
        angle_rad: float,
        map_data: MapData,
        max_override: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Marches a single ray across grid cells via Amanatides-Woo DDA.
        """
        effective_max: float = (
            max_override if max_override is not None else self.max_dist
        )
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

            if hit_t > effective_max:
                return 0.0, 0.0

            if map_data.is_wall(tile_x, tile_y):
                return max(0.0, 1.0 - (hit_t / self.max_dist)), 0.0
