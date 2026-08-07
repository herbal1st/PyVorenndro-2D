"""
Grid-aligned DDA raycaster for wall proximity and global target sensing.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

import config
from core.map_data import MapData


class VisionArcSampler:
    """
    Casts probe rays measuring wall tile proximity and global target visibility across visual arc.
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

    def extract_features_batch(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        headings: NDArray[np.float64],
        speeds: NDArray[np.float64],
        healths: NDArray[np.float64],
        wall_grid: NDArray[np.bool_],
        exit_pos: Tuple[int, int]
    ) -> NDArray[np.float32]:
        """
        Extracts vectorized observation features: wall proximity array + single global target flag.
        Returns a (N, VISION_RAYS + 1) float32 array.
        """
        n_cands: int = int(xs.shape[0])
        if n_cands == 0:
            return np.zeros((0, self.num_rays + 1), dtype=np.float32)

        h, w = wall_grid.shape
        ex, ey = exit_pos

        offsets: NDArray[np.float64] = np.asarray(
            self.relative_angles, dtype=np.float64
        )
        angles: NDArray[np.float64] = headings[:, None] + offsets[None, :]
        dir_x: NDArray[np.float64] = np.cos(angles).ravel()
        dir_y: NDArray[np.float64] = np.sin(angles).ravel()

        ox: NDArray[np.float64] = np.repeat(xs, self.num_rays)
        oy: NDArray[np.float64] = np.repeat(ys, self.num_rays)

        tile_x: NDArray[np.int64] = np.floor(ox).astype(np.int64)
        tile_y: NDArray[np.int64] = np.floor(oy).astype(np.int64)

        n_rays_total: int = int(ox.shape[0])
        prox: NDArray[np.float64] = np.zeros(n_rays_total, dtype=np.float64)
        done: NDArray[np.bool_] = np.zeros(n_rays_total, dtype=np.bool_)
        target_spotted: NDArray[np.bool_] = np.zeros(n_rays_total, dtype=np.bool_)

        inb: NDArray[np.bool_] = (
            (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
        )
        txc: NDArray[np.int64] = np.clip(tile_x, 0, w - 1)
        tyc: NDArray[np.int64] = np.clip(tile_y, 0, h - 1)
        
        inside_wall: NDArray[np.bool_] = inb & wall_grid[tyc, txc]
        prox[inside_wall] = 1.0
        done[inside_wall] = True

        inside_target: NDArray[np.bool_] = inb & (txc == ex) & (tyc == ey)
        target_spotted[inside_target] = True

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
            
            wall_hit: NDArray[np.bool_] = inb & wall_grid[tyc, txc]
            target_hit: NDArray[np.bool_] = inb & (txc == ex) & (tyc == ey)
            
            target_spotted |= active & ~no_hit & target_hit

            blocked: NDArray[np.bool_] = active & ~no_hit & (wall_hit | target_hit | ~inb)
            prox = np.where(
                blocked & wall_hit,
                np.clip(1.0 - (hit_t / max_t), 0.0, 1.0),
                prox,
            )
            done |= blocked

        prox[~done] = 0.0
        
        # Reshape prox to (n_cands, self.num_rays)
        prox_matrix = prox.reshape(n_cands, self.num_rays)
        target_matrix = target_spotted.reshape(n_cands, self.num_rays)
        
        # Compute global boolean flag across any ray per candidate: (n_cands, 1)
        global_target_flag = np.any(target_matrix, axis=1, keepdims=True).astype(np.float32)
        
        # Concatenate wall proximities and global target flag into (n_cands, num_rays + 1)
        features = np.hstack((prox_matrix, global_target_flag))
        return features.astype(np.float32)

    def sample_vision_channels(
        self,
        origin_x: float,
        origin_y: float,
        heading_rad: float,
        map_data: MapData
    ) -> NDArray[np.float32]:
        """
        Casts probe rays and returns wall proximity array.
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