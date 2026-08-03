"""
Candidate movement physics and position LERP interpolation.
"""

import math
from typing import Tuple, Optional

import numpy as np

import config
from core.map_data import MapData


class KinematicsProfile:
    """
    Abstract base steering profile defining rotation dynamics.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Calculates updated heading and returns stationary turning state.
        """
        raise NotImplementedError


class CarProfile(KinematicsProfile):
    """
    Car dynamics: rotation velocity is scaled directly by forward effort.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Applies rotation scaled by forward movement effort.
        """
        clamped_turn: float = max(-1.0, min(1.0, turn_effort))
        clamped_move: float = max(0.0, min(1.0, move_effort))
        effective_turn: float = clamped_turn * clamped_move

        new_heading: float = heading_rad + (effective_turn * rad_per_frame)
        return new_heading % (2.0 * math.pi), False


class TankProfile(KinematicsProfile):
    """
    Tank dynamics: steering works in place; stationary turning drains HP.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Applies differential in-place rotation independent of move effort.
        """
        clamped_turn: float = max(-1.0, min(1.0, turn_effort))
        clamped_move: float = max(0.0, min(1.0, move_effort))

        new_heading: float = heading_rad + (clamped_turn * rad_per_frame)
        is_stationary_turn: bool = (
            abs(clamped_turn) > 0.05 and clamped_move < 0.05
        )
        return new_heading % (2.0 * math.pi), is_stationary_turn


class CandidateKinematics:
    """
    Handles 2D continuous movement physics and Circle-to-AABB collisions.
    """

    def __init__(
        self,
        move_speed: float = config.MOVE_SPEED,
        turn_speed_dpsec: float = config.TURN_SPEED,
        radius_ratio: float = config.PLAYER_RADIUS_RATIO,
        fps: int = config.FPS,
        profile_style: str = config.KINEMATICS_PROFILE
    ) -> None:
        """
        Initializes physical movement constants and steering profile.
        """
        self.move_speed: float = move_speed
        self.radius: float = 0.5 * radius_ratio
        self.rad_per_frame: float = (
            math.radians(turn_speed_dpsec) / float(fps)
        )
        if profile_style.upper() == "TANK":
            self.profile: KinematicsProfile = TankProfile()
        else:
            self.profile = CarProfile()

    def apply_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float = 1.0
    ) -> Tuple[float, bool]:
        """
        Delegates rotational calculation to active steering profile.
        """
        return self.profile.calculate_rotation(
            heading_rad, turn_effort, move_effort, self.rad_per_frame
        )

    def calculate_forward_step(
        self,
        curr_x: float,
        curr_y: float,
        heading_rad: float,
        move_effort: float,
        map_data: MapData
    ) -> Tuple[float, float, bool]:
        """
        Calculates step and resolves Circle-to-AABB penetration pushback.
        """
        clamped_effort: float = max(0.0, min(1.0, move_effort))
        if clamped_effort < 1e-4:
            return curr_x, curr_y, False

        step_dist: float = clamped_effort * self.move_speed
        next_x: float = curr_x + (math.cos(heading_rad) * step_dist)
        next_y: float = curr_y + (math.sin(heading_rad) * step_dist)

        resolved_x, resolved_y, hit = self._resolve_circle_aabb(
            next_x, next_y, map_data
        )
        return resolved_x, resolved_y, hit

    def step_batch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        move_effort: np.ndarray,
        turn_effort: np.ndarray,
        map_data: MapData,
        wall_grid: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized tank rotation, forward step, and Circle-to-AABB resolution
        over (N,) states. Returns (x, y, heading, hit, is_stationary_turn).
        Iteration order matches the scalar path bit-for-bit.
        """
        if wall_grid is None:
            wall_grid = map_data.build_wall_grid()
        h, w = wall_grid.shape
        r: float = self.radius
        n: int = int(xs.shape[0])

        clamped_turn: np.ndarray = np.clip(turn_effort, -1.0, 1.0)
        clamped_move: np.ndarray = np.clip(move_effort, 0.0, 1.0)

        new_headings: np.ndarray = (
            headings + (clamped_turn * self.rad_per_frame)
        ) % (2.0 * math.pi)
        is_stationary_turn: np.ndarray = (
            (np.abs(clamped_turn) > 0.05) & (clamped_move < 0.05)
        )

        no_move: np.ndarray = clamped_move < 1e-4
        step_dist: np.ndarray = clamped_move * self.move_speed
        px: np.ndarray = xs + (np.cos(new_headings) * step_dist)
        py: np.ndarray = ys + (np.sin(new_headings) * step_dist)
        px = np.where(no_move, xs, px)
        py = np.where(no_move, ys, py)

        hit: np.ndarray = np.zeros(n, dtype=np.bool_)

        for _ in range(2):
            base_tx: np.ndarray = np.floor(px).astype(np.int64)
            base_ty: np.ndarray = np.floor(py).astype(np.int64)

            # 3x3 neighborhood always covers the circle's AABB overlap range
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    tx: np.ndarray = base_tx + dx
                    ty: np.ndarray = base_ty + dy
                    inb: np.ndarray = (
                        (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
                    )
                    txc: np.ndarray = np.clip(tx, 0, w - 1)
                    tyc: np.ndarray = np.clip(ty, 0, h - 1)
                    is_wall: np.ndarray = inb & wall_grid[tyc, txc]
                    if not bool(is_wall.any()):
                        continue

                    cx: np.ndarray = np.clip(
                        px, tx.astype(np.float64), (tx + 1).astype(np.float64)
                    )
                    cy: np.ndarray = np.clip(
                        py, ty.astype(np.float64), (ty + 1).astype(np.float64)
                    )
                    ddx: np.ndarray = px - cx
                    ddy: np.ndarray = py - cy
                    dist_sq: np.ndarray = (ddx * ddx) + (ddy * ddy)

                    pen: np.ndarray = is_wall & (dist_sq < (r * r))
                    hit |= pen
                    if not bool(pen.any()):
                        continue

                    dist: np.ndarray = np.sqrt(dist_sq)
                    overlap: np.ndarray = r - dist
                    zero: np.ndarray = pen & (dist_sq < 1e-12)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        nrm_x: np.ndarray = np.where(
                            dist > 1e-6, ddx / dist, 0.0
                        )
                        nrm_y: np.ndarray = np.where(
                            dist > 1e-6, ddy / dist, 0.0
                        )

                    px = np.where(pen, px + (nrm_x * overlap), px)
                    py = np.where(pen, py + (nrm_y * overlap), py)
                    px = np.where(zero, px + 0.01, px)
                    py = np.where(zero, py + 0.01, py)

        px = np.where(no_move, xs, px)
        py = np.where(no_move, ys, py)
        hit = np.where(no_move, False, hit)

        return px, py, new_headings, hit, is_stationary_turn

    def interpolate_pixel_pos(
        self,
        tile_x: float,
        tile_y: float,
        tile_size: int = config.TILE_SIZE
    ) -> Tuple[int, int]:
        """
        Converts continuous tile coordinates to screen pixel positions.
        """
        pixel_x: int = int(round(tile_x * tile_size))
        pixel_y: int = int(round(tile_y * tile_size))
        return pixel_x, pixel_y

    def _resolve_circle_aabb(
        self,
        px: float,
        py: float,
        map_data: MapData,
        passes: int = 2
    ) -> Tuple[float, float, bool]:
        """
        Resolves circle penetration against surrounding wall tile AABBs.
        """
        r: float = self.radius
        has_collided: bool = False

        for _ in range(passes):
            min_tx: int = max(0, int(math.floor(px - r)))
            max_tx: int = min(
                map_data.width - 1, int(math.floor(px + r))
            )
            min_ty: int = max(0, int(math.floor(py - r)))
            max_ty: int = min(
                map_data.height - 1, int(math.floor(py + r))
            )

            for ty in range(min_ty, max_ty + 1):
                for tx in range(min_tx, max_tx + 1):
                    if not map_data.is_wall(tx, ty):
                        continue

                    cx: float = max(
                        float(tx), min(px, float(tx) + 1.0)
                    )
                    cy: float = max(
                        float(ty), min(py, float(ty) + 1.0)
                    )

                    dx: float = px - cx
                    dy: float = py - cy
                    dist_sq: float = (dx * dx) + (dy * dy)

                    if dist_sq < (r * r):
                        has_collided = True
                        dist: float = math.sqrt(dist_sq)

                        if dist > 1e-6:
                            overlap: float = r - dist
                            nx: float = dx / dist
                            ny: float = dy / dist
                            px += nx * overlap
                            py += ny * overlap
                        else:
                            px += 0.01
                            py += 0.01

        return px, py, has_collided
