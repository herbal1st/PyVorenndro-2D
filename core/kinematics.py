"""
Candidate movement physics and position LERP interpolation.
"""

import math
from typing import Tuple

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
