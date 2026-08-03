"""
Player candidate physical state tracking.
"""

from typing import Tuple


class PlayerState:
    """
    Maintains individual candidate position, health, and progress stats.
    """

    def __init__(self, start_x: float, start_y: float) -> None:
        """
        Initializes candidate state at starting tile position.
        """
        self.x: float = start_x
        self.y: float = start_y
        self.heading: float = 0.0
        self.best_step_dist: int = 9999
        self.frames_survived: int = 0
        self.has_collided: bool = False
        self.has_reached_exit: bool = False
        self.health: float = 1.0
        self.is_alive: bool = True

    @property
    def tile_coords(self) -> Tuple[int, int]:
        """
        Returns integer grid coordinates for active candidate location.
        """
        return int(self.x), int(self.y)

    def reset(self, start_x: float, start_y: float) -> None:
        """
        Resets state variables for new evaluation runs.
        """
        self.x = start_x
        self.y = start_y
        self.heading = 0.0
        self.best_step_dist = 9999
        self.frames_survived = 0
        self.has_collided = False
        self.has_reached_exit = False
        self.health = 1.0
        self.is_alive = True
