"""
Candidate emotional state face expression engine.
"""

import config


class PlayerExpress:
    """
    Determines ASCII face strings based on candidate physics states.
    """

    @staticmethod
    def resolve_face(
        has_reached_exit: bool,
        has_collided: bool,
        is_alive: bool
    ) -> str:
        """
        Evaluates active physics flags to select ASCII expression.
        """
        if has_reached_exit:
            return config.FACE_EXIT
        if not is_alive:
            return config.FACE_DEAD
        if has_collided:
            return config.FACE_WALL
        return config.FACE_WALK
