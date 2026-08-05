"""
Continuous min-max fitness evaluation module for candidate ranking.
"""

from typing import List

import config
from entities.player_state import PlayerState


class FitnessEvaluator:
    """
    Computes candidate raw scores and normalized fitness ratios.
    """

    @staticmethod
    def calculate_raw_score(
        state: PlayerState,
        initial_bfs_dist: int,
        max_steps: int = config.MAX_SIMULATION_STEPS,
        move_speed: float = config.MOVE_SPEED,
        dist_ratio: float = config.DIST_TO_TIME_BONUS_RATIO,
        lost_hp_impact: float = config.LOST_HP_SCORE_IMPACT_RATIO
    ) -> float:
        """
        Computes progress score plus time bonus weighted by health factor.
        """
        step_frames: float = 1.0 / max(1e-6, move_speed)
        dist_reduced: float = float(initial_bfs_dist - state.best_step_dist)
        bfs_progress_frames: float = max(0.0, dist_reduced * step_frames)
        weighted_bfs_score: float = bfs_progress_frames * dist_ratio

        if state.has_reached_exit:
            time_bonus: float = float(max_steps - state.frames_survived)
        else:
            time_bonus = 0.0

        raw_total: float = weighted_bfs_score + time_bonus

        clamped_hp: float = max(0.0, min(1.0, state.health))
        hp_factor: float = (1.0 - lost_hp_impact) + (
            lost_hp_impact * clamped_hp
        )

        return raw_total * hp_factor

    @staticmethod
    def calculate_theoretical_max_score(
        initial_bfs_dist: int,
        max_steps: int = config.MAX_SIMULATION_STEPS,
        move_speed: float = config.MOVE_SPEED,
        dist_ratio: float = config.DIST_TO_TIME_BONUS_RATIO,
        num_turns: int = 0,
        corner_savings_per_turn: float = 0.586
    ) -> float:
        """
        Returns theoretical max raw score for a perfect solver.

        Corner savings reduce effective travel distance and minimum
        frames, enlarging the time bonus. The BFS score component
        uses the full initial_bfs_dist — matching calculate_raw_score
        — so the ceiling is never breached by a real candidate.
        """
        step_frames: float = 1.0 / max(1e-6, move_speed)
        turn_savings_tiles: float = (
            float(num_turns) * corner_savings_per_turn
        )
        eff_bfs_dist: float = max(
            0.0, float(initial_bfs_dist) - turn_savings_tiles
        )
        min_travel_frames: float = eff_bfs_dist * step_frames
        max_bfs_score: float = (
            float(initial_bfs_dist) * step_frames * dist_ratio
        )
        max_time_bonus: float = max(
            0.0, float(max_steps) - min_travel_frames
        )
        return max_bfs_score + max_time_bonus

    @staticmethod
    def calculate_scaled_score(
        raw_score: float,
        theoretical_max: float
    ) -> float:
        """
        Normalizes raw score to [0.0, 1000.0] range based on theoretical max.
        """
        if theoretical_max < 1e-6:
            return 0.0

        ratio: float = max(0.0, raw_score / theoretical_max)
        return min(1000.0, ratio * 1000.0)

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        """
        Normalizes a list of raw scores to [0.0, 1.0] ratios.
        """
        if not raw_scores:
            return []

        min_s: float = min(raw_scores)
        max_s: float = max(raw_scores)
        span: float = max_s - min_s

        if span < 1e-6:
            return [1.0 for _ in raw_scores]

        return [(s - min_s) / span for s in raw_scores]
