"""
Novelty & Exploration-Driven Fitness Evaluator.
Implements pure non-cheating intrinsic motivation: agents are rewarded for 
maximizing state-space coverage (exploring unique tiles and branching paths)
rather than knowing the location of the exit.
"""

from typing import List, Dict
import config
from entities.player_state import PlayerState


class FitnessEvaluator:
    """
    Computes candidate raw scores rewarding thorough maze exploration and 
    accidental exit discovery.
    """

    @staticmethod
    def get_weights() -> Dict[str, float]:
        """Loads weights from config, falling back to safe defaults if missing."""
        return getattr(config, "FITNESS_WEIGHTS", {
            "tile_discovery": 10.0,     # Strong reward for stepping onto uncharted floor tiles
            "exit_completion": 2000.0,  # Massive flat reward for successfully finding the exit
            "speed_efficiency": 2.0,    # Reward for finishing quickly once found
            "health_penalty": 0.4,      # Health preservation factor
            "collision_penalty": 0.1,   # Very light wall penalty so they aren't afraid to move
            "spin_penalty": 1.0,        # Discourages spinning in place
        })

    @classmethod
    def calculate_raw_score(
        cls,
        state: PlayerState,
        initial_bfs_dist: int,  # Preserved for engine signature compatibility
        max_steps: int = config.MAX_SIMULATION_STEPS,
        move_speed: float = config.MOVE_SPEED,
    ) -> float:
        """
        Computes raw score based on territory expansion (unique tiles discovered)
        and terminal goal discovery.
        """
        weights = cls.get_weights()

        # --- NOVELTY / EXPLORATION REWARD ---
        # Tracks unique grid coordinates visited during the run
        unique_tiles = len(getattr(state, "visited_tiles", set()))
        exploration_score = float(unique_tiles) * weights["tile_discovery"]

        # --- TERMINAL GOAL REWARD ---
        completion_score = 0.0
        speed_bonus = 0.0
        
        if state.has_reached_exit:
            completion_score = weights["exit_completion"]
            frames_saved = float(max_steps - state.frames_survived)
            speed_bonus = frames_saved * weights["speed_efficiency"]

        # --- HEALTH MULTIPLIER ---
        hp_weight = weights.get("health_penalty", 0.4)
        clamped_hp: float = max(0.0, min(1.0, state.health))
        hp_factor: float = (1.0 - hp_weight) + (hp_weight * clamped_hp)

        # --- PENALTIES ---
        collision_count = getattr(state, "collision_count", 0)
        spin_count = getattr(state, "spin_infraction_count", 0)
        
        penalties = (
            (collision_count * weights["collision_penalty"]) +
            (spin_count * weights["spin_penalty"])
        )

        # --- AGGREGATION ---
        base_score = exploration_score + completion_score + speed_bonus - penalties
        final_score = base_score * hp_factor

        return max(0.0, final_score)

    @staticmethod
    def calculate_theoretical_max_score(
        initial_bfs_dist: int,
        max_steps: int = config.MAX_SIMULATION_STEPS,
        move_speed: float = config.MOVE_SPEED,
        num_turns: int = 0,
        corner_savings_per_turn: float = 0.586
    ) -> float:
        """Calculates theoretical max raw score for a full map exploration run."""
        weights = FitnessEvaluator.get_weights()
        step_frames: float = 1.0 / max(1e-6, move_speed)
        
        turn_savings_tiles: float = float(num_turns) * corner_savings_per_turn
        eff_bfs_dist: float = max(0.0, float(initial_bfs_dist) - turn_savings_tiles)
        min_travel_frames: float = eff_bfs_dist * step_frames

        # Estimate max unique tiles as roughly total walkable area in a 12x9 grid (~70 tiles)
        max_tiles_estimated = 75.0
        max_exploration = max_tiles_estimated * weights["tile_discovery"]
        max_completion = weights["exit_completion"]
        max_speed = max(0.0, float(max_steps) - min_travel_frames) * weights["speed_efficiency"]

        return max_exploration + max_completion + max_speed

    @staticmethod
    def calculate_scaled_score(raw_score: float, theoretical_max: float) -> float:
        if theoretical_max < 1e-6:
            return 0.0
        ratio: float = max(0.0, raw_score / theoretical_max)
        return min(1000.0, ratio * 1000.0)

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        if not raw_scores:
            return []
        min_s: float = min(raw_scores)
        max_s: float = max(raw_scores)
        span: float = max_s - min_s
        if span < 1e-6:
            return [1.0 for _ in raw_scores]
        return [(s - min_s) / span for s in raw_scores]