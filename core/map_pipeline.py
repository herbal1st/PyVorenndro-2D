"""
Modular map generation pipelines for training and display control checks.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from perception.spatial_transformer import SpatialTransformer
from training.fitness import FitnessEvaluator


class TrainingMapPipeline:
    """
    High-throughput batch map generation pipeline for training sessions.
    """

    def __init__(
        self,
        map_generator: Optional[MapGenerator] = None,
        transformer: Optional[SpatialTransformer] = None,
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:
        """
        Initializes map generator, spatial transformer, and max step limits.
        """
        self.map_generator: MapGenerator = (
            map_generator or MapGenerator()
        )
        self.transformer: SpatialTransformer = (
            transformer or SpatialTransformer()
        )
        self.max_steps: int = max_steps

    def generate_run_map(
        self
    ) -> Tuple[MapData, np.ndarray, int, int, float, float]:
        """
        Generates a solvable maze, distance matrix, and spawn heading.
        """
        if config.MAP_DIFFICULTY_MIN >= config.MAP_DIFFICULTY_MAX:
            difficulty: float = config.MAP_DIFFICULTY_MIN
        else:
            difficulty = random.uniform(
                config.MAP_DIFFICULTY_MIN, config.MAP_DIFFICULTY_MAX
            )

        map_data: MapData = self.map_generator.generate_solvable_map(
            difficulty_ratio=difficulty
        )

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        initial_bfs_dist: int = pathfinder.get_step_distance(
            *map_data.start_pos
        )
        num_turns: int = pathfinder.count_shortest_path_turns()

        theoretical_max: float = (
            FitnessEvaluator.calculate_theoretical_max_score(
                initial_bfs_dist, self.max_steps, num_turns=num_turns
            )
        )

        spawn_heading: float = self.transformer.generate_random_heading(
            map_data, map_data.start_pos
        )

        dist_grid: np.ndarray = np.asarray(
            pathfinder.distance_matrix, dtype=np.int64
        )

        return (
            map_data,
            dist_grid,
            initial_bfs_dist,
            num_turns,
            theoretical_max,
            spawn_heading
        )

    def generate_run_maps(self, count: int) -> List[Tuple]:
        """
        Generates a list of count independently seeded maze run tuples.
        """
        return [self.generate_run_map() for _ in range(count)]


class DisplayMapFactory:
    """
    Generates fresh control-check maps for display runner visualizer.
    """

    def __init__(
        self,
        map_generator: Optional[MapGenerator] = None,
        transformer: Optional[SpatialTransformer] = None,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes map generator and transformer with profile context.
        """
        self.map_generator: MapGenerator = (
            map_generator or MapGenerator()
        )
        self.transformer: SpatialTransformer = (
            transformer or SpatialTransformer(
                profile_config=profile_config
            )
        )

    def set_profile_config(
        self, profile_config: Dict[str, Any]
    ) -> None:
        """
        Updates spatial transformer with current profile settings.
        """
        self.transformer = SpatialTransformer(
            profile_config=profile_config
        )

    def create_display_map(
        self
    ) -> Tuple[MapData, np.ndarray, int, float]:
        """
        Generates a fresh unseen display maze, dist matrix, and heading.
        """
        if config.MAP_DIFFICULTY_MIN >= config.MAP_DIFFICULTY_MAX:
            difficulty: float = config.MAP_DIFFICULTY_MIN
        else:
            difficulty = random.uniform(
                config.MAP_DIFFICULTY_MIN, config.MAP_DIFFICULTY_MAX
            )

        map_data: MapData = self.map_generator.generate_solvable_map(
            difficulty_ratio=difficulty
        )

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        initial_bfs: int = pathfinder.get_step_distance(
            *map_data.start_pos
        )
        dist_grid: np.ndarray = np.asarray(
            pathfinder.distance_matrix, dtype=np.int64
        )
        spawn_heading: float = self.transformer.generate_random_heading(
            map_data, map_data.start_pos
        )

        return map_data, dist_grid, initial_bfs, spawn_heading
