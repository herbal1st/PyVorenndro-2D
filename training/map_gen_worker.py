"""
Standalone map-generation worker for the GPU trainer's parallel map pool.

Kept deliberately free of torch imports so ``spawn`` workers start up fast: it
only pulls in the pure-numpy map/pathfinding modules and seeds the global
``random`` from the caller-provided seed so every task yields an independent
maze no matter which worker executes it.
"""

import random
from typing import Any, Tuple

import numpy as np

import config
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from perception.spatial_transformer import SpatialTransformer
from training.fitness import FitnessEvaluator

_WORKER_MAP_GENERATOR: Any = None
_WORKER_HEADING_GENERATOR: Any = None


def _generators() -> Tuple[Any, Any]:
    """
    Lazily builds the per-worker map/heading generators (imported classes are
    already in the module namespace, so this only instantiates them).
    """
    global _WORKER_MAP_GENERATOR, _WORKER_HEADING_GENERATOR
    if _WORKER_MAP_GENERATOR is None:
        _WORKER_MAP_GENERATOR = MapGenerator()
        _WORKER_HEADING_GENERATOR = SpatialTransformer()
    return _WORKER_MAP_GENERATOR, _WORKER_HEADING_GENERATOR


def _generate_run_map_worker(
    seed: int,
    max_steps: int,
) -> Tuple[Any, ...]:
    """
    Mirrors ``HeadlessTrainer._generate_run_map`` but reseeds the global RNG
    first so the maze, its difficulty, and the spawn heading are all derived
    from ``seed`` instead of the main process's stream.
    """
    random.seed(seed)

    if config.MAP_DIFFICULTY_MIN >= config.MAP_DIFFICULTY_MAX:
        difficulty: float = config.MAP_DIFFICULTY_MIN
    else:
        difficulty = random.uniform(
            config.MAP_DIFFICULTY_MIN,
            config.MAP_DIFFICULTY_MAX,
        )

    map_generator, heading_generator = _generators()

    map_data = map_generator.generate_solvable_map(
        difficulty_ratio=difficulty,
    )

    pathfinder = BFSPathfinder(map_data)
    pathfinder.compute_distance_matrix()

    initial_bfs_dist: int = pathfinder.get_step_distance(
        *map_data.start_pos
    )
    num_turns: int = pathfinder.count_shortest_path_turns()

    theoretical_max: float = (
        FitnessEvaluator.calculate_theoretical_max_score(
            initial_bfs_dist,
            max_steps,
            num_turns=num_turns,
        )
    )

    spawn_heading: float = heading_generator.generate_random_heading(
        map_data,
        map_data.start_pos,
    )

    dist_grid: np.ndarray = np.asarray(
        pathfinder.distance_matrix,
        dtype=np.int64,
    )

    return (
        map_data,
        dist_grid,
        initial_bfs_dist,
        num_turns,
        theoretical_max,
        spawn_heading,
    )
