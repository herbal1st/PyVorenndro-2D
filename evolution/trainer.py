"""
Optimized Headless CPU trainer using Numba-accelerated raycasting and matrix passes
with persistent worker pool execution.
"""

from concurrent.futures import ProcessPoolExecutor
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import config
from core.kinematics import CandidateKinematics
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from entities.player_express import PlayerExpress
from entities.player_state import PlayerState
from evolution.fitness import FitnessEvaluator
from evolution.population import PopulationManager, numba_layer_forward
from evolution.recorder import FrameRecorder
from perception.spatial_transformer import SpatialTransformer


def _worker_eval(args: Tuple[int, List[Tuple[NDArray[np.float64], NDArray[np.float64]]], MapData, NDArray[np.bool_], NDArray[np.int32], int, float, int]) -> Tuple[int, PlayerState, List[Dict[str, Any]]]:
    """
    Lightweight worker process function evaluating a single candidate.
    """
    (
        c_idx,
        brain_weights,
        map_data,
        wall_grid,
        dist_grid,
        initial_bfs_dist,
        spawn_heading,
        max_steps
    ) = args

    transformer = SpatialTransformer()
    kinematics = CandidateKinematics()

    start_x, start_y = map_data.start_pos
    exit_x, exit_y = map_data.exit_pos
    width, height = int(map_data.width), int(map_data.height)

    x: float = float(start_x) + 0.5
    y: float = float(start_y) + 0.5
    heading: float = float(spawn_heading)
    health: float = 1.0
    is_alive: bool = True
    has_reached_exit: bool = False
    has_collided: bool = False
    frames_survived: int = 0
    best_dist: int = initial_bfs_dist

    cum_rotation: float = 0.0
    stagnation_ticks: int = 0
    spin_thresh_rad: float = math.radians(config.SPIN_ANGLE_THRESHOLD_DEG)
    spin_reset_rad: float = math.radians(config.SPIN_RESET_ANGLE_DEG)
    stagnation_limit: int = int(max_steps * config.STAGNATION_TIMEOUT_RATIO)
    move_speed: float = kinematics.move_speed

    frames: List[Dict[str, Any]] = []

    for step_idx in range(max_steps):
        if not is_alive or has_reached_exit:
            break

        # Spatial sensing
        features = transformer.compile_feature_vector(
            x, y, heading, move_speed, health, wall_grid
        )

        # Numba-accelerated network evaluation
        activations: List[NDArray[np.float64]] = [features]
        curr_in = features

        for w, b in brain_weights:
            curr_in = numba_layer_forward(curr_in, w, b)
            activations.append(curr_in)

        outputs = activations[-1]
        move_eff, turn_eff = float(outputs[0]), float(outputs[1])

        # Step physics
        new_x_a, new_y_a, new_h_a, hit_a, stat_a = kinematics.step_batch(
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            np.array([heading], dtype=np.float64),
            np.array([move_eff], dtype=np.float64),
            np.array([turn_eff], dtype=np.float64),
            map_data,
            wall_grid
        )

        nx, ny, nh = float(new_x_a[0]), float(new_y_a[0]), float(new_h_a[0])
        hit, stationary_turn = bool(hit_a[0]), bool(stat_a[0])

        h_delta = abs(nh - heading)
        if h_delta > math.pi:
            h_delta = (2.0 * math.pi) - h_delta

        cum_rotation = cum_rotation + h_delta if h_delta >= spin_reset_rad else 0.0
        stagnation_ticks += 1

        old_x, old_y = x, y
        x, y, heading = nx, ny, nh
        has_collided = hit
        frames_survived += 1

        is_idle = (
            (move_eff < config.HEALTH_IDLE_MOVE_THRESHOLD) or
            (abs(nx - old_x) < 1e-4 and abs(ny - old_y) < 1e-4) or
            stationary_turn
        )

        if hit:
            health -= config.HEALTH_COLL_DMG_PER_FRAME
        if is_idle:
            health -= config.HEALTH_IDLE_DMG_PER_FRAME
        if config.SPIN_PENALTY_ENABLED and cum_rotation >= spin_thresh_rad:
            health -= config.SPIN_DMG_PER_FRAME
        if config.STAGNATION_ENABLED and stagnation_ticks >= stagnation_limit:
            health -= config.STAGNATION_DMG_PER_FRAME

        health = max(0.0, health)
        if health <= 0.0:
            is_alive = False

        tx, ty = int(math.floor(x)), int(math.floor(y))
        in_b = (0 <= tx < width) and (0 <= ty < height)
        sx, sy = min(max(tx, 0), width - 1), min(max(ty, 0), height - 1)
        cur_d = int(dist_grid[sy, sx]) if in_b else 9999

        if cur_d < best_dist:
            health = min(1.0, health + (float(best_dist - cur_d) * config.HEALTH_COLL_DMG_PER_FRAME * config.HEALTH_RECOVERY_RATIO))
            best_dist = cur_d
            stagnation_ticks = 0

        if tx == exit_x and ty == exit_y:
            has_reached_exit = True

        frames.append({
            "step": step_idx + 1,
            "x": x,
            "y": y,
            "heading": heading,
            "face": PlayerExpress.resolve_face(has_reached_exit, has_collided, is_alive),
            "hit_wall": has_collided,
            "health": health,
            "is_alive": is_alive,
            "reached_exit": has_reached_exit,
            "dist": cur_d,
            "activations": [a.tolist() for a in activations]
        })

    state = PlayerState(x, y)
    state.heading, state.health, state.is_alive = heading, health, is_alive
    state.has_reached_exit, state.has_collided = has_reached_exit, has_collided
    state.frames_survived, state.best_step_dist = frames_survived, best_dist

    return c_idx, state, frames


class HeadlessTrainer:
    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:
        grid_capacity: int = config.GRID_ROWS * config.GRID_COLS
        self.pop_size: int = min(pop_size, grid_capacity)
        self.max_steps: int = max_steps

        self.map_generator = MapGenerator()
        self.transformer = SpatialTransformer()
        self.kinematics = CandidateKinematics()
        self.recorder = FrameRecorder()

        sample_map = self.map_generator.generate_solvable_map()
        sample_features = self.transformer.compile_feature_vector(
            0.5, 0.5, 0.0, 1.0, 1.0, sample_map.build_wall_grid()
        )

        self.input_size = int(sample_features.shape[0])
        self.output_size = 2

        self.population = PopulationManager(
            self.pop_size,
            input_size=self.input_size,
            output_size=self.output_size
        )

        self._wall_grid: Optional[NDArray[np.bool_]] = None
        self._dist_grid: Optional[NDArray[np.int32]] = None
        self._executor = ProcessPoolExecutor(max_workers=max(1, config.SIMULATION_WORKERS))

    def run_training_session(self, num_generations: int = config.LEARNING_GENERATIONS) -> FrameRecorder:
        print("\n=== CPU NEUROEVOLUTION SIMULATION (PARALLEL + NUMBA) ===")
        print(f"Population: {self.pop_size} | Workers: {config.SIMULATION_WORKERS}\n")

        header_str = f"{'GEN':>7s} | {'TOP':>7s} | {'AVG':>7s} | {'EXITS':>7s} | {'TIME':>7s}"
        print(header_str)
        print("-" * len(header_str))

        try:
            for gen_index in range(num_generations):
                gen_start = time.perf_counter()

                map_data, _, initial_bfs_dist, _, theoretical_max, spawn_headings = self._prepare_map(4)

                candidate_states, candidate_frames = self._simulate_generation(
                    map_data, initial_bfs_dist, spawn_headings
                )

                raw_scores = [
                    FitnessEvaluator.calculate_raw_score(state, initial_bfs_dist, self.max_steps)
                    for state in candidate_states
                ]
                scaled_scores = [
                    FitnessEvaluator.calculate_scaled_score(score, theoretical_max)
                    for score in raw_scores
                ]
                norm_scores = FitnessEvaluator.normalize_scores(raw_scores)

                self.recorder.record_generation(gen_index, map_data, candidate_frames, scaled_scores, norm_scores)
                self.population.evolve_next_generation(norm_scores)

                gen_time = time.perf_counter() - gen_start
                top_score = int(round(max(scaled_scores)))
                avg_score = sum(scaled_scores) / float(len(scaled_scores))
                exits = sum(1 for s in candidate_states if s.has_reached_exit)

                print(f"{gen_index + 1:>7d} | {top_score:>7d} | {avg_score:>7.1f} | {exits:>7d} | {gen_time:>6.2f}s")

        finally:
            self._executor.shutdown()

        return self.recorder

    def _prepare_map(self, target_bfs: int):
        map_data = self.map_generator.generate_solvable_map()
        pathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        initial_bfs_dist = pathfinder.get_step_distance(*map_data.start_pos)
        num_turns = pathfinder.count_shortest_path_turns()
        theoretical_max = FitnessEvaluator.calculate_theoretical_max_score(
            initial_bfs_dist, self.max_steps, num_turns=num_turns
        )

        spawn_headings = np.asarray(
            [self.transformer.generate_random_heading(map_data, map_data.start_pos) for _ in range(self.pop_size)],
            dtype=np.float64
        )

        self._wall_grid = np.asarray(map_data.build_wall_grid(), dtype=np.bool_)
        self._dist_grid = np.asarray(pathfinder.distance_matrix, dtype=np.int32)

        return map_data, pathfinder, initial_bfs_dist, num_turns, theoretical_max, spawn_headings

    def _simulate_generation(self, map_data: MapData, initial_bfs_dist: int, spawn_headings: NDArray[np.float64]):
        tasks = []
        for c_idx in range(self.pop_size):
            weights = self.population.get_candidate_weights(c_idx)
            tasks.append((
                c_idx,
                weights,
                map_data,
                self._wall_grid,
                self._dist_grid,
                initial_bfs_dist,
                spawn_headings[c_idx],
                self.max_steps
            ))

        if config.SIMULATION_WORKERS > 1:
            results = list(self._executor.map(_worker_eval, tasks))
        else:
            results = [_worker_eval(t) for t in tasks]

        results.sort(key=lambda r: r[0])
        return [r[1] for r in results], [r[2] for r in results]