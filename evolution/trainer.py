"""
Headless CPU neuroevolution simulation trainer running batched candidates.
"""

import random
import time
import math
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
from evolution.population import PopulationManager
from evolution.recorder import FrameRecorder
from perception.spatial_transformer import SpatialTransformer


class HeadlessTrainer:
    """
    Coordinates headless training loops and frame timeline recording.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:
        """
        Initializes trainer components, bounds, and population matrices.
        """
        grid_capacity: int = config.GRID_ROWS * config.GRID_COLS
        self.pop_size: int = min(pop_size, grid_capacity)
        self.max_steps: int = max_steps

        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer()
        self.kinematics: CandidateKinematics = CandidateKinematics()
        self.recorder: FrameRecorder = FrameRecorder()

        sample_map: MapData = self.map_generator.generate_solvable_map()
        sample_features: NDArray[np.float32] = (
            self.transformer.compile_feature_vector(
                0.5, 0.5, 0.0, 1.0, 1.0, sample_map
            )
        )

        self.input_size: int = int(sample_features.shape[0])
        self.output_size: int = 2

        self.population: PopulationManager = PopulationManager(
            self.pop_size,
            input_size=self.input_size,
            output_size=self.output_size
        )

        self._wall_grid: Optional[NDArray[np.bool_]] = None
        self._dist_grid: Optional[NDArray[np.int32]] = None

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> FrameRecorder:
        """
        Runs headless neuroevolution simulation across multiple generations.
        """
        print("\n=== CPU NEUROEVOLUTION SIMULATION ===")
        print(
            f"Population: {self.pop_size} | Max Steps: {self.max_steps} | "
            f"Map Regime: {config.MAP_REGIME_GENERATIONS} gens max | "
            f"Curriculum: {config.CURRICULUM_ENABLED}\n"
        )

        header_str: str = (
            f"{'GEN':>7s} | {'TOP':>7s} | {'AVG':>7s} | {'WAY':>7s} | "
            f"{'TARG':>7s} | {'FRAME':>7s} | {'EXITS':>7s} | {'TIME':>7s}"
        )
        print(header_str)
        print("-" * len(header_str))

        target_bfs: int = config.CURRICULUM_START_BFS
        switch_next: bool = True
        gens_on_map: int = 0
        consecutive_failures: int = 0

        map_data: Optional[MapData] = None
        initial_bfs_dist: int = 0
        theoretical_max: float = 0.0
        spawn_headings: Optional[NDArray[np.float64]] = None

        for gen_index in range(num_generations):
            gen_start: float = time.perf_counter()
            new_map: bool = switch_next

            if switch_next:
                (
                    map_data,
                    _pathfinder,
                    initial_bfs_dist,
                    _num_turns,
                    theoretical_max,
                    spawn_headings
                ) = self._prepare_map(target_bfs)
                switch_next = False
                gens_on_map = 0

            candidate_states, candidate_frames = self._simulate_generation(
                map_data, initial_bfs_dist, spawn_headings
            )

            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    state, initial_bfs_dist, self.max_steps
                )
                for state in candidate_states
            ]
            scaled_scores: List[float] = [
                FitnessEvaluator.calculate_scaled_score(
                    score, theoretical_max
                )
                for score in raw_scores
            ]
            norm_scores: List[float] = FitnessEvaluator.normalize_scores(
                raw_scores
            )

            self.recorder.record_generation(
                gen_index,
                map_data,
                candidate_frames,
                scaled_scores,
                norm_scores
            )

            boost: float = config.REGIME_TRANSITION_MUTATION_BOOST
            self.population.mutation_scale = (
                config.MUTATION_SCALE * boost if new_map
                else config.MUTATION_SCALE
            )
            self.population.evolve_next_generation(norm_scores)
            self.population.mutation_scale = config.MUTATION_SCALE

            gen_time: float = time.perf_counter() - gen_start
            top_score: int = int(round(max(scaled_scores)))
            avg_score: float = sum(scaled_scores) / float(len(scaled_scores))

            solvers: List[Tuple[int, int]] = [
                (c_idx, state.frames_survived)
                for c_idx, state in enumerate(candidate_states)
                if state.has_reached_exit
            ]
            solve_count: int = len(solvers)
            solve_str: str = (str(solve_count) if solve_count > 0 else "-")
            frame_str: str = (
                str(min(steps for _, steps in solvers)) if solve_count > 0
                else "-"
            )
            
            exits_str: str = f"{solve_str}"
            time_str: str = f"{gen_time:.2f}s"

            row_str: str = (
                f"{gen_index + 1:>7d} | {top_score:>7d} | "
                f"{avg_score:>7.1f} | {initial_bfs_dist:>7d} | "
                f"{target_bfs:>7d} | {frame_str:>7s} | "
                f"{exits_str:>7s} | {time_str:>7s}"
            )
            print(row_str)

            gens_on_map += 1
            solved: bool = solve_count >= config.REGIME_SOLVE_TARGET

            if solved and gens_on_map >= config.REGIME_MIN_GENERATIONS:
                switch_next = True
                target_bfs = min(
                    target_bfs + config.CURRICULUM_BFS_STEP,
                    config.CURRICULUM_MAX_BFS
                )
                consecutive_failures = 0

            elif gens_on_map >= config.MAP_REGIME_GENERATIONS:
                switch_next = True
                if solved:
                    target_bfs = min(
                        target_bfs + config.CURRICULUM_BFS_STEP,
                        config.CURRICULUM_MAX_BFS
                    )
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if (
                        consecutive_failures >=
                        config.CURRICULUM_FAILURES_BEFORE_EASE
                    ):
                        target_bfs = max(
                            target_bfs - config.CURRICULUM_BFS_STEP,
                            config.CURRICULUM_START_BFS
                        )
                        consecutive_failures = 0

        print("-" * len(header_str))
        print(
            "CPU Training complete! Booting interactive visualizer GUI...\n"
        )
        return self.recorder

    def _prepare_map(
        self,
        target_bfs: int
    ) -> Tuple[
        MapData,
        BFSPathfinder,
        int,
        int,
        float,
        NDArray[np.float64]
    ]:
        """
        Generates/prepares the active map and reusable lookup arrays.
        """
        if config.CURRICULUM_ENABLED:
            map_data: MapData = self._generate_map_for_target(target_bfs)
        else:
            if config.MAP_DIFFICULTY_MIN >= config.MAP_DIFFICULTY_MAX:
                difficulty: float = config.MAP_DIFFICULTY_MIN
            else:
                difficulty = random.uniform(
                    config.MAP_DIFFICULTY_MIN, config.MAP_DIFFICULTY_MAX
                )
            map_data = self.map_generator.generate_solvable_map(
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

        spawn_headings: NDArray[np.float64] = np.asarray(
            [
                self.transformer.generate_random_heading(
                    map_data, map_data.start_pos
                )
                for _ in range(self.pop_size)
            ],
            dtype=np.float64
        )

        self._wall_grid = np.asarray(
            map_data.build_wall_grid(), dtype=np.bool_
        )
        self._dist_grid = np.asarray(
            pathfinder.distance_matrix, dtype=np.int32
        )

        return (
            map_data,
            pathfinder,
            initial_bfs_dist,
            num_turns,
            theoretical_max,
            spawn_headings
        )

    def _generate_map_for_target(self, target_bfs: int) -> MapData:
        """
        Generates a solvable map close to the target curriculum distance.
        """
        minimum: int = max(2, target_bfs)
        maximum: int = minimum + config.CURRICULUM_BFS_WINDOW

        best_map: Optional[MapData] = None
        best_delta: int = 1 << 30

        for _ in range(config.CURRICULUM_MAP_ATTEMPTS):
            map_data: MapData = self.map_generator.generate_solvable_map(
                difficulty_ratio=config.CURRICULUM_DIFFICULTY_RATIO
            )
            pathfinder: BFSPathfinder = BFSPathfinder(map_data)
            pathfinder.compute_distance_matrix()

            distance: int = pathfinder.get_step_distance(*map_data.start_pos)
            if minimum <= distance <= maximum:
                return map_data

            delta: int = (
                minimum - distance if distance < minimum
                else distance - maximum
            )
            if delta < best_delta:
                best_delta = delta
                best_map = map_data

        if best_map is not None:
            return best_map

        return self.map_generator.generate_solvable_map()

    def _simulate_generation(
        self,
        map_data: MapData,
        initial_bfs_dist: int,
        spawn_headings: NDArray[np.float64]
    ) -> Tuple[List[PlayerState], List[List[Dict[str, Any]]]]:
        """
        Simulates the complete population using batched array operations.
        """
        n: int = self.pop_size
        start_x, start_y = map_data.start_pos
        exit_x, exit_y = map_data.exit_pos

        wall_grid: NDArray[np.bool_] = self._wall_grid
        dist_grid: NDArray[np.int32] = self._dist_grid
        width, height = int(map_data.width), int(map_data.height)
        move_speed: float = self.kinematics.move_speed

        xs: NDArray[np.float64] = np.full(
            n, float(start_x) + 0.5, dtype=np.float64
        )
        ys: NDArray[np.float64] = np.full(
            n, float(start_y) + 0.5, dtype=np.float64
        )
        headings: NDArray[np.float64] = spawn_headings.copy()
        healths: NDArray[np.float64] = np.ones(n, dtype=np.float64)
        alive: NDArray[np.bool_] = np.ones(n, dtype=np.bool_)
        reached_exit: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
        has_collided: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
        frames_survived: NDArray[np.int32] = np.zeros(n, dtype=np.int32)
        best_dist: NDArray[np.int32] = np.full(
            n, initial_bfs_dist, dtype=np.int32
        )

        cum_rotation: NDArray[np.float64] = np.zeros(
            n, dtype=np.float64
        )
        stagnation_ticks: NDArray[np.int32] = np.zeros(
            n, dtype=np.int32
        )
        spin_thresh_rad: float = math.radians(
            config.SPIN_ANGLE_THRESHOLD_DEG
        )
        spin_reset_rad: float = math.radians(
            config.SPIN_RESET_ANGLE_DEG
        )
        stagnation_limit: int = int(
            self.max_steps * config.STAGNATION_TIMEOUT_RATIO
        )

        frame_x: NDArray[np.float64] = np.empty(
            (self.max_steps, n), dtype=np.float64
        )
        frame_y: NDArray[np.float64] = np.empty_like(frame_x)
        frame_heading: NDArray[np.float64] = np.empty_like(frame_x)
        frame_health: NDArray[np.float64] = np.empty_like(frame_x)
        frame_dist: NDArray[np.int32] = np.empty(
            (self.max_steps, n), dtype=np.int32
        )
        frame_hit: NDArray[np.bool_] = np.empty(
            (self.max_steps, n), dtype=np.bool_
        )
        frame_alive: NDArray[np.bool_] = np.empty(
            (self.max_steps, n), dtype=np.bool_
        )
        frame_reached: NDArray[np.bool_] = np.empty(
            (self.max_steps, n), dtype=np.bool_
        )

        activation_history: List[NDArray[np.float64]] = [
            np.empty((self.max_steps, n, size), dtype=np.float64)
            for size in self.population.sizes
        ]

        active: NDArray[np.bool_] = alive.copy()
        steps_completed: int = 0
        speeds: NDArray[np.float64] = np.full(
            n, move_speed, dtype=np.float64
        )

        for step_idx in range(self.max_steps):
            if not bool(active.any()):
                break

            active_idx: NDArray[np.int64] = np.flatnonzero(active)

            features: NDArray[np.float32] = (
                self.transformer.compile_feature_batch(
                    xs[active_idx],
                    ys[active_idx],
                    headings[active_idx],
                    speeds[active_idx],
                    healths[active_idx],
                    map_data,
                    wall_grid
                )
            )

            outputs, activations = self.population.forward_batch(
                active_idx.tolist(), features
            )

            for a_idx, act in enumerate(activations):
                activation_history[a_idx][step_idx, active_idx] = act

            move_eff: NDArray[np.float64] = outputs[:, 0]
            turn_eff: NDArray[np.float64] = outputs[:, 1]

            (
                new_x,
                new_y,
                new_heading,
                hit,
                stationary_turn
            ) = self.kinematics.step_batch(
                xs[active_idx],
                ys[active_idx],
                headings[active_idx],
                move_eff,
                turn_eff,
                map_data,
                wall_grid
            )

            old_x: NDArray[np.float64] = xs[active_idx].copy()
            old_y: NDArray[np.float64] = ys[active_idx].copy()

            heading_delta: NDArray[np.float64] = np.abs(
                new_heading - headings[active_idx]
            )
            heading_delta = np.where(
                heading_delta > math.pi,
                (2.0 * math.pi) - heading_delta,
                heading_delta
            )
            cum_rotation[active_idx] = np.where(
                heading_delta >= spin_reset_rad,
                cum_rotation[active_idx] + heading_delta,
                0.0
            )
            stagnation_ticks[active_idx] += 1

            xs[active_idx] = new_x
            ys[active_idx] = new_y
            headings[active_idx] = new_heading
            has_collided[active_idx] = hit
            frames_survived[active_idx] += 1

            is_idle: NDArray[np.bool_] = (
                (move_eff < config.HEALTH_IDLE_MOVE_THRESHOLD) |
                (
                    (np.abs(new_x - old_x) < 1e-4) &
                    (np.abs(new_y - old_y) < 1e-4)
                ) |
                stationary_turn
            )

            active_hp: NDArray[np.float64] = healths[active_idx]
            active_hp -= (
                hit.astype(float) * config.HEALTH_COLL_DMG_PER_FRAME
            )
            active_hp -= (
                is_idle.astype(float) *
                config.HEALTH_IDLE_DMG_PER_FRAME
            )

            if config.SPIN_PENALTY_ENABLED:
                is_spinning: NDArray[np.bool_] = (
                    cum_rotation[active_idx] >= spin_thresh_rad
                )
                active_hp -= (
                    is_spinning.astype(float) *
                    config.SPIN_DMG_PER_FRAME
                )

            if config.STAGNATION_ENABLED:
                is_stagnated: NDArray[np.bool_] = (
                    stagnation_ticks[active_idx] >= stagnation_limit
                )
                active_hp -= (
                    is_stagnated.astype(float) *
                    config.STAGNATION_DMG_PER_FRAME
                )

            np.maximum(active_hp, 0.0, out=active_hp)
            healths[active_idx] = active_hp

            died: NDArray[np.bool_] = active_hp <= 0.0

            tile_x: NDArray[np.int32] = np.floor(new_x).astype(
                np.int32
            )
            tile_y: NDArray[np.int32] = np.floor(new_y).astype(
                np.int32
            )
            inb: NDArray[np.bool_] = (
                (tile_x >= 0) & (tile_x < width) &
                (tile_y >= 0) & (tile_y < height)
            )
            safe_x: NDArray[np.int32] = np.clip(tile_x, 0, width - 1)
            safe_y: NDArray[np.int32] = np.clip(tile_y, 0, height - 1)

            current_dist: NDArray[np.int32] = np.where(
                inb, dist_grid[safe_y, safe_x], 9999
            ).astype(np.int32, copy=False)

            prev_best: NDArray[np.int32] = best_dist[active_idx]
            improved: NDArray[np.bool_] = current_dist < prev_best

            if bool(improved.any()):
                imp_idx: NDArray[np.int64] = active_idx[improved]
                improvement: NDArray[np.float64] = (
                    prev_best[improved] - current_dist[improved]
                ).astype(np.float64)

                stagnation_ticks[imp_idx] = 0

                recovery: NDArray[np.float64] = (
                    improvement *
                    config.HEALTH_COLL_DMG_PER_FRAME *
                    config.HEALTH_RECOVERY_RATIO
                )
                healths[imp_idx] = np.minimum(
                    1.0, healths[imp_idx] + recovery
                )
                best_dist[imp_idx] = current_dist[improved]

            reached_now: NDArray[np.bool_] = (
                (tile_x == exit_x) & (tile_y == exit_y)
            )
            if bool(reached_now.any()):
                reached_indices: NDArray[np.int64] = (
                    active_idx[reached_now]
                )
                reached_exit[reached_indices] = True
                active[reached_indices] = False

            if bool(died.any()):
                died_indices: NDArray[np.int64] = active_idx[died]
                alive[died_indices] = False
                active[died_indices] = False

            frame_x[step_idx] = xs
            frame_y[step_idx] = ys
            frame_heading[step_idx] = headings
            frame_health[step_idx] = healths

            all_tx: NDArray[np.int32] = np.floor(xs).astype(np.int32)
            all_ty: NDArray[np.int32] = np.floor(ys).astype(np.int32)
            all_inb: NDArray[np.bool_] = (
                (all_tx >= 0) & (all_tx < width) &
                (all_ty >= 0) & (all_ty < height)
            )
            all_sx: NDArray[np.int32] = np.clip(all_tx, 0, width - 1)
            all_sy: NDArray[np.int32] = np.clip(all_ty, 0, height - 1)

            frame_dist[step_idx] = np.where(
                all_inb, dist_grid[all_sy, all_sx], 9999
            )
            frame_hit[step_idx] = has_collided
            frame_alive[step_idx] = alive
            frame_reached[step_idx] = reached_exit
            steps_completed = step_idx + 1

        candidate_states: List[PlayerState] = []
        for c_idx in range(n):
            state: PlayerState = PlayerState(
                float(xs[c_idx]), float(ys[c_idx])
            )
            state.heading = float(headings[c_idx])
            state.health = float(healths[c_idx])
            state.is_alive = bool(alive[c_idx])
            state.has_reached_exit = bool(reached_exit[c_idx])
            state.has_collided = bool(has_collided[c_idx])
            state.frames_survived = int(frames_survived[c_idx])
            state.best_step_dist = int(best_dist[c_idx])
            candidate_states.append(state)

        candidate_frames: List[List[Dict[str, Any]]] = [
            [] for _ in range(n)
        ]
        for c_idx in range(n):
            frames: List[Dict[str, Any]] = candidate_frames[c_idx]
            for s_idx in range(steps_completed):
                alive_val: bool = bool(frame_alive[s_idx, c_idx])
                reached_val: bool = bool(frame_reached[s_idx, c_idx])
                hit_val: bool = bool(frame_hit[s_idx, c_idx])

                face: str = PlayerExpress.resolve_face(
                    reached_val, hit_val, alive_val
                )
                activations: List[List[float]] = [
                    activation_history[l_idx][s_idx, c_idx].tolist()
                    for l_idx in range(len(activation_history))
                ]

                frames.append({
                    "step": s_idx + 1,
                    "x": float(frame_x[s_idx, c_idx]),
                    "y": float(frame_y[s_idx, c_idx]),
                    "heading": float(frame_heading[s_idx, c_idx]),
                    "face": face,
                    "hit_wall": hit_val,
                    "health": float(frame_health[s_idx, c_idx]),
                    "is_alive": alive_val,
                    "reached_exit": reached_val,
                    "dist": int(frame_dist[s_idx, c_idx]),
                    "activations": activations
                })

        return candidate_states, candidate_frames
