"""
Headless CPU neuroevolution trainer.
Candidate genomes are stacked weight matrices; the per-step simulation uses
scalar per-candidate stepping (optimal at the fixed small population) with a
single batched einsum forward pass per step, cached numpy lookup grids, and
mastery-based curriculum training that keeps a map until it is solved or a
generation cap is reached.
"""

import time
import random
from typing import List, Dict, Any, Tuple

import numpy as np

import config
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from core.kinematics import CandidateKinematics
from perception.spatial_transformer import SpatialTransformer
from entities.player_state import PlayerState
from entities.player_express import PlayerExpress
from evolution.fitness import FitnessEvaluator
from evolution.population import PopulationManager
from evolution.recorder import FrameRecorder


class HeadlessTrainer:
    """
    Runs CPU neuroevolution and frame timeline recording.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:
        """
        Initializes trainer components and simulation bounds.
        """
        grid_capacity: int = config.GRID_ROWS * config.GRID_COLS
        clamped_pop: int = min(pop_size, grid_capacity)

        self.pop_size: int = clamped_pop
        self.max_steps: int = max_steps
        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer()
        self.kinematics: CandidateKinematics = CandidateKinematics()
        self.recorder: FrameRecorder = FrameRecorder()

        # Dynamically determine input size from actual compiled features
        sample_map = self.map_generator.generate_solvable_map()
        sample_features = self.transformer.compile_feature_vector(
            0.5, 0.5, 0.0, 1.0, 1.0, sample_map
        )
        self.input_size: int = len(sample_features)
        self.output_size: int = 2

        self.population: PopulationManager = PopulationManager(
            clamped_pop,
            input_size=self.input_size,
            output_size=self.output_size
        )

        # Cached numpy lookup grids for the active map
        self._wall_grid: np.ndarray = None
        self._dist_grid: np.ndarray = None

    def _generate_map_for_target(self, target_bfs: int) -> MapData:
        """
        Generates a solvable map whose path length is near the curriculum
        target, retrying until the window is hit or attempts are exhausted.
        """
        lo: int = max(2, target_bfs)
        hi: int = lo + config.CURRICULUM_BFS_WINDOW

        best_map: MapData = None
        best_delta: int = 1 << 30

        for _ in range(config.CURRICULUM_MAP_ATTEMPTS):
            map_data: MapData = self.map_generator.generate_solvable_map(
                difficulty_ratio=config.CURRICULUM_DIFFICULTY_RATIO
            )
            pathfinder: BFSPathfinder = BFSPathfinder(map_data)
            pathfinder.compute_distance_matrix()
            start_dist: int = pathfinder.get_step_distance(*map_data.start_pos)

            if lo <= start_dist <= hi:
                return map_data

            delta: int = 0
            if start_dist < lo:
                delta = lo - start_dist
            elif start_dist > hi:
                delta = start_dist - hi
            if delta < best_delta:
                best_delta = delta
                best_map = map_data

        if best_map is not None:
            return best_map
        return self.map_generator.generate_solvable_map()

    def _prepare_map(self, target_bfs: int) -> Tuple:
        """
        Prepares the active map and its cached numpy grids.
        Returns (map_data, pathfinder, initial_bfs_dist, num_turns,
                 theoretical_max, spawn_headings).
        """
        if config.CURRICULUM_ENABLED:
            map_data = self._generate_map_for_target(target_bfs)
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
        initial_bfs_dist: int = pathfinder.get_step_distance(*map_data.start_pos)

        num_turns: int = pathfinder.count_shortest_path_turns()

        theoretical_max: float = (
            FitnessEvaluator.calculate_theoretical_max_score(
                initial_bfs_dist, self.max_steps, num_turns=num_turns
            )
        )

        # Per-candidate spawn headings fixed for the regime: diverse
        # exploration at spawn with consistent per-slot selection pressure
        spawn_headings: List[float] = [
            self.transformer.generate_random_heading(
                map_data, map_data.start_pos
            )
            for _ in range(self.pop_size)
        ]

        self._wall_grid = map_data.build_wall_grid()
        self._dist_grid = np.array(
            pathfinder.distance_matrix, dtype=np.int64
        )

        return (
            map_data,
            pathfinder,
            initial_bfs_dist,
            num_turns,
            theoretical_max,
            spawn_headings
        )

    def _simulate_generation(
        self,
        map_data: MapData,
        initial_bfs_dist: int,
        spawn_headings: List[float]
    ) -> Tuple[List[PlayerState], List[List[Dict[str, Any]]]]:
        """
        Runs one generation: per-candidate scalar stepping (fast at the small
        fixed population) with a single batched einsum forward pass per step
        and cached numpy grids for distance lookups.
        Returns (candidate_states, candidate_frames).
        """
        n_pop: int = self.pop_size
        start_x, start_y = map_data.start_pos
        exit_x, exit_y = map_data.exit_pos
        dist_grid: np.ndarray = self._dist_grid
        w: int = map_data.width
        h: int = map_data.height
        move_speed: float = self.kinematics.move_speed

        candidate_states: List[PlayerState] = [
            PlayerState(float(start_x) + 0.5, float(start_y) + 0.5)
            for _ in range(n_pop)
        ]
        for state, heading in zip(candidate_states, spawn_headings):
            state.heading = heading
            state.best_step_dist = initial_bfs_dist

        candidate_frames: List[List[Dict[str, Any]]] = [
            [] for _ in range(n_pop)
        ]

        for step in range(1, self.max_steps + 1):
            # Gather features for candidates still simulating
            active_idx: List[int] = []
            feats: List[np.ndarray] = []

            for i, state in enumerate(candidate_states):
                if state.has_reached_exit or not state.is_alive:
                    if candidate_frames[i]:
                        last_f: Dict[str, Any] = candidate_frames[i][-1].copy()
                        last_f["step"] = step
                        candidate_frames[i].append(last_f)
                    continue
                active_idx.append(i)
                feats.append(self.transformer.compile_feature_vector(
                    state.x,
                    state.y,
                    state.heading,
                    move_speed,
                    state.health,
                    map_data
                ))

            if not active_idx:
                break

            features: np.ndarray = np.stack(feats)
            outputs, acts = self.population.forward_batch(active_idx, features)
            act_lists: List[List] = [a.tolist() for a in acts]

            for k, i in enumerate(active_idx):
                state: PlayerState = candidate_states[i]
                move_eff: float = float(outputs[k, 0])
                turn_eff: float = float(outputs[k, 1])

                state.heading, is_stationary_turn = (
                    self.kinematics.apply_rotation(
                        state.heading, turn_eff, move_eff
                    )
                )
                nx, ny, hit = self.kinematics.calculate_forward_step(
                    state.x, state.y, state.heading, move_eff, map_data
                )

                is_idle: bool = (
                    move_eff < 0.05
                    or (
                        abs(nx - state.x) < 1e-4
                        and abs(ny - state.y) < 1e-4
                    )
                    or is_stationary_turn
                )

                state.x = nx
                state.y = ny
                state.has_collided = hit
                state.frames_survived += 1

                if hit:
                    state.health = max(
                        0.0,
                        state.health - config.HEALTH_COLL_DMG_PER_FRAME
                    )
                if is_idle:
                    state.health = max(
                        0.0,
                        state.health - config.HEALTH_IDLE_DMG_PER_FRAME
                    )
                if state.health <= 0.0:
                    state.is_alive = False

                tx: int = int(state.x)
                ty: int = int(state.y)
                if 0 <= tx < w and 0 <= ty < h:
                    curr_dist: int = int(dist_grid[ty, tx])
                else:
                    curr_dist = 9999

                if curr_dist < state.best_step_dist:
                    heal_amount: float = (
                        (state.best_step_dist - curr_dist)
                        * config.HEALTH_COLL_DMG_PER_FRAME
                        * config.HEALTH_RECOVERY_RATIO
                    )
                    state.health = min(1.0, state.health + heal_amount)
                    state.best_step_dist = curr_dist

                if tx == exit_x and ty == exit_y:
                    state.has_reached_exit = True

                face: str = PlayerExpress.resolve_face(
                    state.has_reached_exit,
                    state.has_collided,
                    state.is_alive
                )

                layer_acts: List[List[float]] = [
                    act_lists[0][k]
                ] + [al[k] for al in act_lists[1:]]

                candidate_frames[i].append({
                    "step": step,
                    "x": state.x,
                    "y": state.y,
                    "heading": state.heading,
                    "face": face,
                    "hit_wall": hit,
                    "health": state.health,
                    "is_alive": state.is_alive,
                    "reached_exit": state.has_reached_exit,
                    "dist": curr_dist,
                    "activations": layer_acts,
                })

        return candidate_states, candidate_frames

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> FrameRecorder:
        """
        Runs CPU candidate simulations over multiple generations.
        """
        print("\n=== CPU NEUROEVOLUTION SIMULATION ===")
        print(
            f"Population: {self.pop_size} | Max Steps: {self.max_steps} | "
            f"Map Regime: {config.MAP_REGIME_GENERATIONS} gens max | "
            f"Curriculum: {config.CURRICULUM_ENABLED}\n"
        )
        header_str: str = (
            f"{'GEN':>7s} | {'TOP':>7s} | {'AVG':>7s} | {'WAY':>7s} | "
            f"{'TARG':>7s} | {'FRAME':>7s} | {'EXITS':>7s} | TIME"
        )
        print(header_str)
        print("-" * len(header_str))

        target_bfs: int = config.CURRICULUM_START_BFS
        switch_next: bool = True
        new_map_this_gen: bool = True
        gens_on_map: int = 0
        consecutive_failures: int = 0

        map_data = None
        initial_bfs_dist = 0
        theoretical_max = 0.0
        spawn_headings: List[float] = []

        for gen_idx in range(num_generations):
            gen_start: float = time.time()

            # Regenerate the map at regime boundaries for a stable landscape
            new_map_this_gen = switch_next
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

            # Fitness evaluation
            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    c_state, initial_bfs_dist, self.max_steps
                )
                for c_state in candidate_states
            ]

            scaled_scores: List[float] = [
                FitnessEvaluator.calculate_scaled_score(score, theoretical_max)
                for score in raw_scores
            ]

            norm_scores: List[float] = FitnessEvaluator.normalize_scores(
                raw_scores
            )

            # Record generation
            self.recorder.record_generation(
                gen_idx,
                map_data,
                candidate_frames,
                scaled_scores,
                norm_scores
            )

            # Evolve with a mutation boost right after a map transition
            self.population.mutation_scale = (
                config.MUTATION_SCALE
                * config.REGIME_TRANSITION_MUTATION_BOOST
                if new_map_this_gen else config.MUTATION_SCALE
            )
            self.population.evolve_next_generation(norm_scores)
            self.population.mutation_scale = config.MUTATION_SCALE

            # Statistics
            gen_time: float = time.time() - gen_start
            top_int: int = int(round(max(scaled_scores)))
            avg_scaled: float = (
                sum(scaled_scores) / float(len(scaled_scores))
            )
            winner_idx: int = int(np.argmax(norm_scores))

            solvers: List[Tuple[int, int]] = [
                (c_idx, c_state.frames_survived)
                for c_idx, c_state in enumerate(candidate_states)
                if c_state.has_reached_exit
            ]
            solve_count: int = len(solvers)
            exits_str: str = f"{solve_count}/{self.pop_size}"

            if solve_count > 0:
                fastest_step: int = min(
                    step_cnt for _, step_cnt in solvers
                )
                frame_str: str = str(fastest_step)
            else:
                frame_str = "-"

            winner_str: str = f"# {winner_idx}"

            row_str: str = (
                f"{gen_idx + 1:>7d} | {top_int:>7d} | {avg_scaled:>7.1f} | "
                f"{initial_bfs_dist:>7d} | {target_bfs:>7d} | "
                f"{frame_str:>7s} | {exits_str:>7s} | {gen_time:>5.2f}s"
            )
            print(row_str)

            # Mastery-based curriculum: switch on solve (early) or on cap
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
                        consecutive_failures
                        >= config.CURRICULUM_FAILURES_BEFORE_EASE
                    ):
                        target_bfs = max(
                            target_bfs - config.CURRICULUM_BFS_STEP,
                            config.CURRICULUM_START_BFS
                        )
                        consecutive_failures = 0

        print("-" * len(header_str))
        print("CPU Training complete! Booting interactive visualizer GUI...\n")
        return self.recorder
