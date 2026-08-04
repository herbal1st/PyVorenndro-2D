"""
Headless CPU neuroevolution trainer.

Candidate genomes are stacked weight matrices. Generation simulation is
parallelized across CPU processes. Each worker receives a chunk of genomes
and performs batched neural-network inference for that chunk.

The multiprocessing implementation uses Python's portable "spawn" context,
so it works on Windows, Linux and macOS.
"""

import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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


def _forward_batch(
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    features: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Batched neural-network forward pass for a worker's local population.

    The worker receives complete genome chunks, so local candidate indices
    are sufficient here and no parent-process PopulationManager is needed.
    """
    x: np.ndarray = features.astype(np.float64, copy=False)

    acts: List[np.ndarray] = [x]

    hidden_layers: int = len(weights) - 1

    for layer_idx, (w, b) in enumerate(zip(weights, biases)):
        x = np.einsum(
            "ni,nij->nj",
            x,
            w,
            optimize=False
        ) + b.squeeze(1)

        acts.append(x)

        if layer_idx < hidden_layers:
            x = np.maximum(0.0, x)

    move_eff: np.ndarray = 1.0 / (
        1.0 + np.exp(-np.clip(x[:, 0:1], -500.0, 500.0))
    )

    turn_eff: np.ndarray = np.tanh(x[:, 1:2])

    outputs: np.ndarray = np.hstack([
        move_eff,
        turn_eff
    ])

    return outputs, acts


def _simulate_chunk_worker(
    chunk_indices: List[int],
    chunk_weights: List[np.ndarray],
    chunk_biases: List[np.ndarray],
    map_data: MapData,
    initial_bfs_dist: int,
    spawn_headings: List[float],
    max_steps: int
) -> Tuple[
    List[Tuple[int, PlayerState]],
    List[Tuple[int, List[Dict[str, Any]]]]
]:
    """
    Simulates a chunk of candidates in one worker process.

    Everything required by the worker is explicitly passed as an argument.
    This is what makes the implementation compatible with multiprocessing
    spawn on Windows, Linux and macOS.
    """

    transformer: SpatialTransformer = SpatialTransformer()
    kinematics: CandidateKinematics = CandidateKinematics()

    dist_grid: np.ndarray = np.array(
        BFSPathfinder(map_data).distance_matrix,
        dtype=np.int64
    )

    # The parent already computed the distance matrix, but rebuilding it
    # locally would be wasteful. MapData is small, so the worker instead
    # reconstructs the matrix from the map's pathfinder state if available.
    #
    # If unavailable, compute it once here.
    if dist_grid.size == 0:
        pathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()
        dist_grid = np.array(
            pathfinder.distance_matrix,
            dtype=np.int64
        )

    start_x, start_y = map_data.start_pos
    exit_x, exit_y = map_data.exit_pos

    move_speed: float = kinematics.move_speed

    n_chunk: int = len(chunk_indices)

    candidate_states: List[PlayerState] = [
        PlayerState(
            float(start_x) + 0.5,
            float(start_y) + 0.5
        )
        for _ in range(n_chunk)
    ]

    for local_idx, global_idx in enumerate(chunk_indices):
        state: PlayerState = candidate_states[local_idx]
        state.heading = spawn_headings[global_idx]
        state.best_step_dist = initial_bfs_dist

    candidate_frames: List[List[Dict[str, Any]]] = [
        []
        for _ in range(n_chunk)
    ]

    for step in range(1, max_steps + 1):

        active_local: List[int] = []
        features_list: List[np.ndarray] = []

        for local_idx, state in enumerate(candidate_states):

            if state.has_reached_exit or not state.is_alive:
                if candidate_frames[local_idx]:
                    last_frame: Dict[str, Any] = (
                        candidate_frames[local_idx][-1].copy()
                    )

                    last_frame["step"] = step

                    candidate_frames[local_idx].append(
                        last_frame
                    )

                continue

            active_local.append(local_idx)

            features_list.append(
                transformer.compile_feature_vector(
                    state.x,
                    state.y,
                    state.heading,
                    move_speed,
                    state.health,
                    map_data
                )
            )

        if not active_local:
            break

        features: np.ndarray = np.stack(features_list)

        outputs, acts = _forward_batch(
            chunk_weights,
            chunk_biases,
            features
        )

        act_lists: List[List] = [
            activation.tolist()
            for activation in acts
        ]

        for k, local_idx in enumerate(active_local):

            state: PlayerState = candidate_states[local_idx]

            move_eff: float = float(outputs[k, 0])
            turn_eff: float = float(outputs[k, 1])

            state.heading, is_stationary_turn = (
                kinematics.apply_rotation(
                    state.heading,
                    turn_eff,
                    move_eff
                )
            )

            nx, ny, hit = (
                kinematics.calculate_forward_step(
                    state.x,
                    state.y,
                    state.heading,
                    move_eff,
                    map_data
                )
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
                    state.health
                    - config.HEALTH_COLL_DMG_PER_FRAME
                )

            if is_idle:
                state.health = max(
                    0.0,
                    state.health
                    - config.HEALTH_IDLE_DMG_PER_FRAME
                )

            if state.health <= 0.0:
                state.is_alive = False

            tx: int = int(state.x)
            ty: int = int(state.y)

            if (
                0 <= tx < map_data.width
                and 0 <= ty < map_data.height
            ):
                curr_dist: int = int(
                    dist_grid[ty, tx]
                )
            else:
                curr_dist = 9999

            if curr_dist < state.best_step_dist:

                heal_amount: float = (
                    (
                        state.best_step_dist
                        - curr_dist
                    )
                    * config.HEALTH_COLL_DMG_PER_FRAME
                    * config.HEALTH_RECOVERY_RATIO
                )

                state.health = min(
                    1.0,
                    state.health + heal_amount
                )

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
            ] + [
                layer[k]
                for layer in act_lists[1:]
            ]

            candidate_frames[local_idx].append({
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

    states_result: List[Tuple[int, PlayerState]] = list(
        zip(
            chunk_indices,
            candidate_states
        )
    )

    frames_result: List[
        Tuple[int, List[Dict[str, Any]]]
    ] = list(
        zip(
            chunk_indices,
            candidate_frames
        )
    )

    return states_result, frames_result


class HeadlessTrainer:
    """
    Runs CPU neuroevolution and frame timeline recording.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:

        grid_capacity: int = (
            config.GRID_ROWS * config.GRID_COLS
        )

        clamped_pop: int = min(
            pop_size,
            grid_capacity
        )

        self.pop_size: int = clamped_pop
        self.max_steps: int = max_steps

        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer()
        self.kinematics: CandidateKinematics = CandidateKinematics()
        self.recorder: FrameRecorder = FrameRecorder()

        sample_map = (
            self.map_generator.generate_solvable_map()
        )

        sample_features = (
            self.transformer.compile_feature_vector(
                0.5,
                0.5,
                0.0,
                1.0,
                1.0,
                sample_map
            )
        )

        self.input_size: int = len(sample_features)
        self.output_size: int = 2

        self.population: PopulationManager = (
            PopulationManager(
                clamped_pop,
                input_size=self.input_size,
                output_size=self.output_size
            )
        )

        self._wall_grid: np.ndarray = None
        self._dist_grid: np.ndarray = None

        configured_workers: int = getattr(
            config,
            "SIMULATION_WORKERS",
            0
        )

        if configured_workers <= 0:
            cpu_count: int = mp.cpu_count() or 1

            # Leave one logical CPU available for the main process.
            if cpu_count > 2:
                configured_workers = cpu_count - 1
            else:
                configured_workers = 1

        self.simulation_workers: int = max(
            1,
            min(
                int(configured_workers),
                self.pop_size
            )
        )

        self._process_pool: ProcessPoolExecutor = None

        # Portable spawn context.
        self._mp_context = mp.get_context("spawn")

    def _generate_map_for_target(
        self,
        target_bfs: int
    ) -> MapData:

        lo: int = max(
            2,
            target_bfs
        )

        hi: int = (
            lo
            + config.CURRICULUM_BFS_WINDOW
        )

        best_map: MapData = None
        best_delta: int = 1 << 30

        for _ in range(
            config.CURRICULUM_MAP_ATTEMPTS
        ):

            map_data: MapData = (
                self.map_generator.generate_solvable_map(
                    difficulty_ratio=(
                        config.CURRICULUM_DIFFICULTY_RATIO
                    )
                )
            )

            pathfinder: BFSPathfinder = (
                BFSPathfinder(map_data)
            )

            pathfinder.compute_distance_matrix()

            start_dist: int = (
                pathfinder.get_step_distance(
                    *map_data.start_pos
                )
            )

            if lo <= start_dist <= hi:
                return map_data

            if start_dist < lo:
                delta = lo - start_dist
            else:
                delta = start_dist - hi

            if delta < best_delta:
                best_delta = delta
                best_map = map_data

        if best_map is not None:
            return best_map

        return (
            self.map_generator.generate_solvable_map()
        )

    def _prepare_map(
        self,
        target_bfs: int
    ) -> Tuple:

        if config.CURRICULUM_ENABLED:

            map_data = (
                self._generate_map_for_target(
                    target_bfs
                )
            )

        else:

            if (
                config.MAP_DIFFICULTY_MIN
                >= config.MAP_DIFFICULTY_MAX
            ):
                difficulty: float = (
                    config.MAP_DIFFICULTY_MIN
                )
            else:
                difficulty = random.uniform(
                    config.MAP_DIFFICULTY_MIN,
                    config.MAP_DIFFICULTY_MAX
                )

            map_data = (
                self.map_generator.generate_solvable_map(
                    difficulty_ratio=difficulty
                )
            )

        pathfinder: BFSPathfinder = (
            BFSPathfinder(map_data)
        )

        pathfinder.compute_distance_matrix()

        initial_bfs_dist: int = (
            pathfinder.get_step_distance(
                *map_data.start_pos
            )
        )

        num_turns: int = (
            pathfinder.count_shortest_path_turns()
        )

        theoretical_max: float = (
            FitnessEvaluator.calculate_theoretical_max_score(
                initial_bfs_dist,
                self.max_steps,
                num_turns=num_turns
            )
        )

        spawn_headings: List[float] = [
            self.transformer.generate_random_heading(
                map_data,
                map_data.start_pos
            )
            for _ in range(self.pop_size)
        ]

        self._wall_grid = (
            map_data.build_wall_grid()
        )

        self._dist_grid = np.array(
            pathfinder.distance_matrix,
            dtype=np.int64
        )

        return (
            map_data,
            pathfinder,
            initial_bfs_dist,
            num_turns,
            theoretical_max,
            spawn_headings
        )

    def _make_chunks(self) -> List[List[int]]:
        """
        Splits the population into approximately equal chunks.
        """

        worker_count: int = min(
            self.simulation_workers,
            self.pop_size
        )

        chunks: List[List[int]] = [
            []
            for _ in range(worker_count)
        ]

        for idx in range(self.pop_size):
            chunks[
                idx % worker_count
            ].append(idx)

        return [
            chunk
            for chunk in chunks
            if chunk
        ]

    def _ensure_process_pool(self) -> None:
        """
        Creates the persistent portable worker pool.
        """

        if self._process_pool is None:

            self._process_pool = (
                ProcessPoolExecutor(
                    max_workers=self.simulation_workers,
                    mp_context=self._mp_context
                )
            )

    def _shutdown_process_pool(self) -> None:
        """
        Safely shuts down the worker pool.
        """

        if self._process_pool is not None:

            self._process_pool.shutdown(
                wait=True
            )

            self._process_pool = None

    def _simulate_generation(
        self,
        map_data: MapData,
        initial_bfs_dist: int,
        spawn_headings: List[float]
    ) -> Tuple[
        List[PlayerState],
        List[List[Dict[str, Any]]]
    ]:
        """
        Runs one generation using multiple portable worker processes.

        The parent owns the authoritative population. Each worker receives
        a copy of its genome chunk and performs the complete simulation.
        """

        # Single-process path.
        if self.simulation_workers <= 1:

            chunks = [
                list(range(self.pop_size))
            ]

        else:

            chunks = self._make_chunks()

        # Take immutable snapshots of the current population.

        weight_chunks: List[List[np.ndarray]] = []

        bias_chunks: List[List[np.ndarray]] = []

        for chunk in chunks:

            weight_chunks.append([
                self.population.weights[layer][chunk].copy()
                for layer in range(
                    len(self.population.weights)
                )
            ])

            bias_chunks.append([
                self.population.biases[layer][chunk].copy()
                for layer in range(
                    len(self.population.biases)
                )
            ])

        # Single-process execution avoids multiprocessing overhead for
        # tiny populations or explicitly configured one-worker operation.

        if self.simulation_workers <= 1:

            states_result, frames_result = (
                _simulate_chunk_worker(
                    chunks[0],
                    weight_chunks[0],
                    bias_chunks[0],
                    map_data,
                    initial_bfs_dist,
                    spawn_headings,
                    self.max_steps
                )
            )

            candidate_states: List[PlayerState] = [
                None
                for _ in range(self.pop_size)
            ]

            candidate_frames: List[
                List[Dict[str, Any]]
            ] = [
                []
                for _ in range(self.pop_size)
            ]

            for idx, state in states_result:
                candidate_states[idx] = state

            for idx, frames in frames_result:
                candidate_frames[idx] = frames

            return (
                candidate_states,
                candidate_frames
            )

        self._ensure_process_pool()

        futures = []

        for chunk, weights, biases in zip(
            chunks,
            weight_chunks,
            bias_chunks
        ):

            futures.append(
                self._process_pool.submit(
                    _simulate_chunk_worker,
                    chunk,
                    weights,
                    biases,
                    map_data,
                    initial_bfs_dist,
                    spawn_headings,
                    self.max_steps
                )
            )

        candidate_states: List[PlayerState] = [
            None
            for _ in range(self.pop_size)
        ]

        candidate_frames: List[
            List[Dict[str, Any]]
        ] = [
            []
            for _ in range(self.pop_size)
        ]

        for future in futures:

            states_result, frames_result = (
                future.result()
            )

            for idx, state in states_result:
                candidate_states[idx] = state

            for idx, frames in frames_result:
                candidate_frames[idx] = frames

        return (
            candidate_states,
            candidate_frames
        )

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> FrameRecorder:

        print(
            "\n=== CPU NEUROEVOLUTION SIMULATION ==="
        )

        print(
            f"Population: {self.pop_size} | "
            f"Max Steps: {self.max_steps} | "
            f"Workers: {self.simulation_workers} | "
            f"Map Regime: "
            f"{config.MAP_REGIME_GENERATIONS} gens max | "
            f"Curriculum: "
            f"{config.CURRICULUM_ENABLED}\n"
        )

        header_str: str = (
            f"{'GEN':>7s} | "
            f"{'TOP':>7s} | "
            f"{'AVG':>7s} | "
            f"{'WAY':>7s} | "
            f"{'TARG':>7s} | "
            f"{'FRAME':>7s} | "
            f"{'EXITS':>7s} | TIME"
        )

        print(header_str)
        print("-" * len(header_str))

        target_bfs: int = (
            config.CURRICULUM_START_BFS
        )

        switch_next: bool = True
        new_map_this_gen: bool = True
        gens_on_map: int = 0
        consecutive_failures: int = 0

        map_data = None
        initial_bfs_dist = 0
        theoretical_max = 0.0
        spawn_headings: List[float] = []

        try:

            for gen_idx in range(
                num_generations
            ):

                gen_start: float = time.time()

                new_map_this_gen = switch_next

                if switch_next:

                    (
                        map_data,
                        _pathfinder,
                        initial_bfs_dist,
                        _num_turns,
                        theoretical_max,
                        spawn_headings
                    ) = self._prepare_map(
                        target_bfs
                    )

                    switch_next = False
                    gens_on_map = 0

                (
                    candidate_states,
                    candidate_frames
                ) = self._simulate_generation(
                    map_data,
                    initial_bfs_dist,
                    spawn_headings
                )

                raw_scores: List[float] = [
                    FitnessEvaluator.calculate_raw_score(
                        state,
                        initial_bfs_dist,
                        self.max_steps
                    )
                    for state in candidate_states
                ]

                scaled_scores: List[float] = [
                    FitnessEvaluator.calculate_scaled_score(
                        score,
                        theoretical_max
                    )
                    for score in raw_scores
                ]

                norm_scores: List[float] = (
                    FitnessEvaluator.normalize_scores(
                        raw_scores
                    )
                )

                self.recorder.record_generation(
                    gen_idx,
                    map_data,
                    candidate_frames,
                    scaled_scores,
                    norm_scores
                )

                self.population.mutation_scale = (
                    config.MUTATION_SCALE
                    * config.REGIME_TRANSITION_MUTATION_BOOST
                    if new_map_this_gen
                    else config.MUTATION_SCALE
                )

                self.population.evolve_next_generation(
                    norm_scores
                )

                self.population.mutation_scale = (
                    config.MUTATION_SCALE
                )

                gen_time: float = (
                    time.time() - gen_start
                )

                top_int: int = int(
                    round(max(scaled_scores))
                )

                avg_scaled: float = (
                    sum(scaled_scores)
                    / float(len(scaled_scores))
                )

                winner_idx: int = int(
                    np.argmax(norm_scores)
                )

                solvers: List[Tuple[int, int]] = [
                    (
                        c_idx,
                        c_state.frames_survived
                    )
                    for c_idx, c_state
                    in enumerate(candidate_states)
                    if c_state.has_reached_exit
                ]

                solve_count: int = len(solvers)

                exits_str: str = (
                    f"{solve_count}/{self.pop_size}"
                )

                if solve_count > 0:

                    fastest_step: int = min(
                        step_count
                        for _, step_count
                        in solvers
                    )

                    frame_str: str = str(
                        fastest_step
                    )

                else:

                    frame_str = "-"

                row_str: str = (
                    f"{gen_idx + 1:>7d} | "
                    f"{top_int:>7d} | "
                    f"{avg_scaled:>7.1f} | "
                    f"{initial_bfs_dist:>7d} | "
                    f"{target_bfs:>7d} | "
                    f"{frame_str:>7s} | "
                    f"{exits_str:>7s} | "
                    f"{gen_time:>5.2f}s"
                )

                print(row_str)

                gens_on_map += 1

                solved: bool = (
                    solve_count
                    >= config.REGIME_SOLVE_TARGET
                )

                if (
                    solved
                    and gens_on_map
                    >= config.REGIME_MIN_GENERATIONS
                ):

                    switch_next = True

                    target_bfs = min(
                        target_bfs
                        + config.CURRICULUM_BFS_STEP,
                        config.CURRICULUM_MAX_BFS
                    )

                    consecutive_failures = 0

                elif (
                    gens_on_map
                    >= config.MAP_REGIME_GENERATIONS
                ):

                    switch_next = True

                    if solved:

                        target_bfs = min(
                            target_bfs
                            + config.CURRICULUM_BFS_STEP,
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
                                target_bfs
                                - config.CURRICULUM_BFS_STEP,
                                config.CURRICULUM_START_BFS
                            )

                            consecutive_failures = 0

        finally:

            self._shutdown_process_pool()

        print("-" * len(header_str))

        print(
            "CPU Training complete! "
            "Booting interactive visualizer GUI...\n"
        )

        return self.recorder
