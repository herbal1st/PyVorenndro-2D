"""
Headless single-agent CPU trainer.

One champion genome controls every parallel simulation run of a generation.
Each run is an independently seeded, independently generated maze; the trainer
corroborates all run data back into the single agent (aggregate fitness) and
applies (1+1)-elitist evolution: the next generation mutates the champion and
replaces it only when its corroborated fitness is at least as good.

Run simulation is batched/vectorized across the generation's runs and further
split across portable spawn worker processes. The trainer publishes the latest
champion model state into a shared manager dict so a separate display runner
process can pull it and re-simulate the agent in real time without ever
slowing down training.
"""

import time
import random
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

import config
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from core.kinematics import CandidateKinematics
from perception.spatial_transformer import SpatialTransformer
from entities.player_state import PlayerState
from training.agent import Agent
from training.fitness import FitnessEvaluator
from training.metrics import MetricsRecorder


def _simulate_runs_worker(
    global_indices: List[int],
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    maps: List[MapData],
    dist_grids: List[np.ndarray],
    initial_bfs_dists: List[int],
    spawn_headings: List[float],
    max_steps: int
) -> List[Tuple[int, PlayerState]]:
    """
    Simulates a chunk of independently-seeded runs for the shared agent genome.

    Every run owns its own map, so perception and collision use per-row wall
    grids stacked into (N, H, W) arrays while the neural forward pass reuses
    the single agent's weight matrices for the whole chunk.
    """
    n: int = len(global_indices)

    wall_grids: np.ndarray = np.stack(
        [m.build_wall_grid() for m in maps]
    )
    dist_grid: np.ndarray = np.stack(
        [np.asarray(d, dtype=np.int64) for d in dist_grids]
    )
    exit_pos: np.ndarray = np.array(
        [m.exit_pos for m in maps], dtype=np.int64
    )

    h, w = wall_grids.shape[1], wall_grids.shape[2]

    candidate_states: List[PlayerState] = [
        PlayerState(
            float(m.start_pos[0]) + 0.5,
            float(m.start_pos[1]) + 0.5
        )
        for m in maps
    ]

    for i in range(n):
        candidate_states[i].heading = spawn_headings[i]
        candidate_states[i].best_step_dist = initial_bfs_dists[i]

    xs: np.ndarray = np.empty(n, dtype=np.float64)
    ys: np.ndarray = np.empty(n, dtype=np.float64)
    headings: np.ndarray = np.empty(n, dtype=np.float64)
    healths: np.ndarray = np.ones(n, dtype=np.float64)

    for i in range(n):
        xs[i] = candidate_states[i].x
        ys[i] = candidate_states[i].y
        headings[i] = candidate_states[i].heading
    best_dists: np.ndarray = np.asarray(
        initial_bfs_dists, dtype=np.float64
    )
    frames_survived: np.ndarray = np.zeros(n, dtype=np.int64)
    alive: np.ndarray = np.ones(n, dtype=np.bool_)
    reached: np.ndarray = np.zeros(n, dtype=np.bool_)
    last_hit: np.ndarray = np.zeros(n, dtype=np.bool_)

    move_speed: float = config.MOVE_SPEED

    kinematics: CandidateKinematics = CandidateKinematics()
    transformer: SpatialTransformer = SpatialTransformer()

    for step in range(1, max_steps + 1):
        active: np.ndarray = np.flatnonzero(alive & ~reached)
        if active.size == 0:
            break

        axs: np.ndarray = xs[active]
        ays: np.ndarray = ys[active]
        ahs: np.ndarray = headings[active]
        ahp: np.ndarray = healths[active]

        if config.INCLUDE_BFS_SENSOR:
            sx: np.ndarray = np.floor(axs).astype(np.int64)
            sy: np.ndarray = np.floor(ays).astype(np.int64)
            in_s: np.ndarray = (
                (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            )
            sxc: np.ndarray = np.clip(sx, 0, w - 1)
            syc: np.ndarray = np.clip(sy, 0, h - 1)
            sensor_dist: np.ndarray = np.where(
                in_s,
                dist_grid[active, syc, sxc],
                9999
            ).astype(np.float64)
        else:
            sensor_dist = None

        features = transformer.compile_feature_batch(
            axs,
            ays,
            ahs,
            np.full(active.size, move_speed, dtype=np.float64),
            ahp,
            wall_grids=wall_grids[active],
            exit_positions=exit_pos[active],
            current_dists=sensor_dist
        )

        outputs, _ = Agent.forward_batch(
            weights, biases, features
        )

        move_eff: np.ndarray = outputs[:, 0]
        turn_eff: np.ndarray = outputs[:, 1]

        px, py, new_heading, hit, is_stationary = (
            kinematics.step_batch(
                axs,
                ays,
                ahs,
                move_eff,
                turn_eff,
                wall_grids=wall_grids[active]
            )
        )

        hlt: np.ndarray = ahp
        hlt = hlt - hit * config.HEALTH_COLL_DMG_PER_FRAME

        not_moved: np.ndarray = (
            (np.abs(px - axs) < 1e-4)
            & (np.abs(py - ays) < 1e-4)
        )
        idle: np.ndarray = (
            (move_eff < 0.05)
            | not_moved
            | is_stationary
        )
        hlt = hlt - idle * config.HEALTH_IDLE_DMG_PER_FRAME
        hlt = np.maximum(hlt, 0.0)

        tx: np.ndarray = np.floor(px).astype(np.int64)
        ty: np.ndarray = np.floor(py).astype(np.int64)

        inb: np.ndarray = (
            (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        )
        txc: np.ndarray = np.clip(tx, 0, w - 1)
        tyc: np.ndarray = np.clip(ty, 0, h - 1)

        curr_dist: np.ndarray = np.where(
            inb,
            dist_grid[active, tyc, txc],
            9999
        )

        better: np.ndarray = (
            curr_dist < best_dists[active]
        )
        heal: np.ndarray = (
            (best_dists[active] - curr_dist)
            * config.HEALTH_COLL_DMG_PER_FRAME
            * config.HEALTH_RECOVERY_RATIO
        )
        hlt = np.where(
            better,
            np.minimum(1.0, hlt + heal),
            hlt
        )
        best_dists[active] = np.where(
            better,
            curr_dist,
            best_dists[active]
        )

        ex: np.ndarray = exit_pos[active, 0]
        ey: np.ndarray = exit_pos[active, 1]
        reached_run: np.ndarray = inb & (tx == ex) & (ty == ey)

        xs[active] = px
        ys[active] = py
        headings[active] = new_heading
        healths[active] = hlt
        alive[active] = hlt > 0.0
        reached[active] |= reached_run
        last_hit[active] = hit
        frames_survived[active] += 1

    results: List[Tuple[int, PlayerState]] = []
    for i in range(n):
        state: PlayerState = candidate_states[i]
        state.x = xs[i]
        state.y = ys[i]
        state.heading = headings[i]
        state.health = healths[i]
        state.has_collided = bool(last_hit[i])
        state.is_alive = bool(alive[i])
        state.has_reached_exit = bool(reached[i])
        state.best_step_dist = int(best_dists[i])
        state.frames_survived = int(frames_survived[i])
        results.append((global_indices[i], state))

    return results


def run_training_process(
    state: Dict[str, Any],
    stop_event: Any
) -> None:
    """
    Spawn-process entry point: publishes initial state, trains, and keeps the
    final champion available in shared state until told to stop. Picks the GPU
    backend automatically when CUDA is available (unless overridden).
    """
    backend: str = str(
        getattr(config, "TRAINING_BACKEND", "auto")
    ).lower()

    use_gpu: bool = backend == "gpu"

    if backend == "auto":
        try:
            from training.trainer_gpu import torch_available
            use_gpu = torch_available()
        except Exception:
            use_gpu = False

    if use_gpu:
        from training.trainer_gpu import GpuHeadlessTrainer
        trainer: HeadlessTrainer = GpuHeadlessTrainer(
            state=state,
            stop_event=stop_event
        )
    else:
        trainer = HeadlessTrainer(
            state=state,
            stop_event=stop_event
        )

    trainer.publish_state()
    trainer.run_training_session()

    while not stop_event.is_set():
        stop_event.wait(1.0)


class HeadlessTrainer:
    """
    Runs single-agent neuroevolution over parallel seeded simulation runs.
    """

    def __init__(
        self,
        num_runs: int = 0,
        max_steps: int = config.MAX_SIMULATION_STEPS,
        state: Optional[Dict[str, Any]] = None,
        stop_event: Any = None,
        auto_tune: bool = True
    ) -> None:
        """
        Initializes one champion agent and the parallel simulation pool.
        """
        if num_runs <= 0:
            num_runs = int(getattr(config, "SIMULATION_RUNS", 100))
            self._explicit_num_runs = False
        else:
            self._explicit_num_runs = True

        self.num_runs: int = max(1, int(num_runs))
        self.max_steps: int = max_steps
        self.population_size: int = max(
            2,
            int(getattr(config, "POPULATION_SIZE", 2))
        )
        self.shared_runs: int = max(
            1,
            self.num_runs // self.population_size
        )
        self._state: Optional[Dict[str, Any]] = state
        self._stop_event: Any = stop_event

        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer()
        self.kinematics: CandidateKinematics = CandidateKinematics()
        self.recorder: MetricsRecorder = MetricsRecorder()

        sample_map = (
            self.map_generator.generate_solvable_map()
        )

        self._map_height: int = int(sample_map.height)
        self._map_width: int = int(sample_map.width)
        self._backend_label: str = "CPU"

        sample_dist: Optional[float] = None
        if config.INCLUDE_BFS_SENSOR:
            sample_pathfinder: BFSPathfinder = BFSPathfinder(
                sample_map
            )
            sample_pathfinder.compute_distance_matrix()
            sample_dist = float(
                sample_pathfinder.get_step_distance(
                    *sample_map.start_pos
                )
            )

        sample_features = (
            self.transformer.compile_feature_vector(
                0.5,
                0.5,
                0.0,
                1.0,
                1.0,
                sample_map,
                current_dist=sample_dist
            )
        )

        self.input_size: int = len(sample_features)
        self.output_size: int = 2

        layer_sizes: List[int] = [
            config.NEURAL_NEURONS
            for _ in range(config.NEURAL_HIDDEN_LAYERS)
        ]

        self.agent: Agent = Agent(
            [self.input_size] + layer_sizes + [self.output_size]
        )

        self.best_fitness: float = 0.0
        self.gen_fitness: float = 0.0
        self.solve_count: int = 0
        self._current_gen: int = 0
        self.training_complete: bool = False

        self.mutation_scale: float = float(
            config.MUTATION_SCALE
        )
        self.mutation_scale_min: float = float(
            config.MUTATION_SCALE_MIN
        )
        self.mutation_scale_max: float = float(
            config.MUTATION_SCALE_MAX
        )
        self._accept_window: int = max(
            1,
            int(getattr(config, "MUTATION_ADAPT_WINDOW", 15))
        )
        self._stagnation_gens: int = max(
            1,
            int(getattr(config, "STAGNATION_BUMP_GENERATIONS", 20))
        )
        self._stagnation_factor: float = float(
            getattr(config, "STAGNATION_BUMP_FACTOR", 4.0)
        )
        self._accept_history: deque = deque(
            maxlen=self._accept_window
        )
        self._gens_since_accept: int = 0

        configured_workers: int = getattr(
            config,
            "SIMULATION_WORKERS",
            0
        )

        if configured_workers <= 0:
            cpu_count: int = mp.cpu_count() or 1

            # Leave one logical CPU available for the display runner.
            if cpu_count > 2:
                configured_workers = cpu_count - 1
            else:
                configured_workers = 1

        self.simulation_workers: int = max(
            1,
            min(
                int(configured_workers),
                self.num_runs
            )
        )

        self._process_pool: ProcessPoolExecutor = None

        self._mp_context = mp.get_context("spawn")

        if auto_tune:
            from training import autotune

            autotune.apply_auto_tuning(self)

    def publish_state(self) -> None:
        """
        Writes the current champion genome and training metrics into the
        shared manager dict consumed by the display runner.
        """
        if self._state is None:
            return

        weights, biases = self.agent.to_state()

        self._state["input_size"] = int(self.input_size)
        self._state["output_size"] = int(self.output_size)
        self._state["layer_sizes"] = list(self.agent.sizes)
        self._state["weights"] = weights
        self._state["biases"] = biases
        self._state["generation"] = int(self._current_gen)
        self._state["num_generations"] = int(
            config.LEARNING_GENERATIONS
        )
        self._state["num_runs"] = int(self.shared_runs)
        self._state["max_steps"] = int(self.max_steps)
        self._state["best_fitness"] = float(self.best_fitness)
        self._state["gen_fitness"] = float(self.gen_fitness)
        self._state["solve_count"] = int(self.solve_count)
        self._state["training_complete"] = bool(
            self.training_complete
        )
        self._state["gen_history"] = list(
            self.recorder.generations_history
        )
        self._state["initialized"] = True

    def _generate_run_map(self) -> Tuple[
        MapData,
        np.ndarray,
        int,
        int,
        float,
        float
    ]:
        """
        Builds one independently-seeded run: a fresh solvable maze with a
        randomized difficulty, its BFS distance grid, and a safe spawn heading.
        """
        if (
            config.MAP_DIFFICULTY_MIN
            >= config.MAP_DIFFICULTY_MAX
        ):
            difficulty: float = config.MAP_DIFFICULTY_MIN
        else:
            difficulty = random.uniform(
                config.MAP_DIFFICULTY_MIN,
                config.MAP_DIFFICULTY_MAX
            )

        map_data: MapData = (
            self.map_generator.generate_solvable_map(
                difficulty_ratio=difficulty
            )
        )

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
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

        spawn_heading: float = (
            self.transformer.generate_random_heading(
                map_data,
                map_data.start_pos
            )
        )

        dist_grid: np.ndarray = np.asarray(
            pathfinder.distance_matrix,
            dtype=np.int64
        )

        return (
            map_data,
            dist_grid,
            initial_bfs_dist,
            num_turns,
            theoretical_max,
            spawn_heading
        )

    def _generate_run_maps(self, count: int) -> List[Tuple]:
        """
        Builds ``count`` fresh run tuples (serial on the main thread). GPU
        subclasses override this to parallelize across cores.
        """
        return [self._generate_run_map() for _ in range(count)]

    def _make_chunks(self, num_runs: int = 0) -> List[List[int]]:
        """
        Splits the shared map set into approximately equal worker chunks.
        """
        num_runs = num_runs if num_runs > 0 else self.shared_runs

        worker_count: int = min(
            self.simulation_workers,
            num_runs
        )

        chunks: List[List[int]] = [
            []
            for _ in range(worker_count)
        ]

        for idx in range(num_runs):
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

    def _simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """
        Runs every candidate genome across the SAME shared map set so their
        fitnesses are directly comparable (paired evaluation). Returns one
        (fitness, scaled_scores, candidate_states) tuple per candidate.
        """
        num_runs: int = len(runs)

        if self.simulation_workers <= 1:
            chunks: List[List[int]] = [
                list(range(num_runs))
            ]
        else:
            chunks = self._make_chunks(num_runs)

        all_results: List[Tuple[float, List[float], List[PlayerState]]] = []

        for agent in agents:
            weight_snap: List[np.ndarray] = list(agent.weights)
            bias_snap: List[np.ndarray] = list(agent.biases)

            if self.simulation_workers <= 1:
                chunk = chunks[0]
                states_result = _simulate_runs_worker(
                    chunk,
                    weight_snap,
                    bias_snap,
                    [runs[i][0] for i in chunk],
                    [runs[i][1] for i in chunk],
                    [runs[i][2] for i in chunk],
                    [runs[i][5] for i in chunk],
                    self.max_steps
                )
                results: List[Tuple[int, PlayerState]] = states_result
            else:
                self._ensure_process_pool()

                futures = []
                for chunk in chunks:
                    futures.append(
                        self._process_pool.submit(
                            _simulate_runs_worker,
                            chunk,
                            weight_snap,
                            bias_snap,
                            [runs[i][0] for i in chunk],
                            [runs[i][1] for i in chunk],
                            [runs[i][2] for i in chunk],
                            [runs[i][5] for i in chunk],
                            self.max_steps
                        )
                    )

                results = []
                for future in futures:
                    results.extend(future.result())

            candidate_states: List[PlayerState] = [
                None
                for _ in range(num_runs)
            ]

            for idx, state in results:
                candidate_states[idx] = state

            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    candidate_states[i],
                    runs[i][2],
                    self.max_steps
                )
                for i in range(num_runs)
            ]

            scaled_scores: List[float] = [
                FitnessEvaluator.calculate_scaled_score(
                    raw_scores[i],
                    runs[i][4]
                )
                for i in range(num_runs)
            ]

            fitness: float = float(
                np.mean(scaled_scores)
            ) if scaled_scores else 0.0

            all_results.append(
                (fitness, scaled_scores, candidate_states)
            )

        return all_results

    def _build_candidates(self) -> List[Agent]:
        """
        Returns the champion plus mutated offspring at diverse scales.
        """
        candidates: List[Agent] = [self.agent]

        scale_multipliers: List[float] = [
            1.0, 2.0, 0.5, 3.0, 0.33, 4.0, 0.25
        ]

        for idx in range(1, self.population_size):
            multiplier: float = (
                scale_multipliers[idx % len(scale_multipliers)]
            )

            scale: float = min(
                self.mutation_scale_max,
                max(
                    self.mutation_scale_min,
                    self.mutation_scale * multiplier
                )
            )

            candidates.append(
                self.agent.mutate(
                    mutation_scale=scale
                )
            )

        return candidates

    def _update_mutation_scale(self, accepted: bool) -> None:
        """
        Applies the 1/5 success rule and stagnation bumps to the mutation
        scale so the search self-tunes between exploration and exploitation.
        """
        self._accept_history.append(accepted)
        self._gens_since_accept = (
            0 if accepted else self._gens_since_accept + 1
        )

        if len(self._accept_history) == self._accept_window:
            success_ratio: float = (
                sum(self._accept_history) / self._accept_window
            )

            if success_ratio > 0.2:
                self.mutation_scale = min(
                    self.mutation_scale_max,
                    self.mutation_scale * 1.1
                )
            elif success_ratio < 0.2:
                self.mutation_scale = max(
                    self.mutation_scale_min,
                    self.mutation_scale * 0.9
                )

            self._accept_history.clear()

        if self._gens_since_accept >= self._stagnation_gens:
            self.mutation_scale = min(
                self.mutation_scale_max,
                self.mutation_scale * self._stagnation_factor
            )
            self._gens_since_accept = 0

    def _adapt_run_budget(self, gen_time: float) -> None:
        """
        Runtime self-correction for auto-tuning: if the previous generation ran
        faster or slower than the hardware target, rescale the run budget so
        subsequent generations land back on target on this specific machine.
        Only kicks in when the deviation is significant.
        """
        target: float = getattr(config, "TARGET_GEN_TIME", 0.0)
        if target <= 0.0 or gen_time <= 0.0:
            return

        backend: str = str(getattr(self, "_backend_label", "CPU"))
        if backend == "GPU":
            return

        ratio: float = gen_time / target
        if 0.6 <= ratio <= 1.5:
            return

        new_runs: int = int(self.num_runs * (target / gen_time))
        new_runs = int(
            min(
                getattr(config, "AUTO_TUNE_MAX_RUNS", 8192),
                max(
                    getattr(config, "AUTO_TUNE_MIN_RUNS", 64),
                    new_runs
                ),
            )
        )

        if new_runs == self.num_runs:
            return

        self.num_runs = new_runs
        self.shared_runs = max(
            1, self.num_runs // self.population_size
        )
        config.SIMULATION_RUNS = self.num_runs
        config.SIMULATION_RUNS_GPU = self.num_runs

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> MetricsRecorder:
        """
        Runs the full single-agent evolution loop and publishes each
        generation's champion to the shared state.
        """
        print(
            "\n=== SINGLE-AGENT CPU NEUROEVOLUTION SIMULATION ==="
        )

        print(
            f"Champion: 1 | Population: {self.population_size} | "
            f"Maps/Gen: {self.shared_runs} | "
            f"Sims/Gen: {self.shared_runs * self.population_size} | "
            f"Max Steps: {self.max_steps} | "
            f"Backend: {self._backend_label}\n"
        )

        header_str: str = (
            f"{'GEN':>7s} | "
            f"{'BEST':>7s} | "
            f"{'FIT':>7s} | "
            f"{'WAY':>7s} | "
            f"{'FRAME':>7s} | "
            f"{'EXITS':>7s} | "
            f"{'SCALE':>7s} | TIME"
        )

        print(header_str)
        print("-" * len(header_str))

        try:
            for gen_idx in range(
                int(num_generations)
            ):
                if (
                    self._stop_event is not None
                    and self._stop_event.is_set()
                ):
                    break

                gen_start: float = time.time()
                self._current_gen = gen_idx

                runs: List[Tuple] = self._generate_run_maps(
                    self.shared_runs
                )

                candidates: List[Agent] = self._build_candidates()

                candidate_results: List[
                    Tuple[float, List[float], List[PlayerState]]
                ] = self._simulate_candidates(candidates, runs)

                best_idx: int = max(
                    range(len(candidate_results)),
                    key=lambda i: candidate_results[i][0]
                )

                fitness: float = candidate_results[best_idx][0]
                candidate_states: List[PlayerState] = (
                    candidate_results[best_idx][2]
                )

                accepted: bool = best_idx > 0
                self._update_mutation_scale(accepted)

                if accepted:
                    self.agent = candidates[best_idx].copy()

                self.gen_fitness = fitness
                self.solve_count = sum(
                    1
                    for s in candidate_states
                    if s.has_reached_exit
                )

                if fitness >= self.best_fitness:
                    self.best_fitness = fitness

                avg_bfs: float = (
                    sum(runs[i][2] for i in range(self.shared_runs))
                    / float(self.shared_runs)
                )

                self.recorder.record_generation(
                    gen_idx,
                    self.best_fitness,
                    self.gen_fitness,
                    self.solve_count,
                    self.shared_runs,
                    avg_bfs
                )

                self.publish_state()

                solvers: List[int] = [
                    c_state.frames_survived
                    for c_state in candidate_states
                    if c_state.has_reached_exit
                ]

                if solvers:
                    frame_str: str = str(min(solvers))
                else:
                    frame_str = "-"

                gen_time: float = time.time() - gen_start

                if getattr(config, "AUTO_TUNE", True):
                    self._adapt_run_budget(gen_time)

                row_str: str = (
                    f"{gen_idx + 1:>7d} | "
                    f"{int(round(self.best_fitness)):>7d} | "
                    f"{self.gen_fitness:>7.1f} | "
                    f"{avg_bfs:>7.0f} | "
                    f"{frame_str:>7s} | "
                    f"{f'{self.solve_count}/{self.shared_runs}':>7s} | "
                    f"{self.mutation_scale:>7.3f} | "
                    f"{gen_time:>5.2f}s"
                )

                print(row_str)
        finally:
            self._shutdown_process_pool()

        self.training_complete = True
        self.publish_state()

        print("-" * len(header_str))

        print(
            "Single-agent training complete! "
            "Display runner keeps replaying the final champion "
            "until Ctrl+C...\n"
        )

        return self.recorder
