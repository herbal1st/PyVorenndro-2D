"""
Headless single-agent CPU trainer.

Evaluates candidates across parallel procedural mazes using a unified
NumPy batch matrix pass across all candidates and runs simultaneously.
"""

from collections import deque
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from core.kinematics import CandidateKinematics
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from entities.player_state import PlayerState
from perception.spatial_transformer import SpatialTransformer
from storage.persistence import BrainLibraryRegistry, BrainWeightHandler
from training.agent import Agent
from training.fitness import FitnessEvaluator
from training.metrics import MetricsRecorder


def run_training_process(
    state: Dict[str, Any],
    stop_event: Any
) -> None:
    """
    Spawn-process entry point: trains and updates shared state.
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
    Runs neuroevolution over parallel seeded mazes via vectorized NumPy.
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
        Initializes registry, active profile, champion agent, & persistence.
        """
        self.registry: BrainLibraryRegistry = BrainLibraryRegistry()
        self.profile_id: str = str(
            getattr(config, "ACTIVE_PROFILE_ID", "herbal1st")
        )
        self.profile_config: Dict[str, Any] = self.registry.get_profile(
            self.profile_id
        )

        self.population_size: int = max(
            2, int(getattr(config, "POPULATION_SIZE", 16))
        )
        self.maps_per_candidate: int = max(
            1, int(getattr(config, "MAPS_PER_CANDIDATE", 8))
        )

        if num_runs > 0:
            self.shared_runs: int = max(1, num_runs)
            self._explicit_num_runs: bool = True
        else:
            self.shared_runs = self.maps_per_candidate
            self._explicit_num_runs = False

        self.num_runs: int = self.population_size * self.shared_runs
        self.max_steps: int = max_steps
        self._state: Optional[Dict[str, Any]] = state
        self._stop_event: Any = stop_event

        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer(
            profile_config=self.profile_config
        )
        self.kinematics: CandidateKinematics = CandidateKinematics(
            profile_config=self.profile_config
        )
        self.recorder: MetricsRecorder = MetricsRecorder()

        sample_map: MapData = self.map_generator.generate_solvable_map()
        self._map_height: int = int(sample_map.height)
        self._map_width: int = int(sample_map.width)
        self._backend_label: str = "CPU"

        sensory: Dict[str, Any] = self.profile_config["sensory"]
        include_bfs: bool = bool(sensory["include_bfs_sensor"])

        sample_dist: Optional[float] = None
        if include_bfs:
            sample_pathfinder: BFSPathfinder = BFSPathfinder(sample_map)
            sample_pathfinder.compute_distance_matrix()
            sample_dist = float(
                sample_pathfinder.get_step_distance(
                    *sample_map.start_pos
                )
            )

        move_speed: float = float(
            self.profile_config["kinematics"]["move_speed"]
        )
        sample_features: np.ndarray = (
            self.transformer.compile_feature_vector(
                0.5,
                0.5,
                0.0,
                move_speed,
                1.0,
                sample_map,
                current_dist=sample_dist,
                profile_config=self.profile_config
            )
        )

        self.input_size: int = len(sample_features)
        self.output_size: int = 2

        self.agent: Agent = Agent.from_profile(
            self.profile_config, self.input_size, self.output_size
        )

        self.weight_handler: BrainWeightHandler = BrainWeightHandler()
        reloaded: bool = self.weight_handler.load_champion(
            self.agent, self.profile_config
        )
        if reloaded:
            print(
                f"[Trainer] Reloaded champion weights for profile "
                f"'{self.profile_id}' from disk."
            )

        self.best_fitness: float = 0.0
        self.gen_fitness: float = 0.0
        self.solve_count: int = 0
        self._current_gen: int = 0
        self.training_complete: bool = False

        self.mutation_scale: float = float(config.MUTATION_SCALE)
        self.mutation_scale_min: float = float(config.MUTATION_SCALE_MIN)
        self.mutation_scale_max: float = float(config.MUTATION_SCALE_MAX)
        self._accept_window: int = max(
            1, int(getattr(config, "MUTATION_ADAPT_WINDOW", 8))
        )
        self._stagnation_gens: int = max(
            1, int(getattr(config, "STAGNATION_BUMP_GENERATIONS", 8))
        )
        self._stagnation_factor: float = float(
            getattr(config, "STAGNATION_BUMP_FACTOR", 4.0)
        )
        self._accept_history: deque = deque(maxlen=self._accept_window)
        self._gens_since_accept: int = 0
        self.simulation_workers: int = 1

        if auto_tune:
            from training import autotune
            autotune.apply_auto_tuning(self)

    def publish_state(self) -> None:
        """
        Publishes champion genome and telemetry into shared memory.
        """
        if self._state is None:
            return

        weights, biases = self.agent.to_state()

        self._state["profile_id"] = str(self.profile_id)
        self._state["input_size"] = int(self.input_size)
        self._state["output_size"] = int(self.output_size)
        self._state["layer_sizes"] = list(self.agent.sizes)
        self._state["weights"] = weights
        self._state["biases"] = biases
        self._state["generation"] = int(self._current_gen)
        self._state["num_generations"] = int(config.LEARNING_GENERATIONS)
        self._state["num_runs"] = int(self.shared_runs)
        self._state["max_steps"] = int(self.max_steps)
        self._state["best_fitness"] = float(self.best_fitness)
        self._state["gen_fitness"] = float(self.gen_fitness)
        self._state["solve_count"] = int(self.solve_count)
        self._state["training_complete"] = bool(self.training_complete)
        self._state["gen_history"] = list(
            self.recorder.generations_history
        )
        self._state["initialized"] = True

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> MetricsRecorder:
        """
        Runs neuroevolution and auto-saves champion weights to disk.
        """
        print("\n=== SINGLE-AGENT CPU NEUROEVOLUTION SIMULATION ===")
        print(
            f"Active Profile: '{self.profile_id}' | "
            f"Population: {self.population_size} | "
            f"Maps/Candidate: {self.shared_runs} | "
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

        for gen_idx in range(int(num_generations)):
            if self._stop_event is not None and self._stop_event.is_set():
                break

            gen_start: float = time.time()
            self._current_gen = gen_idx

            runs: List[Tuple] = self._generate_run_maps(self.shared_runs)
            candidates: List[Agent] = self._build_candidates()

            candidate_results = self._simulate_candidates(candidates, runs)

            best_idx: int = max(
                range(len(candidate_results)),
                key=lambda i: candidate_results[i][0]
            )

            gen_top_score: float = candidate_results[best_idx][0]
            candidate_states: List[PlayerState] = (
                candidate_results[best_idx][2]
            )

            accepted: bool = best_idx > 0
            self._update_mutation_scale(accepted)

            if accepted:
                self.agent = candidates[best_idx].copy()

            self.gen_fitness = gen_top_score
            self.solve_count = sum(
                1 for s in candidate_states if s.has_reached_exit
            )

            if gen_top_score > self.best_fitness:
                self.best_fitness = gen_top_score
                self.weight_handler.save_champion(
                    self.agent, self.profile_config
                )

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

            frame_str: str = str(min(solvers)) if solvers else "-"
            gen_time: float = time.time() - gen_start

            row_str: str = (
                f"{gen_idx + 1:>7d} | "
                f"{int(round(gen_top_score)):>7d} | "
                f"{self.gen_fitness:>7.1f} | "
                f"{avg_bfs:>7.0f} | "
                f"{frame_str:>7s} | "
                f"{f'{self.solve_count}/{self.shared_runs}':>7s} | "
                f"{self.mutation_scale:>7.3f} | "
                f"{gen_time:>5.2f}s"
            )

            print(row_str)

        self.training_complete = True
        self.weight_handler.save_champion(self.agent, self.profile_config)
        self.publish_state()

        print("-" * len(header_str))
        print(
            "Single-agent training complete! Champion saved to disk. "
            "Display runner keeps replaying champion until Ctrl+C...\n"
        )

        return self.recorder

    def _generate_run_map(self) -> Tuple[
        MapData, np.ndarray, int, int, float, float
    ]:
        """
        Builds one solvable maze, distance grid, and random spawn heading.
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

    def _generate_run_maps(self, count: int) -> List[Tuple]:
        """
        Generates a list of count independently-seeded maze run tuples.
        """
        return [self._generate_run_map() for _ in range(count)]

    def _simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """
        Simulates all candidate genomes across shared mazes in NumPy batch ops.
        """
        cand_count: int = len(agents)
        run_count: int = len(runs)
        total_instances: int = cand_count * run_count

        metabolics: Dict[str, Any] = self.profile_config["metabolics"]
        kinematics_cfg: Dict[str, Any] = self.profile_config["kinematics"]
        sensory: Dict[str, Any] = self.profile_config["sensory"]

        coll_dmg: float = float(metabolics["collision_damage"])
        idle_dmg: float = float(metabolics["idle_damage"])
        rec_ratio: float = float(metabolics["recovery_ratio"])
        move_speed: float = float(kinematics_cfg["move_speed"])
        include_bfs: bool = bool(sensory["include_bfs_sensor"])

        spin_enabled: bool = bool(
            metabolics.get("spin_penalty_enabled", True)
        )
        spin_thresh_rad: float = math.radians(
            float(metabolics.get("spin_angle_threshold_deg", 360.0))
        )
        spin_reset_rad: float = math.radians(
            float(metabolics.get("spin_reset_angle_deg", 5.0))
        )
        spin_hold_frames: int = int(
            metabolics.get("spin_reset_hold_frames", 15)
        )
        spin_dmg: float = float(
            metabolics.get("spin_damage_per_frame", 0.003)
        )

        stag_enabled: bool = bool(
            metabolics.get("stagnation_enabled", True)
        )
        stag_limit: int = int(
            self.max_steps * float(
                metabolics.get("stagnation_timeout_ratio", 0.75)
            )
        )
        stag_dmg: float = float(
            metabolics.get("stagnation_damage_per_frame", 0.001)
        )

        # Build combined tensor stacks across all cand_count x run_count
        wall_grids: np.ndarray = np.stack(
            [runs[i % run_count][0].build_wall_grid()
             for i in range(total_instances)]
        )
        dist_grids: np.ndarray = np.stack(
            [np.asarray(runs[i % run_count][1], dtype=np.int64)
             for i in range(total_instances)]
        )
        exit_positions: np.ndarray = np.array(
            [runs[i % run_count][0].exit_pos for i in range(total_instances)],
            dtype=np.int64
        )
        initial_bfs_dists: np.ndarray = np.array(
            [runs[i % run_count][2] for i in range(total_instances)],
            dtype=np.int64
        )
        spawn_headings: np.ndarray = np.array(
            [runs[i % run_count][5] for i in range(total_instances)],
            dtype=np.float64
        )

        h, w = wall_grids.shape[1], wall_grids.shape[2]

        # Stack weights and biases per layer for batch einsum
        num_layers: int = len(agents[0].weights)
        w_stacks: List[np.ndarray] = []
        b_stacks: List[np.ndarray] = []

        for l_idx in range(num_layers):
            w_layer: np.ndarray = np.stack(
                [agents[i // run_count].weights[l_idx]
                 for i in range(total_instances)]
            )
            b_layer: np.ndarray = np.stack(
                [agents[i // run_count].biases[l_idx]
                 for i in range(total_instances)]
            )
            w_stacks.append(w_layer)
            b_stacks.append(b_layer)

        # Instance state trackers (1D vectors)
        xs: np.ndarray = np.array(
            [float(runs[i % run_count][0].start_pos[0]) + 0.5
             for i in range(total_instances)],
            dtype=np.float64
        )
        ys: np.ndarray = np.array(
            [float(runs[i % run_count][0].start_pos[1]) + 0.5
             for i in range(total_instances)],
            dtype=np.float64
        )
        headings: np.ndarray = spawn_headings.copy()
        healths: np.ndarray = np.ones(total_instances, dtype=np.float64)
        best_dists: np.ndarray = initial_bfs_dists.astype(np.float64)

        cum_rotation: np.ndarray = np.zeros(
            total_instances, dtype=np.float64
        )
        straight_ticks: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )
        stagnation_ticks: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )

        last_hits: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        is_idles: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        is_spinnings: np.ndarray = np.zeros(total_instances, dtype=np.bool_)

        alive: np.ndarray = np.ones(total_instances, dtype=np.bool_)
        reached: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        frames_survived: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )

        hidden_layers: int = num_layers - 1

        for step in range(1, self.max_steps + 1):
            active_mask: np.ndarray = alive & ~reached
            active_idx: np.ndarray = np.flatnonzero(active_mask)
            if active_idx.size == 0:
                break

            axs: np.ndarray = xs[active_idx]
            ays: np.ndarray = ys[active_idx]
            ahs: np.ndarray = headings[active_idx]
            ahp: np.ndarray = healths[active_idx]

            if include_bfs:
                sx: np.ndarray = np.floor(axs).astype(np.int64)
                sy: np.ndarray = np.floor(ays).astype(np.int64)
                in_s: np.ndarray = (
                    (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                )
                sxc: np.ndarray = np.clip(sx, 0, w - 1)
                syc: np.ndarray = np.clip(sy, 0, h - 1)
                sensor_dist: Optional[np.ndarray] = np.where(
                    in_s, dist_grids[active_idx, syc, sxc], 9999
                ).astype(np.float64)
            else:
                sensor_dist = None

            features = self.transformer.compile_feature_batch(
                axs,
                ays,
                ahs,
                np.full(active_idx.size, move_speed, dtype=np.float64),
                ahp,
                wall_grids=wall_grids[active_idx],
                exit_positions=exit_positions[active_idx],
                current_dists=sensor_dist,
                last_hits=last_hits[active_idx],
                is_idles=is_idles[active_idx],
                is_spinnings=is_spinnings[active_idx],
                profile_config=self.profile_config
            )

            x_mat: np.ndarray = features.astype(np.float64, copy=False)
            for l_idx, (w_st, b_st) in enumerate(zip(w_stacks, b_stacks)):
                x_mat = np.einsum(
                    "ni,nij->nj",
                    x_mat,
                    w_st[active_idx],
                    optimize=False
                ) + b_st[active_idx].squeeze(1)

                if l_idx < hidden_layers:
                    x_mat = np.maximum(0.0, x_mat)

            # FLATTEN TO 1D VECTORS using [:, 0] and [:, 1]
            move_eff: np.ndarray = 1.0 / (
                1.0 + np.exp(-np.clip(x_mat[:, 0], -500.0, 500.0))
            )
            turn_eff: np.ndarray = np.tanh(x_mat[:, 1])

            px, py, new_heading, hit, is_stationary = (
                self.kinematics.step_batch(
                    axs,
                    ays,
                    ahs,
                    move_eff,
                    turn_eff,
                    wall_grids=wall_grids[active_idx]
                )
            )

            heading_delta: np.ndarray = np.abs(new_heading - ahs)
            heading_delta = np.where(
                heading_delta > math.pi,
                (2.0 * math.pi) - heading_delta,
                heading_delta
            )

            is_turning: np.ndarray = heading_delta >= spin_reset_rad
            straight_ticks[active_idx] = np.where(
                is_turning, 0, straight_ticks[active_idx] + 1
            )
            cum_rotation[active_idx] = np.where(
                is_turning,
                cum_rotation[active_idx] + heading_delta,
                cum_rotation[active_idx]
            )

            should_reset_spin: np.ndarray = (
                straight_ticks[active_idx] >= spin_hold_frames
            )
            cum_rotation[active_idx] = np.where(
                should_reset_spin, 0.0, cum_rotation[active_idx]
            )

            stagnation_ticks[active_idx] += 1

            hlt: np.ndarray = ahp - (hit * coll_dmg)
            not_moved: np.ndarray = (
                (np.abs(px - axs) < 1e-4) & (np.abs(py - ays) < 1e-4)
            )
            idle: np.ndarray = (
                (move_eff < 0.05) | not_moved | is_stationary
            )
            hlt = hlt - (idle * idle_dmg)

            is_spinning: np.ndarray = np.zeros(
                active_idx.size, dtype=np.bool_
            )
            if spin_enabled:
                is_spinning = cum_rotation[active_idx] >= spin_thresh_rad
                hlt = hlt - (is_spinning.astype(np.float64) * spin_dmg)

            if stag_enabled:
                is_stagnated: np.ndarray = (
                    stagnation_ticks[active_idx] >= stag_limit
                )
                hlt = hlt - (is_stagnated.astype(np.float64) * stag_dmg)

            hlt = np.maximum(hlt, 0.0)

            tx: np.ndarray = np.floor(px).astype(np.int64)
            ty: np.ndarray = np.floor(py).astype(np.int64)

            inb: np.ndarray = (
                (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            )
            txc: np.ndarray = np.clip(tx, 0, w - 1)
            tyc: np.ndarray = np.clip(ty, 0, h - 1)

            curr_dist: np.ndarray = np.where(
                inb, dist_grids[active_idx, tyc, txc], 9999
            )

            better: np.ndarray = curr_dist < best_dists[active_idx]
            heal: np.ndarray = (
                (best_dists[active_idx] - curr_dist) * coll_dmg * rec_ratio
            )
            hlt = np.where(better, np.minimum(1.0, hlt + heal), hlt)
            best_dists[active_idx] = np.where(
                better, curr_dist, best_dists[active_idx]
            )

            improved_idx: np.ndarray = active_idx[better]
            if improved_idx.size > 0:
                stagnation_ticks[improved_idx] = 0
                cum_rotation[improved_idx] = 0.0
                straight_ticks[improved_idx] = 0

            ex: np.ndarray = exit_positions[active_idx, 0]
            ey: np.ndarray = exit_positions[active_idx, 1]
            reached_run: np.ndarray = inb & (tx == ex) & (ty == ey)

            xs[active_idx] = px
            ys[active_idx] = py
            headings[active_idx] = new_heading
            healths[active_idx] = hlt
            alive[active_idx] = hlt > 0.0
            reached[active_idx] |= reached_run
            last_hits[active_idx] = hit
            is_idles[active_idx] = idle
            is_spinnings[active_idx] = is_spinning
            frames_survived[active_idx] += 1

        # Reconstruct per-candidate results
        results: List[Tuple[float, List[float], List[PlayerState]]] = []

        for c_idx in range(cand_count):
            c_states: List[PlayerState] = []
            c_scaled_scores: List[float] = []

            for r_idx in range(run_count):
                inst_idx: int = c_idx * run_count + r_idx
                st: PlayerState = PlayerState(
                    float(runs[r_idx][0].start_pos[0]) + 0.5,
                    float(runs[r_idx][0].start_pos[1]) + 0.5
                )
                st.x = float(xs[inst_idx])
                st.y = float(ys[inst_idx])
                st.heading = float(headings[inst_idx])
                st.health = float(healths[inst_idx])
                st.has_collided = bool(last_hits[inst_idx])
                st.is_alive = bool(alive[inst_idx])
                st.has_reached_exit = bool(reached[inst_idx])
                st.best_step_dist = int(best_dists[inst_idx])
                st.frames_survived = int(frames_survived[inst_idx])
                c_states.append(st)

                raw: float = FitnessEvaluator.calculate_raw_score(
                    st, runs[r_idx][2], self.max_steps
                )
                scaled: float = FitnessEvaluator.calculate_scaled_score(
                    raw, runs[r_idx][4]
                )
                c_scaled_scores.append(scaled)

            c_fitness: float = (
                float(np.mean(c_scaled_scores)) if c_scaled_scores else 0.0
            )
            results.append((c_fitness, c_scaled_scores, c_states))

        return results

    def _build_candidates(self) -> List[Agent]:
        """
        Returns champion plus mutated offspring across scale multipliers.
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
            candidates.append(self.agent.mutate(mutation_scale=scale))

        return candidates

    def _update_mutation_scale(self, accepted: bool) -> None:
        """
        Applies 1/5 success rule and stagnation bumps to mutation scale.
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
                    self.mutation_scale_max, self.mutation_scale * 1.1
                )
            elif success_ratio < 0.2:
                self.mutation_scale = max(
                    self.mutation_scale_min, self.mutation_scale * 0.9
                )

            self._accept_history.clear()

        if self._gens_since_accept >= self._stagnation_gens:
            self.mutation_scale = min(
                self.mutation_scale_max,
                self.mutation_scale * self._stagnation_factor
            )
            self._gens_since_accept = 0
