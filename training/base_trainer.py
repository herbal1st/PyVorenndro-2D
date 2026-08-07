"""
Abstract base neuroevolution trainer module.

Encapsulates genetic algorithm population management, 1/5th success rule
mutation scaling, IPC state publishing, metrics recording, and primary
generation execution loop.
"""

from collections import deque
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from entities.player_state import PlayerState
from storage.persistence import BrainLibraryRegistry, BrainWeightHandler
from training.agent import Agent
from training.metrics import MetricsRecorder


class BaseTrainer:
    """
    Abstract base trainer running evolutionary strategy across parallel runs.
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
        Initializes registry, profile config, champion agent, & telemetry.
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

        self.recorder: MetricsRecorder = MetricsRecorder()
        self.weight_handler: BrainWeightHandler = BrainWeightHandler()

        self.input_size: int = 0
        self.output_size: int = 2
        self.agent: Agent = Agent.from_profile(
            self.profile_config, seed=None
        )

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

    def publish_state(self) -> None:
        """
        Publishes champion genome and telemetry into shared memory as bytes.
        """
        if self._state is None:
            return

        weights, biases = self.agent.to_state()

        self._state["profile_id"] = str(self.profile_id)
        self._state["input_size"] = int(self.input_size)
        self._state["output_size"] = int(self.output_size)
        self._state["layer_sizes"] = list(self.agent.sizes)
        self._state["weights"] = pickle.dumps(weights)
        self._state["biases"] = pickle.dumps(biases)
        self._state["generation"] = int(self._current_gen)
        self._state["num_generations"] = int(config.LEARNING_GENERATIONS)
        self._state["num_runs"] = int(self.shared_runs)
        self._state["max_steps"] = int(self.max_steps)
        self._state["best_fitness"] = float(self.best_fitness)
        self._state["gen_fitness"] = float(self.gen_fitness)
        self._state["solve_count"] = int(self.solve_count)
        self._state["training_complete"] = bool(self.training_complete)
        self._state["gen_history"] = pickle.dumps(
            self.recorder.generations_history
        )
        self._state["initialized"] = True

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> MetricsRecorder:
        """
        Runs neuroevolution session and auto-saves champion weights to disk.
        """
        backend_label: str = getattr(self, "_backend_label", "CPU")
        print(
            f"\n=== SINGLE-AGENT {backend_label} "
            f"NEUROEVOLUTION SIMULATION ==="
        )
        print(
            f"Active Profile: '{self.profile_id}' | "
            f"Population: {self.population_size} | "
            f"Maps/Candidate: {self.shared_runs} | "
            f"Sims/Gen: {self.shared_runs * self.population_size} | "
            f"Max Steps: {self.max_steps} | "
            f"Backend: {backend_label}\n"
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
        self._shutdown_process_pool()

        print("-" * len(header_str))
        print(
            "Single-agent training complete! Champion saved to disk. "
            "Display runner keeps replaying champion until Ctrl+C...\n"
        )

        return self.recorder

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

    def _generate_run_maps(self, count: int) -> List[Tuple]:
        """
        Abstract method: overridden by subclasses to generate run maps.
        """
        raise NotImplementedError

    def _simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """
        Abstract method: overridden by subclasses to simulate candidates.
        """
        raise NotImplementedError

    def _shutdown_process_pool(self) -> None:
        """
        Hook for subclasses to clean up parallel process pools on exit.
        """
        pass
