"""
Rolling in-memory metric recorder for single-agent training.

No candidate frame timelines are stored anymore: the live display runner
re-simulates the champion genome itself. Only lightweight per-generation
summary metrics are kept for the scrubber's solve ticks and CLI reporting.
"""

from typing import List, Dict, Any

import config


class MetricsRecorder:

    def __init__(self, max_generations: int = None) -> None:
        if max_generations is None:
            max_generations = getattr(
                config,
                "RECORDER_MAX_GENERATIONS",
                1000
            )

        self.max_generations = max(
            1,
            int(max_generations)
        )

        self.generations_history: List[Dict[str, Any]] = []

    def record_generation(
        self,
        generation_index: int,
        best_fitness: float,
        gen_fitness: float,
        solve_count: int,
        num_runs: int,
        initial_bfs: float
    ) -> None:
        """
        Appends one generation's corroborated agent metrics.
        """
        generation_data = {
            "generation": int(generation_index),
            "best_fitness": float(best_fitness),
            "gen_fitness": float(gen_fitness),
            "solve_count": int(solve_count),
            "num_runs": int(num_runs),
            "initial_bfs": float(initial_bfs),
            "solved": int(solve_count) > 0
        }

        self.generations_history.append(generation_data)

        excess = (
            len(self.generations_history)
            - self.max_generations
        )

        if excess > 0:
            del self.generations_history[:excess]

    def get_last_generation(self) -> int:
        if not self.generations_history:
            raise IndexError(
                "No generation data has been recorded."
            )

        return int(
            self.generations_history[-1]["generation"]
        )

    def clear(self) -> None:
        self.generations_history.clear()
