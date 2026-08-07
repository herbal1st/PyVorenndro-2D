"""
IPC state bridge for safe communication with the trainer process.
"""

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from perception.spatial_transformer import SpatialTransformer
from storage.persistence import BrainLibraryRegistry


class TrainerBridge:
    """
    Polls trainer process shared state for champion genomes and telemetry.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        profile_id: str = getattr(config, "ACTIVE_PROFILE_ID", "herbal1st")
    ) -> None:
        """
        Initializes registry binding, profile config, and metrics cache.
        """
        self._state: Dict[str, Any] = state
        self.registry: BrainLibraryRegistry = BrainLibraryRegistry()
        self.profile_id: str = str(profile_id)
        self.profile_config: Dict[str, Any] = self.registry.get_profile(
            self.profile_id
        )
        self.transformer: SpatialTransformer = SpatialTransformer(
            profile_config=self.profile_config
        )

        self._last_metrics_time: float = 0.0
        num_runs: int = (
            config.POPULATION_SIZE * config.MAPS_PER_CANDIDATE
        )
        self.metrics: Dict[str, Any] = {
            "generation": 0,
            "num_generations": config.LEARNING_GENERATIONS,
            "best_fitness": 0.0,
            "gen_fitness": 0.0,
            "solve_count": 0,
            "num_runs": num_runs,
        }
        self.gen_history: List[Dict[str, Any]] = []

    def refresh_metrics(self, force: bool = False) -> bool:
        """
        Polls shared state dict for latest generation metrics and telemetry.
        Returns True if profile configuration changed during hot reload.
        """
        now: float = time.time()
        if not force and (now - self._last_metrics_time) < 0.25:
            return False

        self._last_metrics_time = now
        if not self._state.get("initialized"):
            return False

        profile_changed: bool = False
        published_pid: str = str(
            self._state.get("profile_id", self.profile_id)
        )
        if published_pid != self.profile_id:
            self.profile_id = published_pid
            self.profile_config = self.registry.get_profile(published_pid)
            self.transformer = SpatialTransformer(
                profile_config=self.profile_config
            )
            profile_changed = True

        self.metrics = {
            "generation": int(self._state.get("generation", 0)),
            "num_generations": int(
                self._state.get("num_generations", 1)
            ),
            "best_fitness": float(self._state.get("best_fitness", 0.0)),
            "gen_fitness": float(self._state.get("gen_fitness", 0.0)),
            "solve_count": int(self._state.get("solve_count", 0)),
            "num_runs": int(self._state.get("num_runs", 1)),
        }

        raw_history = self._state.get("gen_history")
        if raw_history:
            try:
                self.gen_history = list(pickle.loads(raw_history))
            except Exception:
                pass

        return profile_changed

    def fetch_genome(
        self
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Unpickles champion genome weight and bias arrays from IPC state.
        """
        if not self._state.get("initialized"):
            return None

        raw_weights = self._state.get("weights")
        raw_biases = self._state.get("biases")

        if raw_weights is None or raw_biases is None:
            return None

        try:
            weights: List[np.ndarray] = pickle.loads(raw_weights)
            biases: List[np.ndarray] = pickle.loads(raw_biases)
            return weights, biases
        except Exception:
            return None
