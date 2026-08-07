"""
Headless single-agent CPU trainer.

Evaluates candidates across parallel procedural mazes using a unified
NumPy batch matrix pass across all candidates and runs simultaneously.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from core.kinematics import CandidateKinematics
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.map_pipeline import TrainingMapPipeline
from core.pathfinder import BFSPathfinder
from entities.player_state import PlayerState
from perception.spatial_transformer import SpatialTransformer
from training.agent import Agent
from training.base_trainer import BaseTrainer
from training.cpu_simulator import CpuSimulator


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
        trainer: BaseTrainer = GpuHeadlessTrainer(
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


class HeadlessTrainer(BaseTrainer):
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
        Initializes CPU map pipeline, physics kinematics, & CPU simulator.
        """
        super().__init__(
            num_runs=num_runs,
            max_steps=max_steps,
            state=state,
            stop_event=stop_event,
            auto_tune=False
        )

        self.map_generator: MapGenerator = MapGenerator()
        self.transformer: SpatialTransformer = SpatialTransformer(
            profile_config=self.profile_config
        )
        self.kinematics: CandidateKinematics = CandidateKinematics(
            profile_config=self.profile_config
        )
        self.map_pipeline: TrainingMapPipeline = TrainingMapPipeline(
            map_generator=self.map_generator,
            transformer=self.transformer,
            max_steps=self.max_steps
        )
        self.cpu_simulator: CpuSimulator = CpuSimulator(
            profile_config=self.profile_config,
            max_steps=self.max_steps,
            kinematics=self.kinematics,
            transformer=self.transformer
        )

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

        self.input_size = len(sample_features)
        self.output_size = 2

        self.agent = Agent.from_profile(
            self.profile_config, self.input_size, self.output_size
        )

        reloaded: bool = self.weight_handler.load_champion(
            self.agent, self.profile_config
        )
        if reloaded:
            print(
                f"[Trainer] Reloaded champion weights for profile "
                f"'{self.profile_id}' from disk."
            )

        self.simulation_workers: int = 1

        if auto_tune:
            from training import autotune
            autotune.apply_auto_tuning(self)

    def _generate_run_map(self) -> Tuple[
        MapData, np.ndarray, int, int, float, float
    ]:
        """
        Delegates single solvable maze generation to TrainingMapPipeline.
        """
        return self.map_pipeline.generate_run_map()

    def _generate_run_maps(self, count: int) -> List[Tuple]:
        """
        Delegates batch maze generation to TrainingMapPipeline.
        """
        return self.map_pipeline.generate_run_maps(count)

    def _simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """
        Delegates candidate matrix simulation to CpuSimulator engine.
        """
        return self.cpu_simulator.simulate_candidates(agents, runs)
