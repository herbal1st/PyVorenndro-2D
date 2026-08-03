"""
Headless neuroevolution simulation trainer running candidate runs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

import config
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
    Coordinates headless training loops and frame timeline recording.
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
        self.population: PopulationManager = PopulationManager(clamped_pop)
        self.recorder: FrameRecorder = FrameRecorder()

    def run_training_session(
        self,
        num_generations: int = config.LEARNING_GENERATIONS
    ) -> FrameRecorder:
        """
        Runs headless candidate simulations over multiple generations.
        """
        print("\n=== NEUROEVOLUTION SIMULATION RUN ===")
        print(
            f"Population: {self.pop_size} | Max Steps: {self.max_steps} | "
            f"Profile: {config.KINEMATICS_PROFILE} | Target Score: 1000\n"
        )
        header_str: str = (
            f"{'GEN':>7s} | {'TOP':>7s} | {'AVG':>7s} | {'WAY':>7s} | "
            f"{'FIRST':>7s} | {'FRAME':>7s} | {'EXITS':>7s}"
        )
        print(header_str)
        print("-" * len(header_str))

        for gen_idx in range(num_generations):
            map_data = self.map_generator.generate_solvable_map()
            pathfinder = BFSPathfinder(map_data)
            pathfinder.compute_distance_matrix()

            start_x, start_y = map_data.start_pos
            initial_bfs_dist: int = pathfinder.get_step_distance(
                start_x, start_y
            )
            num_turns: int = pathfinder.count_shortest_path_turns()

            theoretical_max: float = (
                FitnessEvaluator.calculate_theoretical_max_score(
                    initial_bfs_dist, self.max_steps, num_turns=num_turns
                )
            )

            candidate_states: List[PlayerState] = [
                PlayerState(float(start_x) + 0.5, float(start_y) + 0.5)
                for _ in range(self.pop_size)
            ]

            for state in candidate_states:
                state.heading = self.transformer.generate_random_heading(
                    map_data, map_data.start_pos
                )
                state.best_step_dist = initial_bfs_dist

            candidate_frames: List[List[Dict[str, Any]]] = [
                [] for _ in range(self.pop_size)
            ]

            for step in range(1, self.max_steps + 1):
                active_count: int = 0

                for idx in range(self.pop_size):
                    state = candidate_states[idx]
                    net = self.population.networks[idx]

                    if state.has_reached_exit or not state.is_alive:
                        if candidate_frames[idx]:
                            last_f = candidate_frames[idx][-1].copy()
                            last_f["step"] = step
                            candidate_frames[idx].append(last_f)
                        continue

                    active_count += 1
                    features = self.transformer.compile_feature_vector(
                        state.x,
                        state.y,
                        state.heading,
                        self.kinematics.move_speed,
                        state.health,
                        map_data
                    )

                    outputs = net.forward(features)[0]
                    move_eff: float = float(outputs[0])
                    turn_eff: float = float(outputs[1])

                    state.heading, is_stationary_turn = (
                        self.kinematics.apply_rotation(
                            state.heading, turn_eff, move_eff
                        )
                    )
                    nx, ny, hit = self.kinematics.calculate_forward_step(
                        state.x, state.y, state.heading, move_eff, map_data
                    )

                    is_idle: bool = (
                        move_eff < 0.05 or
                        (
                            abs(nx - state.x) < 1e-4 and
                            abs(ny - state.y) < 1e-4
                        ) or
                        is_stationary_turn
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

                    curr_dist = pathfinder.get_step_distance(
                        *state.tile_coords
                    )

                    if curr_dist < state.best_step_dist:
                        dist_reduced: int = state.best_step_dist - curr_dist
                        heal_amount: float = (
                            dist_reduced *
                            config.HEALTH_COLL_DMG_PER_FRAME *
                            config.HEALTH_RECOVERY_RATIO
                        )
                        state.health = min(1.0, state.health + heal_amount)
                        state.best_step_dist = curr_dist

                    if state.tile_coords == map_data.exit_pos:
                        state.has_reached_exit = True

                    face = PlayerExpress.resolve_face(
                        state.has_reached_exit,
                        state.has_collided,
                        state.is_alive
                    )

                    layer_acts: List[List[float]] = [
                        features.astype(np.float64).tolist()
                    ] + [
                        layer.output.flatten().tolist()
                        for layer in net.layers
                    ]

                    candidate_frames[idx].append({
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
                        "activations": layer_acts
                    })

                if active_count == 0:
                    break

            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    c_state, initial_bfs_dist, self.max_steps
                )
                for c_state in candidate_states
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
                gen_idx,
                map_data,
                candidate_frames,
                scaled_scores,
                norm_scores
            )

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
                fastest_step: int = min(step_cnt for _, step_cnt in solvers)
                frame_str: str = str(fastest_step)
            else:
                frame_str = "-"

            winner_str: str = f"# {winner_idx}"

            row_str: str = (
                f"{gen_idx + 1:>7d} | {top_int:>7d} | {avg_scaled:>7.1f} | "
                f"{initial_bfs_dist:>7d} | {winner_str:>7s} | "
                f"{frame_str:>7s} | {exits_str:>7s}"
            )
            print(row_str)

            self.population.evolve_next_generation(norm_scores)

        print("-" * len(header_str))
        print("Training complete! Booting interactive visualizer GUI...\n")
        return self.recorder
