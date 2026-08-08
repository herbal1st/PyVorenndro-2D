"""
Headless evolutionary trainer executing training runs with CPU JIT fallback.
Fully integrated with PopulationManager (parallel batch inference), modular fitness, 
BFSPathfinder, and FrameRecorder.
"""

import time
import gc
from typing import Dict, Any, List
import numpy as np

import config
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from core.kinematics import CandidateKinematics
from perception.vision_arc import VisionArcSampler
from evolution.population import PopulationManager, numba_layer_forward_batch
from evolution.fitness import FitnessEvaluator
from evolution.recorder import FrameRecorder
from entities.player_state import PlayerState
from entities.player_express import PlayerExpress


class HeadlessTrainer:
    """Runs headless neuroevolution simulation across CPU hardware with parallel JIT workers."""

    def __init__(self) -> None:
        self.map_gen = MapGenerator()
        self.kinematics = CandidateKinematics()
        self.sampler = VisionArcSampler()
        
        # Initialize PopulationManager with input/output dimensions from config
        self.ga = PopulationManager(
            pop_size=config.POPULATION_SIZE,
            input_size=config.NEURAL_INPUT_SIZE,
            output_size=2  # [move_effort, turn_effort]
        )
        self.recorder = FrameRecorder()

    def run_training_session(self) -> FrameRecorder:
        """Executes full neuroevolution training loop with parallel batch inference."""
        print("=" * 80)
        print(" PYVORENNDRO 2D - TRAINING SESSION (NUMBA CPU PARALLEL JIT)")
        print("=" * 80)
        print(f"{'GEN':>5} | {'TOP':>6} | {'AVG':>6} | {'WAY':>4} | {'EXITS':>5} | {'TIME':>6}s")
        print("-" * 80)

        current_map = self.map_gen.generate_solvable_map()
        pathfinder = BFSPathfinder(current_map)
        dist_matrix = pathfinder.compute_distance_matrix()
        
        if dist_matrix is None:
            dist_matrix = [[0 for _ in range(current_map.width)] for _ in range(current_map.height)]

        sx, sy = current_map.start_pos
        start_bfs = dist_matrix[sy][sx]

        for gen in range(config.LEARNING_GENERATIONS):
            start_time = time.time()

            # Initialize population kinematic arrays
            xs = np.full(config.POPULATION_SIZE, float(sx) + 0.5, dtype=np.float32)
            ys = np.full(config.POPULATION_SIZE, float(sy) + 0.5, dtype=np.float32)
            headings = np.zeros(config.POPULATION_SIZE, dtype=np.float32)
            speeds = np.zeros(config.POPULATION_SIZE, dtype=np.float32)
            healths = np.ones(config.POPULATION_SIZE, dtype=np.float32)
            alives = np.ones(config.POPULATION_SIZE, dtype=np.bool_)
            exits = np.zeros(config.POPULATION_SIZE, dtype=np.bool_)
            best_bfs = np.full(config.POPULATION_SIZE, start_bfs, dtype=np.int32)
            steps_taken = np.zeros(config.POPULATION_SIZE, dtype=np.int32)

            visited_tiles_list: List[set] = [set() for _ in range(config.POPULATION_SIZE)]
            collision_counts = np.zeros(config.POPULATION_SIZE, dtype=np.int32)
            spin_counts = np.zeros(config.POPULATION_SIZE, dtype=np.int32)

            candidate_frames: List[List[Dict[str, Any]]] = [
                [] for _ in range(config.POPULATION_SIZE)
            ]

            wall_grid = current_map.build_wall_grid()

            # Step simulation frames
            for step in range(config.MAX_SIMULATION_STEPS):
                # 1. Perception
                inputs = self.sampler.extract_features_batch(
                    xs, ys, headings, speeds, healths, wall_grid, current_map.exit_pos
                )
                inputs_f64 = inputs.astype(np.float64)

                # 2. Parallel Batch Neural Inference across all cores via numba_layer_forward_batch
                curr_in = inputs_f64
                for l_idx in range(len(self.ga.weights)):
                    w = self.ga.weights[l_idx]
                    b = self.ga.biases[l_idx]
                    curr_in = numba_layer_forward_batch(curr_in, w, b)
                
                outputs = curr_in
                move_effort = outputs[:, 0].astype(np.float32)
                turn_effort = outputs[:, 1].astype(np.float32)

                # 3. Kinematic Physics Step
                nx, ny, nh, hits, stats = self.kinematics.step_batch(
                    xs, ys, headings, move_effort, turn_effort, current_map, wall_grid
                )

                # Update states and health for active candidates
                for i in range(config.POPULATION_SIZE):
                    if alives[i] and not exits[i]:
                        xs[i], ys[i], headings[i] = nx[i], ny[i], nh[i]
                        speeds[i] = move_effort[i] * config.MOVE_SPEED
                        steps_taken[i] += 1

                        tx, ty = int(xs[i]), int(ys[i])
                        if 0 <= tx < current_map.width and 0 <= ty < current_map.height:
                            visited_tiles_list[i].add((tx, ty))

                        if hits[i]:
                            healths[i] -= config.HEALTH_COLL_DMG_PER_FRAME
                            collision_counts[i] += 1
                        if stats[i]:
                            healths[i] -= config.HEALTH_IDLE_DMG_PER_FRAME
                            spin_counts[i] += 1

                        curr_dist = dist_matrix[ty][tx] if (0 <= tx < current_map.width and 0 <= ty < current_map.height) else 9999
                        if curr_dist >= 0 and curr_dist < best_bfs[i]:
                            best_bfs[i] = curr_dist
                            healths[i] = min(1.0, healths[i] + config.HEALTH_RECOVERY_RATIO)

                        if (tx, ty) == current_map.exit_pos:
                            exits[i] = True

                        if healths[i] <= 0.0:
                            healths[i] = 0.0
                            alives[i] = False

                    face_str = PlayerExpress.resolve_face(
                        has_reached_exit=bool(exits[i]),
                        has_collided=bool(hits[i]),
                        is_alive=bool(alives[i])
                    )

                    candidate_frames[i].append({
                        "x": float(xs[i]),
                        "y": float(ys[i]),
                        "heading": float(headings[i]),
                        "health": float(healths[i]),
                        "is_alive": bool(alives[i]),
                        "reached_exit": bool(exits[i]),
                        "hit_wall": bool(hits[i]),
                        "face": face_str,
                        "activations": [inputs_f64[i].tolist(), outputs[i].tolist()]
                    })

            raw_scores = np.zeros(config.POPULATION_SIZE, dtype=np.float32)
            for i in range(config.POPULATION_SIZE):
                state = PlayerState(float(sx) + 0.5, float(sy) + 0.5)
                state.visited_tiles = visited_tiles_list[i]
                state.frames_survived = int(steps_taken[i])
                state.has_reached_exit = bool(exits[i])
                state.health = float(healths[i])
                state.best_step_dist = int(best_bfs[i])
                state.collision_count = int(collision_counts[i])
                state.spin_infraction_count = int(spin_counts[i])

                raw_scores[i] = FitnessEvaluator.calculate_raw_score(
                    state=state,
                    initial_bfs_dist=start_bfs,
                    max_steps=config.MAX_SIMULATION_STEPS,
                    move_speed=config.MOVE_SPEED
                )

            norm_scores_list = FitnessEvaluator.normalize_scores(raw_scores.tolist())
            normalized_scores = np.array(norm_scores_list, dtype=np.float32)

            elapsed = time.time() - start_time
            top_score = float(np.max(raw_scores))
            avg_score = float(np.mean(raw_scores))
            solvers = int(np.sum(exits))

            print(f"{gen+1:5d} | {top_score:6.1f} | {avg_score:6.1f} | {start_bfs:4d} | {solvers:5d} | {elapsed:6.2f}s")

            current_map.encode_bitmask()
            self.recorder.record_generation(
                generation_index=gen,
                map_data=current_map,
                candidate_frames=candidate_frames,
                raw_scores=raw_scores.tolist(),
                normalized_scores=normalized_scores.tolist()
            )

            self.ga.evolve_next_generation(raw_scores.tolist())

            if gen % 5 == 0:
                gc.collect()

        return self.recorder