"""
Headless CPU neuroevolution trainer.

The simulation hot path is batched across the population:
feature batch -> neural batch -> physics batch

This avoids Python-level per-candidate simulation work while remaining
portable across operating systems and CPU architectures.
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
Runs CPU neuroevolution using batched candidate simulation.
"""

```
def __init__(
    self,
    pop_size: int = config.POPULATION_SIZE,
    max_steps: int = config.MAX_SIMULATION_STEPS
) -> None:
    """
    Initializes trainer components and simulation bounds.
    """
    grid_capacity: int = (
        config.GRID_ROWS * config.GRID_COLS
    )

    self.pop_size: int = min(
        pop_size,
        grid_capacity
    )

    self.max_steps: int = max_steps

    self.map_generator: MapGenerator = MapGenerator()
    self.transformer: SpatialTransformer = SpatialTransformer()
    self.kinematics: CandidateKinematics = CandidateKinematics()
    self.recorder: FrameRecorder = FrameRecorder()

    # Determine the actual neural input width from the feature compiler.
    sample_map: MapData = (
        self.map_generator.generate_solvable_map()
    )

    sample_features: np.ndarray = (
        self.transformer.compile_feature_vector(
            0.5,
            0.5,
            0.0,
            1.0,
            1.0,
            sample_map
        )
    )

    self.input_size: int = int(
        sample_features.shape[0]
    )

    self.output_size: int = 2

    self.population: PopulationManager = (
        PopulationManager(
            self.pop_size,
            input_size=self.input_size,
            output_size=self.output_size
        )
    )

    # Cached map arrays. These remain unchanged throughout a map regime.
    self._wall_grid: np.ndarray = None
    self._dist_grid: np.ndarray = None

def _generate_map_for_target(
    self,
    target_bfs: int
) -> MapData:
    """
    Generates a solvable map close to the requested curriculum target.
    """
    minimum: int = max(
        2,
        target_bfs
    )

    maximum: int = (
        minimum
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

        distance: int = (
            pathfinder.get_step_distance(
                *map_data.start_pos
            )
        )

        if minimum <= distance <= maximum:
            return map_data

        if distance < minimum:
            delta: int = minimum - distance
        else:
            delta = distance - maximum

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
) -> Tuple[
    MapData,
    BFSPathfinder,
    int,
    int,
    float,
    np.ndarray
]:
    """
    Generates/prepares the active map and its reusable lookup arrays.
    """
    if config.CURRICULUM_ENABLED:
        map_data: MapData = (
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

    spawn_headings: np.ndarray = np.asarray(
        [
            self.transformer.generate_random_heading(
                map_data,
                map_data.start_pos
            )
            for _ in range(self.pop_size)
        ],
        dtype=np.float64
    )

    self._wall_grid = np.asarray(
        map_data.build_wall_grid(),
        dtype=np.bool_
    )

    self._dist_grid = np.asarray(
        pathfinder.distance_matrix,
        dtype=np.int32
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
    spawn_headings: np.ndarray
) -> Tuple[
    List[PlayerState],
    List[List[Dict[str, Any]]]
]:
    """
    Simulates the complete population using batched operations.

    The expensive operations are performed on NumPy arrays:
        - vision
        - neural inference
        - movement
        - collision
        - health
        - distance lookup

    Python PlayerState objects and recorder dictionaries are only created
    after the simulation has finished.
    """
    n: int = self.pop_size

    start_x, start_y = map_data.start_pos
    exit_x, exit_y = map_data.exit_pos

    wall_grid: np.ndarray = self._wall_grid
    dist_grid: np.ndarray = self._dist_grid

    width: int = int(map_data.width)
    height: int = int(map_data.height)

    move_speed: float = (
        self.kinematics.move_speed
    )

    # ------------------------------------------------------------------
    # Vectorized candidate state.
    # ------------------------------------------------------------------

    xs: np.ndarray = np.full(
        n,
        float(start_x) + 0.5,
        dtype=np.float64
    )

    ys: np.ndarray = np.full(
        n,
        float(start_y) + 0.5,
        dtype=np.float64
    )

    headings: np.ndarray = (
        spawn_headings.copy()
    )

    healths: np.ndarray = np.ones(
        n,
        dtype=np.float64
    )

    alive: np.ndarray = np.ones(
        n,
        dtype=np.bool_
    )

    reached_exit: np.ndarray = np.zeros(
        n,
        dtype=np.bool_
    )

    has_collided: np.ndarray = np.zeros(
        n,
        dtype=np.bool_
    )

    frames_survived: np.ndarray = np.zeros(
        n,
        dtype=np.int32
    )

    best_dist: np.ndarray = np.full(
        n,
        initial_bfs_dist,
        dtype=np.int32
    )

    # ------------------------------------------------------------------
    # Numeric frame history.
    #
    # The visualizer requires the complete timeline, so we retain it,
    # but keep the hot-loop representation compact and numeric.
    # ------------------------------------------------------------------

    frame_x: np.ndarray = np.empty(
        (self.max_steps, n),
        dtype=np.float64
    )

    frame_y: np.ndarray = np.empty_like(
        frame_x
    )

    frame_heading: np.ndarray = np.empty_like(
        frame_x
    )

    frame_health: np.ndarray = np.empty_like(
        frame_x
    )

    frame_dist: np.ndarray = np.empty(
        (self.max_steps, n),
        dtype=np.int32
    )

    frame_hit: np.ndarray = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    frame_alive: np.ndarray = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    frame_reached: np.ndarray = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    # One activation history array per network layer.
    #
    # forward_batch() returns the input activation followed by each
    # layer's pre-activation.
    activation_history: List[np.ndarray] = []

    for layer_index in range(
        len(self.population.weights)
    ):
        if layer_index == 0:
            width_activation: int = self.input_size
        else:
            width_activation = int(
                self.population.weights[
                    layer_index
                ].shape[2]
            )

        activation_history.append(
            np.empty(
                (
                    self.max_steps,
                    n,
                    width_activation
                ),
                dtype=np.float64
            )
        )

    active: np.ndarray = alive.copy()

    steps_completed: int = 0

    # Reusable constant speed vector.
    speeds: np.ndarray = np.full(
        n,
        move_speed,
        dtype=np.float64
    )

    # ------------------------------------------------------------------
    # Hot simulation loop.
    # ------------------------------------------------------------------

    for step_index in range(
        self.max_steps
    ):
        if not bool(active.any()):
            break

        active_idx: np.ndarray = (
            np.flatnonzero(active)
        )

        # --------------------------------------------------------------
        # Batch sensory processing.
        # --------------------------------------------------------------

        features: np.ndarray = (
            self.transformer.compile_feature_batch(
                xs[active_idx],
                ys[active_idx],
                headings[active_idx],
                speeds[active_idx],
                healths[active_idx],
                map_data,
                wall_grid
            )
        )

        # --------------------------------------------------------------
        # Batch neural inference.
        # --------------------------------------------------------------

        outputs, activations = (
            self.population.forward_batch(
                active_idx.tolist(),
                features
            )
        )

        activation_history[0][
            step_index,
            active_idx
        ] = activations[0]

        for activation_index in range(
            1,
            len(activations)
        ):
            activation_history[
                activation_index
            ][
                step_index,
                active_idx
            ] = activations[
                activation_index
            ]

        move_eff: np.ndarray = (
            outputs[:, 0]
        )

        turn_eff: np.ndarray = (
            outputs[:, 1]
        )

        # --------------------------------------------------------------
        # Batch movement and collision.
        # --------------------------------------------------------------

        (
            new_x,
            new_y,
            new_heading,
            hit,
            stationary_turn
        ) = self.kinematics.step_batch(
            xs[active_idx],
            ys[active_idx],
            headings[active_idx],
            move_eff,
            turn_eff,
            map_data,
            wall_grid
        )

        old_x: np.ndarray = (
            xs[active_idx].copy()
        )

        old_y: np.ndarray = (
            ys[active_idx].copy()
        )

        xs[active_idx] = new_x
        ys[active_idx] = new_y
        headings[active_idx] = new_heading

        has_collided[
            active_idx
        ] = hit

        frames_survived[
            active_idx
        ] += 1

        # --------------------------------------------------------------
        # Batch health.
        # --------------------------------------------------------------

        is_idle: np.ndarray = (
            (move_eff < 0.05)
            | (
                (
                    np.abs(new_x - old_x)
                    < 1e-4
                )
                & (
                    np.abs(new_y - old_y)
                    < 1e-4
                )
            )
            | stationary_turn
        )

        active_health: np.ndarray = (
            healths[active_idx]
        )

        active_health -= (
            hit.astype(np.float64)
            * config.HEALTH_COLL_DMG_PER_FRAME
        )

        active_health -= (
            is_idle.astype(np.float64)
            * config.HEALTH_IDLE_DMG_PER_FRAME
        )

        np.maximum(
            active_health,
            0.0,
            out=active_health
        )

        healths[
            active_idx
        ] = active_health

        died: np.ndarray = (
            active_health <= 0.0
        )

        # --------------------------------------------------------------
        # Batch grid coordinates.
        # --------------------------------------------------------------

        tile_x: np.ndarray = (
            np.floor(new_x).astype(np.int32)
        )

        tile_y: np.ndarray = (
            np.floor(new_y).astype(np.int32)
        )

        in_bounds: np.ndarray = (
            (tile_x >= 0)
            & (tile_x < width)
            & (tile_y >= 0)
            & (tile_y < height)
        )

        safe_x: np.ndarray = np.clip(
            tile_x,
            0,
            width - 1
        )

        safe_y: np.ndarray = np.clip(
            tile_y,
            0,
            height - 1
        )

        current_dist: np.ndarray = np.where(
            in_bounds,
            dist_grid[safe_y, safe_x],
            9999
        ).astype(
            np.int32,
            copy=False
        )

        # --------------------------------------------------------------
        # Batch distance improvement / health recovery.
        # --------------------------------------------------------------

        previous_best: np.ndarray = (
            best_dist[active_idx]
        )

        improved: np.ndarray = (
            current_dist < previous_best
        )

        if bool(improved.any()):

            improved_indices: np.ndarray = (
                active_idx[improved]
            )

            improvement: np.ndarray = (
                previous_best[improved]
                - current_dist[improved]
            ).astype(
                np.float64
            )

            recovery: np.ndarray = (
                improvement
                * config.HEALTH_COLL_DMG_PER_FRAME
                * config.HEALTH_RECOVERY_RATIO
            )

            healths[
                improved_indices
            ] = np.minimum(
                1.0,
                healths[
                    improved_indices
                ] + recovery
            )

            best_dist[
                improved_indices
            ] = current_dist[
                improved
            ]

        # --------------------------------------------------------------
        # Exit detection.
        # --------------------------------------------------------------

        reached_now: np.ndarray = (
            (tile_x == exit_x)
            & (tile_y == exit_y)
        )

        if bool(reached_now.any()):
            reached_indices: np.ndarray = (
                active_idx[reached_now]
            )

            reached_exit[
                reached_indices
            ] = True

        # Candidates reaching the exit or dying become inactive.
        if bool(reached_now.any()):
            active[
                active_idx[reached_now]
            ] = False

        if bool(died.any()):
            active[
                active_idx[died]
            ] = False

        # --------------------------------------------------------------
        # Store numeric frame state.
        # --------------------------------------------------------------

        frame_x[
            step_index
        ] = xs

        frame_y[
            step_index
        ] = ys

        frame_heading[
            step_index
        ] = headings

        frame_health[
            step_index
        ] = healths

        # Calculate distance for the complete population.
        all_tile_x: np.ndarray = (
            np.floor(xs).astype(np.int32)
        )

        all_tile_y: np.ndarray = (
            np.floor(ys).astype(np.int32)
        )

        all_in_bounds: np.ndarray = (
            (all_tile_x >= 0)
            & (all_tile_x < width)
            & (all_tile_y >= 0)
            & (all_tile_y < height)
        )

        all_safe_x: np.ndarray = np.clip(
            all_tile_x,
            0,
            width - 1
        )

        all_safe_y: np.ndarray = np.clip(
            all_tile_y,
            0,
            height - 1
        )

        frame_dist[
            step_index
        ] = np.where(
            all_in_bounds,
            dist_grid[
                all_safe_y,
                all_safe_x
            ],
            9999
        )

        frame_hit[
            step_index
        ] = has_collided

        frame_alive[
            step_index
        ] = alive

        frame_reached[
            step_index
        ] = reached_exit

        steps_completed = (
            step_index + 1
        )

    # ------------------------------------------------------------------
    # Convert final vector state to PlayerState objects.
    # ------------------------------------------------------------------

    candidate_states: List[PlayerState] = []

    for candidate_index in range(n):

        state: PlayerState = PlayerState(
            float(xs[candidate_index]),
            float(ys[candidate_index])
        )

        state.heading = float(
            headings[candidate_index]
        )

        state.health = float(
            healths[candidate_index]
        )

        state.is_alive = bool(
            alive[candidate_index]
        )

        state.has_reached_exit = bool(
            reached_exit[candidate_index]
        )

        state.has_collided = bool(
            has_collided[candidate_index]
        )

        state.frames_survived = int(
            frames_survived[candidate_index]
        )

        state.best_step_dist = int(
            best_dist[candidate_index]
        )

        candidate_states.append(
            state
        )

    # ------------------------------------------------------------------
    # Convert compact history to the existing recorder format.
    # ------------------------------------------------------------------

    candidate_frames: List[
        List[Dict[str, Any]]
    ] = [
        []
        for _ in range(n)
    ]

    for candidate_index in range(n):

        frames: List[
            Dict[str, Any]
        ] = candidate_frames[
            candidate_index
        ]

        for step_index in range(
            steps_completed
        ):

            alive_value: bool = bool(
                frame_alive[
                    step_index,
                    candidate_index
                ]
            )

            reached_value: bool = bool(
                frame_reached[
                    step_index,
                    candidate_index
                )
            )

            hit_value: bool = bool(
                frame_hit[
                    step_index,
                    candidate_index
                )
            )

            face: str = (
                PlayerExpress.resolve_face(
                    reached_value,
                    hit_value,
                    alive_value
                )
            )

            activations: List[
                List[float]
            ] = [
                activation_history[
                    layer_index
                ][
                    step_index,
                    candidate_index
                ].tolist()
                for layer_index in range(
                    len(activation_history)
                )
            ]

            frames.append({
                "step": step_index + 1,
                "x": float(
                    frame_x[
                        step_index,
                        candidate_index
                    ]
                ),
                "y": float(
                    frame_y[
                        step_index,
                        candidate_index
                    ]
                ),
                "heading": float(
                    frame_heading[
                        step_index,
                        candidate_index
                    ]
                ),
                "face": face,
                "hit_wall": hit_value,
                "health": float(
                    frame_health[
                        step_index,
                        candidate_index
                    ]
                ),
                "is_alive": alive_value,
                "reached_exit": reached_value,
                "dist": int(
                    frame_dist[
                        step_index,
                        candidate_index
                    ]
                ),
                "activations": activations
            })

    return (
        candidate_states,
        candidate_frames
    )

def run_training_session(
    self,
    num_generations: int = config.LEARNING_GENERATIONS
) -> FrameRecorder:
    """
    Runs the complete headless training session.
    """
    print(
        "\n=== CPU NEUROEVOLUTION SIMULATION ==="
    )

    print(
        f"Population: {self.pop_size} | "
        f"Max Steps: {self.max_steps} | "
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
    gens_on_map: int = 0
    consecutive_failures: int = 0

    map_data: MapData = None
    initial_bfs_dist: int = 0
    theoretical_max: float = 0.0
    spawn_headings: np.ndarray = None

    for gen_index in range(
        num_generations
    ):

        generation_start: float = (
            time.perf_counter()
        )

        new_map_this_generation: bool = (
            switch_next
        )

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

        # --------------------------------------------------------------
        # Fitness.
        # --------------------------------------------------------------

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

        normalized_scores: List[float] = (
            FitnessEvaluator.normalize_scores(
                raw_scores
            )
        )

        # --------------------------------------------------------------
        # Recording.
        # --------------------------------------------------------------

        self.recorder.record_generation(
            gen_index,
            map_data,
            candidate_frames,
            scaled_scores,
            normalized_scores
        )

        # --------------------------------------------------------------
        # Evolution.
        # --------------------------------------------------------------

        self.population.mutation_scale = (
            config.MUTATION_SCALE
            * config.REGIME_TRANSITION_MUTATION_BOOST
            if new_map_this_generation
            else config.MUTATION_SCALE
        )

        self.population.evolve_next_generation(
            normalized_scores
        )

        self.population.mutation_scale = (
            config.MUTATION_SCALE
        )

        generation_time: float = (
            time.perf_counter()
            - generation_start
        )

        top_score: int = int(
            round(
                max(scaled_scores)
            )
        )

        average_score: float = (
            sum(scaled_scores)
            / float(len(scaled_scores))
        )

        solvers: List[Tuple[int, int]] = [
            (candidate_index, state.frames_survived)
            for candidate_index, state in enumerate(candidate_states)
            if state.has_reached_exit
        ]

        solve_count: int = len(
            solvers
        )

        if solve_count > 0:
            fastest_step: int = min(
                steps
                for _, steps in solvers
            )
            frame_string: str = str(
                fastest_step
            )
        else:
            frame_string = "-"

        print(
            f"{gen_index + 1:>7d} | "
            f"{top_score:>7d} | "
            f"{average_score:>7.1f} | "
            f"{initial_bfs_dist:>7d} | "
            f"{target_bfs:>7d} | "
            f"{frame_string:>7s} | "
            f"{solve_count}/{self.pop_size:>3d} | "
            f"{generation_time:>5.2f}s"
        )

        # --------------------------------------------------------------
        # Curriculum / map regime.
        # --------------------------------------------------------------

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

    print("-" * len(header_str))

    print(
        "CPU Training complete! "
        "Booting interactive visualizer GUI...\n"
    )

    return self.recorder
```
