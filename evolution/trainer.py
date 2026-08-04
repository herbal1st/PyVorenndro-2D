"""
Headless CPU neuroevolution trainer.

The simulation hot path is fully batched across candidates:
feature batch -> neural batch -> physics batch

This avoids Python-level per-candidate simulation work while retaining the
existing PlayerState, FitnessEvaluator, and FrameRecorder interfaces.

The implementation is platform-independent and does not rely on multiprocessing,
fork(), platform-specific SIMD libraries, or GPU availability.
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

    # Determine the network input size from the real feature compiler.
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

    # Cached map arrays.
    self._wall_grid: np.ndarray = None
    self._dist_grid: np.ndarray = None

def _generate_map_for_target(
    self,
    target_bfs: int
) -> MapData:
    """
    Generates a solvable map near the requested curriculum distance.
    """
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
            delta: int = lo - start_dist
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
    """
    Prepares the active map and cached lookup grids.

    Returns:
        map_data,
        pathfinder,
        initial_bfs_dist,
        num_turns,
        theoretical_max,
        spawn_headings
    """
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

    # These are reused for every timestep of every generation on
    # this map. Rebuilding them inside the hot loop is expensive.
    self._wall_grid = (
        map_data.build_wall_grid()
    )

    self._dist_grid = np.asarray(
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

@staticmethod
def _make_frame(
    step: int,
    x: float,
    y: float,
    heading: float,
    face: str,
    hit_wall: bool,
    health: float,
    is_alive: bool,
    reached_exit: bool,
    dist: int,
    activations: List[List[float]]
) -> Dict[str, Any]:
    """
    Constructs the existing recorder frame format.
    """
    return {
        "step": step,
        "x": x,
        "y": y,
        "heading": heading,
        "face": face,
        "hit_wall": hit_wall,
        "health": health,
        "is_alive": is_alive,
        "reached_exit": reached_exit,
        "dist": dist,
        "activations": activations,
    }

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
    Runs one generation using batched sensory processing, neural
    inference, and physics.

    The simulation state remains in NumPy arrays during the hot loop.
    Python PlayerState objects and recorder dictionaries are created only
    after simulation has completed.
    """

    n: int = self.pop_size

    start_x, start_y = map_data.start_pos
    exit_x, exit_y = map_data.exit_pos

    exit_x_center: float = (
        float(exit_x) + 0.5
    )
    exit_y_center: float = (
        float(exit_y) + 0.5
    )

    dist_grid: np.ndarray = self._dist_grid
    wall_grid: np.ndarray = self._wall_grid

    move_speed: float = (
        self.kinematics.move_speed
    )

    # ------------------------------------------------------------------
    # Persistent vectorized simulation state.
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

    headings: np.ndarray = np.asarray(
        spawn_headings,
        dtype=np.float64
    ).copy()

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
    # Frame storage.
    #
    # Numeric state is kept in compact NumPy arrays during simulation.
    # This avoids constructing thousands of Python dictionaries in the
    # hottest part of training.
    # ------------------------------------------------------------------

    frame_x = np.empty(
        (self.max_steps, n),
        dtype=np.float64
    )

    frame_y = np.empty_like(frame_x)
    frame_heading = np.empty_like(frame_x)
    frame_health = np.empty_like(frame_x)

    frame_dist = np.empty(
        (self.max_steps, n),
        dtype=np.int32
    )

    frame_hit = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    frame_alive = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    frame_reached = np.empty(
        (self.max_steps, n),
        dtype=np.bool_
    )

    # Network activations are required by the visualizer.
    #
    # The first activation is the input feature vector. Later entries
    # are the dense-layer pre-activations returned by forward_batch().
    activation_history: List[np.ndarray] = []

    for layer_weights in self.population.weights:
        activation_history.append(
            np.empty(
                (
                    self.max_steps,
                    n,
                    layer_weights.shape[2]
                    if layer_weights.ndim == 3
                    else self.output_size
                ),
                dtype=np.float64
            )
        )

    # The input activation has a different width, so allocate it after
    # the actual feature compiler is known.
    activation_history.insert(
        0,
        np.empty(
            (
                self.max_steps,
                n,
                self.input_size
            ),
            dtype=np.float64
        )
    )

    active: np.ndarray = alive.copy()

    last_activations: List[np.ndarray] = [
        np.zeros(
            self.input_size,
            dtype=np.float64
        )
    ]

    for layer_weights in self.population.weights:
        last_activations.append(
            np.zeros(
                layer_weights.shape[2],
                dtype=np.float64
            )
        )

    steps_completed: int = 0

    # ------------------------------------------------------------------
    # Main simulation loop.
    # ------------------------------------------------------------------

    for step_idx in range(self.max_steps):

        if not bool(active.any()):
            break

        active_idx: np.ndarray = (
            np.flatnonzero(active)
        )

        # --------------------------------------------------------------
        # 1. Batch sensory processing.
        # --------------------------------------------------------------

        features: np.ndarray = (
            self.transformer.compile_feature_batch(
                xs[active_idx],
                ys[active_idx],
                headings[active_idx],
                np.full(
                    active_idx.shape,
                    move_speed,
                    dtype=np.float64
                ),
                healths[active_idx],
                map_data,
                wall_grid
            )
        )

        # --------------------------------------------------------------
        # 2. Batch neural inference.
        # --------------------------------------------------------------

        outputs, activations = (
            self.population.forward_batch(
                active_idx.tolist(),
                features
            )
        )

        # Store the activation arrays for active candidates.
        activation_history[0][
            step_idx,
            active_idx
        ] = features

        for layer_idx, layer_activation in enumerate(
            activations[1:],
            start=1
        ):
            activation_history[layer_idx][
                step_idx,
                active_idx
            ] = layer_activation

        # --------------------------------------------------------------
        # 3. Batch movement / collision physics.
        # --------------------------------------------------------------

        move_eff: np.ndarray = (
            outputs[:, 0]
        )

        turn_eff: np.ndarray = (
            outputs[:, 1]
        )

        old_xs: np.ndarray = (
            xs[active_idx].copy()
        )

        old_ys: np.ndarray = (
            ys[active_idx].copy()
        )

        (
            new_xs,
            new_ys,
            new_headings,
            hit,
            stationary_turn
        ) = self.kinematics.step_batch(
            old_xs,
            old_ys,
            headings[active_idx],
            move_eff,
            turn_eff,
            map_data,
            wall_grid
        )

        xs[active_idx] = new_xs
        ys[active_idx] = new_ys
        headings[active_idx] = new_headings

        has_collided[active_idx] = hit

        frames_survived[active_idx] += 1

        # --------------------------------------------------------------
        # 4. Batch health update.
        # --------------------------------------------------------------

        clamped_move: np.ndarray = np.clip(
            move_eff,
            0.0,
            1.0
        )

        is_idle: np.ndarray = (
            (clamped_move < 0.05)
            | (
                (
                    np.abs(new_xs - old_xs)
                    < 1e-4
                )
                & (
                    np.abs(new_ys - old_ys)
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

        active_health = np.maximum(
            active_health,
            0.0
        )

        healths[active_idx] = active_health

        died: np.ndarray = (
            active_health <= 0.0
        )

        if bool(died.any()):
            alive[
                active_idx[died]
            ] = False

        # --------------------------------------------------------------
        # 5. Batch distance lookup.
        # --------------------------------------------------------------

        tile_x: np.ndarray = (
            np.floor(new_xs).astype(np.int64)
        )

        tile_y: np.ndarray = (
            np.floor(new_ys).astype(np.int64)
        )

        in_bounds: np.ndarray = (
            (tile_x >= 0)
            & (tile_x < map_data.width)
            & (tile_y >= 0)
            & (tile_y < map_data.height)
        )

        safe_x: np.ndarray = np.clip(
            tile_x,
            0,
            map_data.width - 1
        )

        safe_y: np.ndarray = np.clip(
            tile_y,
            0,
            map_data.height - 1
        )

        current_dist: np.ndarray = np.where(
            in_bounds,
            dist_grid[safe_y, safe_x],
            9999
        ).astype(
            np.int32,
            copy=False
        )

        previous_best: np.ndarray = (
            best_dist[active_idx]
        )

        improved: np.ndarray = (
            current_dist < previous_best
        )

        if bool(improved.any()):

            improved_amount: np.ndarray = (
                previous_best[improved]
                - current_dist[improved]
            ).astype(
                np.float64
            )

            heal_amount: np.ndarray = (
                improved_amount
                * config.HEALTH_COLL_DMG_PER_FRAME
                * config.HEALTH_RECOVERY_RATIO
            )

            improved_indices: np.ndarray = (
                active_idx[improved]
            )

            healths[improved_indices] = np.minimum(
                1.0,
                healths[improved_indices]
                + heal_amount
            )

            best_dist[improved_indices] = (
                current_dist[improved]
            )

        # --------------------------------------------------------------
        # 6. Exit detection.
        # --------------------------------------------------------------

        reached_now: np.ndarray = (
            (
                np.floor(new_xs).astype(np.int64)
                == exit_x
            )
            & (
                np.floor(new_ys).astype(np.int64)
                == exit_y
            )
        )

        if bool(reached_now.any()):
            reached_exit[
                active_idx[reached_now]
            ] = True

        # Reached candidates stop simulating after this frame.
        if bool(reached_now.any()):
            active[
                active_idx[reached_now]
            ] = False

        # Dead candidates also stop after this frame.
        if bool(died.any()):
            active[
                active_idx[died]
            ] = False

        # --------------------------------------------------------------
        # 7. Store numeric frame state.
        # --------------------------------------------------------------

        frame_x[
            step_idx
        ] = xs

        frame_y[
            step_idx
        ] = ys

        frame_heading[
            step_idx
        ] = headings

        frame_health[
            step_idx
        ] = healths

        frame_dist[
            step_idx
        ] = np.where(
            active,
            np.where(
                (
                    np.floor(xs).astype(np.int64)
                    >= 0
                )
                & (
                    np.floor(xs).astype(np.int64)
                    < map_data.width
                )
                & (
                    np.floor(ys).astype(np.int64)
                    >= 0
                )
                & (
                    np.floor(ys).astype(np.int64)
                    < map_data.height
                ),
                dist_grid[
                    np.clip(
                        np.floor(ys).astype(np.int64),
                        0,
                        map_data.height - 1
                    ),
                    np.clip(
                        np.floor(xs).astype(np.int64),
                        0,
                        map_data.width - 1
                    )
                ],
                9999
            ),
            np.where(
                (
                    np.floor(xs).astype(np.int64)
                    >= 0
                )
                & (
                    np.floor(xs).astype(np.int64)
                    < map_data.width
                )
                & (
                    np.floor(ys).astype(np.int64)
                    >= 0
                )
                & (
                    np.floor(ys).astype(np.int64)
                    < map_data.height
                ),
                dist_grid[
                    np.clip(
                        np.floor(ys).astype(np.int64),
                        0,
                        map_data.height - 1
                    ),
                    np.clip(
                        np.floor(xs).astype(np.int64),
                        0,
                        map_data.width - 1
                    )
                ],
                9999
            )
        )

        frame_hit[
            step_idx
        ] = has_collided

        frame_alive[
            step_idx
        ] = alive

        frame_reached[
            step_idx
        ] = reached_exit

        steps_completed = step_idx + 1

    # ------------------------------------------------------------------
    # Convert final vectorized state into the existing PlayerState API.
    # This happens once per candidate rather than once per timestep.
    # ------------------------------------------------------------------

    candidate_states: List[PlayerState] = []

    for i in range(n):

        state = PlayerState(
            float(xs[i]),
            float(ys[i])
        )

        state.heading = float(
            headings[i]
        )

        state.health = float(
            healths[i]
        )

        state.is_alive = bool(
            alive[i]
        )

        state.has_reached_exit = bool(
            reached_exit[i]
        )

        state.has_collided = bool(
            has_collided[i]
        )

        state.frames_survived = int(
            frames_survived[i]
        )

        state.best_step_dist = int(
            best_dist[i]
        )

        candidate_states.append(
            state
        )

    # ------------------------------------------------------------------
    # Convert compact numeric history into the exact dictionary format
    # expected by the visualizer/recorder.
    # ------------------------------------------------------------------

    candidate_frames: List[
        List[Dict[str, Any]]
    ] = [
        []
        for _ in range(n)
    ]

    for i in range(n):

        frames = candidate_frames[i]

        for step_idx in range(
            steps_completed
        ):

            alive_value: bool = bool(
                frame_alive[
                    step_idx,
                    i
                ]
            )

            reached_value: bool = bool(
                frame_reached[
                    step_idx,
                    i
                ]
            )

            hit_value: bool = bool(
                frame_hit[
                    step_idx,
                    i
                ]
            )

            face: str = (
                PlayerExpress.resolve_face(
                    reached_value,
                    hit_value,
                    alive_value
                )
            )

            activations: List[List[float]] = [
                activation_history[layer][
                    step_idx,
                    i
                ].tolist()
                for layer in range(
                    len(activation_history)
                )
            ]

            frames.append(
                self._make_frame(
                    step_idx + 1,
                    float(
                        frame_x[
                            step_idx,
                            i
                        ]
                    ),
                    float(
                        frame_y[
                            step_idx,
                            i
                        ]
                    ),
                    float(
                        frame_heading[
                            step_idx,
                            i
                        ]
                    ),
                    face,
                    hit_value,
                    float(
                        frame_health[
                            step_idx,
                            i
                        ]
                    ),
                    alive_value,
                    reached_value,
                    int(
                        frame_dist[
                            step_idx,
                            i
                        ]
                    ),
                    activations
                )
            )

    return (
        candidate_states,
        candidate_frames
    )

def run_training_session(
    self,
    num_generations: int = config.LEARNING_GENERATIONS
) -> FrameRecorder:
    """
    Runs CPU candidate simulations over multiple generations.
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

    map_data = None
    initial_bfs_dist: int = 0
    theoretical_max: float = 0.0
    spawn_headings: np.ndarray = None

    for gen_idx in range(
        num_generations
    ):

        gen_start: float = time.time()

        new_map_this_gen: bool = (
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

        norm_scores: List[float] = (
            FitnessEvaluator.normalize_scores(
                raw_scores
            )
        )

        # --------------------------------------------------------------
        # Recording.
        # --------------------------------------------------------------

        self.recorder.record_generation(
            gen_idx,
            map_data,
            candidate_frames,
            scaled_scores,
            norm_scores
        )

        # --------------------------------------------------------------
        # Evolution.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Statistics.
        # --------------------------------------------------------------

        gen_time: float = (
            time.time() - gen_start
        )

        top_int: int = int(
            round(
                max(scaled_scores)
            )
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
                candidate_idx,
                state.frames_survived
            )
            for candidate_idx, state
            in enumerate(candidate_states)
            if state.has_reached_exit
        ]

        solve_count: int = len(
            solvers
        )

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

        # --------------------------------------------------------------
        # Curriculum.
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

    print(
        "-" * len(header_str)
    )

    print(
        "CPU Training complete! "
        "Booting interactive visualizer GUI...\n"
    )

    return self.recorder
```
