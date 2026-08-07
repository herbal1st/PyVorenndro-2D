"""
Live display runner.

Separated from the headless trainer: asks trainer for active profile and
champion model state, re-simulating on a fresh maze using active profile
parameters at human speed.
"""

import random
import time
import math
import pickle
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pygame

import config
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.pathfinder import BFSPathfinder
from core.kinematics import CandidateKinematics
from perception.spatial_transformer import SpatialTransformer
from entities.player_state import PlayerState
from entities.player_express import PlayerExpress
from storage.persistence import BrainLibraryRegistry
from training.agent import Agent
from display.viewport import Viewport
from display.network_graph import NetworkGraph
from display.timeline_scrubber import TimelineScrubber
from display.overlay_panel import OverlayPanel


class Episode:
    """
    One human-speed re-simulation of the champion genome on a single maze.
    """

    def __init__(
        self,
        genome: Tuple[List[np.ndarray], List[np.ndarray]],
        map_data: MapData,
        dist_grid: np.ndarray,
        initial_bfs: int,
        spawn_heading: float,
        run_number: int,
        profile_config: Dict[str, Any],
        max_steps: int = config.MAX_SIMULATION_STEPS,
        generation: int = 0
    ) -> None:
        """
        Initializes a fresh live run episode with profile parameters.
        """
        self.weights, self.biases = genome
        self.map_data: MapData = map_data
        self.dist_grid: np.ndarray = dist_grid
        self.initial_bfs: int = initial_bfs
        self.spawn_heading: float = spawn_heading
        self.run_number: int = run_number
        self.generation: int = generation
        self.profile_config: Dict[str, Any] = profile_config
        self.max_steps: int = max_steps

        self.state: PlayerState = PlayerState(
            float(map_data.start_pos[0]) + 0.5,
            float(map_data.start_pos[1]) + 0.5
        )
        self.state.heading = spawn_heading
        self.state.best_step_dist = initial_bfs

        self.frames: List[Dict[str, Any]] = []
        self.finished: bool = False
        self.solved: bool = False
        self.finish_step: int = max_steps
        self._celebration_count: int = 0

        self._kinematics: CandidateKinematics = CandidateKinematics(
            profile_config=profile_config
        )
        self._transformer: SpatialTransformer = SpatialTransformer(
            profile_config=profile_config
        )

        metabolics: Dict[str, Any] = profile_config.get("metabolics", {})
        kinematics_cfg: Dict[str, Any] = profile_config.get("kinematics", {})

        self.coll_damage: float = float(
            metabolics.get("collision_damage", 0.003)
        )
        self.idle_damage: float = float(
            metabolics.get("idle_damage", 0.001)
        )
        self.recovery_ratio: float = float(
            metabolics.get("recovery_ratio", 0.05)
        )
        self.move_speed: float = float(
            kinematics_cfg.get("move_speed", 0.1)
        )

        self.spin_enabled: bool = bool(
            metabolics.get("spin_penalty_enabled", True)
        )
        self.spin_thresh_rad: float = math.radians(
            float(metabolics.get("spin_angle_threshold_deg", 360.0))
        )
        self.spin_reset_rad: float = math.radians(
            float(metabolics.get("spin_reset_angle_deg", 5.0))
        )
        self.spin_hold_frames: int = int(
            metabolics.get("spin_reset_hold_frames", 15)
        )
        self.spin_dmg: float = float(
            metabolics.get("spin_damage_per_frame", 0.003)
        )

        self.stag_enabled: bool = bool(
            metabolics.get("stagnation_enabled", True)
        )
        self.stag_limit: int = int(
            self.max_steps * float(
                metabolics.get("stagnation_timeout_ratio", 0.75)
            )
        )
        self.stag_dmg: float = float(
            metabolics.get("stagnation_damage_per_frame", 0.001)
        )

        self.cum_rotation: float = 0.0
        self.straight_ticks: int = 0
        self.stagnation_ticks: int = 0

        self.last_hit: bool = False
        self.is_idle: bool = False
        self.is_spinning: bool = False

        # Pre-simulate run to completion immediately (<1ms)
        self.simulate_up_to(self.max_steps)

    def run_data(self) -> Dict[str, Any]:
        """
        Exposes episode telemetry dictionary including generation index.
        """
        return {
            "map_data": self.map_data,
            "frames": self.frames,
            "run_number": self.run_number,
            "generation": self.generation,
            "initial_bfs": self.initial_bfs,
            "max_steps": self.max_steps,
            "finished": self.finished,
            "solved": self.solved,
            "finish_step": self.finish_step
        }

    def simulate_up_to(self, target_step: int) -> None:
        """
        Advances the episode until target_step frames exist or it ends.
        """
        while len(self.frames) < target_step and not self.finished:
            self._step()

    def _step(self) -> None:
        """
        Runs one simulation tick for the agent and records its frame.
        """
        if self.solved:
            # Handle post-solve celebration frames
            step: int = len(self.frames) + 1
            self._celebration_count += 1

            celebration_cap: int = getattr(
                config, "WINNER_CELEBRATION_FRAMES", 60
            )

            last_frame: Dict[str, Any] = (
                self.frames[-1] if self.frames else {}
            )
            self.frames.append({
                "step": step,
                "x": self.state.x,
                "y": self.state.y,
                "heading": self.state.heading,
                "face": config.FACE_EXIT,
                "hit_wall": False,
                "health": 1.0,
                "is_alive": True,
                "reached_exit": True,
                "dist": 0,
                "activations": last_frame.get("activations", [])
            })

            if (
                self._celebration_count >= celebration_cap
                or step >= self.max_steps
            ):
                self.finished = True
                self.finish_step = step
            return

        state: PlayerState = self.state
        step: int = len(self.frames) + 1

        tx: int = int(state.x)
        ty: int = int(state.y)

        if (
            0 <= tx < self.map_data.width
            and 0 <= ty < self.map_data.height
        ):
            current_dist: float = float(self.dist_grid[ty, tx])
        else:
            current_dist = 9999.0

        features = self._transformer.compile_feature_vector(
            state.x,
            state.y,
            state.heading,
            self.move_speed,
            state.health,
            self.map_data,
            current_dist=current_dist,
            last_hit=self.last_hit,
            is_idle=self.is_idle,
            is_spinning=self.is_spinning,
            profile_config=self.profile_config
        )

        outputs, acts = Agent.forward_batch(
            self.weights,
            self.biases,
            features.reshape(1, -1),
            return_acts=True
        )

        move_eff: float = float(outputs[0, 0])
        turn_eff: float = float(outputs[0, 1])

        old_heading: float = state.heading
        state.heading, is_stationary_turn = (
            self._kinematics.apply_rotation(
                state.heading,
                turn_eff,
                move_eff
            )
        )

        heading_delta: float = abs(state.heading - old_heading)
        if heading_delta > math.pi:
            heading_delta = (2.0 * math.pi) - heading_delta

        if heading_delta >= self.spin_reset_rad:
            self.straight_ticks = 0
            self.cum_rotation += heading_delta
        else:
            self.straight_ticks += 1

        if self.straight_ticks >= self.spin_hold_frames:
            self.cum_rotation = 0.0

        self.stagnation_ticks += 1

        nx, ny, hit = self._kinematics.calculate_forward_step(
            state.x,
            state.y,
            state.heading,
            move_eff,
            self.map_data
        )

        is_idle: bool = (
            move_eff < 0.05
            or (
                abs(nx - state.x) < 1e-4
                and abs(ny - state.y) < 1e-4
            )
            or is_stationary_turn
        )

        state.x = nx
        state.y = ny
        state.has_collided = hit
        state.frames_survived += 1

        if hit:
            state.health = max(
                0.0,
                state.health - self.coll_damage
            )

        if is_idle:
            state.health = max(
                0.0,
                state.health - self.idle_damage
            )

        is_spinning: bool = False
        if self.spin_enabled and self.cum_rotation >= self.spin_thresh_rad:
            is_spinning = True
            state.health = max(
                0.0,
                state.health - self.spin_dmg
            )

        if self.stag_enabled and self.stagnation_ticks >= self.stag_limit:
            state.health = max(
                0.0,
                state.health - self.stag_dmg
            )

        if state.health <= 0.0:
            state.is_alive = False

        tx: int = int(state.x)
        ty: int = int(state.y)

        if (
            0 <= tx < self.map_data.width
            and 0 <= ty < self.map_data.height
        ):
            curr_dist: int = int(self.dist_grid[ty, tx])
        else:
            curr_dist = 9999

        if curr_dist < state.best_step_dist:
            heal_amount: float = (
                (state.best_step_dist - curr_dist)
                * self.coll_damage
                * self.recovery_ratio
            )
            state.health = min(1.0, state.health + heal_amount)
            state.best_step_dist = curr_dist
            self.stagnation_ticks = 0
            self.cum_rotation = 0.0
            self.straight_ticks = 0

        if tx == self.map_data.exit_pos[0] and ty == self.map_data.exit_pos[1]:
            state.has_reached_exit = True

        self.last_hit = hit
        self.is_idle = is_idle
        self.is_spinning = is_spinning

        face: str = PlayerExpress.resolve_face(
            state.has_reached_exit,
            state.has_collided,
            state.is_alive
        )

        layer_acts: List[List[float]] = [
            acts[0][0].tolist()
        ] + [
            layer[0].tolist()
            for layer in acts[1:]
        ]

        self.frames.append({
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
            "activations": layer_acts,
        })

        if state.has_reached_exit:
            self.solved = True
            self.finish_step = step
        elif not state.is_alive:
            self.finished = True
            self.solved = False
            self.finish_step = step
        elif step >= self.max_steps:
            self.finished = True
            self.finish_step = step


class DisplayRunner:
    """
    Owns the pygame loop and continuously replays the trainer's champion.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        stop_event: Any = None
    ) -> None:
        """
        Initializes registry, active profile, and display loop state.
        """
        self._state: Dict[str, Any] = state
        self._stop_event: Any = stop_event

        self.registry: BrainLibraryRegistry = BrainLibraryRegistry()
        self.profile_id: str = str(
            getattr(config, "ACTIVE_PROFILE_ID", "herbal1st")
        )
        self.profile_config: Dict[str, Any] = self.registry.get_profile(
            self.profile_id
        )

        self.map_generator: MapGenerator = MapGenerator()
        self._transformer: SpatialTransformer = SpatialTransformer(
            profile_config=self.profile_config
        )

        self.viewport: Optional[Viewport] = None
        self.panel: Optional[OverlayPanel] = None
        self.graph: Optional[NetworkGraph] = None
        self.scrubber: Optional[TimelineScrubber] = None

        self.episode: Optional[Episode] = None
        self.active_frame: int = 0
        self.active_gen: int = 0
        self.run_number: int = 0
        self.genome: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None
        self._last_metrics_time: float = 0.0
        self._last_click_time: int = 0

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

    def _refresh_metrics(self, force: bool = False) -> None:
        """
        Pulls latest telemetry and profile ID from shared state.
        """
        now: float = time.time()
        if not force and (now - self._last_metrics_time) < 0.25:
            return

        self._last_metrics_time = now

        if self._state.get("initialized"):
            published_pid: str = str(
                self._state.get("profile_id", self.profile_id)
            )
            if published_pid != self.profile_id:
                self.profile_id = published_pid
                self.profile_config = self.registry.get_profile(published_pid)
                self._transformer = SpatialTransformer(
                    profile_config=self.profile_config
                )
                if self.viewport is not None:
                    self.viewport.set_profile_config(self.profile_config)

            self.metrics = {
                "generation": int(
                    self._state.get("generation", 0)
                ),
                "num_generations": int(
                    self._state.get("num_generations", 1)
                ),
                "best_fitness": float(
                    self._state.get("best_fitness", 0.0)
                ),
                "gen_fitness": float(
                    self._state.get("gen_fitness", 0.0)
                ),
                "solve_count": int(
                    self._state.get("solve_count", 0)
                ),
                "num_runs": int(
                    self._state.get("num_runs", 1)
                ),
            }
            raw_history = self._state.get("gen_history")
            if raw_history:
                try:
                    self.gen_history = list(pickle.loads(raw_history))
                except Exception:
                    pass

    def _fetch_genome(
        self
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Asks trainer for champion model state & unpickles bytes to arrays.
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

    def _make_run_map(self) -> Tuple[MapData, np.ndarray, int, float]:
        """
        Builds a fresh randomly-seeded maze for the next display run.
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

        initial_bfs: int = pathfinder.get_step_distance(
            *map_data.start_pos
        )
        dist_grid: np.ndarray = np.asarray(
            pathfinder.distance_matrix, dtype=np.int64
        )
        spawn_heading: float = self._transformer.generate_random_heading(
            map_data, map_data.start_pos
        )

        return map_data, dist_grid, initial_bfs, spawn_heading

    def _start_episode(
        self,
        genome: Tuple[List[np.ndarray], List[np.ndarray]],
        replay: Optional[Tuple] = None
    ) -> None:
        """
        Begins a new live run, reusing an old episode's maze when replaying.
        """
        if replay is not None:
            map_data, dist_grid, initial_bfs, spawn_heading = replay
        else:
            map_data, dist_grid, initial_bfs, spawn_heading = (
                self._make_run_map()
            )
            self.run_number += 1

        self.episode = Episode(
            genome,
            map_data,
            dist_grid,
            initial_bfs,
            spawn_heading,
            self.run_number,
            profile_config=self.profile_config,
            generation=self.active_gen
        )
        self.active_frame = 0

    def _draw_splash(self, screen: pygame.Surface) -> None:
        """
        Renders the waiting screen before the first model state arrives.
        """
        screen.fill(config.COLOR_BG)
        font: pygame.font.Font = pygame.font.SysFont(
            "monospace", 18, bold=True
        )
        lbl = font.render(
            "Waiting for trainer to publish initial agent state...",
            True,
            config.COLOR_PLAYER_HIGHLIGHT
        )
        screen.blit(
            lbl,
            lbl.get_rect(center=(config.SCREEN_WIDTH // 2, 120))
        )

    def run(self) -> None:
        """
        Runs the live display loop until the window closes or Ctrl+C.
        """
        pygame.init()
        screen: pygame.Surface = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption(
            "PyVorenndro 2D - Live Single-Agent Runner"
        )
        clock: pygame.time.Clock = pygame.time.Clock()

        self.viewport = Viewport(
            config.LAYOUT_GRID_RECT, profile_config=self.profile_config
        )
        self.panel = OverlayPanel(config.LAYOUT_PANEL_RECT)
        self.graph = NetworkGraph(config.LAYOUT_GRAPH_RECT)
        self.scrubber = TimelineScrubber(config.LAYOUT_SCRUBBER_RECT)

        running: bool = True

        while running:
            if self._stop_event is not None and self._stop_event.is_set():
                running = False
                break

            self._refresh_metrics()
            latest_gen: int = int(self.metrics.get("generation", 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        if self._stop_event is not None:
                            self._stop_event.set()

                    elif event.key == pygame.K_RETURN:
                        self.viewport.toggle_camera_mode()

                    elif event.key == pygame.K_SPACE:
                        self.scrubber.is_playing = (
                            not self.scrubber.is_playing
                        )

                    elif event.key == pygame.K_LEFT:
                        self.active_frame = max(0, self.active_frame - 25)

                    elif event.key == pygame.K_RIGHT:
                        self.active_frame = min(
                            config.MAX_SIMULATION_STEPS - 1,
                            self.active_frame + 25
                        )

                    elif event.key == pygame.K_PAGEUP:
                        if self.active_gen < latest_gen:
                            self.active_gen += 1
                            self.episode = None

                    elif event.key == pygame.K_PAGEDOWN:
                        if self.active_gen > 0:
                            self.active_gen -= 1
                            self.episode = None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    now_ms: int = pygame.time.get_ticks()
                    is_double: bool = (
                        event.button == 1
                        and (now_ms - self._last_click_time) < 300
                    )
                    if event.button == 1:
                        self._last_click_time = now_ms

                    self.viewport.handle_click(
                        event.pos,
                        is_double_click=is_double
                    )

                    new_gen, new_frame = self.scrubber.handle_click(
                        event.pos,
                        latest_gen,
                        self.metrics["num_generations"],
                        config.MAX_SIMULATION_STEPS,
                        mouse_button=event.button
                    )

                    if new_gen is not None:
                        if new_gen != self.active_gen:
                            self.active_gen = new_gen
                            self.episode = None

                    if new_frame is not None:
                        self.active_frame = new_frame

            if self.episode is None:
                if self._state.get("initialized"):
                    self.genome = self._fetch_genome()
                    if self.genome is not None:
                        self._start_episode(self.genome)
                    else:
                        self._draw_splash(screen)
                        pygame.display.flip()
                        clock.tick(config.FPS)
                        continue
                else:
                    self._draw_splash(screen)
                    pygame.display.flip()
                    clock.tick(config.FPS)
                    continue

            if self.scrubber.is_playing:
                self.active_frame += self.scrubber.playback_speed
                max_clip_step: int = max(1, self.episode.finish_step)
                self.active_frame = min(self.active_frame, max_clip_step)

                # Clip reached end on screen - advance or replay
                if self.active_frame >= max_clip_step:
                    if self.scrubber.repeat_all:
                        if self.active_gen < latest_gen:
                            self.active_gen += 1
                        else:
                            self.active_gen = 0
                        self.episode = None
                        continue
                    else:
                        self._start_episode(
                            self.genome,
                            replay=(
                                self.episode.map_data,
                                self.episode.dist_grid,
                                self.episode.initial_bfs,
                                self.episode.spawn_heading
                            )
                        )

            run_data: Dict[str, Any] = self.episode.run_data()

            screen.fill(config.COLOR_BG)

            self.viewport.draw(
                screen,
                run_data,
                self.active_frame,
                profile_config=self.profile_config
            )
            self.panel.draw_panel(
                screen,
                run_data,
                self.active_frame,
                self.metrics,
                self.active_gen
            )
            self.graph.draw_graph(
                screen,
                run_data,
                self.active_frame,
                profile_config=self.profile_config
            )
            self.scrubber.draw_controls(
                screen,
                self.active_gen,
                latest_gen,
                self.metrics["num_generations"],
                self.active_frame,
                config.MAX_SIMULATION_STEPS,
                gen_history=self.gen_history,
                selected_cand_frames=run_data["frames"]
            )

            pygame.display.flip()
            clock.tick(config.FPS)

        pygame.quit()
