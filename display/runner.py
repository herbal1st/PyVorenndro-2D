"""
Live display runner.

This process is kept entirely separate from the headless trainer: instead of
slowing training down with rendering, it repeatedly ASKS the trainer for the
latest champion model state, re-simulates that genome on a fresh random maze
in real time, and only when the run finishes does it ask again.

The runner keeps running after training completes (replaying the final
champion) until the window is closed or Ctrl+C is pressed.
"""

import random
import time
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
        max_steps: int = config.MAX_SIMULATION_STEPS
    ) -> None:
        """
        Initializes a fresh live run episode.
        """
        self.weights, self.biases = genome
        self.map_data: MapData = map_data
        self.dist_grid: np.ndarray = dist_grid
        self.initial_bfs: int = initial_bfs
        self.spawn_heading: float = spawn_heading
        self.run_number: int = run_number
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
        self.finish_step: int = 0

        self._kinematics: CandidateKinematics = CandidateKinematics()
        self._transformer: SpatialTransformer = SpatialTransformer()

    def run_data(self) -> Dict[str, Any]:
        """
        Exposes the episode in the shape the display widgets expect.
        """
        return {
            "map_data": self.map_data,
            "frames": self.frames,
            "run_number": self.run_number,
            "initial_bfs": self.initial_bfs,
            "max_steps": self.max_steps,
            "finished": self.finished,
            "solved": self.solved
        }

    def simulate_up_to(self, target_step: int) -> None:
        """
        Advances the episode until ``target_step`` frames exist or it ends.
        """
        while len(self.frames) < target_step and not self.finished:
            self._step()

    def _step(self) -> None:
        """
        Runs one simulation tick for the single agent and records its frame.
        """
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
            config.MOVE_SPEED,
            state.health,
            self.map_data,
            current_dist=current_dist
        )

        outputs, acts = Agent.forward_batch(
            self.weights,
            self.biases,
            features.reshape(1, -1),
            return_acts=True
        )

        move_eff: float = float(outputs[0, 0])
        turn_eff: float = float(outputs[0, 1])

        state.heading, is_stationary_turn = (
            self._kinematics.apply_rotation(
                state.heading,
                turn_eff,
                move_eff
            )
        )

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
                state.health - config.HEALTH_COLL_DMG_PER_FRAME
            )

        if is_idle:
            state.health = max(
                0.0,
                state.health - config.HEALTH_IDLE_DMG_PER_FRAME
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
                * config.HEALTH_COLL_DMG_PER_FRAME
                * config.HEALTH_RECOVERY_RATIO
            )
            state.health = min(1.0, state.health + heal_amount)
            state.best_step_dist = curr_dist

        if tx == self.map_data.exit_pos[0] and ty == self.map_data.exit_pos[1]:
            state.has_reached_exit = True

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
            self.finished = True
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
        Initializes the live-run loop state.
        """
        self._state: Dict[str, Any] = state
        self._stop_event: Any = stop_event

        self.map_generator: MapGenerator = MapGenerator()
        self._transformer: SpatialTransformer = SpatialTransformer()

        self.viewport: Optional[Viewport] = None
        self.panel: Optional[OverlayPanel] = None
        self.graph: Optional[NetworkGraph] = None
        self.scrubber: Optional[TimelineScrubber] = None

        self.episode: Optional[Episode] = None
        self.active_frame: int = 0
        self.run_number: int = 0
        self.genome: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None
        self._last_metrics_time: float = 0.0
        self._last_click_time: int = 0

        self.metrics: Dict[str, Any] = {
            "generation": 0,
            "num_generations": config.LEARNING_GENERATIONS,
            "best_fitness": 0.0,
            "gen_fitness": 0.0,
            "solve_count": 0,
            "num_runs": config.SIMULATION_RUNS,
        }
        self.gen_history: List[Dict[str, Any]] = []

    # ----- Trainer interaction -----

    def _refresh_metrics(self, force: bool = False) -> None:
        """
        Pulls the latest trainer telemetry from shared state (rate-limited).
        """
        now: float = time.time()
        if not force and (now - self._last_metrics_time) < 0.25:
            return

        self._last_metrics_time = now

        if self._state.get("initialized"):
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
            self.gen_history = list(
                self._state.get("gen_history", [])
            )

    def _fetch_genome(
        self
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Asks the trainer for the latest champion model state.
        """
        if not self._state.get("initialized"):
            return None

        weights = self._state.get("weights")
        biases = self._state.get("biases")

        if weights is None or biases is None:
            return None

        return weights, biases

    # ----- Episode lifecycle -----

    def _make_run_map(self) -> Tuple[MapData, np.ndarray, int, float]:
        """
        Builds a fresh randomly-seeded maze for the next display run.
        """
        if (
            config.MAP_DIFFICULTY_MIN
            >= config.MAP_DIFFICULTY_MAX
        ):
            difficulty: float = config.MAP_DIFFICULTY_MIN
        else:
            difficulty = random.uniform(
                config.MAP_DIFFICULTY_MIN,
                config.MAP_DIFFICULTY_MAX
            )

        map_data: MapData = (
            self.map_generator.generate_solvable_map(
                difficulty_ratio=difficulty
            )
        )

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        initial_bfs: int = pathfinder.get_step_distance(
            *map_data.start_pos
        )

        dist_grid: np.ndarray = np.asarray(
            pathfinder.distance_matrix,
            dtype=np.int64
        )

        spawn_heading: float = (
            self._transformer.generate_random_heading(
                map_data,
                map_data.start_pos
            )
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
            self.run_number
        )
        self.active_frame = 0

    # ----- Main loop -----

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

        self.viewport = Viewport(config.LAYOUT_GRID_RECT)
        self.panel = OverlayPanel(config.LAYOUT_PANEL_RECT)
        self.graph = NetworkGraph(config.LAYOUT_GRAPH_RECT)
        self.scrubber = TimelineScrubber(config.LAYOUT_SCRUBBER_RECT)

        running: bool = True

        while running:
            if (
                self._stop_event is not None
                and self._stop_event.is_set()
            ):
                running = False
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.viewport.toggle_camera_mode()

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

                    _new_gen, new_frame = (
                        self.scrubber.handle_click(
                            event.pos,
                            self.metrics["num_generations"],
                            config.MAX_SIMULATION_STEPS,
                            mouse_button=event.button
                        )
                    )

                    if new_frame is not None:
                        self.active_frame = new_frame

            self._refresh_metrics()

            # Episode lifecycle: ask trainer for the model, then run it.
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

            # Human-speed advancement (multiplied by the turbo setting).
            if self.scrubber.is_playing:
                target: int = (
                    self.active_frame + self.scrubber.playback_speed
                )
                self.episode.simulate_up_to(target)

                if self.episode.finished:
                    self.active_frame = self.episode.finish_step
                else:
                    self.active_frame = min(
                        target,
                        len(self.episode.frames)
                    )

                # Run finished: repeat by asking the trainer again.
                if (
                    self.episode.finished
                    and self.active_frame >= self.episode.finish_step
                ):
                    if self.scrubber.repeat_all:
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

            self.viewport.draw(screen, run_data, self.active_frame)
            self.panel.draw_panel(
                screen,
                run_data,
                self.active_frame,
                self.metrics
            )
            self.graph.draw_graph(screen, run_data, self.active_frame)
            self.scrubber.draw_controls(
                screen,
                self.metrics["generation"],
                self.metrics["num_generations"],
                self.active_frame,
                config.MAX_SIMULATION_STEPS,
                gen_history=self.gen_history,
                selected_cand_frames=run_data["frames"]
            )

            pygame.display.flip()
            clock.tick(config.FPS)

        pygame.quit()
