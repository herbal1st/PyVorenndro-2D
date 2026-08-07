"""
Live display runner orchestrator.

Coordinates Pygame initialization, window management, IPC state sync via
TrainerBridge, input event dispatching via DisplayEventHandler, and UI drawing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

import config
from core.map_data import MapData
from core.map_generator import MapGenerator
from core.map_pipeline import DisplayMapFactory
from display.episode import Episode
from display.event_handler import DisplayEventHandler, EventResult
from display.network_graph import NetworkGraph
from display.overlay_panel import OverlayPanel
from display.timeline_scrubber import TimelineScrubber
from display.trainer_bridge import TrainerBridge
from display.viewport import Viewport


class DisplayRunner:
    """
    Orchestrates Pygame window, input dispatching, and champion replay.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        stop_event: Any = None
    ) -> None:
        """
        Initializes IPC bridge, event listener, map factory, and state flags.
        """
        self._stop_event: Any = stop_event
        self.bridge: TrainerBridge = TrainerBridge(state)
        self.event_handler: DisplayEventHandler = DisplayEventHandler()
        self.map_generator: MapGenerator = MapGenerator()
        self.display_map_factory: DisplayMapFactory = DisplayMapFactory(
            map_generator=self.map_generator,
            transformer=self.bridge.transformer,
            profile_config=self.bridge.profile_config
        )

        self.viewport: Optional[Viewport] = None
        self.panel: Optional[OverlayPanel] = None
        self.graph: Optional[NetworkGraph] = None
        self.scrubber: Optional[TimelineScrubber] = None

        self.episode: Optional[Episode] = None
        self.active_frame: int = 0
        self.active_gen: int = 0
        self.run_number: int = 0
        self.genome: Optional[
            Tuple[List[np.ndarray], List[np.ndarray]]
        ] = None

    def run(self) -> None:
        """
        Runs the live display window loop until user quits or Ctrl+C.
        """
        pygame.init()
        self._load_window_icon()

        screen: pygame.Surface = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption(
            "PyVorenndro 2D - Live Single-Agent Runner"
        )
        clock: pygame.time.Clock = pygame.time.Clock()

        self.viewport = Viewport(
            config.LAYOUT_GRID_RECT,
            profile_config=self.bridge.profile_config
        )
        self.panel = OverlayPanel(config.LAYOUT_PANEL_RECT)
        self.graph = NetworkGraph(config.LAYOUT_GRAPH_RECT)
        self.scrubber = TimelineScrubber(config.LAYOUT_SCRUBBER_RECT)

        running: bool = True

        while running:
            if self._stop_event is not None and self._stop_event.is_set():
                running = False
                break

            profile_changed: bool = self.bridge.refresh_metrics()
            if profile_changed:
                if self.viewport is not None:
                    self.viewport.set_profile_config(
                        self.bridge.profile_config
                    )
                self.display_map_factory.set_profile_config(
                    self.bridge.profile_config
                )

            latest_gen: int = int(self.bridge.metrics.get("generation", 0))

            res: EventResult = self.event_handler.process_events(
                self.viewport,
                self.scrubber,
                self.active_gen,
                latest_gen,
                self.bridge.metrics["num_generations"]
            )

            if res.should_quit:
                running = False
                if self._stop_event is not None:
                    self._stop_event.set()

            if res.toggle_camera and self.viewport is not None:
                self.viewport.toggle_camera_mode()

            if res.toggle_play and self.scrubber is not None:
                self.scrubber.is_playing = not self.scrubber.is_playing

            if res.frame_delta != 0:
                self.active_frame = max(
                    0,
                    min(
                        config.MAX_SIMULATION_STEPS - 1,
                        self.active_frame + res.frame_delta
                    )
                )

            if res.gen_delta != 0:
                target_gen: int = self.active_gen + res.gen_delta
                if 0 <= target_gen <= latest_gen:
                    self.active_gen = target_gen
                    self.episode = None

            if res.set_gen is not None and res.set_gen != self.active_gen:
                self.active_gen = res.set_gen
                self.episode = None

            if res.set_frame is not None:
                self.active_frame = res.set_frame

            if self.episode is None:
                if self.bridge._state.get("initialized"):
                    self.genome = self.bridge.fetch_genome()
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

            if self.scrubber is not None and self.scrubber.is_playing:
                self.active_frame += self.scrubber.playback_speed
                max_clip_step: int = max(1, self.episode.finish_step)
                self.active_frame = min(self.active_frame, max_clip_step)

                if self.active_frame >= max_clip_step:
                    if self.scrubber.repeat_all:
                        if self.active_gen < latest_gen:
                            self.active_gen += 1
                        else:
                            self.active_gen = 0
                        self.episode = None
                        continue

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
                profile_config=self.bridge.profile_config
            )
            self.panel.draw_panel(
                screen,
                run_data,
                self.active_frame,
                self.bridge.metrics,
                self.active_gen
            )
            self.graph.draw_graph(
                screen,
                run_data,
                self.active_frame,
                profile_config=self.bridge.profile_config
            )
            self.scrubber.draw_controls(
                screen,
                self.active_gen,
                latest_gen,
                self.bridge.metrics["num_generations"],
                self.active_frame,
                config.MAX_SIMULATION_STEPS,
                gen_history=self.bridge.gen_history,
                selected_cand_frames=run_data["frames"]
            )

            pygame.display.flip()
            clock.tick(config.FPS)

        pygame.quit()

    def _load_window_icon(self) -> None:
        """
        Loads icon.png from project root and sets Pygame window icon.
        """
        icon_path: Path = Path(__file__).resolve().parents[1] / "icon.png"
        if not icon_path.exists():
            return

        try:
            icon_surf: pygame.Surface = pygame.image.load(str(icon_path))
            pygame.display.set_icon(icon_surf)
        except Exception:
            pass

    def _make_run_map(self) -> Tuple[MapData, np.ndarray, int, float]:
        """
        Delegates single control-check display map creation to factory.
        """
        return self.display_map_factory.create_display_map()

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
            profile_config=self.bridge.profile_config,
            generation=self.active_gen
        )
        self.active_frame = 0

    def _draw_splash(self, screen: pygame.Surface) -> None:
        """
        Renders waiting screen before first model state arrives.
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
