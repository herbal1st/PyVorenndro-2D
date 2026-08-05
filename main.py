"""
Application entry point executing pre-training and launching visualizer GUI.
"""

import sys
import pygame

import config
from evolution.trainer import HeadlessTrainer
from visualization.viewport_grid import ViewportGrid
from visualization.network_graph import NetworkGraph
from visualization.timeline_scrubber import TimelineScrubber
from visualization.overlay_panel import OverlayPanel


def main() -> None:
    """
    Runs headless neuroevolution, then initializes Pygame interactive GUI.
    """
    # 1. Run Headless Training Loop & Pre-calculate Generations
    trainer = HeadlessTrainer()
    recorder = trainer.run_training_session()

    total_gens: int = len(recorder.generations_history)
    if total_gens == 0:
        print("[Error] No generation history recorded.")
        sys.exit(1)

    # 2. Initialize Pygame GUI Window
        # 2. Initialize Pygame GUI Window
    pygame.init()
    pygame.key.set_repeat(300, 50)
    screen: pygame.Surface = pygame.display.set_mode(
        (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    )
    pygame.display.set_caption(
        "PyVorenndro 2D - Neuroevolution Visualizer"
    )
    clock: pygame.time.Clock = pygame.time.Clock()

    # 3. Instantiate Layout Sub-Systems using config rectangles
    viewport_grid = ViewportGrid(config.LAYOUT_GRID_RECT)
    overlay_panel = OverlayPanel(config.LAYOUT_PANEL_RECT)
    network_graph = NetworkGraph(config.LAYOUT_GRAPH_RECT)
    timeline_scrubber = TimelineScrubber(config.LAYOUT_SCRUBBER_RECT)

    active_gen: int = 0
    active_frame: int = 0
    last_click_time: int = 0

    running: bool = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    viewport_grid.toggle_camera_mode()

                elif event.key == pygame.K_SPACE:
                    timeline_scrubber.is_playing = (
                        not timeline_scrubber.is_playing
                    )

                elif event.key == pygame.K_RIGHT:
                    active_frame = min(
                        active_frame +
                        config.SCRUBBER_ARROW_JUMP_FRAMES,
                        config.MAX_SIMULATION_STEPS - 1
                    )

                elif event.key == pygame.K_LEFT:
                    active_frame = max(
                        active_frame -
                        config.SCRUBBER_ARROW_JUMP_FRAMES,
                        0
                    )

                elif event.key == pygame.K_PAGEUP:
                    active_gen = min(
                        active_gen +
                        config.SCRUBBER_PAGE_JUMP_GENS,
                        total_gens - 1
                    )
                    active_frame = 0

                elif event.key == pygame.K_PAGEDOWN:
                    active_gen = max(
                        active_gen -
                        config.SCRUBBER_PAGE_JUMP_GENS,
                        0
                    )
                    active_frame = 0

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 2, 3, 4, 5):
                    now_ms: int = pygame.time.get_ticks()
                    is_double: bool = (
                        event.button == 1 and
                        (now_ms - last_click_time) < 300
                    )
                    if event.button == 1:
                        last_click_time = now_ms

                    m_pos = event.pos

                    if event.button == 1:
                        viewport_grid.handle_click(
                            m_pos, is_double_click=is_double
                        )

                    gen_data = recorder.get_generation_data(active_gen)
                    total_f = config.MAX_SIMULATION_STEPS

                    was_playing_before: bool = (
                        timeline_scrubber.is_playing
                    )
                    new_gen, new_frame = timeline_scrubber.handle_click(
                        m_pos,
                        total_gens,
                        total_f,
                        mouse_button=event.button
                    )

                    # Auto-restart if Play is pressed at end boundary
                    if (
                        not was_playing_before and
                        timeline_scrubber.is_playing and
                        active_frame >= total_f - 1
                    ):
                        if (
                            active_gen >= total_gens - 1 and
                            timeline_scrubber.repeat_all
                        ):
                            active_gen = 0
                        active_frame = 0

                    if new_gen is not None:
                        active_gen = new_gen
                        active_frame = 0

                    if new_frame is not None:
                        active_frame = new_frame

        gen_data = recorder.get_generation_data(active_gen)
        cand_idx = viewport_grid.selected_idx
        cand_frames = gen_data["candidate_frames"]

        safe_cand_idx: int = min(cand_idx, len(cand_frames) - 1)
        total_f = config.MAX_SIMULATION_STEPS
        selected_frames = (
            cand_frames[safe_cand_idx] if cand_frames else []
        )

        # Auto-play generation advancement & repeat mode handling
        if timeline_scrubber.is_playing:
            active_frame += timeline_scrubber.playback_speed
            if active_frame >= total_f:
                if timeline_scrubber.repeat_all:
                    if active_gen < total_gens - 1:
                        active_gen += 1
                        active_frame = 0
                    else:
                        active_gen = 0
                        active_frame = 0
                else:
                    active_frame = 0

        screen.fill(config.COLOR_BG)

        viewport_grid.draw_grid(screen, gen_data, active_frame)
        overlay_panel.draw_panel(
            screen,
            gen_data,
            safe_cand_idx,
            active_frame,
            config.MAX_SIMULATION_STEPS,
            total_gens
        )
        network_graph.draw_graph(
            screen, gen_data, safe_cand_idx, active_frame
        )
        timeline_scrubber.draw_controls(
            screen,
            active_gen,
            total_gens,
            active_frame,
            total_f,
            gen_history=recorder.generations_history,
            selected_cand_frames=selected_frames
        )

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
