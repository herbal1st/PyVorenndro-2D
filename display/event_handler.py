"""
User input event listener and interaction dispatcher for display runner.
"""

from typing import Optional
import pygame

import config
from display.timeline_scrubber import TimelineScrubber
from display.viewport import Viewport


class EventResult:
    """
    Data container holding action flags generated during event processing.
    """

    def __init__(self) -> None:
        """
        Initializes action flags and target frame/generation updates.
        """
        self.should_quit: bool = False
        self.toggle_camera: bool = False
        self.toggle_play: bool = False
        self.frame_delta: int = 0
        self.gen_delta: int = 0
        self.set_gen: Optional[int] = None
        self.set_frame: Optional[int] = None


class DisplayEventHandler:
    """
    Translates Pygame keyboard and mouse input events into UI state actions.
    """

    def __init__(self) -> None:
        """
        Initializes double-click timer threshold tracking.
        """
        self._last_click_time: int = 0

    def process_events(
        self,
        viewport: Optional[Viewport],
        scrubber: Optional[TimelineScrubber],
        active_gen: int,
        latest_gen: int,
        total_gens: int
    ) -> EventResult:
        """
        Polls Pygame event queue and dispatches interactive input commands.
        """
        result: EventResult = EventResult()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result.should_quit = True

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key, result)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mousedown(
                    event,
                    viewport,
                    scrubber,
                    latest_gen,
                    total_gens,
                    result
                )

        return result

    def _handle_keydown(
        self,
        key: int,
        result: EventResult
    ) -> None:
        """
        Maps keypress bindings to action commands on EventResult.
        """
        if key == pygame.K_ESCAPE:
            result.should_quit = True
        elif key == pygame.K_RETURN:
            result.toggle_camera = True
        elif key == pygame.K_SPACE:
            result.toggle_play = True
        elif key == pygame.K_LEFT:
            result.frame_delta = -25
        elif key == pygame.K_RIGHT:
            result.frame_delta = 25
        elif key == pygame.K_PAGEUP:
            result.gen_delta = 1
        elif key == pygame.K_PAGEDOWN:
            result.gen_delta = -1

    def _handle_mousedown(
        self,
        event: pygame.event.Event,
        viewport: Optional[Viewport],
        scrubber: Optional[TimelineScrubber],
        latest_gen: int,
        total_gens: int,
        result: EventResult
    ) -> None:
        """
        Processes viewport clicks and timeline scrubber scrubbing clicks.
        """
        now_ms: int = pygame.time.get_ticks()
        is_double: bool = (
            event.button == 1
            and (now_ms - self._last_click_time) < 300
        )
        if event.button == 1:
            self._last_click_time = now_ms

        if viewport is not None:
            viewport.handle_click(
                event.pos,
                is_double_click=is_double
            )

        if scrubber is not None:
            new_gen, new_frame = scrubber.handle_click(
                event.pos,
                latest_gen,
                total_gens,
                config.MAX_SIMULATION_STEPS,
                mouse_button=event.button
            )
            if new_gen is not None:
                result.set_gen = new_gen
            if new_frame is not None:
                result.set_frame = new_frame
