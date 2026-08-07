"""
Interactive UI transport controls, dual timeline bars, and marker tags.
"""

from typing import Tuple, List, Optional, Dict, Any
import pygame

import config


class TimelineScrubber:
    """
    Renders Play/Pause/Repeat/Speed buttons, dual timelines, and markers.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_SCRUBBER_RECT
    ) -> None:
        """
        Initializes transport buttons, timeline bar rects, and state flags.
        """
        self.x, self.y, self.w, self.h = rect
        self.is_playing: bool = True
        self.repeat_all: bool = True  # True = REP ALL, False = REP GEN
        self.playback_speed: int = config.DEFAULT_PLAYBACK_SPEED
        self.speed_options: List[int] = [1, 2, 4, 8, 16]

        btn_h: int = config.HUD_SCRUBBER_BTN_HEIGHT
        self.btn_toggle_rect: pygame.Rect = pygame.Rect(
            self.x, self.y, 70, btn_h
        )
        self.btn_repeat_rect: pygame.Rect = pygame.Rect(
            self.x + 75, self.y, 70, btn_h
        )
        self.btn_speed_rect: pygame.Rect = pygame.Rect(
            self.x + 150, self.y, 50, btn_h
        )

        bar_x: int = self.x + 210
        bar_w: int = self.w - 210
        bar_h: int = config.HUD_SCRUBBER_BAR_HEIGHT
        self.frame_bar_rect: pygame.Rect = pygame.Rect(
            bar_x, self.y, bar_w, bar_h
        )
        self.gen_bar_rect: pygame.Rect = pygame.Rect(
            bar_x, self.y + bar_h + 8, bar_w, bar_h
        )

        self.font: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.HUD_SCRUBBER_BTN_FONT_SIZE, bold=True
        )

    def draw_controls(
        self,
        surface: pygame.Surface,
        active_gen: int,
        latest_gen: int,
        total_gens: int,
        active_frame: int,
        total_frames: int,
        gen_history: Optional[List[Dict[str, Any]]] = None,
        selected_cand_frames: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Renders transport buttons, dual timeline bars, and marker tick tags.
        """
        # Play / Pause toggle button
        t_color = (
            config.COLOR_BUTTON_ACTIVE if self.is_playing
            else config.COLOR_BUTTON
        )
        pygame.draw.rect(surface, t_color, self.btn_toggle_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, self.btn_toggle_rect, 1
        )
        t_str: str = "PAUSE" if self.is_playing else "PLAY"
        t_lbl = self.font.render(t_str, True, (255, 255, 255))
        surface.blit(
            t_lbl, t_lbl.get_rect(center=self.btn_toggle_rect.center)
        )

        # Repeat mode button (REP ALL / REP GEN)
        r_color = (
            config.COLOR_BUTTON_ACTIVE if self.repeat_all
            else config.COLOR_BUTTON
        )
        pygame.draw.rect(surface, r_color, self.btn_repeat_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, self.btn_repeat_rect, 1
        )
        r_str: str = "REP ALL" if self.repeat_all else "REP GEN"
        r_lbl = self.font.render(r_str, True, (255, 255, 255))
        surface.blit(
            r_lbl, r_lbl.get_rect(center=self.btn_repeat_rect.center)
        )

        # Speed toggle button
        pygame.draw.rect(surface, config.COLOR_BUTTON, self.btn_speed_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, self.btn_speed_rect, 1
        )
        sp_lbl = self.font.render(
            f"{self.playback_speed}x", True, (255, 255, 255)
        )
        surface.blit(
            sp_lbl, sp_lbl.get_rect(center=self.btn_speed_rect.center)
        )

        # Timeline Bars Background
        pygame.draw.rect(
            surface, config.COLOR_TIMELINE_BAR, self.frame_bar_rect
        )
        pygame.draw.rect(
            surface, config.COLOR_TIMELINE_BAR, self.gen_bar_rect
        )

        # Draw active completed progress fill on generation bar
        tot_g: int = max(1, total_gens - 1)
        progress_ratio: float = float(latest_gen) / float(tot_g)
        progress_w: int = int(progress_ratio * self.gen_bar_rect.w)
        if progress_w > 0:
            prog_rect = pygame.Rect(
                self.gen_bar_rect.x,
                self.gen_bar_rect.y,
                progress_w,
                self.gen_bar_rect.h
            )
            pygame.draw.rect(surface, config.COLOR_BUTTON_ACTIVE, prog_rect)

        # Draw green solve tick marks on generation timeline bar
        if gen_history:
            for g_idx in range(min(len(gen_history), latest_gen + 1)):
                g_data = gen_history[g_idx]
                if g_data.get("solved", False):
                    g_r: float = float(g_idx) / float(tot_g)
                    gx: int = int(
                        self.gen_bar_rect.x + (g_r * self.gen_bar_rect.w)
                    )
                    pygame.draw.line(
                        surface,
                        config.COLOR_EXIT,
                        (gx, self.gen_bar_rect.top),
                        (gx, self.gen_bar_rect.bottom),
                        2
                    )

        # Draw green finish tick mark on frame bar for selected run
        if selected_cand_frames:
            tot_f: int = max(1, total_frames - 1)
            for f_step, f_dict in enumerate(selected_cand_frames):
                if f_dict.get("reached_exit", False):
                    f_r: float = float(f_step) / float(tot_f)
                    fx: int = int(
                        self.frame_bar_rect.x + (f_r * self.frame_bar_rect.w)
                    )
                    pygame.draw.line(
                        surface,
                        config.COLOR_EXIT,
                        (fx, self.frame_bar_rect.top),
                        (fx, self.frame_bar_rect.bottom),
                        2
                    )
                    break

        marker_r: int = config.HUD_SCRUBBER_MARKER_RADIUS

        # Frame Timeline Marker
        f_ratio: float = (
            float(active_frame) / float(max(1, total_frames - 1))
        )
        f_marker_x: int = int(
            self.frame_bar_rect.x + (f_ratio * self.frame_bar_rect.w)
        )
        f_marker_center: Tuple[int, int] = (
            f_marker_x, self.frame_bar_rect.centery
        )
        pygame.draw.circle(
            surface, config.COLOR_MARKER, f_marker_center, marker_r
        )

        f_lbl = self.font.render(
            f"# {active_frame}", True, config.COLOR_MARKER
        )
        surface.blit(f_lbl, (f_marker_x - 12, self.frame_bar_rect.y - 14))

        # Generation Timeline Marker
        g_ratio: float = float(active_gen) / float(tot_g)
        g_marker_x: int = int(
            self.gen_bar_rect.x + (g_ratio * self.gen_bar_rect.w)
        )
        g_marker_center: Tuple[int, int] = (
            g_marker_x, self.gen_bar_rect.centery
        )
        pygame.draw.circle(
            surface, config.COLOR_MARKER, g_marker_center, marker_r
        )

        g_lbl = self.font.render(
            f"# {active_gen + 1}/{latest_gen + 1}", True, config.COLOR_MARKER
        )
        surface.blit(g_lbl, (g_marker_x - 12, self.gen_bar_rect.y + 18))

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        latest_gen: int,
        total_gens: int,
        total_frames: int,
        mouse_button: int = 1
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Processes transport button clicks and dual timeline bar scrubbing.
        """
        cx, cy = click_pos

        if self.btn_toggle_rect.collidepoint(cx, cy):
            self.is_playing = not self.is_playing
            return None, None

        if self.btn_repeat_rect.collidepoint(cx, cy):
            self.repeat_all = not self.repeat_all
            return None, None

        if self.btn_speed_rect.collidepoint(cx, cy):
            curr_idx: int = self.speed_options.index(self.playback_speed)
            if mouse_button == 3:  # Right Click = Speed Down
                next_idx: int = (curr_idx - 1) % len(self.speed_options)
            else:  # Left Click / Default = Speed Up
                next_idx = (curr_idx + 1) % len(self.speed_options)
            self.playback_speed = self.speed_options[next_idx]
            return None, None

        new_frame: Optional[int] = None
        new_gen: Optional[int] = None

        if self.frame_bar_rect.collidepoint(cx, cy):
            rel_x: float = float(cx - self.frame_bar_rect.x)
            ratio: float = max(
                0.0, min(1.0, rel_x / float(self.frame_bar_rect.w))
            )
            new_frame = int(round(ratio * (total_frames - 1)))

        if self.gen_bar_rect.collidepoint(cx, cy):
            rel_x = float(cx - self.gen_bar_rect.x)
            ratio = max(0.0, min(1.0, rel_x / float(self.gen_bar_rect.w)))
            raw_gen: int = int(round(ratio * (total_gens - 1)))
            new_gen = min(raw_gen, latest_gen)

        return new_gen, new_frame
