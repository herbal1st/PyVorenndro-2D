"""
Renders header title, telemetry dashboard, and live run information.
"""

from typing import Tuple, Dict, Any, List
import pygame

import config


class OverlayPanel:
    """
    Renders dashboard headers, active run stats, and trainer metrics.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_PANEL_RECT
    ) -> None:
        """
        Initializes bounding rect and font properties.
        """
        self.x, self.y, self.w, self.h = rect
        self.title_font: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.HUD_PANEL_TITLE_FONT_SIZE, bold=True
        )
        self.body_font: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.HUD_PANEL_BODY_FONT_SIZE, bold=True
        )

    def draw_panel(
        self,
        surface: pygame.Surface,
        run_data: Dict[str, Any],
        active_step: int,
        metrics: Dict[str, Any],
        active_gen: int = 0
    ) -> None:
        """
        Renders title header and 2-column live telemetry metrics.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h), 1
        )

        # Title header in Light Blue
        title_text: str = "PYVORENNDRO 2D - LIVE AGENT"
        title_surf = self.title_font.render(
            title_text, True, config.COLOR_START
        )
        surface.blit(title_surf, (self.x + 12, self.y + 10))

        cand_idx: int = int(run_data.get("cand_idx", 0))
        max_f: int = int(
            run_data.get("max_steps", config.MAX_SIMULATION_STEPS)
        )
        gen_num: int = active_gen + 1
        best_fitness: float = float(metrics.get("best_fitness", 0.0))
        gen_fitness: float = float(metrics.get("gen_fitness", 0.0))

        # Column X coordinates and Row Y coordinates
        col1_x: int = self.x + 12
        col2_x: int = self.x + 215
        row_y1: int = self.y + 40
        row_y2: int = self.y + 70
        row_y3: int = self.y + 100

        # Left Column Metrics
        lbl_sel = self.body_font.render(
            f"GEN      : #{gen_num}",
            True,
            config.COLOR_PLAYER_HIGHLIGHT
        )
        surface.blit(lbl_sel, (col1_x, row_y1))

        lbl_step = self.body_font.render(
            f"FRAME STEP: {active_step}/{max_f}",
            True,
            (255, 255, 255)
        )
        surface.blit(lbl_step, (col1_x, row_y2))

        lbl_gen = self.body_font.render(
            f"AGENT    : #{cand_idx}",
            True,
            (255, 255, 255)
        )
        surface.blit(lbl_gen, (col1_x, row_y3))

        # Right Column Metrics
        lbl_win = self.body_font.render(
            f"WINNER   : #{gen_num if best_fitness > 0 else 0}",
            True,
            config.COLOR_EXIT
        )
        surface.blit(lbl_win, (col2_x, row_y1))

        lbl_top = self.body_font.render(
            f"TOP SCORE : {best_fitness:.1f}",
            True,
            config.COLOR_EXIT
        )
        surface.blit(lbl_top, (col2_x, row_y2))

        lbl_avg = self.body_font.render(
            f"AVG SCORE : {gen_fitness:.1f}",
            True,
            (200, 200, 200)
        )
        surface.blit(lbl_avg, (col2_x, row_y3))
