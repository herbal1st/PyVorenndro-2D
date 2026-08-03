"""
Renders header title, telemetry dashboard, and active candidate information.
"""

from typing import Tuple, Dict, Any, List
import pygame

import config


class OverlayPanel:
    """
    Renders dashboard headers, active scores, and winner callouts in 2 columns.
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
        gen_data: Dict[str, Any],
        active_cand_idx: int,
        active_step: int,
        total_steps: int,
        total_gens: int
    ) -> None:
        """
        Renders title header and 2-column scaled dashboard metrics.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h), 1
        )

        # Title header in Light Blue
        title_text: str = "PYVORENNDRO 2D - NEUROEVOLUTION"
        title_surf = self.title_font.render(
            title_text, True, config.COLOR_START
        )
        surface.blit(title_surf, (self.x + 12, self.y + 10))

        gen_num: int = gen_data["generation"] + 1
        raw_scores: List[float] = gen_data["raw_scores"]
        top_score: float = max(raw_scores)
        avg_score: float = sum(raw_scores) / float(len(raw_scores))
        winner_idx: int = gen_data["winner_index"]
        max_f: int = total_frames_limit(gen_data)

        # Column X coordinates and Row Y coordinates
        col1_x: int = self.x + 12
        col2_x: int = self.x + 215
        row_y1: int = self.y + 40
        row_y2: int = self.y + 70
        row_y3: int = self.y + 100

        # Left Column Metrics
        lbl_sel = self.body_font.render(
            f"SELECTED : #{active_cand_idx}",
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
            f"GENERATION: {gen_num}/{total_gens}",
            True,
            (255, 255, 255)
        )
        surface.blit(lbl_gen, (col1_x, row_y3))

        # Right Column Metrics
        lbl_win = self.body_font.render(
            f"WINNER   : #{winner_idx}",
            True,
            config.COLOR_EXIT
        )
        surface.blit(lbl_win, (col2_x, row_y1))

        lbl_top = self.body_font.render(
            f"TOP SCORE : {top_score:.1f}",
            True,
            config.COLOR_EXIT
        )
        surface.blit(lbl_top, (col2_x, row_y2))

        lbl_avg = self.body_font.render(
            f"AVG SCORE : {avg_score:.1f}",
            True,
            (200, 200, 200)
        )
        surface.blit(lbl_avg, (col2_x, row_y3))


def total_frames_limit(gen_data: Dict[str, Any]) -> int:
    """
    Returns max frames recorded across all candidates in a generation.
    """
    frames_list = gen_data["candidate_frames"]
    if not frames_list:
        return 0
    return max(len(c_frames) for c_frames in frames_list)
