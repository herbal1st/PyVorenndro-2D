"""
Renders self-adjusting candidate viewport grid and candidate zoom mode.
"""

import math
from typing import Tuple, Dict, Any, List, Optional
import pygame

import config
from core.map_data import MapData
from perception.vision_arc import VisionArcSampler


class ViewportGrid:
    """
    Renders candidate grid viewports, single-click selection, & zoom mode.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT,
        rows: int = config.GRID_ROWS,
        cols: int = config.GRID_COLS
    ) -> None:
        """
        Initializes viewport rect, grid bounds, font cache, & surface buffers.
        """
        self.x, self.y, self.w, self.h = rect
        self.rows: int = max(1, min(8, rows))
        self.cols: int = max(1, min(8, cols))
        self.selected_idx: int = 0
        self.is_zoomed: bool = False
        self.is_player_centered: bool = False
        self.sampler: VisionArcSampler = VisionArcSampler()

        self._font_cache: Dict[int, pygame.font.Font] = {}
        self._map_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}
        self._arc_scratchpad: pygame.Surface = pygame.Surface(
            (self.w, self.h), pygame.SRCALPHA
        )

    def toggle_camera_mode(self) -> None:
        """
        Toggles between Map-Centered and Player-Centered tracking views.
        """
        self.is_player_centered = not self.is_player_centered

    def draw_grid(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        active_step: int
    ) -> None:
        """
        Renders sub-viewports for all candidate maps or single zoomed view.
        """
        map_w: int = gen_data.get(
            "map_width", gen_data.get("map_size", 20)
        )
        map_h: int = gen_data.get(
            "map_height", gen_data.get("map_size", 20)
        )
        map_data: MapData = MapData(
            map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"]
        )
        map_data.decode_bitmask(gen_data["bitmask_chunks"])

        if self.is_zoomed:
            self._draw_single_candidate_viewport(
                surface,
                map_data,
                gen_data,
                self.selected_idx,
                active_step,
                (self.x, self.y, self.w, self.h),
                is_selected=True
            )
            return

        sub_w: int = self.w // self.cols
        sub_h: int = self.h // self.rows

        for row in range(self.rows):
            for col in range(self.cols):
                idx: int = (row * self.cols) + col
                cand_frames = gen_data["candidate_frames"]
                if idx >= len(cand_frames):
                    continue

                sub_rect: Tuple[int, int, int, int] = (
                    self.x + (col * sub_w),
                    self.y + (row * sub_h),
                    sub_w,
                    sub_h
                )
                is_sel: bool = (idx == self.selected_idx)
                self._draw_single_candidate_viewport(
                    surface,
                    map_data,
                    gen_data,
                    idx,
                    active_step,
                    sub_rect,
                    is_selected=is_sel
                )

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        is_double_click: bool = False
    ) -> bool:
        """
        Processes single-click selection and double-click zoom toggles.
        """
        cx, cy = click_pos
        if not (
            self.x <= cx <= self.x + self.w and
            self.y <= cy <= self.y + self.h
        ):
            return False

        if self.is_zoomed:
            if is_double_click:
                self.is_zoomed = False
            return True

        sub_w: int = self.w // self.cols
        sub_h: int = self.h // self.rows

        col: int = (cx - self.x) // sub_w
        row: int = (cy - self.y) // sub_h

        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))

        clicked_idx: int = (row * self.cols) + col
        self.selected_idx = clicked_idx

        if is_double_click:
            self.is_zoomed = True

        return True

    def _get_cached_font(self, font_size: int) -> pygame.font.Font:
        """
        Retrieves cached font instance or creates new one for point size.
        """
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        font: pygame.font.Font = pygame.font.SysFont(
            "monospace", font_size, bold=True
        )
        self._font_cache[font_size] = font
        return font

    def _get_rendered_map_surface(
        self,
        map_data: MapData,
        gen_idx: int,
        rw: int,
        rh: int
    ) -> pygame.Surface:
        """
        Retrieves cached pre-rendered map surface or renders background once.
        """
        cache_key: Tuple[int, int, int] = (gen_idx, rw, rh)
        if cache_key in self._map_cache:
            return self._map_cache[cache_key]

        map_surf: pygame.Surface = pygame.Surface((rw, rh))
        map_surf.fill(config.COLOR_BG)

        tile_size: float = min(
            float(rw) / float(map_data.width),
            float(rh) / float(map_data.height)
        )

        for y in range(map_data.height):
            for x in range(map_data.width):
                t_x: int = int(float(x) * tile_size)
                t_y: int = int(float(y) * tile_size)
                t_rect: Tuple[int, int, int, int] = (
                    t_x, t_y, int(tile_size) + 1, int(tile_size) + 1
                )

                if (x, y) == map_data.start_pos:
                    pygame.draw.rect(map_surf, config.COLOR_START, t_rect)
                elif (x, y) == map_data.exit_pos:
                    pygame.draw.rect(map_surf, config.COLOR_EXIT, t_rect)
                elif map_data.is_wall(x, y):
                    pygame.draw.rect(map_surf, config.COLOR_WALL, t_rect)
                    pygame.draw.rect(
                        map_surf, config.COLOR_WALL_BORDER, t_rect, 1
                    )
                else:
                    pygame.draw.rect(map_surf, config.COLOR_FLOOR, t_rect)
                    pygame.draw.rect(
                        map_surf, config.COLOR_FLOOR_BORDER, t_rect, 1
                    )

        self._map_cache[cache_key] = map_surf
        return map_surf

    def _draw_single_candidate_viewport(
        self,
        surface: pygame.Surface,
        map_data: MapData,
        gen_data: Dict[str, Any],
        cand_idx: int,
        active_step: int,
        rect: Tuple[int, int, int, int],
        is_selected: bool
    ) -> None:
        """
        Renders a single candidate sub-viewport with vision arc and body.
        """
        rx, ry, rw, rh = rect

        frames: List[Dict[str, Any]] = gen_data["candidate_frames"][cand_idx]
        if not frames:
            return

        frame_idx: int = min(active_step, len(frames) - 1)
        curr_frame: Dict[str, Any] = frames[frame_idx]

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)

        pygame.draw.rect(surface, config.COLOR_BG, rect)

        if self.is_zoomed:
            base_sub_w: float = float(self.w) / float(self.cols)
            ui_scale: float = (float(rw) / base_sub_w) * 0.5
        else:
            ui_scale = 1.0

        font_norm = self._get_cached_font(max(10, int(12 * ui_scale)))
        font_small = self._get_cached_font(max(8, int(10 * ui_scale)))

        cx: float = float(curr_frame["x"])
        cy: float = float(curr_frame["y"])
        heading: float = float(curr_frame["heading"])
        gen_idx: int = int(gen_data.get("generation", 0))

        if self.is_player_centered:
            grid_multiplier: float = (
                float(max(self.rows, self.cols)) if self.is_zoomed else 1.0
            )
            tile_size: float = (
                config.TILE_SIZE * config.PLAYER_CAMERA_ZOOM * grid_multiplier
            )
            center_px: float = float(rx) + (float(rw) / 2.0)
            center_py: float = float(ry) + (float(rh) / 2.0)

            def get_tile_rect(tx: int, ty: int) -> Tuple[int, int, int, int]:
                t_x: int = int(round(center_px + (float(tx) - cx) * tile_size))
                t_y: int = int(round(center_py + (float(ty) - cy) * tile_size))
                return (t_x, t_y, int(tile_size) + 1, int(tile_size) + 1)

            origin_pixel: Tuple[int, int] = (
                int(round(center_px)),
                int(round(center_py))
            )

            for y in range(map_data.height):
                for x in range(map_data.width):
                    t_rect = get_tile_rect(x, y)
                    if (
                        t_rect[0] + t_rect[2] < rx or
                        t_rect[0] > rx + rw or
                        t_rect[1] + t_rect[3] < ry or
                        t_rect[1] > ry + rh
                    ):
                        continue

                    if (x, y) == map_data.start_pos:
                        pygame.draw.rect(surface, config.COLOR_START, t_rect)
                    elif (x, y) == map_data.exit_pos:
                        pygame.draw.rect(surface, config.COLOR_EXIT, t_rect)
                    elif map_data.is_wall(x, y):
                        pygame.draw.rect(surface, config.COLOR_WALL, t_rect)
                        pygame.draw.rect(
                            surface, config.COLOR_WALL_BORDER, t_rect, 1
                        )
                    else:
                        pygame.draw.rect(surface, config.COLOR_FLOOR, t_rect)
                        pygame.draw.rect(
                            surface, config.COLOR_FLOOR_BORDER, t_rect, 1
                        )
        else:
            tile_size = min(
                float(rw) / float(map_data.width),
                float(rh) / float(map_data.height)
            )

            origin_pixel = (
                int(rx + (cx * tile_size)),
                int(ry + (cy * tile_size))
            )

            bg_surf = self._get_rendered_map_surface(
                map_data, gen_idx, rw, rh
            )
            surface.blit(bg_surf, (rx, ry))

        # Sample vision arc fan points
        cone_points: List[Tuple[int, int]] = [origin_pixel]
        for rel_angle in self.sampler.relative_angles:
            ray_angle: float = heading + rel_angle
            wall_prox, _ = self.sampler._cast_single_ray(
                cx, cy, ray_angle, map_data
            )
            dist_tiles: float = (1.0 - wall_prox) * config.VISION_MAX_DIST
            ex: float = cx + (math.cos(ray_angle) * dist_tiles)
            ey: float = cy + (math.sin(ray_angle) * dist_tiles)

            if self.is_player_centered:
                px_e: int = int(
                    round(center_px + (ex - cx) * tile_size)
                )
                py_e: int = int(
                    round(center_py + (ey - cy) * tile_size)
                )
            else:
                px_e = int(rx + (ex * tile_size))
                py_e = int(ry + (ey * tile_size))

            cone_points.append((px_e, py_e))

        # Render vision fan and translucent heading line onto alpha scratchpad
        self._arc_scratchpad.fill((0, 0, 0, 0))

        if len(cone_points) > 2:
            rel_points = [
                (pt[0] - rx, pt[1] - ry) for pt in cone_points
            ]
            pygame.draw.polygon(
                self._arc_scratchpad, config.COLOR_VISION_ARC, rel_points
            )

        head_line_len: float = (
            config.PLAYER_HEADING_LINE_LENGTH * tile_size
        )
        rel_origin: Tuple[int, int] = (
            origin_pixel[0] - rx, origin_pixel[1] - ry
        )
        hx: int = int(rel_origin[0] + (math.cos(heading) * head_line_len))
        hy: int = int(rel_origin[1] + (math.sin(heading) * head_line_len))
        pygame.draw.line(
            self._arc_scratchpad,
            config.COLOR_PLAYER_HEADING_LINE,
            rel_origin,
            (hx, hy),
            config.PLAYER_HEADING_LINE_WIDTH
        )

        surface.blit(
            self._arc_scratchpad, (rx, ry), area=pygame.Rect(0, 0, rw, rh)
        )

        # Draw candidate body circle scaled by config.PLAYER_RADIUS_RATIO
        px, py = origin_pixel
        p_radius: int = max(
            3, int(tile_size * 0.5 * config.PLAYER_RADIUS_RATIO)
        )

        body_color = (
            config.COLOR_PLAYER_HIGHLIGHT if is_selected
            else config.COLOR_PLAYER
        )
        pygame.draw.circle(surface, body_color, (px, py), p_radius)

        face_str: str = curr_frame["face"]
        raw_face = font_norm.render(
            face_str, True, config.COLOR_PLAYER_TEXT
        )
        target_side: int = max(
            2, int(p_radius * 2 * config.PLAYER_FACE_TEXT_SCALE)
        )
        scaled_face = pygame.transform.smoothscale(
            raw_face, (target_side, target_side)
        )
        f_rect = scaled_face.get_rect(center=(px, py))
        surface.blit(scaled_face, f_rect)

        # Draw candidate index label (#0, #1, ...) in top-left
        idx_surf = font_norm.render(
            f"#{cand_idx}", True, config.COLOR_PLAYER_HIGHLIGHT
        )
        surface.blit(idx_surf, (rx + 4, ry + 4))

        # Draw Right-Anchored Health Bar & Percentage Label in top-right
        health_val: float = float(curr_frame.get("health", 1.0))
        pct_val: float = health_val * 100.0

        if health_val >= 0.50:
            hp_color = config.COLOR_HEALTH_FULL
        elif health_val >= 0.20:
            hp_color = config.COLOR_HEALTH_MID
        else:
            hp_color = config.COLOR_HEALTH_LOW

        bar_w: int = int(rw * 0.20 * ui_scale)
        bar_h: int = max(4, int(6 * ui_scale))
        bar_x: int = rx + rw - bar_w - int(6 * ui_scale)
        bar_y: int = ry + int(6 * ui_scale)

        fill_w: int = int(bar_w * health_val)
        bar_fill_x: int = bar_x + (bar_w - fill_w)

        pct_surf = font_small.render(
            f"{pct_val:5.1f}%", True, hp_color
        )
        pct_rect = pct_surf.get_rect(
            midright=(bar_x - 4, bar_y + (bar_h // 2))
        )
        surface.blit(pct_surf, pct_rect)

        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, (bar_x, bar_y, bar_w, bar_h)
        )
        if fill_w > 0:
            pygame.draw.rect(
                surface, hp_color, (bar_fill_x, bar_y, fill_w, bar_h)
            )

        surface.set_clip(None)

        reached_exit: bool = bool(curr_frame.get("reached_exit", False))
        is_alive: bool = bool(curr_frame.get("is_alive", True))

        if reached_exit:
            inset_rect: Tuple[int, int, int, int] = (
                rx + 2, ry + 2, rw - 4, rh - 4
            )
            pygame.draw.rect(
                surface, config.COLOR_FRAME_SOLVED, inset_rect, 2
            )
        elif not is_alive:
            inset_rect = (rx + 2, ry + 2, rw - 4, rh - 4)
            pygame.draw.rect(
                surface, config.COLOR_FRAME_DEAD, inset_rect, 2
            )

        border_color = (
            config.COLOR_PLAYER_HIGHLIGHT if is_selected
            else config.COLOR_WALL_BORDER
        )
        border_width: int = 3 if is_selected else 1
        pygame.draw.rect(surface, border_color, rect, border_width)
