"""
Renders candidate viewport grid and single candidate zoom view.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import pygame

import config
from core.map_data import MapData
from perception.vision_arc import VisionArcSampler


class Viewport:
    """
    Renders candidate viewports, single candidate zoom, and camera modes.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT,
        rows: int = getattr(config, "VIEWPORT_GRID_ROWS", 4),
        cols: int = getattr(config, "VIEWPORT_GRID_COLS", 4),
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes viewport dimensions, font cache, and alpha scratchpad.
        """
        self.x, self.y, self.w, self.h = rect
        self.rows: int = max(1, min(8, rows))
        self.cols: int = max(1, min(8, cols))
        self.selected_idx: int = 0
        self.is_zoomed: bool = True
        self.is_player_centered: bool = False
        self.profile_config: Optional[Dict[str, Any]] = profile_config
        self.sampler: VisionArcSampler = VisionArcSampler(
            profile_config=profile_config
        )

        self._font_cache: Dict[int, pygame.font.Font] = {}
        self._map_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}
        self._arc_scratchpad: pygame.Surface = pygame.Surface(
            (self.w, self.h), pygame.SRCALPHA
        )

    def set_profile_config(self, profile_config: Dict[str, Any]) -> None:
        """
        Updates active profile configuration and sampler angles.
        """
        self.profile_config = profile_config
        self.sampler = VisionArcSampler(profile_config=profile_config)

    def toggle_camera_mode(self) -> None:
        """
        Toggles between Map-Centered and Player-Centered tracking views.
        """
        self.is_player_centered = not self.is_player_centered

    def toggle_zoom(self) -> None:
        """
        Toggles between single-candidate full view and multi-grid view.
        """
        self.is_zoomed = not self.is_zoomed

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
            self.x <= cx <= self.x + self.w
            and self.y <= cy <= self.y + self.h
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

    def draw(
        self,
        surface: pygame.Surface,
        run_data: Dict[str, Any],
        active_step: int,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Renders single zoomed view or multi-candidate grid viewports.
        """
        if profile_config is not None:
            self.set_profile_config(profile_config)

        map_data: MapData = run_data["map_data"]
        frames: List[Dict[str, Any]] = run_data["frames"]

        if not frames:
            return

        if self.is_zoomed:
            self._draw_single_viewport(
                surface,
                map_data,
                run_data,
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
                sub_rect: Tuple[int, int, int, int] = (
                    self.x + (col * sub_w),
                    self.y + (row * sub_h),
                    sub_w,
                    sub_h
                )
                is_sel: bool = (idx == self.selected_idx)
                self._draw_single_viewport(
                    surface,
                    map_data,
                    run_data,
                    active_step,
                    sub_rect,
                    is_selected=is_sel
                )

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
        run_idx: int,
        rw: int,
        rh: int
    ) -> pygame.Surface:
        """
        Retrieves cached pre-rendered map surface using unique map_id.
        """
        cache_key: Tuple[int, int, int] = (map_data.map_id, rw, rh)
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

    def _draw_single_viewport(
        self,
        surface: pygame.Surface,
        map_data: MapData,
        run_data: Dict[str, Any],
        active_step: int,
        rect: Tuple[int, int, int, int],
        is_selected: bool
    ) -> None:
        """
        Renders a single viewport section with candidate body and vision arc.
        """
        rx, ry, rw, rh = rect

        p_cfg: Dict[str, Any] = self.profile_config or {}
        kin: Dict[str, Any] = p_cfg.get("kinematics", {})
        sens: Dict[str, Any] = p_cfg.get("sensory", {})

        radius_ratio: float = float(kin.get("player_radius_ratio", 0.5))
        max_dist: float = float(sens.get("goal_sensor_max_range", 5.0))
        if max_dist <= 0.0:
            max_dist = 5.0

        frames: List[Dict[str, Any]] = run_data["frames"]
        frame_idx: int = min(active_step, len(frames) - 1)
        curr_frame: Dict[str, Any] = frames[frame_idx]

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)

        pygame.draw.rect(surface, config.COLOR_BG, rect)

        if self.is_zoomed:
            ui_scale: float = 1.0
        else:
            base_sub_w: float = float(self.w) / float(self.cols)
            ui_scale = (float(rw) / base_sub_w) * 0.5

        font_norm = self._get_cached_font(max(10, int(12 * ui_scale)))
        font_small = self._get_cached_font(max(8, int(10 * ui_scale)))

        cx: float = float(curr_frame["x"])
        cy: float = float(curr_frame["y"])
        heading: float = float(curr_frame["heading"])
        gen_num: int = int(run_data.get("generation", 0)) + 1
        cand_idx: int = int(run_data.get("cand_idx", 0))

        if self.is_player_centered:
            player_zoom: float = getattr(config, "PLAYER_CAMERA_ZOOM", 1.5)
            tile_size: float = (
                config.TILE_SIZE * player_zoom * ui_scale
            )
            center_px: float = float(rx) + (float(rw) / 2.0)
            center_py: float = float(ry) + (float(rh) / 2.0)

            def get_tile_rect(tx: int, ty: int) -> Tuple[int, int, int, int]:
                t_x: int = int(round(center_px + (float(tx) - cx) * tile_size))
                t_y: int = int(round(center_py + (float(ty) - cy) * tile_size))
                return (t_x, t_y, int(tile_size) + 1, int(tile_size) + 1)

            origin_pixel: Tuple[int, int] = (
                int(round(center_px)), int(round(center_py))
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
                map_data, gen_num, rw, rh
            )
            surface.blit(bg_surf, (rx, ry))

        cone_points: List[Tuple[int, int]] = [origin_pixel]
        for rel_angle in self.sampler.relative_angles:
            ray_angle: float = heading + rel_angle
            wall_prox, _ = self.sampler._cast_single_ray(
                cx, cy, ray_angle, map_data
            )
            dist_tiles: float = (1.0 - wall_prox) * max_dist
            ex: float = cx + (math.cos(ray_angle) * dist_tiles)
            ey: float = cy + (math.sin(ray_angle) * dist_tiles)

            if self.is_player_centered:
                px_e: int = int(round(center_px + (ex - cx) * tile_size))
                py_e: int = int(round(center_py + (ey - cy) * tile_size))
            else:
                px_e = int(rx + (ex * tile_size))
                py_e = int(ry + (ey * tile_size))

            cone_points.append((px_e, py_e))

        self._arc_scratchpad.fill((0, 0, 0, 0))

        if len(cone_points) > 2:
            rel_points = [(pt[0] - rx, pt[1] - ry) for pt in cone_points]
            pygame.draw.polygon(
                self._arc_scratchpad, config.COLOR_VISION_ARC, rel_points
            )

        head_line_len: float = config.PLAYER_HEADING_LINE_LENGTH * tile_size
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

        px, py = origin_pixel
        p_radius: int = max(
            3, int(tile_size * 0.5 * radius_ratio)
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

        idx_surf = font_norm.render(
            f"GEN #{gen_num} | AGENT #{cand_idx}",
            True,
            config.COLOR_PLAYER_HIGHLIGHT
        )
        surface.blit(idx_surf, (rx + 4, ry + 4))

        health_val: float = float(curr_frame.get("health", 1.0))
        pct_val: float = health_val * 100.0

        if health_val >= 0.50:
            hp_color = config.COLOR_HEALTH_FULL
        elif health_val >= 0.20:
            hp_color = config.COLOR_HEALTH_MID
        else:
            hp_color = config.COLOR_HEALTH_LOW

        bar_w: int = int(rw * 0.15 * ui_scale)
        bar_h: int = max(4, int(6 * ui_scale))
        bar_x: int = rx + rw - bar_w - int(6 * ui_scale)
        bar_y: int = ry + int(6 * ui_scale)

        fill_w: int = int(bar_w * health_val)
        bar_fill_x: int = bar_x + (bar_w - fill_w)

        pct_surf = font_small.render(f"{pct_val:5.1f}%", True, hp_color)
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
