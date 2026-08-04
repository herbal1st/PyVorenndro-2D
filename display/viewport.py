"""
Renders the single live candidate viewport with camera tracking and zoom.

The old multi-view grid is gone: the display runner shows one champion run at
a time in a single large viewport. ENTER toggles map/player-centred camera;
double-click toggles zoom.
"""

import math
from typing import Tuple, Dict, Any, List
import pygame

import config
from core.map_data import MapData
from perception.vision_arc import VisionArcSampler


class Viewport:
    """
    Renders a single candidate's live run with vision arc and health HUD.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT
    ) -> None:
        """
        Initializes viewport rect, font cache, and alpha scratchpad buffers.
        """
        self.x, self.y, self.w, self.h = rect
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

    def toggle_zoom(self) -> None:
        """
        Toggles between fit-to-map and zoomed-in tile scaling.
        """
        self.is_zoomed = not self.is_zoomed

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        is_double_click: bool = False
    ) -> bool:
        """
        Processes double-click zoom toggling within the viewport rect.
        """
        cx, cy = click_pos
        if not (
            self.x <= cx <= self.x + self.w and
            self.y <= cy <= self.y + self.h
        ):
            return False

        if is_double_click:
            self.toggle_zoom()

        return True

    def draw(
        self,
        surface: pygame.Surface,
        run_data: Dict[str, Any],
        active_step: int
    ) -> None:
        """
        Renders the live run: map, candidate body, vision arc, and HUD.
        """
        map_data: MapData = run_data["map_data"]
        frames: List[Dict[str, Any]] = run_data["frames"]

        if not frames:
            return

        frame_idx: int = min(active_step, len(frames) - 1)
        curr_frame: Dict[str, Any] = frames[frame_idx]

        clip_rect = pygame.Rect(self.x, self.y, self.w, self.h)
        surface.set_clip(clip_rect)

        pygame.draw.rect(surface, config.COLOR_BG, (self.x, self.y, self.w, self.h))

        zoom_scale: float = 2.0 if self.is_zoomed else 1.0

        font_norm = self._get_cached_font(max(10, int(12 * zoom_scale)))
        font_small = self._get_cached_font(max(8, int(10 * zoom_scale)))

        cx: float = float(curr_frame["x"])
        cy: float = float(curr_frame["y"])
        heading: float = float(curr_frame["heading"])
        gen_idx: int = int(run_data.get("run_number", 0))

        if self.is_player_centered:
            tile_size: float = (
                config.TILE_SIZE
                * config.PLAYER_CAMERA_ZOOM
                * zoom_scale
            )
            center_px: float = float(self.x) + (float(self.w) / 2.0)
            center_py: float = float(self.y) + (float(self.h) / 2.0)

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
                        t_rect[0] + t_rect[2] < self.x or
                        t_rect[0] > self.x + self.w or
                        t_rect[1] + t_rect[3] < self.y or
                        t_rect[1] > self.y + self.h
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
                float(self.w) / float(map_data.width),
                float(self.h) / float(map_data.height)
            ) * zoom_scale

            origin_pixel = (
                int(self.x + (cx * tile_size)),
                int(self.y + (cy * tile_size))
            )

            if not self.is_zoomed:
                bg_surf = self._get_rendered_map_surface(
                    map_data, gen_idx, self.w, self.h
                )
                surface.blit(bg_surf, (self.x, self.y))
            else:
                self._draw_zoomed_map_tiles(
                    surface,
                    map_data,
                    tile_size,
                    self.x,
                    self.y
                )

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
                px_e = int(self.x + (ex * tile_size))
                py_e = int(self.y + (ey * tile_size))

            cone_points.append((px_e, py_e))

        # Render vision fan and translucent heading line onto alpha scratchpad
        self._arc_scratchpad.fill((0, 0, 0, 0))

        if len(cone_points) > 2:
            rel_points = [
                (pt[0] - self.x, pt[1] - self.y) for pt in cone_points
            ]
            pygame.draw.polygon(
                self._arc_scratchpad, config.COLOR_VISION_ARC, rel_points
            )

        head_line_len: float = (
            config.PLAYER_HEADING_LINE_LENGTH * tile_size
        )
        rel_origin: Tuple[int, int] = (
            origin_pixel[0] - self.x, origin_pixel[1] - self.y
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
            self._arc_scratchpad,
            (self.x, self.y),
            area=pygame.Rect(0, 0, self.w, self.h)
        )

        # Draw candidate body circle scaled by config.PLAYER_RADIUS_RATIO
        px, py = origin_pixel
        p_radius: int = max(
            3, int(tile_size * 0.5 * config.PLAYER_RADIUS_RATIO)
        )

        pygame.draw.circle(surface, config.COLOR_PLAYER, (px, py), p_radius)

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

        # Draw run index label in top-left
        idx_surf = font_norm.render(
            f"RUN #{run_data.get('run_number', 0)}",
            True,
            config.COLOR_PLAYER_HIGHLIGHT
        )
        surface.blit(idx_surf, (self.x + 4, self.y + 4))

        # Draw Right-Anchored Health Bar & Percentage Label in top-right
        health_val: float = float(curr_frame.get("health", 1.0))
        pct_val: float = health_val * 100.0

        if health_val >= 0.50:
            hp_color = config.COLOR_HEALTH_FULL
        elif health_val >= 0.20:
            hp_color = config.COLOR_HEALTH_MID
        else:
            hp_color = config.COLOR_HEALTH_LOW

        bar_w: int = int(self.w * 0.15 * zoom_scale)
        bar_h: int = max(4, int(6 * zoom_scale))
        bar_x: int = self.x + self.w - bar_w - int(6 * zoom_scale)
        bar_y: int = self.y + int(6 * zoom_scale)

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
                self.x + 2, self.y + 2, self.w - 4, self.h - 4
            )
            pygame.draw.rect(
                surface, config.COLOR_FRAME_SOLVED, inset_rect, 2
            )
        elif not is_alive:
            inset_rect = (self.x + 2, self.y + 2, self.w - 4, self.h - 4)
            pygame.draw.rect(
                surface, config.COLOR_FRAME_DEAD, inset_rect, 2
            )

        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h), 1
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
        Retrieves cached pre-rendered map surface or renders background once.
        """
        cache_key: Tuple[int, int, int] = (run_idx, rw, rh)
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

    def _draw_zoomed_map_tiles(
        self,
        surface: pygame.Surface,
        map_data: MapData,
        tile_size: float,
        origin_x: int,
        origin_y: int
    ) -> None:
        """
        Renders the whole map manually at the (larger) zoomed tile scale.
        """
        for y in range(map_data.height):
            for x in range(map_data.width):
                t_x: int = int(float(x) * tile_size) + origin_x
                t_y: int = int(float(y) * tile_size) + origin_y
                t_rect: Tuple[int, int, int, int] = (
                    t_x, t_y, int(tile_size) + 1, int(tile_size) + 1
                )

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
