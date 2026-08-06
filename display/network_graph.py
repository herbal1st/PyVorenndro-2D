"""
Renders real-time neural network node activation graph with color shifts.
"""

from typing import Tuple, List, Dict, Any, Optional
import pygame

import config


class NetworkGraph:
    """
    Renders activation graph topology (Input -> Hidden -> Output layers).
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRAPH_RECT
    ) -> None:
        """
        Initializes bounding rect and node display properties.
        """
        self.x, self.y, self.w, self.h = rect
        self.font: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.HUD_GRAPH_FONT_SIZE, bold=True
        )

    def draw_graph(
        self,
        surface: pygame.Surface,
        run_data: Dict[str, Any],
        active_step: int,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Renders candidate neural layers with dark red to bright orange nodes.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h), 1
        )

        frames: List[Dict[str, Any]] = run_data["frames"]
        if not frames:
            return

        frame_idx: int = min(active_step, len(frames) - 1)
        activations: List[List[float]] = frames[frame_idx]["activations"]

        num_layers: int = len(activations)
        if num_layers == 0:
            return

        col_spacing: float = float(self.w) / float(num_layers + 1)
        node_radius: int = config.HUD_GRAPH_NODE_RADIUS
        label_padding: int = config.HUD_GRAPH_LABEL_PADDING

        for l_idx, layer_vals in enumerate(activations):
            cx: int = int(self.x + ((l_idx + 1) * col_spacing))
            num_nodes: int = len(layer_vals)
            row_spacing: float = float(self.h - 40) / float(
                max(1, num_nodes)
            )

            for n_idx, val in enumerate(layer_vals):
                cy: int = int(self.y + 20 + (n_idx * row_spacing))

                clamped_val: float = max(0.0, min(1.0, float(val)))
                r: int = int(
                    config.COLOR_NODE_INACTIVE[0] +
                    clamped_val * (
                        config.COLOR_NODE_ACTIVE[0] -
                        config.COLOR_NODE_INACTIVE[0]
                    )
                )
                g: int = int(
                    config.COLOR_NODE_INACTIVE[1] +
                    clamped_val * (
                        config.COLOR_NODE_ACTIVE[1] -
                        config.COLOR_NODE_INACTIVE[1]
                    )
                )
                b: int = int(
                    config.COLOR_NODE_INACTIVE[2] +
                    clamped_val * (
                        config.COLOR_NODE_ACTIVE[2] -
                        config.COLOR_NODE_INACTIVE[2]
                    )
                )
                node_color: Tuple[int, int, int] = (r, g, b)

                pygame.draw.circle(
                    surface, node_color, (cx, cy), node_radius
                )
                pygame.draw.circle(
                    surface, config.COLOR_WALL_BORDER,
                    (cx, cy), node_radius, 1
                )

                # Node shorthand labels
                if l_idx == 0:
                    lbl_text = self._get_input_shorthand(
                        n_idx, profile_config
                    )
                    lbl_surf = self.font.render(
                        lbl_text, True, config.COLOR_PLAYER_HIGHLIGHT
                    )
                    lbl_rect = lbl_surf.get_rect(
                        midleft=(cx + node_radius + label_padding, cy)
                    )
                    surface.blit(lbl_surf, lbl_rect)

                elif l_idx == num_layers - 1:
                    lbl_text = "MOVE" if n_idx == 0 else "TURN"
                    lbl_surf = self.font.render(
                        lbl_text, True, config.COLOR_PLAYER_HIGHLIGHT
                    )
                    lbl_rect = lbl_surf.get_rect(
                        midleft=(cx + node_radius + label_padding, cy)
                    )
                    surface.blit(lbl_surf, lbl_rect)

        # Layer title labels below graph
        labels: List[str] = ["INPUT"] + [
            f"HIDDEN {i+1}" for i in range(num_layers - 2)
        ] + ["OUTPUT"]

        for l_idx, label in enumerate(labels):
            cx = int(self.x + ((l_idx + 1) * col_spacing))
            lbl_surf = self.font.render(
                label, True, config.COLOR_PLAYER_HIGHLIGHT
            )
            lbl_rect = lbl_surf.get_rect(
                center=(cx, self.y + self.h - 10)
            )
            surface.blit(lbl_surf, lbl_rect)

    def _get_input_shorthand(
        self,
        node_idx: int,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates shorthand labels for Wall Rays, TG-L, TG-R, SPD, and HP.
        """
        sensory: Dict[str, Any] = (
            profile_config.get("sensory", {}) if profile_config else {}
        )
        num_rays: int = int(sensory.get("vision_rays", 9))
        half_arc: float = float(sensory.get("vision_arc_angle", 120.0)) / 2.0
        include_compass: bool = bool(sensory.get("include_compass", False))

        if node_idx < num_rays:
            step: float = (2.0 * half_arc) / float(max(1, num_rays - 1))
            deg: int = int(round(-half_arc + (node_idx * step)))
            return f"{deg:+d}°"

        curr_idx: int = num_rays
        if include_compass:
            if node_idx == curr_idx:
                return "TG-L"
            if node_idx == curr_idx + 1:
                return "TG-R"
            curr_idx += 2

        if node_idx == curr_idx:
            return "SPD"
        if node_idx == curr_idx + 1:
            return "HP"

        return "BFS"
