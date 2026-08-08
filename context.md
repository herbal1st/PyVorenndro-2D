# tokenix pack

Project: `/home/scooterking38/pyvorenndro-2d`
Profile: `Plan`
Budget: `20000` tokens
Packed: `3556` tokens

## Focused Context

<!-- tokenix_context: 'repository architecture entry points public interfaces' mode=Plan -->

## Preference Memory
- If the user states a durable preference, migration decision, workflow rule, or project policy, save it with `tokenix_memory_add` when MCP is available.
- Use `scope=project` for repository-specific rules and `scope=global` only for cross-repository preferences.
- Do not save secrets, credentials, private tokens, one-off bug details, or guesses.

## Entry Points
- visualization/viewport_grid.py:14-414 [class] ViewportGrid
- MANUAL.txt:299-335 [block] (file chunk)
- visualization/viewport_grid.py:205-451 [function] _draw_single_candidate_viewport
- visualization/viewport_grid.py:205-451 [function] _draw_single_candidate_viewport
- visualization/viewport_grid.py:14-414 [class] ViewportGrid
- README.md:59-81 [block] (file chunk)

## Relevant Source
<!-- tokenix: 6 chunks for 'repository architecture entry points public interfaces' -->

### MANUAL.txt
```  L299-335
LAYOUT_SCRUBBER_RECT          : Position and size (x, y, width, height) of
                                the interactive transport scrubber panel.
                                Metric: Rectangle (int, int, int, int).


[HUD TYPOGRAPHY & ELEMENT SIZING]
-------------------------------------------------------------------------------
HUD_PANEL_TITLE_FONT_SIZE     : Font point size for panel header titles.
                                Metric: Points (int).

HUD_PANEL_BODY_FONT_SIZE      : Font point size for panel telemetry text.
                                Metric: Points (int).

HUD_SCRUBBER_BTN_HEIGHT       : Pixel height for scrubber UI buttons.
                                Metric: Pixels (int).

HUD_SCRUBBER_BTN_FONT_SIZE    : Font point size for scrubber button text.
                                Metric: Points (int).

HUD_SCRUBBER_BAR_HEIGHT       : Pixel height for timeline track bars.
                                Metric: Pixels (int).

HUD_SCRUBBER_MARKER_RADIUS    : Pixel radius for timeline handle markers.
                                Metric: Pixels (int).

HUD_GRAPH_NODE_RADIUS         : Pixel radius for neural network graph nodes.
                                Metric: Pixels (int).

HUD_GRAPH_FONT_SIZE           : Font point size for neural graph labels.
                                Metric: Points (int).

HUD_GRAPH_LABEL_PADDING       : Spacing padding between graph nodes and
                                text labels.
                                Metric: Pixels (int).


[VISUAL THEME & COLORS]
```

### README.md
```  L59-81
[3.0 SPATIAL PERCEPTION (9 TO 13 INPUT CHANNELS)]
-------------------------------------------------------------------------------
Vision Fan     : 9 probe rays distributed evenly across a 120-degree Field
                 of View (VISION_ARC_ANGLE), calculated via grid-aligned
                 Amanatides-Woo DDA. Measures proximity to walls:
                 - Channels 0..8 (-60° to +60°): 0.0 for open space up to 1.0
                   for a point-blank wall collision.
Proprioception : 2 physical status channels:
                 - Speed Channel (SPD): Current forward translation speed.
                 - Health Channel (HP): Active candidate health ratio [0, 1].
Optional       : Toggleable via INCLUDE_COMPASS in config. Adds 2 local
Stereo Compass   target-guidance channels (TG-L, TG-R) encoding exit bearing
                 relative to heading (expanding inputs from 11 to 13).
                 Disabled by default so agents must learn maze structure from
                 wall rays alone (e.g. wall-following).

[4.0 NEURAL NETWORK ARCHITECTURE (MLP ENGINE)]
-------------------------------------------------------------------------------
Inputs         : 11 continuous sensory channels by default (9 Wall Rays +
                 Speed + Health Ratio), or 13 when INCLUDE_COMPASS is True.
Batched Pass   : Tensorized matrix multiplication (`np.einsum`) evaluating the
                 entire population's neural decisions simultaneously.
Hidden Layers  : Configurable multi-layer dense stack (NEURAL_HIDDEN_LAYERS,
```

### visualization/viewport_grid.py
```  L14-414 [class] ViewportGrid
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
                (pt[0] - rx, pt[1
```
```  L14-414 [class] ViewportGrid
] - ry) for pt in cone_points
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
      
```
```  L205-451 [function] _draw_single_candidate_viewport
nder vision fan and translucent heading line onto alpha scratchpad
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
     
```
```  L205-451 [function] _draw_single_candidate_viewport
raw.rect(surface, config.COLOR_FLOOR, t_rect)
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

        # Re
```

<!-- 2373 tokens -->

## File Outlines

### MANUAL.txt
(outline omitted: budget exhausted)


## Safety Report
- Sensitive paths omitted: 0
- Budget omissions: 0


## Repository Map

### README.md (~844 tok, semantic)
```text
____           __  __                                          __
/\  _`\        /\ \/\ \                                        /\ \
\ \ \L\ \__  __\ \ \ \ \    ___   _ __    __    ___     ___    \_\ \  _ __   ___
\ \ ,__/\ \/\ \\ \ \ \ \  / __`\/\`'__\/'__`\/' _ `\ /' _ `\  /'_` \/\`'__\/ __`\
\ \ \/\ \ \_\ \\ \ \_/ \/\ \L\ \ \ \//\  __//\ \/\ \/\ \/\ \/\ \L\ \ \ \//\ \L\ \
\ \_\ \/`____ \\ `\___/\ \____/\ \_\\ \____\ \_\ \_\ \_\ \_\ \___,_\ \_\\ \____/
\/_/  `/___/> \`\/__/  \/___/  \/_/ \/____/\/_/\/_/\/_/\/_/\/__,_ /\/_/ \/___/
/\___/
\/__/
===============================================================================
PYVORENNDRO 2D - SYSTEM SPECIFICATIONS & GUIDE
===============================================================================

[1.0 SYSTEM OVERVIEW]
-------------------------------------------------------------------------------
Core Philosophy: Sovereign Compute, Matrix-Isolated, Zero-Dependency.
Architecture   : High-performance 2D Neuroevolution Visualizer powered by a
vectorized NumPy batch simulation engine (feature batch ->
neural batch -> physics batch), PyBiwis 64-bit bitmask grid
compression, grid-aligned Amanatides-Woo DDA raycasting,
continuous Circle-to-AABB smooth wall physics, dual driving
dynamics (Car and Tank profiles), adaptive curriculum
training, and a dynamic Multi-Layer Perceptron (MLP).
Primary Goal   : Train autonomous 2D candidate agents to navigate procedural
labyrinths from randomized start locations to exit tiles
using a 9-ray visual sensor fan and optional target guidance.
Presentation   : Interactive GUI featuring dual camera tracking modes (Map-
Centered and Player-Centered), 16x turbo speed controls,
dual-bar scrubber transport, pre-rendered background surface
caching, standardized 0-1000 score normalization, real-time
neural activation graphs, and an 8-column CLI console table.

[2.0 MEMORY, MAPS & PROCEDURAL GENERATION (PYBIWIS & NUMPY)]
-------------------------------------------------------------------------------
Grid Storage   : Rectangular tile grids (MAP_WIDTH x MAP_HEIGHT) stored in
C-contiguous NumPy arrays for instant vectorized lookups.
PyBiwis Chunks : Packed 64-bit integer words storing wall collision layouts.
Compresses tile maps into tiny save snapshots and fast memory.
100% Floor Fill: All level generators use flood-fill analysis to guarantee
100% floor connectivity with zero isolated dead-end pockets.
BFS Distance   : Every level builds an O(1) step-distance matrix. Each tile
stores its exact topological step-distance to the exit.
Dual Map Styles: Selectable procedural level generation algorithms:
- "BRANCHING WALLS": Organic tree-like maze crawler growing
continuous wall stems, 90-degree corners, and T-junctions
without 2x2 solid wall blocks or isolated dead zones.
- "RANDOM": Classic random scattered wall noise layout.
Map Regimes    : Organizes training generations into controlled "regimes" on
a single map layout. This gives agents time to master a
specific maze before rotating environments. Transitioning
to a new map temporarily boosts mutation rates to encourage
exploration on new layouts.
Curriculum     : Adaptive difficulty system that acts like a training ladder.
Learning         It starts agents on short, easy paths (e.g. 8 tiles) and
automatically steps up maze complexity as agents succeed. If
(outline truncated by tokenix pack budget)
```
