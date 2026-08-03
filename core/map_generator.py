"""
Procedural map generator producing solvable grid levels validated via BFS.
"""

import random
from collections import deque
from typing import List, Tuple, Optional, Set

import config
from core.map_data import MapData
from core.pathfinder import BFSPathfinder


class MapGenerator:
    """
    Generates procedural grid maps with guaranteed solvable paths.
    """

    def __init__(
        self,
        width: int = config.MAP_WIDTH,
        height: int = config.MAP_HEIGHT,
        map_type: str = config.MAP_TYPE
    ) -> None:
        """
        Initializes generator dimensions and map layout generation style.
        """
        self.width: int = width
        self.height: int = height
        self.map_type: str = map_type

    def generate_solvable_map(
        self,
        wall_density: float = config.WALL_DENSITY,
        difficulty_ratio: float = config.MIN_PATH_DIFFICULTY_RATIO,
        max_attempts: int = config.MAX_MAP_GEN_ATTEMPTS
    ) -> MapData:
        """
        Generates a solvable layout with start/exit and difficulty check.
        """
        for _ in range(max_attempts):
            map_data: Optional[MapData] = self._try_generate_map(
                wall_density, difficulty_ratio
            )
            if map_data is not None:
                map_data.encode_bitmask()
                return map_data

        return self._build_fallback_map()

    def _try_generate_map(
        self,
        wall_density: float,
        difficulty_ratio: float
    ) -> Optional[MapData]:
        """
        Builds a map layout and verifies 100% floor connectivity via flood-fill.
        """
        dummy_start: Tuple[int, int] = (1, 1)
        dummy_exit: Tuple[int, int] = (self.width - 2, self.height - 2)
        map_data: MapData = MapData(
            self.width, self.height, dummy_start, dummy_exit
        )

        if self.map_type == "BRANCHING WALLS":
            self._generate_branching_walls_map(map_data, wall_density)
        else:
            self._generate_random_scatter_map(map_data, wall_density)

        open_tiles: List[Tuple[int, int]] = [
            (x, y) for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if map_data.is_walkable(x, y)
        ]

        if len(open_tiles) < 2:
            return None

        # Flood-fill connectivity check from first open tile
        connected_tiles: Set[Tuple[int, int]] = self._flood_fill_region(
            open_tiles[0], map_data
        )

        # Reject layout if any isolated islands exist
        if len(connected_tiles) != len(open_tiles):
            return None

        # Pick random start_pos from fully connected floor
        start_pos: Tuple[int, int] = random.choice(open_tiles)
        map_data.start_pos = start_pos

        # Run BFS from start_pos to find exact max reachable step distance
        dist_from_start: Optional[List[List[int]]] = (
            self._compute_start_bfs(start_pos, map_data)
        )
        if dist_from_start is None:
            return None

        reachable_distances: List[int] = [
            dist_from_start[pos[1]][pos[0]]
            for pos in open_tiles
            if pos != start_pos and dist_from_start[pos[1]][pos[0]] < 9999
        ]

        if not reachable_distances:
            return None

        max_bfs_dist: int = max(reachable_distances)
        target_min_dist: int = int(max_bfs_dist * difficulty_ratio)

        valid_exits: List[Tuple[int, int]] = [
            pos for pos in open_tiles
            if pos != start_pos and
            dist_from_start[pos[1]][pos[0]] >= target_min_dist
            and dist_from_start[pos[1]][pos[0]] < 9999
        ]

        if not valid_exits:
            return None

        map_data.exit_pos = random.choice(valid_exits)

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        return map_data

    def _generate_random_scatter_map(
        self,
        map_data: MapData,
        wall_density: float
    ) -> None:
        """
        Populates grid with random scattered wall layout.
        """
        for y in range(self.height):
            for x in range(self.width):
                if (
                    x == 0 or x == self.width - 1 or
                    y == 0 or y == self.height - 1
                ):
                    map_data.set_wall(x, y, True)
                elif random.random() < wall_density:
                    map_data.set_wall(x, y, True)

    def _generate_branching_walls_map(
        self,
        map_data: MapData,
        wall_density: float
    ) -> None:
        """
        Grows organic branching wall structures extending from existing walls.
        """
        border_walls: Set[Tuple[int, int]] = set()
        unrelated_walls: Set[Tuple[int, int]] = set()

        for y in range(self.height):
            for x in range(self.width):
                if (
                    x == 0 or x == self.width - 1 or
                    y == 0 or y == self.height - 1
                ):
                    map_data.set_wall(x, y, True)
                    border_walls.add((x, y))

        inner_area: int = (self.width - 2) * (self.height - 2)
        target_walls: int = int(inner_area * wall_density)
        placed_count: int = 0
        max_attempts: int = target_walls * 100

        cardinal_dirs: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        active_stem: List[Tuple[int, int]] = []
        current_dir: Tuple[int, int] = (0, 0)

        attempts: int = 0
        while placed_count < target_walls and attempts < max_attempts:
            attempts += 1

            if not active_stem:
                # Phase 1: Seed a new wall branch off border or existing wall
                seed_pool: List[Tuple[int, int]] = list(
                    border_walls | unrelated_walls
                )
                if not seed_pool:
                    break

                seed_x, seed_y = random.choice(seed_pool)
                dx, dy = random.choice(cardinal_dirs)

                nx: int = seed_x + dx
                ny: int = seed_y + dy

                if not (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1):
                    continue

                if map_data.is_wall(nx, ny):
                    continue

                if self._is_valid_seed_placement(
                    nx, ny, seed_x, seed_y, border_walls, unrelated_walls
                ):
                    map_data.set_wall(nx, ny, True)
                    active_stem.append((nx, ny))
                    placed_count += 1
                    current_dir = (dx, dy)

            else:
                # Phase 2: Extend active stem until naturally blocked
                last_x, last_y = active_stem[-1]

                # 70% chance straight, 30% chance 90 deg turn
                if random.random() < 0.7:
                    dx, dy = current_dir
                else:
                    dx, dy = random.choice(
                        [(-current_dir[1], current_dir[0]),
                         (current_dir[1], -current_dir[0])]
                    )

                nx = last_x + dx
                ny = last_y + dy

                if not (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1):
                    unrelated_walls.update(active_stem)
                    active_stem.clear()
                    continue

                if map_data.is_wall(nx, ny):
                    unrelated_walls.update(active_stem)
                    active_stem.clear()
                    continue

                if self._is_valid_stem_extension(
                    nx, ny, active_stem, border_walls, unrelated_walls
                ):
                    map_data.set_wall(nx, ny, True)
                    active_stem.append((nx, ny))
                    placed_count += 1
                    current_dir = (dx, dy)
                else:
                    unrelated_walls.update(active_stem)
                    active_stem.clear()

        if active_stem:
            unrelated_walls.update(active_stem)
            active_stem.clear()

    def _is_valid_seed_placement(
        self,
        nx: int,
        ny: int,
        seed_x: int,
        seed_y: int,
        border_walls: Set[Tuple[int, int]],
        unrelated_walls: Set[Tuple[int, int]]
    ) -> bool:
        """
        Validates first tile placement (C1) of a new branch next to anchor P.
        """
        seed_is_border: bool = (seed_x, seed_y) in border_walls

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                bx: int = nx + dx
                by: int = ny + dy

                if seed_is_border:
                    if (bx, by) in unrelated_walls:
                        return False
                    if (bx, by) in border_walls:
                        if abs(bx - seed_x) > 1 or abs(by - seed_y) > 1:
                            return False
                else:
                    if (bx, by) in border_walls:
                        return False
                    if (bx, by) in unrelated_walls:
                        if abs(bx - seed_x) > 1 or abs(by - seed_y) > 1:
                            return False

        return True

    def _is_valid_stem_extension(
        self,
        nx: int,
        ny: int,
        active_stem: List[Tuple[int, int]],
        border_walls: Set[Tuple[int, int]],
        unrelated_walls: Set[Tuple[int, int]]
    ) -> bool:
        """
        Validates extending active stem tile including 90-degree turns.
        """
        allowed_stem: Set[Tuple[int, int]] = set(active_stem[-2:])

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                bx: int = nx + dx
                by: int = ny + dy

                if (bx, by) in border_walls or (bx, by) in unrelated_walls:
                    return False

                if (bx, by) in active_stem and (bx, by) not in allowed_stem:
                    return False

        return True

    def _flood_fill_region(
        self,
        start_node: Tuple[int, int],
        map_data: MapData
    ) -> Set[Tuple[int, int]]:
        """
        Discovers all walkable tiles reachable from a starting node.
        """
        visited: Set[Tuple[int, int]] = {start_node}
        queue: deque[Tuple[int, int]] = deque([start_node])
        cardinal_moves: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in cardinal_moves:
                nx: int = cx + dx
                ny: int = cy + dy
                if map_data.is_walkable(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return visited

    def _compute_start_bfs(
        self,
        start_pos: Tuple[int, int],
        map_data: MapData
    ) -> Optional[List[List[int]]]:
        """
        Computes BFS step distance grid outward from start_pos.
        """
        unreachable_val: int = 9999
        dist_grid: List[List[int]] = [
            [unreachable_val for _ in range(self.width)]
            for _ in range(self.height)
        ]

        sx, sy = start_pos
        dist_grid[sy][sx] = 0

        queue: deque[Tuple[int, int]] = deque([(sx, sy)])
        cardinal_moves: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        while queue:
            cx, cy = queue.popleft()
            current_dist: int = dist_grid[cy][cx]

            for dx, dy in cardinal_moves:
                nx: int = cx + dx
                ny: int = cy + dy
                if map_data.is_walkable(nx, ny):
                    if dist_grid[ny][nx] == unreachable_val:
                        dist_grid[ny][nx] = current_dist + 1
                        queue.append((nx, ny))

        return dist_grid

    def _build_fallback_map(self) -> MapData:
        """
        Constructs an open fallback map with border walls only.
        """
        start_pos: Tuple[int, int] = (1, 1)
        exit_pos: Tuple[int, int] = (self.width - 2, self.height - 2)
        map_data: MapData = MapData(
            self.width, self.height, start_pos, exit_pos
        )

        for y in range(self.height):
            for x in range(self.width):
                if (
                    x == 0 or x == self.width - 1 or
                    y == 0 or y == self.height - 1
                ):
                    map_data.set_wall(x, y, True)

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()
        map_data.encode_bitmask()
        return map_data
