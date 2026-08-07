"""
Data structures for 2D map grids and compact PyBiwis bitmask encoding.
"""

from typing import List, Tuple
import numpy as np


class MapData:
    """
    Stores 2D tile layout data natively in NumPy with PyBiwis bitmask packing.
    """

    _next_map_id: int = 1

    def __init__(
        self,
        width: int,
        height: int,
        start_pos: Tuple[int, int],
        exit_pos: Tuple[int, int]
    ) -> None:
        """
        Initializes map layout array, unique map_id, and entry/exit coords.
        """
        self.map_id: int = MapData._next_map_id
        MapData._next_map_id += 1

        self.width: int = width
        self.height: int = height
        self.start_pos: Tuple[int, int] = start_pos
        self.exit_pos: Tuple[int, int] = exit_pos
        self.grid: np.ndarray = np.zeros((height, width), dtype=np.bool_)
        self.bitmask_chunks: List[int] = []

    def set_wall(self, x: int, y: int, is_wall: bool = True) -> None:
        """
        Sets wall state at the designated tile coordinate.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = is_wall

    def is_wall(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinates contain a wall or are out of bounds.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return bool(self.grid[y, x])

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinates are open for traversal.
        """
        return not self.is_wall(x, y)

    def build_wall_grid(self) -> np.ndarray:
        """
        Returns boolean (height, width) wall lookup grid for vectorized ops.
        """
        return self.grid.copy()

    def encode_bitmask(self) -> List[int]:
        """
        Packs the 2D grid into 64-bit PyBiwis integer chunks.
        """
        flat_grid: np.ndarray = self.grid.ravel()
        total_tiles: int = flat_grid.size
        num_chunks: int = (total_tiles + 63) // 64
        chunks: List[int] = [0] * num_chunks

        wall_indices: np.ndarray = np.flatnonzero(flat_grid)
        for flat_idx in wall_indices:
            chunk_idx: int = int(flat_idx) // 64
            bit_off: int = int(flat_idx) % 64
            chunks[chunk_idx] |= (1 << bit_off)

        self.bitmask_chunks = chunks
        return chunks

    def decode_bitmask(self, chunks: List[int]) -> None:
        """
        Unpacks 64-bit PyBiwis integer chunks into the 2D tile grid.
        """
        self.bitmask_chunks = chunks
        self.grid.fill(False)
        total_tiles: int = self.width * self.height

        for flat_idx in range(total_tiles):
            chunk_idx: int = flat_idx // 64
            bit_off: int = flat_idx % 64
            if (chunks[chunk_idx] >> bit_off) & 1:
                y: int = flat_idx // self.width
                x: int = flat_idx % self.width
                self.grid[y, x] = True
