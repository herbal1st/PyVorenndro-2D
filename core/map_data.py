"""
Data structures for 2D map grids and compact PyBiwis bitmask encoding.
"""

from typing import List, Tuple


class MapData:
    """
    Stores 2D tile layout data with PyBiwis 64-bit bitmask packing.
    """

    def __init__(
        self,
        width: int,
        height: int,
        start_pos: Tuple[int, int],
        exit_pos: Tuple[int, int]
    ) -> None:
        """
        Initializes map layout grids and entry/exit coordinates.
        """
        self.width: int = width
        self.height: int = height
        self.start_pos: Tuple[int, int] = start_pos
        self.exit_pos: Tuple[int, int] = exit_pos
        self.grid: List[List[int]] = [
            [0 for _ in range(width)] for _ in range(height)
        ]
        self.bitmask_chunks: List[int] = []

    def set_wall(self, x: int, y: int, is_wall: bool = True) -> None:
        """
        Sets wall state at the designated tile coordinate.
        """
        val: int = 1 if is_wall else 0
        self.grid[y][x] = val

    def is_wall(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinates contain a wall or are out of bounds.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid[y][x] == 1

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinates are open for traversal.
        """
        return not self.is_wall(x, y)

    def encode_bitmask(self) -> List[int]:
        """
        Packs the 2D grid into 64-bit PyBiwis integer chunks.
        """
        total_tiles: int = self.width * self.height
        num_chunks: int = (total_tiles + 63) // 64
        chunks: List[int] = [0] * num_chunks

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1:
                    flat_idx: int = x + (y * self.width)
                    chunk_idx: int = flat_idx // 64
                    bit_off: int = flat_idx % 64
                    chunks[chunk_idx] |= (1 << bit_off)

        self.bitmask_chunks = chunks
        return chunks

    def decode_bitmask(self, chunks: List[int]) -> None:
        """
        Unpacks 64-bit PyBiwis integer chunks into the 2D tile grid.
        """
        self.bitmask_chunks = chunks
        for y in range(self.height):
            for x in range(self.width):
                flat_idx: int = x + (y * self.width)
                chunk_idx: int = flat_idx // 64
                bit_off: int = flat_idx % 64
                is_set: bool = bool(
                    (chunks[chunk_idx] >> bit_off) & 1
                )
                self.grid[y][x] = 1 if is_set else 0
