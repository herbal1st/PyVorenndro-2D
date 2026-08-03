"""
In-memory frame recorder logging generation simulation histories for playback.
"""

from typing import List, Dict, Any
import numpy as np

from core.map_data import MapData


class FrameRecorder:
    """
    Stores playback frame data and neural activations per generation.
    """

    def __init__(self) -> None:
        """
        Initializes generational history storage.
        """
        self.generations_history: List[Dict[str, Any]] = []

    def record_generation(
        self,
        generation_index: int,
        map_data: MapData,
        candidate_frames: List[List[Dict[str, Any]]],
        raw_scores: List[float],
        normalized_scores: List[float]
    ) -> None:
        """
        Stores recorded frame timelines and metrics for a generation.
        """
        gen_data: Dict[str, Any] = {
            "generation": generation_index,
            "bitmask_chunks": map_data.bitmask_chunks,
            "start_pos": map_data.start_pos,
            "exit_pos": map_data.exit_pos,
            "map_width": map_data.width,
            "map_height": map_data.height,
            "candidate_frames": candidate_frames,
            "raw_scores": raw_scores,
            "normalized_scores": normalized_scores,
            "winner_index": int(np.argmax(normalized_scores))
        }
        self.generations_history.append(gen_data)

    def get_generation_data(self, gen_idx: int) -> Dict[str, Any]:
        """
        Retrieves recorded history data for a specific generation.
        """
        safe_idx: int = max(
            0, min(gen_idx, len(self.generations_history) - 1)
        )
        return self.generations_history[safe_idx]
