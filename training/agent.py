"""
Single-agent genome holder.

Instead of a population of parallel candidate genomes, the trainer owns
a champion genome that steers every parallel simulation run. The genome
is built from a data-driven profile or explicit layer size lists.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import config


class Agent:
    """
    Owns a single MLP genome as per-layer weight and bias matrices.
    """

    def __init__(
        self,
        sizes: List[int],
        seed: Optional[int] = None
    ) -> None:
        """
        Initializes Gaussian weight matrices and zero biases per layer.
        """
        self.sizes: List[int] = list(sizes)
        rng: np.random.Generator = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for fin, fout in zip(self.sizes[:-1], self.sizes[1:]):
            scale: float = np.sqrt(2.0 / float(fin + fout))
            self.weights.append(
                scale * rng.standard_normal((fin, fout))
            )
            self.biases.append(
                np.zeros((1, fout), dtype=np.float64)
            )

    @classmethod
    def from_profile(
        cls,
        profile_config: Dict[str, Any],
        input_size: int = 0,
        output_size: int = 2,
        seed: Optional[int] = None
    ) -> "Agent":
        """
        Constructs an Agent directly from a resolved YAML profile dictionary.
        """
        topology: Dict[str, Any] = profile_config["topology"]
        sensory: Dict[str, Any] = profile_config["sensory"]

        hidden_layers: int = int(topology.get("hidden_layers", 2))
        neurons: int = int(topology.get("neurons_per_layer", 32))
        memory_frames: int = max(1, int(topology.get("memory_frames", 1)))

        if input_size <= 0:
            num_rays: int = int(sensory.get("vision_rays", 9))
            include_compass: bool = bool(sensory.get("include_compass", False))
            include_bfs: bool = bool(sensory.get("include_bfs_sensor", False))

            base_channels: int = (
                num_rays
                + (2 if include_compass else 0)
                + 2  # SPD, HP
                + (1 if include_bfs else 0)
                + 3  # HIT, IDL, SPN
            )
            input_size = base_channels * memory_frames

        layer_sizes: List[int] = [input_size] + [
            neurons for _ in range(hidden_layers)
        ] + [output_size]

        return cls(sizes=layer_sizes, seed=seed)

    @classmethod
    def from_state(
        cls,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        sizes: List[int]
    ) -> "Agent":
        """
        Rebuilds an agent from plain weight and bias lists.
        """
        agent = cls.__new__(cls)
        agent.sizes = list(sizes)
        agent.weights = [np.asarray(w) for w in weights]
        agent.biases = [np.asarray(b) for b in biases]
        return agent

    def copy(self) -> "Agent":
        """
        Returns a deep copy sharing no mutable numpy storage.
        """
        return Agent.from_state(self.weights, self.biases, self.sizes)

    def mutate(
        self,
        mutation_rate: float = config.MUTATION_RATE,
        mutation_scale: float = config.MUTATION_SCALE
    ) -> "Agent":
        """
        Returns a mutated offspring copy of the champion genome.
        """
        child = self.copy()

        for layer_idx in range(len(child.weights)):
            w: np.ndarray = child.weights[layer_idx]
            b: np.ndarray = child.biases[layer_idx]

            w_mask: np.ndarray = (
                np.random.rand(*w.shape) < mutation_rate
            )
            child.weights[layer_idx] = w + (
                np.random.normal(0.0, mutation_scale, w.shape)
                * w_mask
            )

            b_mask: np.ndarray = (
                np.random.rand(*b.shape) < mutation_rate
            )
            child.biases[layer_idx] = b + (
                np.random.normal(0.0, mutation_scale, b.shape)
                * b_mask
            )

        return child

    def to_state(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns plain per-layer weight and bias lists for sharing.
        """
        return self.weights, self.biases

    @staticmethod
    def forward_batch(
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        features: np.ndarray,
        return_acts: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Batched MLP forward pass sharing one genome across (N,) rows.
        """
        x: np.ndarray = features.astype(np.float64, copy=False)

        acts: Optional[List[np.ndarray]] = [x] if return_acts else None
        hidden_layers: int = len(weights) - 1

        for layer_idx, (w, b) in enumerate(zip(weights, biases)):
            x = np.einsum(
                "ni,ij->nj",
                x,
                w,
                optimize=False
            ) + b.squeeze()

            if return_acts:
                acts.append(x)

            if layer_idx < hidden_layers:
                x = np.maximum(x, 0.0)

        move_eff: np.ndarray = 1.0 / (
            1.0 + np.exp(-np.clip(x[:, 0:1], -500.0, 500.0))
        )
        turn_eff: np.ndarray = np.tanh(x[:, 1:2])

        outputs: np.ndarray = np.hstack([move_eff, turn_eff])

        return outputs, acts
