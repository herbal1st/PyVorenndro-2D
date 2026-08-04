"""
Single-agent genome holder.

Instead of a population of parallel candidate genomes, the trainer now owns
exactly one champion genome. That genome steers every parallel simulation run
of the generation; the trainer corroborates all run data back into the agent
before deciding whether the next mutated copy should replace it.
"""

from typing import List, Tuple, Optional

import numpy as np

import config


class Agent:
    """
    Owns a single MLP genome as per-layer weight/bias matrices.
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
    def from_state(
        cls,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        sizes: List[int]
    ) -> "Agent":
        """
        Rebuilds an agent from plain weight/bias lists (e.g. shared state).
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

        Returns (outputs, layer_activations). Activations are only materialized
        when requested (the trainer never needs them; the display does).
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
