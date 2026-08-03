"""
Numpy population manager. Candidate genomes are stored as stacked weight
matrices and evolved in place via elitism, tournament selection, uniform
crossover, and Gaussian mutation.
"""

import random
from typing import List, Tuple
import numpy as np

import config


class PopulationManager:
    """
    Owns batched MLP weight matrices and executes vectorized neuroevolution.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        input_size: int = config.VISION_RAYS + (
            2 if config.INCLUDE_COMPASS else 0
        ) + 2,
        hidden_layers: int = config.NEURAL_HIDDEN_LAYERS,
        neurons: int = config.NEURAL_NEURONS,
        output_size: int = 2,
        mutation_rate: float = config.MUTATION_RATE,
        mutation_scale: float = config.MUTATION_SCALE,
        elitism_ratio: float = config.ELITISM_RATIO
    ) -> None:
        """
        Initializes evolution hyper-parameters and candidate weight stacks.
        """
        self.pop_size: int = pop_size
        self.mutation_rate: float = mutation_rate
        self.mutation_scale: float = mutation_scale
        self.elitism_ratio: float = elitism_ratio

        self.layer_sizes: List[int] = [neurons] * hidden_layers
        self.input_size: int = input_size
        self.output_size: int = output_size

        self.sizes: List[int] = [input_size] + self.layer_sizes + [output_size]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weight_stacks()

    def _init_weight_stacks(self) -> None:
        """
        Builds batched Gaussian-initialized weight and bias matrices per layer.
        """
        for fin, fout in zip(self.sizes[:-1], self.sizes[1:]):
            scale: float = np.sqrt(2.0 / float(fin + fout))
            weights: np.ndarray = scale * np.random.randn(
                self.pop_size, fin, fout
            )
            biases: np.ndarray = np.zeros(
                (self.pop_size, 1, fout), dtype=np.float64
            )
            self.weights.append(weights)
            self.biases.append(biases)

    def forward_batch(
        self,
        active_idx: List[int],
        features: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Batched forward pass over active candidates using indexed weight stacks.
        Returns (outputs, layer_activations); layer_activations[0] is the input
        layer and later entries are each dense layer's pre-activation values.
        """
        idx: np.ndarray = np.asarray(active_idx, dtype=np.int64)
        x: np.ndarray = features.astype(np.float64, copy=False)

        acts: List[np.ndarray] = [x]

        for l_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.einsum(
                "ni,nij->nj", x, w[idx], optimize=False
            ) + b[idx].squeeze(1)
            acts.append(x)
            if l_idx < len(self.layer_sizes):
                x = np.maximum(0.0, x)

        move_eff: np.ndarray = 1.0 / (
            1.0 + np.exp(-np.clip(x[:, 0:1], -500.0, 500.0))
        )
        turn_eff: np.ndarray = np.tanh(x[:, 1:2])
        outputs: np.ndarray = np.hstack([move_eff, turn_eff])

        return outputs, acts

    def evolve_next_generation(
        self,
        fitness_scores: List[float]
    ) -> None:
        """
        Performs elitism, tournament selection, uniform crossover, and Gaussian
        mutation across the stacked candidate genomes.
        """
        indexed_scores: List[Tuple[int, float]] = list(
            enumerate(fitness_scores)
        )
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        num_elites: int = max(1, int(self.pop_size * self.elitism_ratio))
        elite_idx: List[int] = [idx for idx, _ in indexed_scores[:num_elites]]

        n_offspring: int = self.pop_size - num_elites
        parent_a: np.ndarray = self._tournament_select(
            fitness_scores, n_offspring
        )
        parent_b: np.ndarray = self._tournament_select(
            fitness_scores, n_offspring
        )

        new_weights: List[np.ndarray] = []
        new_biases: List[np.ndarray] = []

        # Per layer: keep elites unchanged, then breed offspring to refill
        for l_idx in range(len(self.sizes) - 1):
            elite_w: np.ndarray = self.weights[l_idx][elite_idx].copy()
            elite_b: np.ndarray = self.biases[l_idx][elite_idx].copy()

            wa = self.weights[l_idx][parent_a]
            wb = self.weights[l_idx][parent_b]
            ba = self.biases[l_idx][parent_a]
            bb = self.biases[l_idx][parent_b]

            mask_w: np.ndarray = np.random.rand(*wa.shape) < 0.5
            mask_b: np.ndarray = np.random.rand(*ba.shape) < 0.5
            child_w: np.ndarray = np.where(mask_w, wa, wb)
            child_b: np.ndarray = np.where(mask_b, ba, bb)

            mut_mask: np.ndarray = (
                np.random.rand(n_offspring, 1, 1) < self.mutation_rate
            )
            child_w = child_w + (
                np.random.normal(0.0, self.mutation_scale, child_w.shape)
                * mut_mask
            )
            child_b = child_b + (
                np.random.normal(0.0, self.mutation_scale, child_b.shape)
                * mut_mask
            )

            new_weights.append(np.concatenate([elite_w, child_w], axis=0))
            new_biases.append(np.concatenate([elite_b, child_b], axis=0))

        self.weights = new_weights
        self.biases = new_biases

    def _tournament_select(
        self,
        scores: List[float],
        count: int,
        k: int = 3
    ) -> np.ndarray:
        """
        Returns (count,) parent indices from k-way random tournament sampling.
        """
        k = min(k, self.pop_size)
        picks: np.ndarray = np.array([
            random.sample(range(self.pop_size), k) for _ in range(count)
        ])
        cand_scores: np.ndarray = np.asarray(scores)[picks]
        best: np.ndarray = cand_scores.argmax(axis=1)
        return picks[np.arange(count), best]
