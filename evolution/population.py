"""
Batched neural network population manager and neuroevolution engine.
"""

import random
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

import config


class PopulationManager:
    """
    Owns batched MLP weight matrices and executes vectorized neuroevolution.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        input_size: Optional[int] = None,
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

        if input_size is None:
            compass_ch: int = 2 if config.INCLUDE_COMPASS else 0
            input_size = config.VISION_RAYS + compass_ch + 2

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.layer_sizes: List[int] = [neurons] * hidden_layers

        self.sizes: List[int] = (
            [input_size] + self.layer_sizes + [output_size]
        )
        self.weights: List[NDArray[np.float64]] = []
        self.biases: List[NDArray[np.float64]] = []
        self._init_weight_stacks()

    def forward_batch(
        self,
        active_idx: List[int],
        features: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
        """
        Executes batched forward pass over active candidates.
        """
        idx: NDArray[np.int64] = np.asarray(active_idx, dtype=np.int64)
        x: NDArray[np.float64] = features.astype(np.float64, copy=False)

        acts: List[NDArray[np.float64]] = [x]

        for l_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.einsum(
                "ni,nij->nj", x, w[idx], optimize=False
            ) + b[idx].squeeze(1)
            acts.append(x)
            if l_idx < len(self.layer_sizes):
                x = np.maximum(0.0, x)

        move_eff: NDArray[np.float64] = 1.0 / (
            1.0 + np.exp(-np.clip(x[:, 0:1], -500.0, 500.0))
        )
        turn_eff: NDArray[np.float64] = np.tanh(x[:, 1:2])
        outputs: NDArray[np.float64] = np.hstack([move_eff, turn_eff])

        return outputs, acts

    def evolve_next_generation(
        self,
        fitness_scores: List[float]
    ) -> None:
        """
        Performs elitism, tournament selection, crossover, and mutation.
        """
        indexed_scores: List[Tuple[int, float]] = list(
            enumerate(fitness_scores)
        )
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        num_elites: int = max(1, int(self.pop_size * self.elitism_ratio))
        elite_idx: List[int] = [
            idx for idx, _ in indexed_scores[:num_elites]
        ]

        n_offspring: int = self.pop_size - num_elites
        parent_a: NDArray[np.int64] = self._tournament_select(
            fitness_scores, n_offspring
        )
        parent_b: NDArray[np.int64] = self._tournament_select(
            fitness_scores, n_offspring
        )

        new_weights: List[NDArray[np.float64]] = []
        new_biases: List[NDArray[np.float64]] = []

        for l_idx in range(len(self.sizes) - 1):
            elite_w: NDArray[np.float64] = (
                self.weights[l_idx][elite_idx].copy()
            )
            elite_b: NDArray[np.float64] = (
                self.biases[l_idx][elite_idx].copy()
            )

            wa: NDArray[np.float64] = self.weights[l_idx][parent_a]
            wb: NDArray[np.float64] = self.weights[l_idx][parent_b]
            ba: NDArray[np.float64] = self.biases[l_idx][parent_a]
            bb: NDArray[np.float64] = self.biases[l_idx][parent_b]

            mask_w: NDArray[np.bool_] = np.random.rand(*wa.shape) < 0.5
            mask_b: NDArray[np.bool_] = np.random.rand(*ba.shape) < 0.5
            child_w: NDArray[np.float64] = np.where(mask_w, wa, wb)
            child_b: NDArray[np.float64] = np.where(mask_b, ba, bb)

            mut_mask: NDArray[np.bool_] = (
                np.random.rand(n_offspring, 1, 1) < self.mutation_rate
            )
            noise_w: NDArray[np.float64] = np.random.normal(
                0.0, self.mutation_scale, child_w.shape
            )
            noise_b: NDArray[np.float64] = np.random.normal(
                0.0, self.mutation_scale, child_b.shape
            )

            child_w += noise_w * mut_mask
            child_b += noise_b * mut_mask

            new_weights.append(np.concatenate([elite_w, child_w], axis=0))
            new_biases.append(np.concatenate([elite_b, child_b], axis=0))

        self.weights = new_weights
        self.biases = new_biases

    def _init_weight_stacks(self) -> None:
        """
        Builds batched Gaussian-initialized weight and bias matrices.
        """
        for fin, fout in zip(self.sizes[:-1], self.sizes[1:]):
            scale: float = np.sqrt(2.0 / float(fin + fout))
            weights: NDArray[np.float64] = scale * np.random.randn(
                self.pop_size, fin, fout
            )
            biases: NDArray[np.float64] = np.zeros(
                (self.pop_size, 1, fout), dtype=np.float64
            )
            self.weights.append(weights)
            self.biases.append(biases)

    def _tournament_select(
        self,
        scores: List[float],
        count: int,
        k: int = 3
    ) -> NDArray[np.int64]:
        """
        Returns (count,) parent indices from k-way tournament sampling.
        """
        k = min(k, self.pop_size)
        picks: NDArray[np.int64] = np.array([
            random.sample(range(self.pop_size), k) for _ in range(count)
        ])
        cand_scores: NDArray[np.float64] = np.asarray(scores)[picks]
        best: NDArray[np.int64] = cand_scores.argmax(axis=1)
        return picks[np.arange(count), best]
