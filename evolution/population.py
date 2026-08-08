"""
Population manager for CPU neuroevolution supporting Numba-accelerated parallel inference
and candidate weight extraction for multithreading.
"""

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

import config


@njit(fastmath=True, nogil=True)
def numba_layer_forward(current_in: NDArray[np.float64], weight: NDArray[np.float64], bias: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Numba-compiled single forward pass step: dot product, bias addition, and tanh activation.
    """
    return np.tanh(np.dot(current_in, weight) + bias)


@njit(fastmath=True, nogil=True, parallel=True)
def numba_layer_forward_batch(
    inputs: NDArray[np.float64],
    weights: NDArray[np.float64],
    biases: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Numba-compiled parallel batch forward pass across all population candidates.
    Spreads the work across all available CPU cores using prange.
    """
    n_cands = inputs.shape[0]
    out_dim = biases.shape[1]
    result = np.empty((n_cands, out_dim), dtype=np.float64)

    for i in prange(n_cands):
        res = np.dot(inputs[i], weights[i]) + biases[i]
        for j in range(out_dim):
            result[i, j] = np.tanh(res[j])

    return result


class PopulationManager:
    """
    Manages neural network weights and evolutionary operations across the population.
    """

    def __init__(
        self,
        pop_size: int,
        input_size: int,
        output_size: int,
        hidden_layers: int = config.NEURAL_HIDDEN_LAYERS,
        neurons_per_hidden: int = config.NEURAL_NEURONS
    ) -> None:
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.mutation_scale: float = config.MUTATION_SCALE

        self.sizes: List[int] = [input_size]
        for _ in range(hidden_layers):
            self.sizes.append(neurons_per_hidden)
        self.sizes.append(output_size)

        self.weights: List[NDArray[np.float64]] = []
        self.biases: List[NDArray[np.float64]] = []

        self._initialize_population()

    def _initialize_population(self) -> None:
        self.weights.clear()
        self.biases.clear()

        for i in range(len(self.sizes) - 1):
            in_dim = self.sizes[i]
            out_dim = self.sizes[i + 1]

            limit = np.sqrt(6.0 / (in_dim + out_dim))
            w_layer = np.random.uniform(
                -limit, limit, size=(self.pop_size, in_dim, out_dim)
            ).astype(np.float64)

            b_layer = np.zeros((self.pop_size, out_dim), dtype=np.float64)

            self.weights.append(w_layer)
            self.biases.append(b_layer)

    def get_candidate_weights(
        self, candidate_idx: int
    ) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Extracts weight and bias matrices for a single candidate.
        Exposes weights cleanly for parallel evaluation workers.
        """
        candidate_layers = []
        for l_idx in range(len(self.weights)):
            w = self.weights[l_idx][candidate_idx].copy()
            b = self.biases[l_idx][candidate_idx].copy()
            candidate_layers.append((w, b))
        return candidate_layers

    def evolve_next_generation(self, fitness_scores: List[float]) -> None:
        scores = np.asarray(fitness_scores, dtype=np.float64)
        sorted_indices = np.argsort(scores)[::-1]

        num_elites = max(1, int(self.pop_size * config.ELITISM_RATIO))
        elite_indices = sorted_indices[:num_elites]

        new_weights: List[NDArray[np.float64]] = [np.empty_like(w) for w in self.weights]
        new_biases: List[NDArray[np.float64]] = [np.empty_like(b) for b in self.biases]

        for l_idx in range(len(self.weights)):
            new_weights[l_idx][:num_elites] = self.weights[l_idx][elite_indices]
            new_biases[l_idx][:num_elites] = self.biases[l_idx][elite_indices]

        num_offspring = self.pop_size - num_elites
        if num_offspring > 0:
            parent1_indices = self._tournament_selection(scores, num_offspring)
            parent2_indices = self._tournament_selection(scores, num_offspring)

            for l_idx in range(len(self.weights)):
                mask_w = np.random.rand(*self.weights[l_idx][parent1_indices].shape) < 0.5
                mask_b = np.random.rand(*self.biases[l_idx][parent1_indices].shape) < 0.5

                child_w = np.where(
                    mask_w,
                    self.weights[l_idx][parent1_indices],
                    self.weights[l_idx][parent2_indices]
                )
                child_b = np.where(
                    mask_b,
                    self.biases[l_idx][parent1_indices],
                    self.biases[l_idx][parent2_indices]
                )

                mut_mask_w = np.random.rand(*child_w.shape) < config.MUTATION_RATE
                mut_mask_b = np.random.rand(*child_b.shape) < config.MUTATION_RATE

                child_w += mut_mask_w * np.random.normal(0.0, self.mutation_scale, size=child_w.shape)
                child_b += mut_mask_b * np.random.normal(0.0, self.mutation_scale, size=child_b.shape)

                new_weights[l_idx][num_elites:] = child_w
                new_biases[l_idx][num_elites:] = child_b

        self.weights = new_weights
        self.biases = new_biases

    def _tournament_selection(
        self, scores: NDArray[np.float64], num_selections: int, tournament_size: int = 3
    ) -> NDArray[np.int64]:
        selected = np.empty(num_selections, dtype=np.int64)
        for i in range(num_selections):
            competitors = np.random.choice(self.pop_size, size=tournament_size, replace=False)
            winner = competitors[np.argmax(scores[competitors])]
            selected[i] = winner
        return selected