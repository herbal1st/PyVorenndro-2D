"""
Genetic algorithm population manager executing reproduction and mutation.
"""

import random
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

import config
from neural.network import NeuralNetwork


class PopulationManager:
    """
    Manages neural network weight matrices across evolutionary generations.
    """

    def __init__(
        self,
        pop_size: int = config.POPULATION_SIZE,
        mutation_rate: float = config.MUTATION_RATE,
        mutation_scale: float = config.MUTATION_SCALE,
        elitism_ratio: float = config.ELITISM_RATIO
    ) -> None:
        """
        Initializes population parameters and candidate networks.
        """
        self.pop_size: int = pop_size
        self.mutation_rate: float = mutation_rate
        self.mutation_scale: float = mutation_scale
        self.elitism_ratio: float = elitism_ratio
        self.networks: List[NeuralNetwork] = [
            NeuralNetwork() for _ in range(pop_size)
        ]

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
        elite_indices: List[int] = [
            idx for idx, _ in indexed_scores[:num_elites]
        ]

        new_networks: List[NeuralNetwork] = []

        # Elitism: Copy elite weights directly without mutation
        for idx in elite_indices:
            elite_net = NeuralNetwork()
            self._copy_weights(self.networks[idx], elite_net)
            new_networks.append(elite_net)

        # Breed offspring to fill remaining population
        while len(new_networks) < self.pop_size:
            parent_a = self._tournament_select(fitness_scores)
            parent_b = self._tournament_select(fitness_scores)

            child_net = NeuralNetwork()
            self._crossover_and_mutate(parent_a, parent_b, child_net)
            new_networks.append(child_net)

        self.networks = new_networks

    def _tournament_select(
        self,
        scores: List[float],
        k: int = 3
    ) -> NeuralNetwork:
        """
        Selects top network candidate from k random tournament entries.
        """
        chosen_indices: List[int] = random.sample(
            range(self.pop_size), min(k, self.pop_size)
        )
        best_idx: int = max(chosen_indices, key=lambda idx: scores[idx])
        return self.networks[best_idx]

    def _copy_weights(
        self,
        src_net: NeuralNetwork,
        dest_net: NeuralNetwork
    ) -> None:
        """
        Copies weight and bias matrices from source to destination.
        """
        for i in range(len(src_net.layers)):
            dest_net.layers[i].weights = src_net.layers[i].weights.copy()
            dest_net.layers[i].biases = src_net.layers[i].biases.copy()

    def _crossover_and_mutate(
        self,
        parent_a: NeuralNetwork,
        parent_b: NeuralNetwork,
        child_net: NeuralNetwork
    ) -> None:
        """
        Applies uniform crossover and Gaussian mutation to child weights.
        """
        for i in range(len(parent_a.layers)):
            wa = parent_a.layers[i].weights
            wb = parent_b.layers[i].weights
            ba = parent_a.layers[i].biases
            bb = parent_b.layers[i].biases

            mask_w: NDArray[np.bool_] = (
                np.random.rand(*wa.shape) < 0.5
            )
            mask_b: NDArray[np.bool_] = (
                np.random.rand(*ba.shape) < 0.5
            )

            cw = np.where(mask_w, wa, wb)
            cb = np.where(mask_b, ba, bb)

            if random.random() < self.mutation_rate:
                cw += np.random.normal(
                    0.0, self.mutation_scale, size=cw.shape
                )
                cb += np.random.normal(
                    0.0, self.mutation_scale, size=cb.shape
                )

            child_net.layers[i].weights = cw
            child_net.layers[i].biases = cb
