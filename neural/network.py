"""
Sequential multi-layer perceptron (MLP) architecture builder.
"""

from typing import List, Any
import numpy as np
from numpy.typing import NDArray

import config
from neural.layers import NeuralDenseLayer
from neural.activations import (
    ActivationReLU,
    ActivationTanh,
    ActivationSigmoid
)


class NeuralNetwork:
    """
    Manages sequential data flow through dense layers and activation modules.
    """

    def __init__(
        self,
        input_size: int = config.VISION_RAYS + 4,
        hidden_layers: int = config.NEURAL_HIDDEN_LAYERS,
        neurons: int = config.NEURAL_NEURONS,
        output_size: int = 2
    ) -> None:
        """
        Constructs sequential multi-layer MLP topology.
        """
        self.layers: List[NeuralDenseLayer] = []
        self.activations: List[Any] = []

        # Input to first hidden layer
        self.layers.append(NeuralDenseLayer(input_size, neurons))
        self.activations.append(ActivationReLU())

        # Intermediary hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(NeuralDenseLayer(neurons, neurons))
            self.activations.append(ActivationReLU())

        # Hidden layer to output layer
        self.layers.append(NeuralDenseLayer(neurons, output_size))
        self.activations.append(None)  # Output activations handled in forward

        self.out_sigmoid: ActivationSigmoid = ActivationSigmoid()
        self.out_tanh: ActivationTanh = ActivationTanh()

    def forward(self, input_data: NDArray) -> NDArray[np.float64]:
        """
        Executes sequential forward pass returning [move_effort, turn_effort].
        """
        curr: NDArray[np.float64] = (
            input_data.astype(np.float64) if input_data.ndim == 2
            else input_data[np.newaxis, :].astype(np.float64)
        )

        for i in range(len(self.layers)):
            curr = self.layers[i].forward(curr)
            if self.activations[i] is not None:
                curr = self.activations[i].forward(curr)

        # Output Channel 0: Move Effort (Sigmoid [0, 1])
        # Output Channel 1: Turn Effort (Tanh [-1, 1])
        move_effort: NDArray[np.float64] = self.out_sigmoid.forward(
            curr[:, 0:1]
        )
        turn_effort: NDArray[np.float64] = self.out_tanh.forward(
            curr[:, 1:2]
        )

        return np.hstack([move_effort, turn_effort])
