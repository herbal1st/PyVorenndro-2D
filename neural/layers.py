"""
Vectorized dense linear layer for neural network forward transformations.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class NeuralDenseLayer:
    """
    Fully connected linear transformation layer (W * X + b).
    """

    def __init__(self, input_count: int, neuron_count: int) -> None:
        """
        Initializes weight and bias matrices using Gaussian distribution.
        """
        scale: float = np.sqrt(2.0 / float(input_count + neuron_count))
        self.weights: NDArray[np.float64] = scale * np.random.randn(
            input_count, neuron_count
        )
        self.biases: NDArray[np.float64] = np.zeros(
            (1, neuron_count), dtype=np.float64
        )
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Executes linear projection forward calculation.
        """
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output
