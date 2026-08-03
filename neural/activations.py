"""
Vectorized activation modules with floating-point arithmetic overflow guards.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class ActivationReLU:
    """
    Rectified Linear Unit activation function.
    """

    def __init__(self) -> None:
        """
        Initializes activation state.
        """
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Applies max(0, x) element-wise.
        """
        self.output = np.maximum(0.0, input_data)
        return self.output


class ActivationTanh:
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self) -> None:
        """
        Initializes activation state.
        """
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Applies tanh(x) element-wise.
        """
        self.output = np.tanh(input_data)
        return self.output


class ActivationSigmoid:
    """
    Logistic sigmoid activation function with overflow protection.
    """

    def __init__(self) -> None:
        """
        Initializes activation state.
        """
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Applies standard sigmoid with [-500, 500] clipping.
        """
        clipped: NDArray[np.float64] = np.clip(input_data, -500.0, 500.0)
        self.output = 1.0 / (1.0 + np.exp(-clipped))
        return self.output
