from bayesian_optimization.kernel.kernel import Kernel
import numpy as np
from typing import *
from typing import NoReturn


class AbstractDecorator(Kernel):
    """Abstract Decorator for a kernel
    """

    def __init__(self, kernel_to_decorate: 'Kernel') -> NoReturn:
        super().__init__()
        self.kernel_to_decorate = kernel_to_decorate


    @staticmethod
    def read_from_config(config, base_kernel) -> 'Kernel':
        pass