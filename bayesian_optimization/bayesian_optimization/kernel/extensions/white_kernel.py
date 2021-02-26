from bayesian_optimization.kernel.extensions.abstract_decorator import AbstractDecorator
from sklearn.gaussian_process.kernels import WhiteKernel


class WhiteKernelDecorator(AbstractDecorator):
    """White Kernel decorator to add to other Kernels.
    """

    def __init__(
            self,
            kernel_to_decorate: 'Kernel',
            noise_level,
            noise_level_bounds,
    ):
        super().__init__(kernel_to_decorate)
        self.kernel = self.kernel_to_decorate.kernel + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds
        )

    @staticmethod
    def read_from_config(config, base_kernel) -> 'Kernel':
        """reads the configuration from a config object (from config file)
        and created Kernel accordingly

        :param config: configuration object
        :param base_kernel: kernen to which a white kernel schould be added
        :return: decorated kernel instance
        """
        return WhiteKernelDecorator(
            kernel_to_decorate=base_kernel,
            noise_level=config.as_float("noise_level"),
            noise_level_bounds=[float(i) for i in config.as_list("noise_level_bounds")]
        )