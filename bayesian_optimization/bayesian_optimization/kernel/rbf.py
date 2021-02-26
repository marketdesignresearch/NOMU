from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from bayesian_optimization.kernel.kernel import Kernel


class RBF_Kernel(Kernel):
    """Wrapper to the sklearn RBF kernel allowing to be
    decorated with other kernel and to be constructed according to a config file
    """

    def __init__(
            self,
            constant_value,
            constant_value_bounds,
            length_scale,
            length_scale_bounds
    ):
        super().__init__()
        self.kernel = C(
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds
        ) \
            * RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            )

    @staticmethod
    def read_from_config(config):
        """reads the configuration from a config object (from config file)
        and created Kernel accordingly

        :param config: configuration object
        :return: Kernel instance
        """
        return RBF_Kernel(
            constant_value=[float(i) for i in config.as_list("constant_value")],
            constant_value_bounds=[float(i) for i in config.as_list("constant_value_bounds")],
            length_scale=[float(i) for i in config.as_list("length_scale")],
            length_scale_bounds=[float(i) for i in config.as_list("length_scale_bounds")],
        )

    @staticmethod
    def fix_params(params):
        """fixes certain parameter so that they can no longer be newly learned
        :param params:
        :return:
        """
        params['k2__length_scale_bounds'] = "fixed"
        return params


