class Kernel:
    """Wrapper class to the kernels from sklearn.
    Allows the kernel to be generated from a config file.
    """

    def __init__(self):
        self.kernel = None

    @staticmethod
    def fix_params(params):
        pass

    @staticmethod
    def read_from_config(config: dict) -> 'Kernel':
        """reads the configuration from a config object (from config file)
        and created Kernel accordingly

        :param config: configuration object
        :return: Kernel instance
        """
        from . import SUPPORTED_KERNELS
        assert "kernel" in config, "config file must include 'Acquisition' section"
        from .extensions import SUPPORTED_EXTENSIONS
        base_kernel = SUPPORTED_KERNELS[config["kernel"]].read_from_config(config)

        added_extensions = {}
        for key, extension in SUPPORTED_EXTENSIONS.items():
            if key in config:
                order = config[key].as_int("order")
                added_extensions[order] = {"key": key, "extension": extension}
        for i in sorted(added_extensions):
            base_kernel = added_extensions[i]["extension"].read_from_config(config[added_extensions[i]["key"]], base_kernel)
        return base_kernel
