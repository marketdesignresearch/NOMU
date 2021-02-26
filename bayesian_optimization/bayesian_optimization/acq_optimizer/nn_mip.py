
# Libs
import docplex.mp.model as cpx
import tensorflow.keras as K
import numpy as np

# Internal
from bayesian_optimization.acq_optimizer.nn_acq_optimizer import NNACQOptimizer
from bayesian_optimization.estimators.single_bounded import SingleMethodBounded

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from configobj import ConfigObj

class NNMIP(NNACQOptimizer):
    """Class for finding the optima of a acquisition function based on the mean and variance predictions of a
    neuronal network function.
    """
    INSP_BASE = "opt_base_results"
    INSP_FINAL = "opt_final_results"

    CALLBACK = "opt_callback"

    INSP_X = "opt_grid"

    INSP_ARG_MAX = "opt_arg_max"
    INSP_MU_ARG_MAX = "opt_mu_arg_max"
    INSP_SIGMA_ARG_MAX = "opt_sigma_arg_max"
    INSP_ACQ_ARG_MAX = "opt_acq_arg_max"
    INSP_MU = "opt_mu"
    INSP_SIGMA = "opt_sigma"
    INSP_ACQ = "opt_acq"

    def __init__(
            self,
            lower_search_bounds: np.array,
            upper_search_bounds: np.array,
            big_m: int = 20,
            upper_bound: int = 10,
            time_limit: Union[int, None] = None,
            relative_gap: Union[float, None] = None,
            integrality: Union[float, None] = None,
            log_ouput: bool = False,
            tighten_IA: bool = False
    ) -> NoReturn:
        """default constructor

        :param lower_search_bounds: lower bounds of the search area must have same dimensions as the sample x space)
        :param upper_search_bounds: upper bounds of the search area must have same dimensions as the sample x space)
        :param big_m: M from the big M-constraint
        :param upper_bound: upper bounds for tighten_bounds_IA
        :param time_limit: time limit for the cplex solver
        :param relative_gap: relative gap limit for the cplex solver
        :param integrality: integrality limit for cplex solver
        :param log_ouput: log the cplex optimization logs
        :param tighten_IA: integrality limit for cplex solver
        """
        super().__init__(lower_search_bounds, upper_search_bounds)
        self.upper_bound = upper_bound
        self.y_main: dict = {}  # vector denoting which node in the corresponding layer is active
        self.z_main: dict = {}  # positive components of the output value o of each layer
        self.s_main: dict = {}  # slack variable representing absolute value of the negative components of output o
        self.y_side: dict = {}  # vector denoting which node in the corresponding layer is active
        self.z_side: dict = {}  # positive components of the output value o of each layer
        self.s_side: dict = {}  # slack variable representing absolute value of the negative components of output o
        self.big_l = big_m
        self.cpx_MIP = cpx.Model("NN_MIP")
        self.layers_main = None
        self.layers_side = None
        self.upper_bounds_z_main: dict = {}
        self.upper_bounds_s_main: dict = {}
        self.upper_bounds_z_side: dict = {}
        self.upper_bounds_s_side: dict = {}
        self.upper_search_bounds = upper_search_bounds
        self.lower_search_bounds = lower_search_bounds
        self.Model = None
        self.time_limit = time_limit
        self.relative_gap = relative_gap
        self.integrality = integrality
        self.log_ouput = log_ouput
        self.tighten_IA = tighten_IA

    def apply_MIP_parameters(self) -> NoReturn:
        """configures the cplex solver with the given parameters if they are specified
        :return:
        """
        if self.time_limit:
            self.cpx_MIP.set_time_limit(self.time_limit)
        if self.relative_gap:
            self.cpx_MIP.parameters.mip.tolerances.mipgap = self.relative_gap
        if self.integrality:
            self.cpx_MIP.parameters.mip.tolerances.integrality = self.integrality

    def get_optima(
            self,
            sample_x: np.ndarray,
            sample_y: np.ndarray,
            run_callback: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the optima of the acquisition function based on the predictions
        which are based on the given samples.

        :param sample_x: input values of the samples
        :param sample_y: target values of the samples
        :param run_callback: target values of the samples
        :return: optimal input value, mean prediction and optima, r prediction at optima
        """

        incumbent = np.max(sample_y)
        self.Model = self.context.estimator.fit(sample_x, sample_y)
        if self.tighten_IA:
            raise Exception('tighten_bounds_IA is currently not supported!')
            # self.tighten_bounds_IA(self.upper_bound)
        self.init_mip()
        self.apply_MIP_parameters()
        solution = self.cpx_MIP.solve(log_output=self.log_ouput)
        optima_x = self.keep_in_boundaries(np.asarray([self.z_main[(0, 0)].solution_value]))
        mus = np.asarray([self.z_main[(len(self.get_layers_main()), 0)].solution_value])
        sigmas = np.asarray([self.z_side[(len(self.get_layers_side()), 0)].solution_value])
        acq_values = self.context.acq.evaluate(mus, sigmas, incumbent)
        self._inspect(optima_x, mus, sigmas, acq_values)
        if run_callback and self.callback:
            raise Exception('The MIP Implementation currently does not support optimization callbacks!')
        return optima_x, sigmas, self.context.acq.evaluate(mu=mus, sigma=sigmas, incumbent=incumbent)

    def _inspect(
            self,
            arg_max: np.array,
            mu_arg_max: np.array,
            sigma_arg_max: np.array,
            acq_arg_max: np.array
    ) -> NoReturn:
        """save inspection data into the inspector.
        This data might be useful later on to verify the process

        :param arg_max: optima input value (input value that produces optimal target value)
        :param mu_arg_max: mean prediction at optima
        :param sigma_arg_max: sigma prediction and optima
        :param acq_arg_max: acquisition function value at optima
        :return:
        """
        if self.context.inspector is not None:
            self.context.inspector.dump_data(
                self.INSP_FINAL,
                {
                    self.INSP_ARG_MAX: arg_max,
                    self.INSP_MU_ARG_MAX: mu_arg_max,
                    self.INSP_SIGMA_ARG_MAX: sigma_arg_max,
                    self.INSP_ACQ_ARG_MAX: acq_arg_max,
                }
            )

    def keep_in_boundaries(self, optima_x: np.ndarray) -> np.ndarray:
        """makes sure that the optima found is within the boundaries,
        is can happen that to rounding issues the optima is slightly outside the boundaries.

        :param optima_x: proposed optima position
        :return: the boundary point closes to the proposed optima
        """
        cleaned_optima = []
        for i, entry in enumerate(optima_x):
            if entry > self.upper_search_bounds[i]:
                cleaned_optima.append(self.upper_search_bounds[i])
            if entry <= self.lower_search_bounds[i]:
                cleaned_optima.append(float(self.lower_search_bounds[i])+0.0001)
            else:
                cleaned_optima.append(entry)
        return np.asarray(optima_x)

    def get_layers(self) -> K.layers:
        """return all the layers of the Model which is given to the NNMIP
        :return: layers of the model
        """
        layers = self.Model.layers
        return [layer for layer in layers if (layer.get_config()["name"].startswith("hidden"))]

    def get_layers_main(self) -> K.layers:
        """return all the layers of the Model which is given to the NNMIP
        :return: layers of the model
        """
        layers = self.Model.layers
        return [layer for layer in layers if (layer.get_config()["name"].startswith("hidden") or layer.get_config()["name"].startswith("output_layer"))]

    def get_layers_side(self) -> K.layers:
        """return all the layers of the Model which is given to the NNMIP
        :return: layers of the model
        """
        layers = self.Model.layers
        return [layer for layer in layers if (layer.get_config()["name"].startswith("r_hidden") or layer.get_config()["name"].startswith("r_output_layer"))]

    @staticmethod
    def get_weight(layer) -> np.array:
        return layer.get_weights()[0]

    @staticmethod
    def get_bias(layer) -> np.array:
        return layer.get_weights()[1]

    def get_input_layer(self) -> K.layers:
        layers = self.Model.layers
        print([layer.get_config()["name"] for layer in layers])
        return [layer for layer in layers if layer.get_config()["name"].startswith("input")]

    def add_mip_variables_main(self) -> NoReturn:
        """instantiates all cplex variables for the main net
        """
        layers = self.get_layers_main()
        k = 1
        for layer in layers:
            W_t = NNMIP.get_weight(layer)
            W = W_t.transpose()
            rows, cols = W.shape
            self.z_main.update({(k, r): self.cpx_MIP.continuous_var(lb=0, name="z_main{}_{}".format(k, r)) for r in range(0, rows)})
            self.s_main.update({(k, r): self.cpx_MIP.continuous_var(lb=0, name="s_main{}_{}".format(k, r)) for r in range(0, rows)})
            self.y_main.update({(k, r): self.cpx_MIP.binary_var(name="y_main{}_{}".format(k, r)) for r in range(0, rows)})
            k += 1
        X = self.get_input_layer()
        self.z_main.update(
            {
                (0, r): self.cpx_MIP.continuous_var(
                    lb=self.lower_search_bounds[r],
                    ub=self.upper_search_bounds[r],
                    name="z_main{}_{}".format(0, r)) for r in range(0, X[0].get_config()["batch_input_shape"][1])
            })

    def add_mip_variables_side(self) -> NoReturn:
        """instantiates all cplex variables for the side net
        """
        layers = self.get_layers_side()
        k = 1
        for layer in layers:
            W_t = NNMIP.get_weight(layer)
            W = W_t.transpose()
            rows, cols = W.shape
            self.z_side.update({(k, r): self.cpx_MIP.continuous_var(lb=0, name="z_side{}_{}".format(k, r)) for r in range(0, rows)})
            self.s_side.update({(k, r): self.cpx_MIP.continuous_var(lb=0, name="s_side{}_{}".format(k, r)) for r in range(0, rows)})
            self.y_side.update({(k, r): self.cpx_MIP.binary_var(name="y_side{}_{}".format(k, r)) for r in range(0, rows)})
            k += 1
        X = self.get_input_layer()

        self.z_side.update(
            {
                (0, r): self.cpx_MIP.continuous_var(
                    lb=self.lower_search_bounds[r],
                    ub=self.upper_search_bounds[r],
                    name="z_side{}_{}".format(0, r)) for r in range(0, X[0].get_config()["batch_input_shape"][1])})
        # input into main and side net should be identical
        for r in range(0, X[0].get_config()["batch_input_shape"][1]):
            self.cpx_MIP.add_constraint(
                ct=self.z_main[(0, r)] == self.z_side[(0, r)])

    def add_zs_WZb_constraints_main(self, layer: int) -> NoReturn:
        """Creates the main constraint of the main net
        """
        W_t = NNMIP.get_weight(self.layers_main[layer])
        b = NNMIP.get_bias(self.layers_main[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.cpx_MIP.sum(W[row, col] * self.z_main[(layer, col)] for col in range(0,cols)) + b[row] == self.z_main[(layer+1, row)] - self.s_main[(layer+1, row)])
            if len(self.upper_bounds_z_main) > 0:
                self.cpx_MIP.add_constraint(
                    ct=self.z_main[(layer+1, row)] <= self.y_main[(layer+1, row)] * self.upper_bounds_z_main[layer+1][row][0],
                    ctname="BinaryCT_Layer{}_Row{}_Z_main".format(layer, row))
            if len(self.upper_bounds_s_main) > 0:
                self.cpx_MIP.add_constraint(
                    ct=self.s_main[(layer+1, row)] <= (1 - self.y_main[(layer+1, row)]) * self.upper_bounds_s_main[layer+1][row][0],
                    ctname="BinaryCT_Layer{}_Row{}_S_main".format(layer, row))

    def add_zs_WZb_constraints_side(self, layer: int) -> NoReturn:
        """Creates the main constraint of the side net
        """
        W_t = NNMIP.get_weight(self.layers_side[layer])
        b = NNMIP.get_bias(self.layers_side[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.cpx_MIP.sum(W[row, col] * self.z_side[(layer, col)] for col in range(0,cols)) + b[row] == self.z_side[(layer+1, row)] - self.s_side[(layer+1, row)])
            if len(self.upper_bounds_z_side) > 0:
                self.cpx_MIP.add_constraint(
                    ct=self.z_side[(layer+1, row)] <= self.y_side[(layer+1, row)] * self.upper_bounds_z_side[layer+1][row][0],
                    ctname="BinaryCT_Layer{}_Row{}_Z_side".format(layer, row))
            if len(self.upper_bounds_s_side) > 0:
                self.cpx_MIP.add_constraint(
                    ct=self.s_side[(layer+1, row)] <= (1 - self.y_side[(layer+1, row)]) * self.upper_bounds_s_side[layer+1][row][0],
                    ctname="BinaryCT_Layer{}_Row{}_S_side".format(layer, row))


    def add_z_smaller_yM_constraint_main(self, layer: int) -> NoReturn:
        """Creates the big M constraint for z of the main net
        """
        W_t = NNMIP.get_weight(self.layers_main[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.z_main[(layer+1, row)] <= self.y_main[(layer+1, row)]*self.big_l)

    def add_z_smaller_yM_constraint_side(self, layer: int) -> NoReturn:
        """Creates the big M constraint for z of the side net
        """
        W_t = NNMIP.get_weight(self.layers_side[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.z_side[(layer+1, row)] <= self.y_side[(layer+1, row)]*self.big_l)


    def add_s_smaller_1minusyM_constraint_main(self, layer: int) -> NoReturn:
        """Creates the big M constraint for s of the main net
        """
        W_t = NNMIP.get_weight(self.layers_main[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.s_main[(layer+1, row)] <= (1-self.y_main[(layer+1, row)])*self.big_l)

    def add_s_smaller_1minusyM_constraint_side(self, layer: int) -> NoReturn:
        """Creates the big M constraint for s of the side net
        """
        W_t = NNMIP.get_weight(self.layers_side[layer])
        W = W_t.transpose()
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.s_side[(layer+1, row)] <= (1-self.y_side[(layer+1, row)])*self.big_l)


    def add_last_layer_constraint_main(self) -> NoReturn:
        """Add the constraint for the last layer of the main net which is responsible for the
        residual r.
        """
        last_index = len(self.layers_main)
        W_t = NNMIP.get_weight(self.layers_main[-1])
        b = NNMIP.get_bias(self.layers_main[-1])
        W = W_t.transpose()
        rows, cols = W.shape
        print("row/col", rows, cols)
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.cpx_MIP.sum(W[row, col] * self.z_main[(last_index-1, col)] for col in range(0, cols)) + b[row] == self.z_main[(last_index, row)])

    def add_last_layer_constraint_side(self) -> NoReturn:
        """Add the constraint for the last layer of the side net which is responsible for the
        residual r.
        """
        last_index = len(self.layers_side)
        W_t = NNMIP.get_weight(self.layers_side[-1])
        b = NNMIP.get_bias(self.layers_side[-1])
        print("bbb", b)
        W = W_t.transpose()
        print("www", W)
        rows, cols = W.shape
        for row in range(0, rows):
            self.cpx_MIP.add_constraint(ct=self.cpx_MIP.sum(W[row, col] * self.z_main[(last_index-1, col)] for col in range(0, int(cols/2))) + self.cpx_MIP.sum(W[row, int(cols/2) + col] * self.z_side[(last_index-1, col)] for col in range(0, int(cols/2))) + b[row] == self.z_side[(last_index, row)])

    def add_objective(self) -> NoReturn:
        """defines the objective function in terms of the available variables
        and adds it to the mixed Integer Problem
        """
        last_index_main = len(self.layers_main)
        last_index_side = len(self.layers_side)
        if isinstance(self.context.estimator, SingleMethodBounded) and self.context.estimator.mip:
            objective = self.context.acq.single_expression(
                self.cpx_MIP,
                self.z_main[(last_index_main,0)],
                self.context.estimator.mip_activation_expression(self.cpx_MIP, self.z_side[(last_index_side,0)]),
                0.0)
        else:
            objective = self.context.acq.single_expression(self.cpx_MIP, self.z_main[(last_index_main,0)], self.z_side[(last_index_side,0)], 0.0)
        self.cpx_MIP.maximize(objective)

    def init_mip(self) -> NoReturn:
        """initialize the mixed integer problem
        """
        self.add_mip_variables_main()
        self.layers_main = self.get_layers_main()
        for i, layer in enumerate(self.layers_main[:-1]):
            self.add_zs_WZb_constraints_main(i)
            self.add_z_smaller_yM_constraint_main(i)
            self.add_s_smaller_1minusyM_constraint_main(i)
        self.add_last_layer_constraint_main()
        self.add_mip_variables_side()
        self.layers_side = self.get_layers_side()
        for i, layer in enumerate(self.layers_side[:-1]):
            self.add_zs_WZb_constraints_side(i)
            self.add_z_smaller_yM_constraint_side(i)
            self.add_s_smaller_1minusyM_constraint_side(i)
        self.add_last_layer_constraint_side()

        self.add_objective()
        self.cpx_MIP.parameters.mip.tolerances.integrality.set(1e-8)

    def solve(self):
        # for m in range(0, self.cpx_MIP.number_of_constraints):
        #     if self.cpx_MIP.get_constraint_by_index(m) is not None:
        #         print(self.cpx_MIP.get_constraint_by_index(m))
        sol = self.cpx_MIP.solve(log_output=True)
        return self.z_main[(0, 0)], self.z_main[(len(self.get_layers_main()), 0)], self.z_side[(0, 0)], self.z_side[(len(self.get_layers_side()), 0)]

    # Tighten Bound IA is not fully compatible mit the NOMU structure with the side net

    # def tighten_bounds_IA(self, upper_bound):
    #     for i_layer in self.get_input_layer():
    #         self.upper_bounds_z_main[0] = np.array(upper_bound).reshape(-1, 1)
    #         self.upper_bounds_s_main[0] = np.array(upper_bound).reshape(-1, 1)
    #     for i, layer in enumerate(self.get_layers_main()):
    #         W_t = NNMIP.get_weight(layer)
    #         W = W_t.transpose()
    #         W_plus = np.maximum(W, 0)
    #         W_minus = np.minimum(W, 0)
    #         b = NNMIP.get_bias(layer)
    #         self.upper_bounds_z_main[i+1] = np.ceil(
    #             np.maximum(W_plus @ self.upper_bounds_z_main[i] + b.reshape(-1, 1), 0)).astype(
    #             int)  # upper bound for z
    #         self.upper_bounds_s_main[i+1] = np.ceil(
    #             np.maximum(-(W_minus @ self.upper_bounds_z_main[i] + b.reshape(-1, 1)), 0)).astype(
    #             int)  # upper bound  for s
    #
    #
    #     for i_layer in self.get_input_layer():
    #         self.upper_bounds_z_side[0] = np.array(upper_bound).reshape(-1, 1)
    #         self.upper_bounds_s_side[0] = np.array(upper_bound).reshape(-1, 1)
    #
    #     for i, layer in enumerate(self.get_layers_side()):
    #         W_t = NNMIP.get_weight(layer)
    #         W = W_t.transpose()
    #         W_plus = np.maximum(W, 0)
    #         W_minus = np.minimum(W, 0)
    #         b = NNMIP.get_bias(layer)
    #
    #
    #         if i == len(self.get_layers_side())-1:
    #             # last layer
    #             self.upper_bounds_z_side[i + 1] = np.ceil(
    #                 np.maximum(W_plus @ [[x] for x in np.ravel(np.column_stack((self.upper_bounds_z_main[i],self.upper_bounds_z_side[i])))] + b.reshape(-1, 1), 0)).astype(
    #                 int)  # upper bound for z
    #             self.upper_bounds_s_side[i + 1] = np.ceil(
    #                 np.maximum(-(W_minus @ [[x] for x in np.ravel(np.column_stack((self.upper_bounds_s_main[i],self.upper_bounds_s_side[i])))] + b.reshape(-1, 1)), 0)).astype(
    #                 int)  # upper bound  for s
    #         else:
    #             self.upper_bounds_z_side[i + 1] = np.ceil(
    #                 np.maximum(W_plus @ self.upper_bounds_z_side[i] + b.reshape(-1, 1), 0)).astype(
    #                 int)  # upper bound for z
    #             self.upper_bounds_s_side[i + 1] = np.ceil(
    #                 np.maximum(-(W_minus @ self.upper_bounds_z_side[i] + b.reshape(-1, 1)), 0)).astype(
    #                 int)  # upper bound  for s

    @classmethod
    def read_from_config(cls, config: 'ConfigObj') -> 'NNMIP':
        """reads the configuration of the gridsearch from the given configfile and returns the
        accordingly created grid search instance
        :param config: config parser instance
        :return: grid search instance
        """
        supported_callbacks = {
        }
        callback = None

        for key, extension in supported_callbacks.items():
            if key in config:
                callback = supported_callbacks[key].read_from_config(config[key])

        lower_bounds = np.array([float(i) for i in config.as_list("lower_search_bounds")])
        upper_bounds = np.array([float(i) for i in config.as_list("upper_search_bounds")])
        return cls(
            lower_search_bounds=lower_bounds,
            upper_search_bounds=upper_bounds,
            tighten_IA=config.as_bool("tighten_IA"),
            relative_gap=1e-2
        )