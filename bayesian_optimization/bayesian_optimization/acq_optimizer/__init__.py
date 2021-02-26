from .gridsearch import GridSearch
from .direct_optimizer import DirectOptimizer
from .nn_mip import NNMIP

SUPPORTED_ACQ_OPTIMIZERS = {
    "grid_search": GridSearch,
    "direct": DirectOptimizer,
    "mip": NNMIP,
}