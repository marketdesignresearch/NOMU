from .mean import MeanOnly
from .uncertainty import UncertaintyOnly
from .upperbound import UpperBound
from .poi import ProbabilityOfImprovement
from .ei import ExpectedImprovement

from .decorators.mean_width_scale import MeanWidthScaled
from .decorators.mean_width_scale_mc import MeanWidthScaledMC
from .decorators.bounded_r import BoundedR

SUPPORTED_ACQS = {
    "mean_only": MeanOnly,
    "uncertainty_only": UncertaintyOnly,
    "upper_bound": UpperBound,
    "probability_of_improvement": ProbabilityOfImprovement,
    "expected_improvement": ExpectedImprovement,
}

SUPPORTED_EXTENSIONS = {
    "mean_width_scaling": MeanWidthScaled,
    "mean_width_scaling_mc": MeanWidthScaledMC,
    "bounded_r": BoundedR
}
