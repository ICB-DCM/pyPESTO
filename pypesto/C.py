"""
Constants
=========
Package-wide consistent constant definitions.
"""

from enum import Enum
from typing import Callable, Tuple, Union

###############################################################################
# OBJECTIVE

MODE_FUN = 'mode_fun'  # mode for function values
MODE_RES = 'mode_res'  # mode for residuals
FVAL = 'fval'  # function value
FVAL0 = 'fval0'  # function value at start
GRAD = 'grad'  # gradient
HESS = 'hess'  # Hessian
HESSP = 'hessp'  # Hessian vector product
RES = 'res'  # residual
SRES = 'sres'  # residual sensitivities
RDATAS = 'rdatas'  # returned simulated data sets

TIME = 'time'  # time
N_FVAL = 'n_fval'  # number of function evaluations
N_GRAD = 'n_grad'  # number of gradient evaluations
N_HESS = 'n_hess'  # number of Hessian evaluations
N_RES = 'n_res'  # number of residual evaluations
N_SRES = 'n_sres'  # number of residual sensitivity evaluations
CHI2 = 'chi2'  # chi2 value
SCHI2 = 'schi2'  # chi2 value gradient
X = 'x'
X0 = 'x0'
ID = 'id'

EXITFLAG = 'exitflag'
MESSAGE = 'message'


###############################################################################
# PRIOR

LIN = 'lin'  # linear
LOG = 'log'  # logarithmic to basis e
LOG10 = 'log10'  # logarithmic to basis 10

UNIFORM = 'uniform'
PARAMETER_SCALE_UNIFORM = 'parameterScaleUniform'
NORMAL = 'normal'
PARAMETER_SCALE_NORMAL = 'parameterScaleNormal'
LAPLACE = 'laplace'
PARAMETER_SCALE_LAPLACE = 'parameterScaleLaplace'
LOG_UNIFORM = 'logUniform'
LOG_NORMAL = 'logNormal'
LOG_LAPLACE = 'logLaplace'


###############################################################################
# PREDICT

OUTPUT_IDS = 'output_ids'  # data member in PredictionConditionResult
PARAMETER_IDS = 'x_names'  # data member in PredictionConditionResult
TIMEPOINTS = 'timepoints'  # data member in PredictionConditionResult
OUTPUT = 'output'  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = 'output_sensi'  # field in the return dict of AmiciPredictor
OUTPUT_WEIGHT = 'output_weight'  # field in the return dict of AmiciPredictor
OUTPUT_SIGMAY = 'output_sigmay'  # field in the return dict of AmiciPredictor

# separator in the conditions_ids between preequilibration and simulation
# condition
CONDITION_SEP = '::'

AMICI_T = 't'  # return field in amici simulation result
AMICI_X = 'x'  # return field in amici simulation result
AMICI_SX = 'sx'  # return field in amici simulation result
AMICI_Y = 'y'  # return field in amici simulation result
AMICI_SY = 'sy'  # return field in amici simulation result
AMICI_LLH = 'llh'  # return field in amici simulation result
AMICI_STATUS = 'status'  # return field in amici simulation result
AMICI_SIGMAY = 'sigmay'  # return field in amici simulation result

CONDITION = 'condition'
CONDITION_IDS = 'condition_ids'

CSV = 'csv'  # return file format
H5 = 'h5'  # return file format


###############################################################################
# ENSEMBLE

PREDICTOR = 'predictor'
PREDICTION_ID = 'prediction_id'
PREDICTION_RESULTS = 'prediction_results'
PREDICTION_ARRAYS = 'prediction_arrays'
PREDICTION_SUMMARY = 'prediction_summary'

HISTORY = 'history'
OPTIMIZE = 'optimize'
SAMPLE = 'sample'

MEAN = 'mean'
MEDIAN = 'median'
STANDARD_DEVIATION = 'std'
PERCENTILE = 'percentile'
SUMMARY = 'summary'
WEIGHTED_SIGMA = 'weighted_sigma'

X_NAMES = 'x_names'
NX = 'n_x'
X_VECTOR = 'x_vectors'
NVECTORS = 'n_vectors'
VECTOR_TAGS = 'vector_tags'
ENSEMBLE_TYPE = 'ensemble_type'
PREDICTIONS = 'predictions'

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'
PREEQUILIBRATION_CONDITION_ID = 'preequilibrationConditionId'
SIMULATION_CONDITION_ID = 'simulationConditionId'

COLOR_HIT_BOTH_BOUNDS = [0.6, 0.0, 0.0, 0.9]
COLOR_HIT_ONE_BOUND = [0.95, 0.6, 0.0, 0.9]
COLOR_HIT_NO_BOUNDS = [0.0, 0.8, 0.0, 0.9]


class EnsembleType(Enum):
    """Specifies different ensemble types."""

    ensemble = 1
    sample = 2
    unprocessed_chain = 3


###############################################################################
# SELECT

TYPE_POSTPROCESSOR = Callable[["ModelProblem"], None]  # noqa: F821


###############################################################################
# VISUALIZE

LEN_RGB = 3  # number of elements in an RGB color
LEN_RGBA = 4  # number of elements in an RGBA color
RGB = Tuple[(float,) * LEN_RGB]  # typing of an RGB color
RGBA = Tuple[(float,) * LEN_RGBA]  # typing of an RGBA color
RGB_RGBA = Union[RGB, RGBA]  # typing of an RGB or RGBA color
RGBA_MIN = 0  # min value for an RGBA element
RGBA_MAX = 1  # max value for an RGBA element
RGBA_ALPHA = 3  # zero-indexed fourth element in RGBA
RGBA_WHITE = (RGBA_MAX, RGBA_MAX, RGBA_MAX, RGBA_MAX)  # white as an RGBA color
RGBA_BLACK = (RGBA_MIN, RGBA_MIN, RGBA_MIN, RGBA_MAX)  # black as an RGBA color

# optimizer history
TRACE_X_TIME = 'time'
TRACE_X_STEPS = 'steps'
# supported values to plot on x-axis
TRACE_X = (TRACE_X_TIME, TRACE_X_STEPS)

TRACE_Y_FVAL = 'fval'
TRACE_Y_GRADNORM = 'gradnorm'
# supported values to plot on y-axis
TRACE_Y = (TRACE_Y_FVAL, TRACE_Y_GRADNORM)


###############################################################################
# Environment variables

PYPESTO_MAX_N_STARTS: str = "PYPESTO_MAX_N_STARTS"
PYPESTO_MAX_N_SAMPLES: str = "PYPESTO_MAX_N_SAMPLES"
