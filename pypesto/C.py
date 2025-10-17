"""
Constants
=========
Package-wide consistent constant definitions.
"""

from enum import Enum
from typing import Literal

###############################################################################
# ENSEMBLE

PREDICTOR = "predictor"
PREDICTION_ID = "prediction_id"
PREDICTION_RESULTS = "prediction_results"
PREDICTION_ARRAYS = "prediction_arrays"
PREDICTION_SUMMARY = "prediction_summary"

HISTORY = "history"
OPTIMIZE = "optimize"
SAMPLE = "sample"

MEAN = "mean"
MEDIAN = "median"
STANDARD_DEVIATION = "std"
PERCENTILE = "percentile"
SUMMARY = "summary"
WEIGHTED_SIGMA = "weighted_sigma"

X_NAMES = "x_names"
NX = "n_x"
X_VECTOR = "x_vectors"
NVECTORS = "n_vectors"
VECTOR_TAGS = "vector_tags"
ENSEMBLE_TYPE = "ensemble_type"
PREDICTIONS = "predictions"

SIMULTANEOUS = "simultaneous"
POINTWISE = "pointwise"

LOWER_BOUND = "lower_bound"
UPPER_BOUND = "upper_bound"
PREEQUILIBRATION_CONDITION_ID = "preequilibrationConditionId"
SIMULATION_CONDITION_ID = "simulationConditionId"

COLOR_HIT_BOTH_BOUNDS = [0.6, 0.0, 0.0, 0.9]
COLOR_HIT_ONE_BOUND = [0.95, 0.6, 0.0, 0.9]
COLOR_HIT_NO_BOUNDS = [0.0, 0.8, 0.0, 0.9]


class EnsembleType(Enum):
    """Specifies different ensemble types."""

    ensemble = 1
    sample = 2
    unprocessed_chain = 3


###############################################################################
# OBJECTIVE

MODE_FUN = "mode_fun"  # mode for function values
MODE_RES = "mode_res"  # mode for residuals
ModeType = Literal["mode_fun", "mode_res"]  # type for `mode` argument
FVAL = "fval"  # function value
FVAL0 = "fval0"  # function value at start
GRAD = "grad"  # gradient
HESS = "hess"  # Hessian
HESSP = "hessp"  # Hessian vector product
RES = "res"  # residual
SRES = "sres"  # residual sensitivities
RDATAS = "rdatas"  # returned simulated data sets
OBJECTIVE_NEGLOGPOST = "neglogpost"  # objective is negative log-posterior
OBJECTIVE_NEGLOGLIKE = "negloglike"  # objective is negative log-likelihood

TIME = "time"  # time
N_FVAL = "n_fval"  # number of function evaluations
N_GRAD = "n_grad"  # number of gradient evaluations
N_HESS = "n_hess"  # number of Hessian evaluations
N_RES = "n_res"  # number of residual evaluations
N_SRES = "n_sres"  # number of residual sensitivity evaluations
START_TIME = "start_time"  # start time
X = "x"
X0 = "x0"
ID = "id"

AMICI = "amici"
ROADRUNNER = "roadrunner"
PETAB = "petab"


###############################################################################
# HIERARCHICAL SCALING + OFFSET

INNER_PARAMETERS = "inner_parameters"
PARAMETER_TYPE = "parameterType"
RELATIVE = "relative"


class InnerParameterType(str, Enum):
    """Specifies different inner parameter types."""

    OFFSET = "offset"
    SCALING = "scaling"
    SIGMA = "sigma"
    ORDINAL = "ordinal"
    SPLINE = "spline"


DUMMY_INNER_VALUE = {
    InnerParameterType.OFFSET: 0.0,
    InnerParameterType.SCALING: 1.0,
    InnerParameterType.SIGMA: 1.0,
    InnerParameterType.ORDINAL: 0.0,
    InnerParameterType.SPLINE: 0.0,
}

INNER_PARAMETER_BOUNDS = {
    InnerParameterType.OFFSET: {
        LOWER_BOUND: -float("inf"),
        UPPER_BOUND: float("inf"),
    },
    InnerParameterType.SCALING: {
        LOWER_BOUND: -float("inf"),
        UPPER_BOUND: float("inf"),
    },
    InnerParameterType.SIGMA: {
        LOWER_BOUND: 0,
        UPPER_BOUND: float("inf"),
    },
    InnerParameterType.ORDINAL: {
        LOWER_BOUND: -float("inf"),
        UPPER_BOUND: float("inf"),
    },
    InnerParameterType.SPLINE: {
        LOWER_BOUND: -float("inf"),
        UPPER_BOUND: float("inf"),
    },
}

###############################################################################
# OPTIMAL SCALING

# Should go to PEtab constants at some point
MEASUREMENT_CATEGORY = "measurementCategory"
MEASUREMENT_TYPE = "measurementType"
CENSORING_BOUNDS = "censoringBounds"

ORDINAL = "ordinal"
CENSORED = "censored"
LEFT_CENSORED = "left-censored"
RIGHT_CENSORED = "right-censored"
INTERVAL_CENSORED = "interval-censored"
CENSORING_TYPES = [LEFT_CENSORED, RIGHT_CENSORED, INTERVAL_CENSORED]

REDUCED = "reduced"
STANDARD = "standard"
MAXMIN = "max-min"
MAX = "max"

METHOD = "method"
REPARAMETERIZED = "reparameterized"
INTERVAL_CONSTRAINTS = "interval_constraints"
MIN_GAP = "min_gap"
ORDINAL_OPTIONS = [
    METHOD,
    REPARAMETERIZED,
    INTERVAL_CONSTRAINTS,
    MIN_GAP,
]

CAT_LB = "cat_lb"
CAT_UB = "cat_ub"

NUM_CATEGORIES = "num_categories"
NUM_DATAPOINTS = "num_datapoints"
SURROGATE_DATA = "surrogate_data"
NUM_INNER_PARAMS = "num_inner_params"
LB_INDICES = "lb_indices"
UB_INDICES = "ub_indices"

QUANTITATIVE_IXS = "quantitative_ixs"
QUANTITATIVE_DATA = "quantitative_data"
NUM_CONSTR_FULL = "num_constr_full"
C_MATRIX = "C_matrix"
W_MATRIX = "W_matrix"
W_DOT_MATRIX = "W_dot_matrix"

SCIPY_SUCCESS = "success"
SCIPY_FUN = "fun"
SCIPY_X = "x"

###############################################################################
# SPLINE APPROXIMATION FOR SEMIQUANTITATIVE DATA

MEASUREMENT_TYPE = "measurementType"

SEMIQUANTITATIVE = "semiquantitative"

SPLINE_RATIO = "spline_ratio"
MIN_DIFF_FACTOR = "min_diff_factor"
REGULARIZE_SPLINE = "regularize_spline"
REGULARIZATION_FACTOR = "regularization_factor"
SPLINE_APPROXIMATION_OPTIONS = [
    SPLINE_RATIO,
    MIN_DIFF_FACTOR,
    REGULARIZE_SPLINE,
    REGULARIZATION_FACTOR,
]

MIN_SIM_RANGE = 1e-16

SPLINE_PAR_TYPE = "spline"
SPLINE_KNOTS = "spline_knots"
N_SPLINE_PARS = "n_spline_pars"
DATAPOINTS = "datapoints"
MIN_DATAPOINT = "min_datapoint"
MAX_DATAPOINT = "max_datapoint"
EXPDATA_MASK = "expdata_mask"
CURRENT_SIMULATION = "current_simulation"
INNER_NOISE_PARS = "inner_noise_pars"
OPTIMIZE_NOISE = "optimize_noise"


###############################################################################
# HISTORY

HISTORY = "history"
TRACE = "trace"
N_ITERATIONS = "n_iterations"
MESSAGES = "messages"
MESSAGE = "message"
EXITFLAG = "exitflag"
TRACE_SAVE_ITER = "trace_save_iter"

SUFFIXES_CSV = ["csv"]
SUFFIXES_HDF5 = ["hdf5", "h5"]
SUFFIXES = SUFFIXES_CSV + SUFFIXES_HDF5

CPU_TIME_TOTAL = "cpu_time_total"
PREEQ_CPU_TIME = "preeq_cpu_time"
PREEQ_CPU_TIME_BACKWARD = "preeq_cpu_timeB"
POSTEQ_CPU_TIME = "posteq_cpu_time"
POSTEQ_CPU_TIME_BACKWARD = "posteq_cpu_timeB"


###############################################################################
# PRIOR

LIN = "lin"  # linear
LOG = "log"  # logarithmic to basis e
LOG10 = "log10"  # logarithmic to basis 10

UNIFORM = "uniform"
PARAMETER_SCALE_UNIFORM = "parameterScaleUniform"
NORMAL = "normal"
PARAMETER_SCALE_NORMAL = "parameterScaleNormal"
LAPLACE = "laplace"
PARAMETER_SCALE_LAPLACE = "parameterScaleLaplace"
LOG_UNIFORM = "logUniform"
LOG_NORMAL = "logNormal"
LOG_LAPLACE = "logLaplace"

###############################################################################
# SAMPLING

EXPONENTIAL_DECAY = (
    "exponential_decay"  # temperature schedule for parallel tempering
)
BETA_DECAY = "beta_decay"  # temperature schedule for parallel tempering
TRAPEZOID = "trapezoid"  # method to compute log evidence
SIMPSON = "simpson"  # method to compute log evidence
STEPPINGSTONE = "steppingstone"  # method to compute log evidence

###############################################################################
# PREDICT

OUTPUT_IDS = "output_ids"  # data member in PredictionConditionResult
PARAMETER_IDS = "x_names"  # data member in PredictionConditionResult
TIMEPOINTS = "timepoints"  # data member in PredictionConditionResult
OUTPUT = "output"  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = "output_sensi"  # field in the return dict of AmiciPredictor
OUTPUT_WEIGHT = "output_weight"  # field in the return dict of AmiciPredictor
OUTPUT_SIGMAY = "output_sigmay"  # field in the return dict of AmiciPredictor

# separator in the conditions_ids between preequilibration and simulation
# condition
CONDITION_SEP = "::"

AMICI_T = "t"  # return field in amici simulation result
AMICI_X = "x"  # return field in amici simulation result
AMICI_SX = "sx"  # return field in amici simulation result
AMICI_Y = "y"  # return field in amici simulation result
AMICI_SY = "sy"  # return field in amici simulation result
AMICI_LLH = "llh"  # return field in amici simulation result
AMICI_STATUS = "status"  # return field in amici simulation result
AMICI_SIGMAY = "sigmay"  # return field in amici simulation result
AMICI_SSIGMAY = "ssigmay"  # return field in amici simulation result
AMICI_SSIGMAZ = "ssigmaz"  # return field in amici simulation result

ROADRUNNER_LLH = "llh"  # return field in roadrunner objective
ROADRUNNER_INSTANCE = "roadrunner_instance"
ROADRUNNER_SIMULATION = "simulation_results"

CONDITION = "condition"
CONDITION_IDS = "condition_ids"

CSV = "csv"  # return file format
H5 = "h5"  # return file format

###############################################################################
# VISUALIZE

LEN_RGB = 3  # number of elements in an RGB color
LEN_RGBA = 4  # number of elements in an RGBA color
RGB = tuple[(float,) * LEN_RGB]  # typing of an RGB color
RGBA = tuple[(float,) * LEN_RGBA]  # typing of an RGBA color
RGB_RGBA = RGB | RGBA  # typing of an RGB or RGBA color
RGBA_MIN = 0  # min value for an RGBA element
RGBA_MAX = 1  # max value for an RGBA element
RGBA_ALPHA = 3  # zero-indexed fourth element in RGBA
RGBA_WHITE = (RGBA_MAX, RGBA_MAX, RGBA_MAX, RGBA_MAX)  # white as an RGBA color
RGBA_BLACK = (RGBA_MIN, RGBA_MIN, RGBA_MIN, RGBA_MAX)  # black as an RGBA color
COLOR = str | RGB | RGBA  # typing of a color recognized by matplotlib

# optimizer history
TRACE_X_TIME = "time"
TRACE_X_STEPS = "steps"
# supported values to plot on x-axis
TRACE_X = (TRACE_X_TIME, TRACE_X_STEPS)

TRACE_Y_FVAL = "fval"
TRACE_Y_GRADNORM = "gradnorm"
# supported values to plot on y-axis
TRACE_Y = (TRACE_Y_FVAL, TRACE_Y_GRADNORM)

# parameter indices
FREE_ONLY = "free_only"  # only estimated parameters
ALL = "all"  # all parameters, also for start indices

# start indices
ALL_CLUSTERED = "all_clustered"  # best + all that are in a cluster of size > 1
FIRST_CLUSTER = "first_cluster"  # all starts that belong to the first cluster

# waterfall max value
WATERFALL_MAX_VALUE = 1e100


###############################################################################
# ENVIRONMENT VARIABLES

PYPESTO_MAX_N_STARTS: str = "PYPESTO_MAX_N_STARTS"
PYPESTO_MAX_N_SAMPLES: str = "PYPESTO_MAX_N_SAMPLES"
