import h5py
from .hdf5 import write_string_array, write_int_array, write_float_array
from ..problem import Problem
from ..result import OptimizeResult


def save_problem(f, problem):
    problem_grp = f.create_group("problem")
    # problem_grp.attrs['config'] = objective.get_config()
    write_float_array(problem_grp, "lb", problem.lb)
    write_float_array(problem_grp, "ub", problem.ub)
    write_float_array(problem_grp, "lb_full", problem.lb_full)
    write_float_array(problem_grp, "ub_full", problem.ub_full)
    # dim [int]
    # dim_full [int]
    write_float_array(problem_grp, "x_fixed_values", problem.x_fixed_vals)
    write_int_array(problem_grp, "x_fixed_indices", problem.x_fixed_indices)
    write_string_array(problem_grp, "x_names", problem.x_names)


def save_optimization(f, result):
    optimization_grp = f.create_group("optimization")
    # settings = optimization_grp.create_dataset("settings", settings, dtype=)
    results_grp = optimization_grp.create_group("results")
    for i, start in enumerate(result.list):
        start_grp = results_grp.create_group(str(i))
        # fval: [float]
        write_float_array(start_grp, "x", start.x)
        write_float_array(start_grp, "grad", start.grad)
        write_float_array(start_grp, "hess", start.hess)
        # - n_fval: [int]
        # - n_grad: [int]
        write_float_array(start_grp, "x0", start.x0)
        # - fval0: [float]
        # - exitflag: [str]
        # - time: [float]
        # - message: [str]


def save_to_hdf5(problem: Problem,
                 result: OptimizeResult,
                 folder):
    with h5py.File("pyPESTO_result.hdf5", "a") as f:
        save_problem(f, problem)
        save_optimization(f, result)


def save_to_csv():
    pass
