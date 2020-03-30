import h5py
from .hdf5 import write_string_array, write_int_array, write_float_array
from ..result import Result
from ..objective.constants import *


def save_problem(f, problem):
    problem_grp = f.create_group("problem")
    # problem_grp.attrs['config'] = objective.get_config()
    problem_grp.attrs['dim'] = problem.dim
    problem_grp.attrs['dim_full'] = problem.dim_full

    write_float_array(problem_grp, "lb", problem.lb)
    write_float_array(problem_grp, "ub", problem.ub)
    write_float_array(problem_grp, "lb_full", problem.lb_full)
    write_float_array(problem_grp, "ub_full", problem.ub_full)
    write_float_array(problem_grp, "x_fixed_values", problem.x_fixed_vals)
    write_int_array(problem_grp, "x_fixed_indices", problem.x_fixed_indices)
    write_string_array(problem_grp, "x_names", problem.x_names)


def save_optimization(f, result):
    optimization_grp = f.create_group("optimization")
    # settings = optimization_grp.create_dataset("settings", settings, dtype=)
    results_grp = optimization_grp.create_group("results")
    for i, start in enumerate(result.list):
        start_grp = results_grp.create_group(str(i))

        start_grp.attrs[FVAL] = start.fval
        start_grp.attrs[N_FVAL] = start.n_fval
        start_grp.attrs[N_GRAD] = start.n_grad
        start_grp.attrs['fval0'] = start.fval0
        start_grp.attrs['exitflag'] = start.exitflag
        start_grp.attrs['time'] = start.time
        start_grp.attrs['message'] = start.message

        write_float_array(start_grp, X, start.x)
        write_float_array(start_grp, GRAD, start.grad)
        if start.hess:
            write_float_array(start_grp, HESS, start.hess)
        write_float_array(start_grp, X0, start.x0)


def save_to_hdf5(result: Result,
                 output_file: str):
    with h5py.File(output_file, "w") as f:
        save_problem(f, result.problem)
        save_optimization(f, result.optimize_result)


def save_to_csv():
    pass
