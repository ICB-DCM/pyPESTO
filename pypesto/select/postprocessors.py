"""Process a model after calibration."""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from petab_select.constants import ESTIMATE, TYPE_PATH, Criterion

from .. import store, visualize
from ..C import TYPE_POSTPROCESSOR
from .model_problem import ModelProblem


def multi_postprocessor(
    problem: ModelProblem,
    postprocessors: List[TYPE_POSTPROCESSOR] = None,
):
    """Combine multiple postprocessors into a single postprocessor.

    See `save_postprocessor` for usage hints.

    Parameters
    ----------
    problem:
        A model selection problem that has been optimized.
    postprocessors:
        A list of postprocessors, which will be sequentially applied to the
        optimized model `problem`.
        The location where results will be stored.
    """
    for postprocessor in postprocessors:
        postprocessor(problem)


def waterfall_plot_postprocessor(
    problem: ModelProblem,
    output_path: TYPE_PATH = ".",
):
    """Produce a waterfall plot from a model calibration.

    See `save_postprocessor` for usage hints and argument documentation.
    """
    visualize.waterfall(problem.minimize_result)
    plot_output_path = Path(output_path) / (problem.model.model_hash + ".png")
    plt.savefig(str(plot_output_path))


def save_postprocessor(
    problem: ModelProblem,
    output_path: TYPE_PATH = ".",
):
    """Save the parameter estimation results for optimized models.

    When used, first set the output folder for results, e.g. with
    `functools.partial`. This is because postprocessors should take only a
    single parameter: an optimized model.
    .. code-block:: python
       from functools import partial
       output_path = 'results'
       pp = partial(save_postprocessor, output_path=output_path)
       selector = pypesto.select.ModelSelector(
           problem=problem,
           model_postprocessor=pp,
       )

    Parameters
    ----------
    problem:
        A model problem that has been optimized.
    output_path:
        The location where output will be stored.
    """
    store.write_result(
        problem.minimize_result,
        Path(output_path) / (problem.model.model_hash + ".hdf5"),
    )


def model_id_binary_postprocessor(problem: ModelProblem):
    """Change model IDs to binary strings.

    Changes the model ID in-place to be a string like `M_ijk`, where
    `i`, `j`, `k`, etc. are `1` if the parameter in that position is estimated,
    or `0` if the parameter is fixed.

    To ensure that other postprocessors (e.g. `report_postprocessor`) use this
    new model ID, when in use with a `multi_postprocessor`, ensure this is
    before the other postprocessors in the `postprocessors` argument of
    `multi_postprocessor`.

    Parameters
    ----------
    problem:
        A model selection problem that has been optimized.
    """
    model_id = "M_"
    for parameter_value in problem.model.parameters.values():
        model_id += "1" if parameter_value == ESTIMATE else "0"
    problem.model.model_id = model_id


def report_postprocessor(
    problem: ModelProblem,
    output_filepath: TYPE_PATH,
    criteria: List[Criterion] = None,
):
    """Create a TSV table of model calibration results.

    Parameters
    ----------
    problem:
        A model selection problem that has been optimized.
    output_filepath:
        The file path where the report will be saved.
    criteria:
        The criteria that will be in the report. Defaults to nllh, AIC, AICc,
        and BIC.
    """
    output_filepath = Path(output_filepath)
    write_header = False
    # Only write the header if the file doesn't yet exist or is empty.
    if not output_filepath.exists() or output_filepath.stat().st_size == 0:
        write_header = True
    if criteria is None:
        criteria = [
            Criterion.NLLH,
            Criterion.AIC,
            Criterion.AICC,
            Criterion.BIC,
        ]

    start_optimization_times = problem.minimize_result.optimize_result.time

    header = []
    row = []

    header.append('model_id')
    row.append(problem.model.model_id)

    header.append('total_time')
    row.append(str(sum(start_optimization_times)))

    for criterion in criteria:
        header.append(criterion.value)
        row.append(str(problem.model.get_criterion(criterion)))

    # Arbitrary convergence criterion
    header.append('n_converged')
    row.append(
        str(
            (
                np.array(problem.minimize_result.optimize_result.fval)
                < (problem.minimize_result.optimize_result.list[0].fval + 0.1)
            ).sum()
        )
    )

    for start_index, start_optimization_time in enumerate(
        start_optimization_times
    ):
        header.append(f'start_time_{start_index}')
        row.append(str(start_optimization_time))

    with open(output_filepath, 'a+') as f:
        if write_header:
            f.write('\t'.join(header) + '\n')
        f.write('\t'.join(row) + '\n')
