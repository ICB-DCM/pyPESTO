from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from petab_select.constants import TYPE_PATH

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
    plot_output_path = Path(output_path) / (problem.model.model_id + ".png")
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
        Path(output_path) / (problem.model.model_id + ".hdf5"),
    )
