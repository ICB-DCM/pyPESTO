from typing import List

from pathlib import Path

import matplotlib.pyplot as plt

from .. import store
from .. import visualize
from .constants import (
    TYPE_PATH,
    TYPE_POSTPROCESSOR,
)
from .problem import ModelSelectionProblem


def multi_postprocessor(
    problem: ModelSelectionProblem,
    postprocessors: List[TYPE_POSTPROCESSOR] = None,
):
    """A postprocessor that combines multiple other postprocessors.

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
    problem: ModelSelectionProblem,
    output_path: TYPE_PATH = ".",
):
    """A postprocessor to produce a waterfall plot from a model calibration.

    See `save_postprocessor` for usage hints and argument documentation.
    """
    visualize.waterfall(problem.minimize_result)
    plot_output_path = Path(output_path) / (problem.model.model_id + ".png")
    plt.savefig(str(plot_output_path))


def save_postprocessor(
    problem: ModelSelectionProblem,
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
        A model selection problem that has been optimized.
    output_path:
        The location where results will be stored.
    """
    store.write_result(
        problem.minimize_result,
        Path(output_path) / (problem.model.model_id + ".hdf5"),
    )
