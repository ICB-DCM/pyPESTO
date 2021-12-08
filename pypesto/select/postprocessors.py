from typing import Callable, List, Union

from pathlib import Path

import matplotlib.pyplot as plt

from .. import visualize
from .. import store


# TODO align with `petab_select.constants`
TYPE_PATH = Union[str, Path]

TYPE_POSTPROCESSOR = Callable[['ModelSelectionProblem'], None]


def multi_postprocessor(
    problem: 'ModelSelectionProblem',
    postprocessors: List[TYPE_POSTPROCESSOR] = None,
):
    for postprocessor in postprocessors:
        postprocessor(problem)


def waterfall_plot_postprocessor(
    problem: 'ModelSelectionProblem',
    output_path: TYPE_PATH = '.',
):
    """
    Intended use is to first set the output folder for plots with
    `functools.partial`.
    """
    visualize.waterfall(problem.minimize_result)
    plot_output_path = Path(output_path) / (problem.model.model_id + '.png')
    plt.savefig(str(plot_output_path))


def save_postprocessor(
    problem: 'ModelSelectionProblem',
    output_path: TYPE_PATH = '.',
):
    """
    Intended use is to first set the output folder for results with
    `functools.partial`.
    """
    store.write_result(
        problem.minimize_result,
        Path(output_path) / (problem.model.model_id + '.hdf5'),
    )
