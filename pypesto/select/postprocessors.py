from pathlib import Path

import matplotlib.pyplot as plt

from .. import visualize


# TODO align with `petab_select.constants`
TYPE_PATH = Union[str, Path]


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
