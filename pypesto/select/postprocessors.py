from pathlib import Path

import matplotlib.pyplot as plt

from .. import visualize
from .constants import TYPE_PATH


def waterfall_plot_postprocessor(
    problem: 'ModelSelectionProblem',
    output_path: TYPE_PATH = '.',
):
    """A postprocessor to produce a waterfall plot from a model calibration.

    When used, first set the output folder for plots, e.g.:
    .. code-block:: python
       from functools import partial
       output_path = 'waterfall_plots'
       wpp = partial(waterfall_plot_postprocessor, output_path=output_path)
       selector = pypesto.select.ModelSelector(
           problem=selection_problem,
           model_postprocessor=wpp,
       )
    """
    visualize.waterfall(problem.minimize_result)
    plot_output_path = Path(output_path) / (problem.model.model_id + '.png')
    plt.savefig(str(plot_output_path))
