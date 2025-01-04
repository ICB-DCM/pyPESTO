"""Process a model selection :class:`ModelProblem` after calibration."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from petab_select.constants import TYPE_PATH, Criterion

from .. import store, visualize
from .model_problem import TYPE_POSTPROCESSOR, ModelProblem

__all__ = [
    "model_id_binary_postprocessor",
    "multi_postprocessor",
    "report_postprocessor",
    "save_postprocessor",
    "waterfall_plot_postprocessor",
]


def multi_postprocessor(
    problem: ModelProblem,
    postprocessors: list[TYPE_POSTPROCESSOR] = None,
):
    """Combine multiple postprocessors into a single postprocessor.

    See :meth:`save_postprocessor` for usage hints.

    Parameters
    ----------
    problem:
        A model selection :class:`ModelProblem` that has been optimized.
    postprocessors:
        A list of postprocessors, which will be sequentially applied to the
        optimized model ``problem``.
        The location where results will be stored.
    """
    for postprocessor in postprocessors:
        postprocessor(problem)


def waterfall_plot_postprocessor(
    problem: ModelProblem,
    output_path: TYPE_PATH = ".",
):
    """Produce a waterfall plot.

    See :meth:`save_postprocessor` for usage hints and argument documentation.
    """
    visualize.waterfall(problem.minimize_result)
    plot_output_path = Path(output_path) / (str(problem.model.hash) + ".png")
    plt.savefig(str(plot_output_path))


def save_postprocessor(
    problem: ModelProblem,
    output_path: TYPE_PATH = ".",
    use_model_hash: bool = False,
):
    """Save the parameter estimation result.

    When used, first set the output folder for results, e.g. with
    :func:`functools.partial`. This is because postprocessors should take only a
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
        A model selection :class:`ModelProblem` that has been optimized.
    output_path:
        The location where output will be stored.
    use_model_hash:
        Whether the filename should use the model hash. Defaults to ``False``,
        in which case the model ID is used instead.
    """
    stem = problem.model.model_id
    if use_model_hash:
        stem = str(problem.model.hash)
    store.write_result(
        problem.minimize_result,
        Path(output_path) / (stem + ".hdf5"),
    )


def model_id_binary_postprocessor(problem: ModelProblem):
    """Change a PEtab Select model ID to a binary string.

    Changes the model ID in-place to be a string like ``M_ijk``, where
    ``i``, ``j``, ``k``, etc. are ``1`` if the parameter in that position is estimated,
    or ``0`` if the parameter is fixed.

    To ensure that other postprocessors (e.g. :func:`report_postprocessor`) use this
    new model ID, when in use with a :func:`multi_postprocessor`, ensure this is
    before the other postprocessors in the ``postprocessors`` argument of
    :func:`multi_postprocessor`.

    Parameters
    ----------
    problem:
        A model selection :class:`ModelProblem` that has been optimized.
    """
    warnings.warn(
        (
            "This is obsolete. Model IDs are by default the model hash, which "
            "is now similar to the binary string."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    problem.model.model_id = str(problem.model.hash)


def report_postprocessor(
    problem: ModelProblem,
    output_filepath: TYPE_PATH,
    criteria: list[Criterion] = None,
):
    """Create a TSV table of model selection results.

    Parameters
    ----------
    problem:
        A model selection :class:`ModelProblem` that has been optimized.
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

    header.append("model_id")
    row.append(problem.model.model_id)

    header.append("total_time")
    row.append(str(sum(start_optimization_times)))

    for criterion in criteria:
        header.append(criterion.value)
        row.append(str(problem.model.get_criterion(criterion)))

    # Arbitrary convergence criterion
    header.append("n_converged")
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
        header.append(f"start_time_{start_index}")
        row.append(str(start_optimization_time))

    with open(output_filepath, "a+") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(row) + "\n")
