import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple

from petab_select.constants import Criterion
from petab_select import Model

from .. import select as pypesto_select


# TODO move methods to petab_select
RELATIVE_LABEL_FONTSIZE = -2


def default_label_maker(model: Model) -> str:
    """Create a model label, for plotting."""
    return model.model_hash[:4]


# FIXME supply the problem to automatically detect the correct criterion?
def plot_selected_models(
    selected_models: List[Model],
    criterion: str = Criterion.AIC,
    relative: str = True,
    fz: int = 14,
    size: Tuple[float, float] = (5, 4),
    labels: Dict[str, str] = None,
) -> matplotlib.axes.Axes:
    """Plot criterion for calibrated models.

    Parameters
    ----------
    selected_models:
        First result in tuple returned by `ModelSelector.select()`.
    criterion:
        Key of criterion value in `selected_models` to be plotted.
    relative:
        If `True`, criterion values are plotted relative to the lowest
        criterion value. TODO is the lowest value, always the best? May not
        be for different criterion.
    fz:
        fontsize
    size:
        Figure size in inches.
    labels:
        A dictionary of model labels, where keys are model hashes, and
        values are model labels, for plotting. If a model label is not
        provided, it will be generated from its model ID.

    Returns
    -------
    ax:
        The plot axes.
    """
    zero = 0
    if relative:
        zero = selected_models[-1].get_criterion(criterion)

    if labels is None:
        labels = {}

    # FIGURE
    _, ax = plt.subplots(figsize=size)
    linewidth = 3

    criterion_values = {
        labels.get(model.get_hash(), default_label_maker(model)):
        model.get_criterion(criterion) - zero
        for model in selected_models
    }

    ax.plot(
        criterion_values.keys(),
        criterion_values.values(),
        linewidth=linewidth,
        color='lightgrey',
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(criterion_values))))
    ax.set_ylabel(criterion + ('(relative)' if relative else '(absolute)'),
                  fontsize=fz)
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(
        criterion_values.keys(),
        fontsize=fz+RELATIVE_LABEL_FONTSIZE,
    )
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fz+RELATIVE_LABEL_FONTSIZE)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_history_digraph(
    problem: pypesto_select.Problem,
    history: Dict[str, Model] = None,
    criterion: Criterion = None,
    optimal_distance: float = 1,
    relative: bool = True,
    options: Dict = None,
    labels: Dict[str, str] = None,
):
    """Plot all visited models in the model space, as a directed graph.

    TODO replace magic numbers with options/constants

    Args:
        problem:
            The pyPESTO Select problem.
        history:
            The models calibrated during model selection, in the format of
            `pypesto.select.Problem.history`.
        criterion:
            The criterion.
        optimal_distance:
            See docs for argument `k` in `networkx.spring_layout`.
        relative:
            If `True`, criterion values are offset by the minimum criterion
            value.
        options:
            Additional keyword arguments for `networkx.draw_networkx`.
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
    """
    if criterion is None:
        criterion = problem.petab_select_problem.criterion
    if history is None:
        history = problem.history
    if labels is None:
        labels = {}

    G = nx.DiGraph()
    edges = []
    for model in history.values():
        predecessor_model_hash = model.predecessor_model_hash
        if predecessor_model_hash is not None:
            from_ = labels.get(predecessor_model_hash, predecessor_model_hash)
            # may only not be the case for
            # COMPARED_MODEL_ID == INITIAL_VIRTUAL_MODEL
            if predecessor_model_hash in history:
                predecessor_model = history[predecessor_model_hash]
                from_ = labels.get(
                    predecessor_model.get_hash(),
                    default_label_maker(predecessor_model),
                )
        else:
            raise NotImplementedError('Plots for models with `None` as their predecessor model are not yet implemented.')  # noqa: E501
            from_ = 'None'
        to = labels.get(model.get_hash(), default_label_maker(model))
        edges.append((from_, to))

    # edges = [(node_data['compared_modelId'], node)
    #          for node, node_data in selection_history.items()]
    G.add_edges_from(edges)
    default_options = {
        'node_color': 'lightgrey',
        'arrowstyle': '-|>',
        'node_shape': 's',
        'node_size': 2500,
        # 'width': 2,
        # 'arrowsize': 10,
    }
    if options is not None:
        default_options.update(options)
    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G, k=optimal_distance, iterations=20)
    nx.draw_networkx(G, pos, **default_options)
    # if optimal_distance is not None:
    #     pos = nx.spring_layout(G, k=optimal_distance, iterations=20)
    #     nx.draw_networkx(G, pos, **default_options)
    # else:
    #     nx.draw_networkx(G, **default_options)

    plt.show()
