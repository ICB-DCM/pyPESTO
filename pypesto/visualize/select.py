import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple

from petab_select import (
    AIC,
)
from pypesto.select.constants import (
    INITIAL_VIRTUAL_MODEL,
    MODEL_ID,
)


RELATIVE_LABEL_FONTSIZE = -2


# FIXME supply the problem to automatically detect the correct criterion?
def plot_selected_models(
        selected_models: List[Dict],
        criterion: str = AIC,
        relative: str = True,
        fz: int = 14,
        size: Tuple[float, float] = (5, 4)) -> matplotlib.axes.Axes:
    """
    Plot AIC or BIC for different models selected during model selection
    routine.

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

    Returns
    -------
    ax:
        The plot axes.
    """
    if relative:
        zero = selected_models[-1]['model'].get_criterion(criterion)
    else:
        zero = 0

    # FIGURE
    _, ax = plt.subplots(figsize=size)
    linewidth = 3

    model_ids = [m[MODEL_ID] for m in selected_models]
    criterion_values = [
        m['model'].get_criterion(criterion) - zero
        for m in selected_models
    ]

    ax.plot(
        model_ids,
        criterion_values,
        linewidth=linewidth,
        color='lightgrey',
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(model_ids))))
    ax.set_ylabel(criterion + ('(relative)' if relative else '(absolute)'),
                  fontsize=fz)
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(model_ids, fontsize=fz+RELATIVE_LABEL_FONTSIZE)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fz+RELATIVE_LABEL_FONTSIZE)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_history_digraph(selection_history: Dict,
                         criterion: str = AIC,
                         optimal_distance: float = 1,
                         relative: bool = True,
                         options: Dict = None):
    """
    Plots all visited models in the model space, as a directed graph.
    TODO replace magic numbers with options/constants

    Arguments
    ---------
    selection_history:
        The output from a `ModelSelector.select()` call.

    options:
        The values to be used for the optional keyword arguments in the
        `networkx.draw_networkx()` method.
    """

    zero = 0
    if relative:
        criterions = [
            v['model'].get_criterion(criterion)
            for k, v in selection_history.items()
        ]
        zero = min(criterions)

    G = nx.DiGraph()
    edges = []
    for _node, node_data in selection_history.items():
        model = node_data['model']
        model0 = node_data['model0']
        if model0 is not None:
            from_ = model0.model_id
            # may only not be the case for
            # COMPARED_MODEL_ID == INITIAL_VIRTUAL_MODEL
            if model0.model_id in selection_history:
                from_ += '\n' + f'{model0.get_criterion(criterion) - zero:.2f}'
        else:
            from_ = INITIAL_VIRTUAL_MODEL
        to = (
            model.model_id + '\n' +
            f'{model.get_criterion(criterion) - zero:.2f}'
        )
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
