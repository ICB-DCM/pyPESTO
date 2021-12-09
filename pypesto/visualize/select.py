import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple

from petab_select.constants import (
    Criterion,
    VIRTUAL_INITIAL_MODEL,
)
from petab_select import (
    Model,
    MODEL_ID,
)


# TODO move methods to petab_select
RELATIVE_LABEL_FONTSIZE = -2


# FIXME supply the problem to automatically detect the correct criterion?
def plot_selected_models(
    selected_models: List[Dict],
    criterion: str = Criterion.AIC,
    relative: str = True,
    fz: int = 14,
    size: Tuple[float, float] = (5, 4),
    label_attribute: str = MODEL_ID,
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
    label_attribute:
        The attribute of a `petab_select.model.Model` instance, to use as the
        label for the model's representation in the graph.

    Returns
    -------
    ax:
        The plot axes.
    """
    if relative:
        zero = selected_models[-1].get_criterion(criterion)
    else:
        zero = 0

    def label_maker(
        model: Model,
        zero=zero,
        label_attribute=label_attribute,
    ) -> str:
        model_label = getattr(model, label_attribute)
        return model_label

    # FIGURE
    _, ax = plt.subplots(figsize=size)
    linewidth = 3

    model_labels = [label_maker(model) for model in selected_models]
    criterion_values = [
        model.get_criterion(criterion) - zero
        for model in selected_models
    ]

    ax.plot(
        model_labels,
        criterion_values,
        linewidth=linewidth,
        color='lightgrey',
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(model_labels))))
    ax.set_ylabel(criterion + ('(relative)' if relative else '(absolute)'),
                  fontsize=fz)
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(model_labels, fontsize=fz+RELATIVE_LABEL_FONTSIZE)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fz+RELATIVE_LABEL_FONTSIZE)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_history_digraph(
    selection_history: Dict,
    criterion: Criterion = Criterion.AIC,
    optimal_distance: float = 1,
    relative: bool = True,
    options: Dict = None,
    label_attribute: str = MODEL_ID,
):
    """Plot all visited models in the model space, as a directed graph.

    TODO replace magic numbers with options/constants

    Arguments
    ---------
    selection_history:
        The output from a `ModelSelector.select()` call.
    options:
        The values to be used for the optional keyword arguments in the
        `networkx.draw_networkx()` method.
    label_attribute:
        The attribute of a `petab_select.model.Model` instance, to use as the
        label for the model's representation in the graph.
    """
    zero = 0
    if relative:
        criterions = [
            v['model'].get_criterion(criterion)
            for k, v in selection_history.items()
        ]
        zero = min(criterions)

    def label_maker(
        model: Model,
        zero=zero,
        label_attribute=label_attribute,
    ) -> str:
        model_label = getattr(model, label_attribute)
        criterion_label = \
            f'{model.get_criterion(criterion) - zero:.2f}'
        return '\n'.join([model_label, criterion_label])

    G = nx.DiGraph()
    edges = []
    for _node, node_data in selection_history.items():
        model = node_data['model']
        predecessor_model_id = model.predecessor_model_id
        if predecessor_model_id is not None:
            from_ = predecessor_model_id
            # may only not be the case for
            # COMPARED_MODEL_ID == INITIAL_VIRTUAL_MODEL
            if predecessor_model_id in selection_history:
                predecessor_model = \
                    selection_history[predecessor_model_id]['model']
                from_ = label_maker(predecessor_model)
        else:
            raise ValueError(
                'unknown error: not sure why this is reachable anymore'
            )
            from_ = VIRTUAL_INITIAL_MODEL
        to = label_maker(model)
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
