import matplotlib.pyplot as plt
import matplotlib.axes
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from pypesto.model_selection.constants import MODEL_ID, INITIAL_VIRTUAL_MODEL

def plot_selected_models(
        selected_models: List[Dict],
        criterion: str = 'AIC',
        relative: str = True,
        fz: int = 14,
        size: Tuple[float, float] = [5,4]):
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
    fz: int
        fontsize
    size: ndarray
        Figure size in inches.

    Returns
    -------
    ax:
        The plot axes.
    """
    if relative:
        zero = selected_models[-1][criterion]
    else:
        zero = 0

    # FIGURE
    fig, ax = plt.subplots(figsize=size)
    width = 0.75

    model_ids = [m[MODEL_ID] for m in selected_models]
    criterion_values = [m[criterion] - zero for m in selected_models]
    compared_model_ids = [m[f'compared_{MODEL_ID}'] for m in selected_models]

    ax.bar(model_ids,
           criterion_values,
           width,
           color='lightgrey',
           edgecolor='k')

    ax.get_xticks()
    ax.set_xticks(list(range(len(model_ids))))
    ax.set_ylabel(criterion + ('(relative)' if relative else '(absolute)'),
                  fontsize = fz)
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(model_ids, fontsize = fz-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fz-2)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl),max(ytl)])
    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_history_digraph(selection_history: Dict, options: Dict = None):
    """
    Plots all visited models in the model space, as a directed graph.

    Arguments
    ---------
    selection_history:
        The output from a `ModelSelector.select()` call.

    options:
        The values to be used for the optional keyword arguments in the
        `networkx.draw_networkx()` method.
    """
    G = nx.DiGraph()
    edges = [(node_data['compared_modelId'], node)
             for node, node_data in selection_history.items()]
    G.add_edges_from(edges)
    default_options = {
        'node_color': 'lightgrey',
        'arrowstyle': '-|>',
        #'node_size': 1000,
        #'width': 2,
        #'arrowsize': 10,
       }
    if options is not None:
        default_options.update(options)
    plt.figure(figsize=(12,12))
    nx.draw_networkx(G, **default_options)
    plt.show()
