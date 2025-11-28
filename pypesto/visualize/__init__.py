# noqa: D400,D205
"""
Visualize
=========

pypesto comes with various visualization routines. To use these,
import pypesto.visualize.
"""

from pypesto.util import assign_clusters, delete_nan_inf

from .clust_color import assign_clustered_colors, assign_colors
from .dimension_reduction import (
    ensemble_crosstab_scatter_lowlevel,
    ensemble_scatter_lowlevel,
    projection_scatter_pca,
    projection_scatter_umap,
    projection_scatter_umap_original,
)
from .ensemble import ensemble_identifiability
from .misc import process_offset_y, process_result_list, process_y_limits
from .observable_mapping import (
    plot_linear_observable_mappings_from_pypesto_result,
    plot_splines_from_inner_result,
    plot_splines_from_pypesto_result,
    visualize_estimated_observable_mapping,
)
from .optimization_stats import (
    optimization_run_properties_one_plot,
    optimization_run_properties_per_multistart,
    optimization_run_property_per_multistart,
)
from .optimizer_convergence import optimizer_convergence
from .optimizer_history import optimizer_history, optimizer_history_lowlevel
from .ordinal_categories import (
    plot_categories_from_inner_result,
    plot_categories_from_pypesto_result,
)
from .parameters import (
    optimization_scatter,
    parameter_hist,
    parameters,
    parameters_correlation_matrix,
    parameters_lowlevel,
)
from .profile_cis import profile_cis, profile_nested_cis
from .profiles import profile_lowlevel, profiles, profiles_lowlevel
from .reference_points import ReferencePoint, create_references
from .sampling import (
    sampling_1d_marginals,
    sampling_fval_traces,
    sampling_parameter_cis,
    sampling_parameter_traces,
    sampling_prediction_trajectories,
    sampling_scatter,
)
from .waterfall import waterfall, waterfall_lowlevel
