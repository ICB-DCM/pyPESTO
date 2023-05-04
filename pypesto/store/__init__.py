"""
Storage
=======

Saving and loading traces and results objects.
"""

from .auto import autosave
from .hdf5 import write_array
from .read_from_hdf5 import (
    OptimizationResultHDF5Reader,
    ProblemHDF5Reader,
    ProfileResultHDF5Reader,
    SamplingResultHDF5Reader,
    load_objective_config,
    read_result,
)
from .save_to_hdf5 import (
    OptimizationResultHDF5Writer,
    ProblemHDF5Writer,
    ProfileResultHDF5Writer,
    SamplingResultHDF5Writer,
    write_result,
)
