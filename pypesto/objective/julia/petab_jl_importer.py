"""Contains the PetabJlImporter class."""

from __future__ import annotations

import logging
import os.path
from collections.abc import Iterable

import numpy as np

from pypesto.objective.julia import PEtabJlObjective
from pypesto.problem import Problem

logger = logging.getLogger(__name__)

PEtabProblemJl = ["PEtab.jl::PEtabODEProblem"]


class PetabJlImporter:
    """
    Importer for PEtab models in Julia, using PEtab.jl.

    Create an `objective.JuliaObjective` or a `pypesto.Problem` from PEtab
    files or from a julia module.
    """

    def __init__(
        self,
        module: str = None,
        source_file: str = None,
        petab_problem_name: str = "petabProblem",
    ):
        """
        Initialize importer.

        Parameters
        ----------
        module:
            Name of the Julia model
        source_file:
            Path to the Julia source file.
        petab_problem:
            Wrapper around the PEtab.jl problem.
        """
        self.module = module
        self.source_file = source_file
        self._petab_problem_name = petab_problem_name
        # placeholder for the petab.jl problem
        self.petab_jl_problem = None

    @staticmethod
    def from_yaml(
        yaml_file: str,
        ode_solver_options: dict | None = None,
        gradient_method: str | None = None,
        hessian_method: str | None = None,
        sparse_jacobian: bool | None = None,
        verbose: bool | None = None,
        directory: str | None = None,
    ) -> PetabJlImporter:
        """
        Create a `PetabJlImporter` from a yaml file.

        Writes the Julia module to a file in `directory` and returns a
        `PetabJlImporter` for that module.

        Parameters
        ----------
        yaml_file:
            The yaml file of the PEtab problem
        ode_solver_options:
            Dictionary like options for the ode solver in julia
        gradient_method, hessian_method:
            Julia methods to compute gradient and hessian
        sparse_jacobian:
            Whether to compute sparse Jacobians
        verbose:
            Whether to have a more informative log.
        directory:
            Where to write the julia file, defaults to the directory of the
            yaml file.
        """
        # get default values
        options = _get_default_options(
            ode_solver_options=ode_solver_options,
            gradient_method=gradient_method,
            hessian_method=hessian_method,
            sparse_jacobian=sparse_jacobian,
            verbose=verbose,
        )

        # write julia module
        source_file, module = _write_julia_file(
            yaml_file=yaml_file, options=options, directory=directory
        )

        return PetabJlImporter(
            module=module,
            source_file=source_file,
        )

    def create_objective(
        self, precompile: bool | None = True
    ) -> PEtabJlObjective:
        """
        Create a `pypesto.objective.PEtabJlObjective` from the PEtab.jl problem.

        The objective function will be the negative log likelihood or the
        negative log posterior, depending on the PEtab.jl problem.

        Parameters
        ----------
        precompile:
            Whether to precompile the julia module for speed up in
            multistart optimization.
        """
        # lazy imports
        try:
            from julia import Main  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            ) from None
        if self.source_file is None:
            self.source_file = f"{self.module}.jl"

        if not os.path.exists(self.source_file):
            raise ValueError(
                "The julia file does not exist. You can create "
                "it from a petab yaml file path using "
                "`PetabJlImporter.from_yaml(yaml_file)`"
            )

        obj = PEtabJlObjective(
            module=self.module,
            source_file=self.source_file,
            petab_problem_name=self._petab_problem_name,
            precompile=precompile,
        )

        self.petab_jl_problem = obj.petab_jl_problem
        return obj

    def create_problem(
        self,
        x_guesses: Iterable[float] | None = None,
        lb_init: np.ndarray | list[float] | None = None,
        ub_init: np.ndarray | list[float] | None = None,
        precompile: bool | None = True,
    ) -> Problem:
        """
        Create a `pypesto.Problem` from the PEtab.jl problem.

        Parameters
        ----------
        x_guesses:
            Guesses for the parameter values, shape (g, dim), where g denotes the
            number of guesses. These are used as start points in the optimization.
        lb_init, ub_init:
            The lower and upper bounds for initialization, typically for defining
            search start points.
            If not set, set to lb, ub.
        precompile:
            Whether to precompile the julia module for speed up in
            multistart optimization.
        """
        obj = self.create_objective(precompile=precompile)
        lb = np.asarray(self.petab_jl_problem.lower_bounds)
        ub = np.asarray(self.petab_jl_problem.upper_bounds)

        return Problem(
            objective=obj,
            lb=lb,
            ub=ub,
            x_guesses=x_guesses,
            x_names=obj.x_names,
            lb_init=lb_init,
            ub_init=ub_init,
        )


def _get_default_options(
    ode_solver_options: dict | None = None,
    gradient_method: str | None = None,
    hessian_method: str | None = None,
    sparse_jacobian: str | None = None,
    verbose: str | None = None,
) -> dict:
    """
    If values are not specified, get default values for the options.

    Additionally check that the values are valid.

    Parameters
    ----------
    ode_solver_options:
        Options for the ODE solver.
    gradient_method:
        Method for gradient calculation.
    hessian_method:
        Method for hessian calculation.
    sparse_jacobian:
        Whether the jacobian should be sparse.
    verbose:
        Whether to print verbose output.

    Returns
    -------
    dict:
        The options.
    """
    # get default values
    if ode_solver_options is None:
        ode_solver_options = {
            "solver": "Rodas5P",
            "abstol": 1e-8,
            "reltol": 1e-8,
            "maxiters": "Int64(1e4)",
        }
    if not ode_solver_options["solver"].endswith("()"):
        ode_solver_options["solver"] += "()"  # add parentheses
    if gradient_method is None:
        gradient_method = "nothing"
    if hessian_method is None:
        hessian_method = "nothing"
    if sparse_jacobian is None:
        sparse_jacobian = "nothing"
    if verbose is None:
        verbose = "true"

    # check values for gradient_method and hessian_method
    allowed_gradient_methods = [
        "ForwardDiff",
        "ForwardEquations",
        "Adjoint",
        "Zygote",
    ]
    if gradient_method not in allowed_gradient_methods:
        logger.warning(
            f"gradient_method {gradient_method} is not in "
            f"{allowed_gradient_methods}. Defaulting to ForwardDiff."
        )
        gradient_method = "ForwardDiff"
    allowed_hessian_methods = ["ForwardDiff", "BlocForwardDiff", "GaussNewton"]
    if hessian_method not in allowed_hessian_methods:
        logger.warning(
            f"hessian_method {hessian_method} is not in "
            f"{allowed_hessian_methods}. Defaulting to ForwardDiff."
        )
        hessian_method = "ForwardDiff"

    # fill options
    options = {
        "ode_solver_options": ode_solver_options,
        "gradient_method": gradient_method,
        "hessian_method": hessian_method,
        "sparse_jacobian": sparse_jacobian,
        "verbose": verbose,
    }
    return options


def _write_julia_file(
    yaml_file: str, options: dict, directory: str
) -> tuple[str, str]:
    """
    Write the Julia file.

    Parameters
    ----------
    yaml_file:
        The yaml file of the PEtab problem.
    options:
        The options.
    dir:
        The directory to write the file to.

    Returns
    -------
    source_file:
        The name/path of the file.
    module:
        The module name.
    """
    if directory is None:
        directory = os.path.dirname(yaml_file)  # directory of the yaml file
    source_file = os.path.join(directory, "PEtabJl_module.jl")
    module = "MyPEtabJlModule"

    link_to_options = (
        "https://sebapersson.github.io/"
        "PEtab.jl/dev/API_choosen/#PEtab.setupPEtabODEProblem"
    )
    odeSolvOpt_str = ", ".join(
        [f"{k}={v}" for k, v in options["ode_solver_options"].items()]
    )
    # delete "solver=" from string
    odeSolvOpt_str = odeSolvOpt_str.replace("solver=", "")

    content = (
        f"module {module}\n\n"
        f"using OrdinaryDiffEq\n"
        f"using Sundials\n"
        f"using PEtab\n\n"
        f'pathYaml = "{yaml_file}"\n'
        f"petabModel = PEtabModel(pathYaml, verbose=true)\n\n"
        f"# A full list of options for PEtabODEProblem can be "
        f"found at {link_to_options}\n"
        f"petabProblem = PEtabODEProblem(\n\t"
        f"petabModel,\n\t"
        f"odesolver=ODESolver({odeSolvOpt_str}),\n\t"
        f"gradient_method=:{options['gradient_method']},\n\t"
        f"hessian_method=:{options['hessian_method']},\n\t"
        f"sparse_jacobian={options['sparse_jacobian']},\n\t"
        f"verbose={options['verbose']}\n)\n\nend\n"
    )
    # write file
    with open(source_file, "w") as f:
        f.write(content)

    return source_file, module
