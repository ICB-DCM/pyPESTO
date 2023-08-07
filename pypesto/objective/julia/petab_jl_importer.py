"""Contains the PetabJlImporter class."""
from __future__ import annotations

import logging
import os.path
from typing import Iterable, List, Optional, Tuple, Union

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
        odeSolverOptions: Optional[dict] = None,
        gradientMethod: Optional[str] = None,
        hessianMethod: Optional[str] = None,
        sparseJacobian: Optional[bool] = None,
        verbose: Optional[bool] = None,
        directory: Optional[str] = None,
    ) -> PetabJlImporter:
        """
        Create a `PetabJlImporter` from a yaml file.

        Writes the Julia module to a file in `directory` and returns a
        `PetabJlImporter` for that module.

        Parameters
        ----------
        yaml_file:
            The yaml file of the PEtab problem
        odeSolverOptions:
            Dictionary like options for the ode solver in julia
        gradientMethod, hessianMethod:
            Julia methods to compute gradient and hessian
        sparseJacobian:
            Whether to compute sparse Jacobians
        verbose:
            Whether to have a more informative log.
        directory:
            Where to write the julia file, defaults to the directory of the
            yaml file.
        """
        # get default values
        options = _get_default_options(
            odeSolverOptions=odeSolverOptions,
            gradientMethod=gradientMethod,
            hessianMethod=hessianMethod,
            sparseJacobian=sparseJacobian,
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
        self, precompile: Optional[bool] = True
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
            )
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
        x_guesses: Optional[Iterable[float]] = None,
        lb_init: Union[np.ndarray, List[float], None] = None,
        ub_init: Union[np.ndarray, List[float], None] = None,
        precompile: Optional[bool] = True,
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
        lb = np.asarray(self.petab_jl_problem.lowerBounds)
        ub = np.asarray(self.petab_jl_problem.upperBounds)

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
    odeSolverOptions: Union[dict, None] = None,
    gradientMethod: Union[str, None] = None,
    hessianMethod: Union[str, None] = None,
    sparseJacobian: Union[str, None] = None,
    verbose: Union[str, None] = None,
) -> dict:
    """
    If values are not specified, get default values for the options.

    Additionally check that the values are valid.

    Parameters
    ----------
    odeSolverOptions:
        Options for the ODE solver.
    gradientMethod:
        Method for gradient calculation.
    hessianMethod:
        Method for hessian calculation.
    sparseJacobian:
        Whether the jacobian should be sparse.
    verbose:
        Whether to print verbose output.

    Returns
    -------
    dict:
        The options.
    """
    # get default values
    if odeSolverOptions is None:
        odeSolverOptions = {
            "solver": "Rodas5P",
            "abstol": 1e-8,
            "reltol": 1e-8,
            "maxiters": "Int64(1e4)",
        }
    if not odeSolverOptions["solver"].endswith("()"):
        odeSolverOptions["solver"] += "()"  # add parentheses
    if gradientMethod is None:
        gradientMethod = "nothing"
    if hessianMethod is None:
        hessianMethod = "nothing"
    if sparseJacobian is None:
        sparseJacobian = "nothing"
    if verbose is None:
        verbose = "true"

    # check values for gradientMethod and hessianMethod
    allowed_gradient_methods = [
        "ForwardDiff",
        "ForwardEquations",
        "Adjoint",
        "Zygote",
    ]
    if gradientMethod not in allowed_gradient_methods:
        logger.warning(
            f"gradientMethod {gradientMethod} is not in "
            f"{allowed_gradient_methods}. Defaulting to ForwardDiff."
        )
        gradientMethod = "ForwardDiff"
    allowed_hessian_methods = ["ForwardDiff", "BlocForwardDiff", "GaussNewton"]
    if hessianMethod not in allowed_hessian_methods:
        logger.warning(
            f"hessianMethod {hessianMethod} is not in "
            f"{allowed_hessian_methods}. Defaulting to ForwardDiff."
        )
        hessianMethod = "ForwardDiff"

    # fill options
    options = {
        "odeSolverOptions": odeSolverOptions,
        "gradientMethod": gradientMethod,
        "hessianMethod": hessianMethod,
        "sparseJacobian": sparseJacobian,
        "verbose": verbose,
    }
    return options


def _write_julia_file(
    yaml_file: str, options: dict, directory: str
) -> Tuple[str, str]:
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
        [f"{k}={v}" for k, v in options["odeSolverOptions"].items()]
    )
    # delete "solver=" from string
    odeSolvOpt_str = odeSolvOpt_str.replace("solver=", "")

    content = (
        f"module {module}\n\n"
        f"using OrdinaryDiffEq\n"
        f"using Sundials\n"
        f"using PEtab\n\n"
        f"pathYaml = \"{yaml_file}\"\n"
        f"petabModel = readPEtabModel(pathYaml, verbose=true)\n\n"
        f"# A full list of options for createPEtabODEProblem can be "
        f"found at {link_to_options}\n"
        f"petabProblem = createPEtabODEProblem(\n\t"
        f"petabModel,\n\t"
        f"odeSolverOptions=ODESolverOptions({odeSolvOpt_str}),\n\t"
        f"gradientMethod=:{options['gradientMethod']},\n\t"
        f"hessianMethod=:{options['hessianMethod']},\n\t"
        f"sparseJacobian={options['sparseJacobian']},\n\t"
        f"verbose={options['verbose']}\n)\n\nend\n"
    )
    # write file
    with open(source_file, "w") as f:
        f.write(content)

    return source_file, module
