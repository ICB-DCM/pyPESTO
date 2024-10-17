"""Interface to PEtab.jl."""

import logging
import os

import numpy as np

from .base import JuliaObjective, _read_source

logger = logging.getLogger(__name__)

PEtabProblemJl = ["PEtab.jl::PEtabODEProblem"]


class PEtabJlObjective(JuliaObjective):
    """
    Wrapper around an objective defined in PEtab.jl.

    Parameters
    ----------
    module:
        Name of the julia module containing the objective.
    source_file:
        Julia source file. Defaults to "{module}.jl".
    petab_problem_name:
        Name of the petab problem variable in the julia module.
    """

    def __init__(
        self,
        module: str,
        source_file: str = None,
        petab_problem_name: str = "petabProblem",
        precompile: bool = True,
        force_compile: bool = False,
    ):
        """Initialize objective."""
        # lazy imports
        try:
            from julia import Main, Pkg  # noqa: F401

            Pkg.activate(".")
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            ) from None

        self.module = module
        self.source_file = source_file
        self._petab_problem_name = petab_problem_name
        if precompile:
            self.precompile_model(force_compile=force_compile)

        if self.source_file is None:
            self.source_file = f"{module}.jl"

        # Include module if not already included
        _read_source(module, source_file)

        petab_jl_problem = self.get(petab_problem_name)
        self.petab_jl_problem = petab_jl_problem

        # get functions
        fun = self.petab_jl_problem.nllh
        grad = self.petab_jl_problem.grad
        hess = self.petab_jl_problem.hess
        x_names = np.asarray(self.petab_jl_problem.xnames)

        # call the super super super constructor
        super(JuliaObjective, self).__init__(
            fun=fun, grad=grad, hess=hess, x_names=x_names
        )

    def __getstate__(self):
        """Get state for pickling."""
        # if not dumped, dump it via JLD2
        return {
            "module": self.module,
            "source_file": self.source_file,
            "_petab_problem_name": self._petab_problem_name,
        }

    def __setstate__(self, state):
        """Set state from pickling."""
        for key, value in state.items():
            setattr(self, key, value)
        # lazy imports
        try:
            from julia import (
                Main,  # noqa: F401
                Pkg,
            )

            Pkg.activate(".")
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            ) from None
        # Include module if not already included
        _read_source(self.module, self.source_file)

        petab_jl_problem = self.get(self._petab_problem_name)
        self.petab_jl_problem = petab_jl_problem

        # get functions
        fun = self.petab_jl_problem.nllh
        grad = self.petab_jl_problem.grad
        hess = self.petab_jl_problem.hess
        x_names = np.asarray(self.petab_jl_problem.xnames)

        # call the super super constructor
        super(JuliaObjective, self).__init__(fun, grad, hess, x_names)

    def __deepcopy__(self, memodict=None):
        """Deepcopy."""
        return PEtabJlObjective(
            module=self.module,
            source_file=self.source_file,
            petab_problem_name=self._petab_problem_name,
            precompile=False,
        )

    def precompile_model(self, force_compile: bool = False):
        """
        Use Julias PrecompilationTools to precompile the relevant code.

        Only needs to be done once, and speeds up Julia loading drastically.
        """
        directory = os.path.dirname(self.source_file)
        # check whether precompilation is necessary, if the directory exists
        if (
            os.path.exists(f"{directory}/{self.module}_pre")
            and not force_compile
        ):
            logger.info("Precompilation module already exists.")
            return None
        # lazy imports
        try:
            from julia import Main  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            ) from None
        # setting up a local project, where the precompilation will be done in
        from julia import Pkg

        Pkg.activate(".")
        # create a Project f"{self.module}_pre".
        try:
            Pkg.generate(f"{directory}/{self.module}_pre")
        except Exception:
            logger.info("Module is already generated. Skipping generate...")
        # Adjust the precompilation file
        write_precompilation_module(
            module=self.module,
            source_file_orig=self.source_file,
        )
        # add a new line at the top of the original module to use the
        # precompiled module
        with open(self.source_file) as read_f:
            if read_f.readline().endswith("_pre\n"):
                with open("dummy_temp_file.jl", "w+") as write_f:
                    write_f.write(f"using {self.module}_pre\n\n")
                    write_f.write(read_f.read())
                os.remove(self.source_file)
                os.rename("dummy_temp_file.jl", self.source_file)

        try:
            Pkg.develop(path=f"{directory}/{self.module}_pre")
        except Exception:
            logger.info("Module is already developed. Skipping develop...")
        Pkg.activate(f"{directory}/{self.module}_pre/")
        # add dependencies
        Pkg.add("PrecompileTools")
        Pkg.add("OrdinaryDiffEq")
        Pkg.add("PEtab")
        Pkg.add("Sundials")
        Pkg.precompile()


def write_precompilation_module(module, source_file_orig):
    """Write the precompilation module for the PEtabJl module."""
    # read the original source file
    with open(source_file_orig) as f:
        lines = np.array(f.readlines())
    # path to the yaml file
    yaml_path = "\t".join(lines[["yaml" in line for line in lines]])
    # packages
    packages = "\t\t".join(
        lines[[line.startswith("using ") for line in lines]]
    )
    # get everything in between the packages and the end line
    start = int(np.argwhere([line.startswith("using ") for line in lines])[-1])
    end = int(np.argwhere([line.startswith("end") for line in lines])[0])
    petab_loading = "\t\t".join(lines[start:end])

    content = (
        f"module {module}_pre\n\n"
        f"using PrecompileTools\n\n"
        f"# Reduce time for reading a PEtabModel and for "
        f"building a PEtabODEProblem\n"
        f"@setup_workload begin\n"
        f"\t{yaml_path}"
        f"\t@compile_workload begin\n"
        f"\t\t{packages}"
        f"\t\t{petab_loading}"
        f"\tend\n"
        f"end\n\n"
        f"end\n"
    )
    # get the directory of the source file
    directory = os.path.dirname(source_file_orig)
    # write file
    with open(f"{directory}/{module}_pre/src/{module}_pre.jl", "w") as f:
        f.write(content)
