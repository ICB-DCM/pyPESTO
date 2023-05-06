"""Interface to PEtab.jl"""

import numpy as np
import logging

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
        Julia source file. Defaults to "{module_name}.jl".
    petab_problem_name:
        Name of the petab problem variable in the julia module.
    petab_jl_problem:
        Wrapper around the PEtab.jl problem.
    """

    def __init__(
            self,
            module: str = None,
            source_file: str = None,
            petab_problem_name: str = "petabProblem",
            _dump_file_name: str = None,
    ):
        """Initialize objective."""
        # check that not both module_name and petab_jl_problem are None
        if module is None and _dump_file_name is None:
            raise ValueError(
                "Either module_name or _dump_file_name must be specified."
            )

        # lazy imports
        try:
            from julia import Main  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            )

        # create temporary file, when pickled, check whether file exists
        self._dump_file_name = "tmp_petab_jl_problem.jld2"
        self._dumped = False

        # then dump petabProblem via JLD2. get state will load the
        # petabProblem from the file.
        # PotentialProblems: Is it accessible, will it be deleted?

        self.module = module
        self.source_file = source_file
        self._petab_problem_name = petab_problem_name

        # if petab_jl_problem is not specified, create it.
        if module is not None and _dump_file_name is None:
            if source_file is None:
                source_file = f"{module}.jl"

            # Include module if not already included
            _read_source(module, source_file)

            petab_jl_problem = self.get(petab_problem_name)
        else:
            petab_jl_problem = self.load_petab_problem(
                self._dump_file_name
            )
        self.petab_jl_problem = petab_jl_problem

        # get functions
        fun = self.petab_jl_problem.computeCost
        grad = self.petab_jl_problem.computeGradient
        hess = self.petab_jl_problem.computeHessian
        x_names = np.asarray(self.petab_jl_problem.θ_estNames)

        # call the super super constructor
        super(JuliaObjective, self).__init__(fun=fun, grad=grad, hess=hess, x_names=x_names)

    def __getstate__(self):
        """Get state for pickling."""
        # if not dumped, dump it via JLD2
        if not self._dumped:
            self.dump_petab_jl_problem(self._dump_file_name)
            self._dumped = True
        print(f"I have been dumped. You can find me in {self._dump_file_name}")
        return {
            'module': self.module,
            'source_file': self.source_file,
            '_petab_problem_name': self._petab_problem_name,
            '_dump_file_name': self._dump_file_name,
            '_dumped': self._dumped
        }

    def __setstate__(self, state):
        """Set state from pickling."""
        for key, value in state.items():
            setattr(self, key, value)
        # lazy imports
        try:
            from julia import Main  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            )
        # load petabProblem from file
        self.petab_jl_problem = self.load_petab_problem(
            self._dump_file_name
        )
        print(f"loaded petab problem from file done. {self.petab_jl_problem}")

        # get functions
        fun = self.petab_jl_problem.computeCost
        grad = self.petab_jl_problem.computeGradient
        hess = self.petab_jl_problem.computeHessian
        x_names = np.asarray(self.petab_jl_problem.θ_estNames)

        # call the super super constructor
        super(JuliaObjective, self).__init__(fun, grad, hess, x_names)

    def __deepcopy__(self, memodict=None):
        """Deepcopy."""
        # TODO: check whether this is also forces recompilation
        return PEtabJlObjective(
            module=self.module,
            source_file=self.source_file,
            petab_problem_name=self._petab_problem_name,
        )

    def dump_petab_jl_problem(self, file_name):
        """Dump the petab problem to a file."""
        from julia import Main

        try:
            Main.using(f"JLD2")
        except:
            logger.warning(
                "JLD2 not found. Install via `using Pkg; Pkg.add(\"JLD2\")`"
            )
            Main.eval(f"using Pkg; Pkg.add(\"JLD2\")")
            Main.using(f"JLD2")
        # now dump the petab problem
        Main.save(
            file_name,
            self._petab_problem_name,
            self.petab_jl_problem
        )

    def load_petab_problem(self, file_name):
        """Load the petab problem from a file."""
        from julia import Main
        Main.using(f"JLD2")
        # also need to use PEtab
        Main.using(f"PEtab")
        # now load the petab problem
        return Main.load(file_name, self._petab_problem_name)