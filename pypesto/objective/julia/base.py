"""Interface to Julia via pyjulia."""

from typing import Callable, Union

import numpy as np

from ..function import Objective


def _as_array(input):
    """Convert output to numpy array."""
    if callable(input):

        def wrapper(*args, **kwargs):
            return np.asarray(input(*args, **kwargs))

        return wrapper

    return np.asarray(input)


def _read_source(module_name: str, source_file: str) -> None:
    """Read source if module not attached to julia Main yet.

    Parameters
    ----------
    module_name: Julia module name.
    source_file: Qualified Julia source file.
    """
    from julia import Main

    if not hasattr(Main, module_name):
        Main.include(source_file)


class JuliaObjective(Objective):
    """Wrapper around an objective defined in Julia.

    This class provides objective function wrappers around Julia objects.
    It expects the corresponding Julia objects to be defined in a
    `source_file` within a `module`.

    We use the PyJulia package to access Julia from inside Python.
    It can be installed via `pip install pypesto[julia]`, however requires
    additional Julia dependencies to be installed via:

    >>> python -c "import julia; julia.install()"

    For further information, see
    https://pyjulia.readthedocs.io/en/latest/installation.html.

    There are some known problems, e.g. with statically linked Python
    interpreters, see
    https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
    for details.
    Possible solutions are to pass ``compiled_modules=False`` to the Julia
    constructor early in your code:

    >>> from julia.api import Julia
    >>> jl = Julia(compiled_modules=False)

    This however slows down loading and using Julia packages, especially for
    large ones.
    An alternative is to use the ``python-jl`` command shipped with PyJulia:

    >>> python-jl MY_SCRIPT.py

    This basically launches a Python interpreter inside Julia.
    When using Jupyter notebooks, this wrapper can be installed as an
    additional kernel via:

    >>> python -m ipykernel install --name python-jl [--prefix=/path/to/python/env]

    And changing the first argument in
    ``/path/to/python/env/share/jupyter/kernels/python-jl/kernel.json``
    to ``python-jl``.

    Model simulations are eagerly converted to Python objects
    (specifically, `numpy.ndarray` and `pandas.DataFrame`).
    This can introduce overhead and could be avoided by an alternative
    lazy implementation.

    Parameters
    ----------
    module:
        Julia module name.
    source_file:
        Julia source file name. Defaults to `{module}.jl`.
    fun, grad, hess, res, sres:
        Names of callables within the Julia code of the corresponding
        objective functions and derivatives.
    """

    def __init__(
        self,
        module: str,
        source_file: str = None,
        fun: str = None,
        grad: str = None,
        hess: str = None,
        res: str = None,
        sres: str = None,
    ):
        # lazy imports
        try:
            from julia import Main  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            ) from None

        # store module name and source file
        self.module: str = module
        if source_file is None:
            source_file = module + ".jl"
        self.source_file: str = source_file
        _read_source(self.module, self.source_file)

        # store function names
        self._fun: str = fun
        self._grad: str = grad
        self._hess: str = hess
        self._res: str = res
        self._sres: str = sres

        # get callables
        fun, grad, hess, res, sres = self._get_callables()
        super().__init__(fun=fun, grad=grad, hess=hess, res=res, sres=sres)

    def get(self, name: str, as_array: bool = False) -> Union[Callable, None]:
        """Get variable from Julia module.

        Use this function to access any variable from the Julia module.
        """
        from julia import Main

        if name is not None:
            ret = getattr(getattr(Main, self.module), name, None)
            if as_array:
                ret = _as_array(ret)
            return ret
        return None

    def _get_callables(self) -> tuple:
        """Get all callables."""
        fun = self.get(self._fun)
        grad = self.get(self._grad, as_array=True)
        hess = self.get(self._hess, as_array=True)
        res = self.get(self._res, as_array=True)
        sres = self.get(self._sres, as_array=True)
        return fun, grad, hess, res, sres

    def __getstate__(self):
        return {
            "module": self.module,
            "source_file": self.source_file,
            "_fun": self._fun,
            "_grad": self._grad,
            "_hess": self._hess,
            "_res": self._res,
            "_sres": self._sres,
        }

    def __setstate__(self, d):
        for key, val in d.items():
            setattr(self, key, val)
        _read_source(self.module, self.source_file)

        fun, grad, hess, res, sres = self._get_callables()
        super().__init__(fun=fun, grad=grad, hess=hess, res=res, sres=sres)

    def __deepcopy__(self, memodict=None) -> "JuliaObjective":
        return JuliaObjective(
            module=self.module,
            source_file=self.source_file,
            fun=self._fun,
            grad=self._grad,
            hess=self._hess,
            res=self._res,
            sres=self._sres,
        )


def display_source_ipython(source_file: str):
    """Display source code as syntax highlighted HTML within IPython."""
    import IPython.display as display
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import JuliaLexer

    with open(source_file) as f:
        code = f.read()

    formatter = HtmlFormatter()
    return display.HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"),
            highlight(code, JuliaLexer(), formatter),
        )
    )
