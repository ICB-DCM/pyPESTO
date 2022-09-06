"""Interface to Julia via pyjulia."""

from typing import Callable, Union

import numpy as np

try:
    from julia import Main
except ImportError:
    pass

from ..function import Objective


def as_array(fun):
    """Convert output to numpy array."""

    def wrapper(*args, **kwargs):
        return np.asarray(fun(*args, **kwargs))

    return wrapper


def _read_source(module_name: str, source_file: str) -> None:
    """Read source if module not attached to julia Main yet.

    Parameters
    ----------
    module_name: Julia module name.
    source_file: Qualified Julia source file.
    """
    if not hasattr(Main, module_name):
        Main.include(source_file)


class JuliaObjective(Objective):
    """Wrapper around an objective defined in Julia."""

    def __init__(
        self,
        module: str,
        source_file: str = None,
        fun: str = "fun",
        grad: str = "grad",
        hess: str = "hess",
        res: str = "res",
        sres: str = None,
    ):
        # checks
        if Main is None:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pypesto[julia]`, "
                "and see the class documentation",
            )

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

    def get(self, name: str) -> Union[Callable, None]:
        """Get variable from Julia module.

        Use this function to access any variable from the Julia module.
        """
        if name is not None:
            return getattr(getattr(Main, self.module), name, None)
        return None

    def _get_callables(self) -> tuple:
        """Get all callables."""
        fun = self.get(self._fun)
        grad = as_array(self.get(self._grad))
        hess = as_array(self.get(self._hess))
        res = as_array(self.get(self._res))
        sres = as_array(self.get(self._sres))
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

    def __deepcopy__(self, memodict=None) -> 'JuliaObjective':
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
            formatter.get_style_defs('.highlight'),
            highlight(code, JuliaLexer(), formatter),
        )
    )
