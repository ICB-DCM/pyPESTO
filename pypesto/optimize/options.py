from typing import Union


class OptimizeOptions(dict):
    """
    Options for the multistart optimization.

    Parameters
    ----------
    allow_failed_starts:
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.
    report_sres:
        Flag indicating whether sres will be stored in the results object.
        Deactivating this option will improve memory consumption for large
        scale problems.
    report_hess:
        Flag indicating whether hess will be stored in the results object.
        Deactivating this option will improve memory consumption for large
        scale problems.
    history_beats_optimizer:
        Whether the optimal value recorded by pyPESTO in the history has
        priority over the optimal value reported by the optimizer (True)
        or not (False).
    """

    def __init__(
        self,
        allow_failed_starts: bool = True,
        report_sres: bool = True,
        report_hess: bool = True,
        history_beats_optimizer: bool = True,
    ):
        super().__init__()

        self.allow_failed_starts: bool = allow_failed_starts
        self.report_sres: bool = report_sres
        self.report_hess: bool = report_hess
        self.history_beats_optimizer: bool = history_beats_optimizer

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(
        maybe_options: Union["OptimizeOptions", dict],
    ) -> "OptimizeOptions":
        """
        Return a valid options object.

        Parameters
        ----------
        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, OptimizeOptions):
            return maybe_options
        options = OptimizeOptions(**maybe_options)
        return options
