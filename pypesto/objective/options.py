class ObjectiveOptions(dict):
    """
    Options for the objective that are used in optimization, profiles
    and sampling.

    Parameters
    ----------

    trace_record: bool, optional
        Flag indicating whether to record the trace of function calls.
        The trace_record_* flags only become effective if
        trace_record is True.
        Default: False.

    trace_record_grad: bool, optional
        Flag indicating whether to record the gradient in the trace.
        Default: True.

    trace_record_hess: bool, optional
        Flag indicating whether to record the Hessian in the trace.
        Default: False.

    trace_record_res: bool, optional
        Flag indicating whether to record the residual in
        the trace.
        Default: False.

    trace_record_sres: bool, optional.
        Flag indicating whether to record the residual sensitivities in
        the trace.
        Default: False.

    trace_record_chi2: bool, optional
        Flag indicating whether to record the chi2 in the trace.
        Default: True.

    trace_record_schi2: bool, optional
        Flag indicating whether to record the chi2 sensitivities in the
        trace.
        Default: True.

    trace_all: bool, optional
        Flag indicating whether to record all (True, default) or only
        better (False) values.

    trace_file: str or True, optional
        Either pass a string here denoting the file name for storing the
        trace, or True, in which case the default file name
        "tmp_trace_{index}.dat" is used. A contained substring {index}
        is converted to the multistart index.
        Default: None, i.e. no file is created.

    trace_save_iter. index, optional
        Trace is saved every tr_save_iter iterations.
        Default: 10.
    """

    def __init__(self,
                 trace_record=False,
                 trace_record_grad=True,
                 trace_record_hess=False,
                 trace_record_res=False,
                 trace_record_sres=False,
                 trace_record_chi2=True,
                 trace_record_schi2=True,
                 trace_all=True,
                 trace_file=None,
                 trace_save_iter=10):
        super().__init__()

        self.trace_record = trace_record
        self.trace_record_grad = trace_record_grad
        self.trace_record_hess = trace_record_hess
        self.trace_record_res = trace_record_res
        self.trace_record_sres = trace_record_sres
        self.trace_record_chi2 = trace_record_chi2
        self.trace_record_schi2 = trace_record_schi2
        self.trace_all = trace_all

        if trace_file is True:
            trace_file = "tmp_trace_{index}.dat"
        self.trace_file = trace_file

        self.trace_save_iter = trace_save_iter

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(maybe_options):
        """
        Returns a valid options object.

        Parameters
        ----------

        maybe_options: ObjectiveOptions or dict
        """
        if isinstance(maybe_options, ObjectiveOptions):
            return maybe_options
        options = ObjectiveOptions(**maybe_options)
        return options
