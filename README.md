# pyPESTO - Parameter EStimation TOolbox for python

<img src="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/master/doc/logo/logo_wordmark.png" width="50%" alt="pyPESTO logo"/>

**pyPESTO** is a widely applicable and highly customizable toolbox for
parameter estimation.

[![PyPI](https://badge.fury.io/py/pypesto.svg)](https://badge.fury.io/py/pypesto)
[![CI](https://github.com/ICB-DCM/pyPESTO/workflows/CI/badge.svg)](https://github.com/ICB-DCM/pyPESTO/actions)
[![Coverage](https://codecov.io/gh/ICB-DCM/pyPESTO/branch/master/graph/badge.svg)](https://codecov.io/gh/ICB-DCM/pyPESTO)
[![Documentation](https://readthedocs.org/projects/pypesto/badge/?version=latest)](https://pypesto.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2553546.svg)](https://doi.org/10.5281/zenodo.2553546)

## Feature overview

pyPESTO features include:

* Multi-start local optimization
* Profile computation
* Result visualization
* Interface to [AMICI](https://github.com/AMICI-dev/AMICI/) for efficient
  simulation and sensitivity analysis of ordinary differential equation (ODE)
  models
  ([example](https://github.com/ICB-DCM/pyPESTO/blob/main/doc/example/amici_import.ipynb))
* Parameter estimation pipeline for systems biology problems specified in
  [SBML](http://sbml.org/) and [PEtab](https://github.com/PEtab-dev/PEtab)
  ([example](https://github.com/ICB-DCM/pyPESTO/blob/master/doc/example/petab_import.ipynb))
* Parameter estimation with qualitative data as described in
  [Schmiester et al. (2020)](https://doi.org/10.1007/s00285-020-01522-w).
  This is currently implemented in the `feature_ordinal` branch.

## Quick install

The simplest way to install **pyPESTO** is via pip:

```shell
pip3 install pypesto
```

More information is available here:
https://pypesto.readthedocs.io/en/latest/install.html

## Documentation

The documentation is hosted on readthedocs.io:
<https://pypesto.readthedocs.io>

## Examples

Multiple use cases are discussed in the documentation. In particular, there are
jupyter notebooks in the [doc/example](doc/example) directory.

## Contributing

We are happy about any contributions. For more information on how to contribute
to pyPESTO check out
<https://pypesto.readthedocs.io/en/latest/contribute.html>

## References

[**PESTO**](https://github.com/ICB-DCM/PESTO/):
Parameter estimation toolbox for MATLAB. Development is discontinued, but PESTO
comes with additional features waiting to be ported to pyPESTO.
