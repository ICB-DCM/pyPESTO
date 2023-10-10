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
  ([example](https://github.com/ICB-DCM/pyPESTO/blob/main/doc/example/amici.ipynb))
* Parameter estimation pipeline for systems biology problems specified in
  [SBML](http://sbml.org/) and [PEtab](https://github.com/PEtab-dev/PEtab)
  ([example](https://github.com/ICB-DCM/pyPESTO/blob/master/doc/example/petab_import.ipynb))
* Parameter estimation with ordinal data as described in
  [Schmiester et al. (2020)](https://doi.org/10.1007/s00285-020-01522-w) and
  [Schmiester et al. (2021)](https://doi.org/10.1093/bioinformatics/btab512).
  ([example](https://github.com/ICB-DCM/pyPESTO/blob/master/doc/example/example_ordinal.ipynb))
* Parameter estimation with censored data. ([example](https://github.com/ICB-DCM/pyPESTO/blob/master/doc/example/example_censored.ipynb))
* Parameter estimation with nonlinear-monotone data. ([example](https://github.com/ICB-DCM/pyPESTO/blob/master/doc/example/example_nonlinear_monotone.ipynb))

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

## Publications

**Citeable DOI for the latest pyPESTO release:**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2553546.svg)](https://doi.org/10.5281/zenodo.2553546)

There is a list of [publications using pyPESTO](https://pypesto.readthedocs.io/en/latest/references.html).
If you used pyPESTO in your work, we are happy to include
your project, please let us know via a GitHub issue.

When using pyPESTO in your project, please cite
* Schälte, Y., Fröhlich, F., Jost, P. J., Vanhoefer, J., Pathirana, D., Stapor, P.,
  Lakrisenko, P., Wang, D., Raimúndez, E., Merkt, S., Schmiester, L., Städter, P.,
  Grein, S., Dudkin, E., Doresic, D., Weindl, D., & Hasenauer, J. (2023). pyPESTO: A
  modular and scalable tool for parameter estimation for dynamic models [(arXiv:2305.01821)](https://doi.org/10.48550/arXiv.2305.01821).

When presenting work that employs pyPESTO, feel free to use one of the icons in
[doc/logo/](https://github.com/ICB-DCM/pyPESTO/tree/main/doc/logo):

<p align="center">
  <img src="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/master/doc/logo/logo.png" height="75" alt="AMICI Logo">
</p>

## References

pyPESTO supersedes [**PESTO**](https://github.com/ICB-DCM/PESTO/) a parameter estimation
toolbox for MATLAB, whose development is discontinued.
