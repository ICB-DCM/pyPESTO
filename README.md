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

![](https://raw.githubusercontent.com/ICB-DCM/pyPESTO/main/doc/gfx/concept_pypesto.png)
*Feature overview of pyPESTO. Figure taken from the [Bioinformatics publication](https://doi.org/10.1093/bioinformatics/btad711).*

pyPESTO features include:

* Parameter estimation interfacing **multiple optimization algorithms** including
  multi-start local and global optimization. ([example](https://pypesto.readthedocs.io/en/latest/example/getting_started.html),
  [overview of optimizers](https://pypesto.readthedocs.io/en/latest/api/pypesto.optimize.html))
* Interface to **multiple simulators** including
  * [AMICI](https://github.com/AMICI-dev/AMICI/) for efficient simulation and
    sensitivity analysis of ordinary differential equation (ODE) models. ([example](https://pypesto.readthedocs.io/en/latest/example/amici.html))
  * [RoadRunner](https://libroadrunner.org/) for simulation of SBML models. ([example](https://pypesto.readthedocs.io/en/latest/example/roadrunner.html))
  * [Jax](https://jax.readthedocs.io/en/latest/quickstart.html) and
    [Julia](https://julialang.org) for automatic differentiation.
* **Uncertainty quantification** using various methods:
  * **Profile likelihoods**.
  * **Sampling** using Markov chain Monte Carlo (MCMC), parallel tempering, and
    interfacing other samplers including [emcee](https://emcee.readthedocs.io/en/stable/),
    [pymc](https://www.pymc.io/welcome.html) and
    [dynesty](https://dynesty.readthedocs.io/en/stable/).
    ([example](https://pypesto.readthedocs.io/en/latest/example/sampler_study.html))
  * **Variational inference**
* **Complete** parameter estimation **pipeline** for systems biology problems specified in
  [SBML](http://sbml.org/) and [PEtab](https://github.com/PEtab-dev/PEtab).
  ([example](https://pypesto.readthedocs.io/en/latest/example/petab_import.html))
* Parameter estimation pipelines for **different modes of data**:
  * **Relative (scaled and offset) data** as described in
    [Schmiester et al. (2020)](https://doi.org/10.1093/bioinformatics/btz581).
    ([example](https://pypesto.readthedocs.io/en/latest/example/relative_data.html))
  * **Ordinal data** as described in
    [Schmiester et al. (2020)](https://doi.org/10.1007/s00285-020-01522-w) and
    [Schmiester et al. (2021)](https://doi.org/10.1093/bioinformatics/btab512).
    ([example](https://pypesto.readthedocs.io/en/latest/example/ordinal_data.html))
  * **Censored data**. ([example](https://pypesto.readthedocs.io/en/latest/example/censored_data.html))
  * **Semiquantitative data** as described in [Doresic et al. (2024)](https://doi.org/10.1093/bioinformatics/btae210). ([example](https://pypesto.readthedocs.io/en/latest/example/semiquantitative_data.html))
* **Model selection**. ([example](https://pypesto.readthedocs.io/en/latest/example/model_selection.html))
* Various **visualization methods** to analyze parameter estimation results.

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

## How to Cite

**Citeable DOI for the latest pyPESTO release:**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2553546.svg)](https://doi.org/10.5281/zenodo.2553546)

When using pyPESTO in your project, please cite
* Schälte, Y., Fröhlich, F., Jost, P. J., Vanhoefer, J., Pathirana, D., Stapor, P.,
  Lakrisenko, P., Wang, D., Raimúndez, E., Merkt, S., Schmiester, L., Städter, P.,
  Grein, S., Dudkin, E., Doresic, D., Weindl, D., & Hasenauer, J. (2023). pyPESTO: A
  modular and scalable tool for parameter estimation for dynamic models,
  Bioinformatics, 2023, btad711, [doi:10.1093/bioinformatics/btad711](https://doi.org/10.1093/bioinformatics/btad711)

When presenting work that employs pyPESTO, feel free to use one of the icons in
[doc/logo/](doc/logo):

<p align="center">
  <img src="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/main/doc/logo/logo.png" height="75" alt="pyPESTO Logo">
</p>

There is a list of [publications using pyPESTO](https://pypesto.readthedocs.io/en/latest/references.html).
If you used pyPESTO in your work, we are happy to include
your project, please let us know via a GitHub issue.

## References

pyPESTO supersedes [**PESTO**](https://github.com/ICB-DCM/PESTO/) a parameter estimation
toolbox for MATLAB, whose development is discontinued.
