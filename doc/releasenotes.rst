Release notes
=============


0.2 series
..........


0.2.3 (2021-01-18)
------------------

* New Optimizers:
    * FIDES (#506, #503 # 500)
    * NLopt (#493)

* Extended PEtab support:
    * PySB import (#437)
    * Support of PEtab's initializationPriors (#535)
    * Support of prior parameterScale{Normal,Laplace}  (#520)
    * Example notebook for synthetic data generation (#482)

* General new and improved functionality:
    * Predictions (#544)
    * Move tests to GitHub Actions (#524)
    * Parallelize profile calculation (#532)
    * Save `x_guesses` in `pypesto.problem` (#494)
    * Improved finite difference gradients (#464)
    * Support of unconstrained optimization (#519)
    * Additional NaN check for fval, grad and hessian (#521)
    * Add sanity checks for optimizer bounds (#516)

* Improvements in storage:
    * Fix hdf5 export of optimizer history (#536)
    * Fix reading `x_names` from hdf5 history (#528)
    * Storage does not save empty arrays (#489)
    * hdf5 storage sampling (#546)
    * hdf5 storage parameter profiles (#546)

* Improvements in the visualization routines:
    * Plot parameter values as histogram (#485)
    * Fix y axis limits in waterfall plots (#503)
    * Fix color scheme in visualization (#498)
    * Improved visualization of optimization results (#486)

* Several small bug fixes (#547, #541, #538, #533, #512, #508)


0.2.2 (2020-10-05)
------------------

* New optimizer: CMA-ES (#457)
* New plot: Optimizer convergence summary (#446)

* Fixes in visualization:
    * Type checks for reference points (#460)
    * y_limits in waterfall plots with multiple results (#475)
* Support of new amici release (#469)

* Multiple fixes in optimization code:
    * Remove unused argument for dlib optimizer (#466)
    * Add check for installation of ipopt (#470)
    * Add maxiter as default option of dlib (#474)

* Numpy based subindexing in amici_util (#462)
* Check amici/PEtab installation (#477)


0.2.1 (2020-09-07)
------------------

* Example Notebook for prior functionality (#438)
* Changed parameter indexing in profiling routines (#419)
* Basic sanity checking for parameter fixing (#420)

* Bug fixes in:
    * Displaying of multi start optimization (#430)
    * AMICI error output (#428)
    * Axes scaling/limits in waterfall plots (#441)
    * Priors (PEtab import, error handling) (#448, #452, #454)

* Improved sampling diagnostics (e.g. effective samples size) (#426)
* Improvements and bug fixes in parameter plots (#425)


0.2.0 (2020-06-17)
------------------

Major:

* Modularize import, to import optimization, sampling and profiling
  separately (#413)

Minor:

* Bug fixes in
    * sampling (#412)
    * visualization (#405)
    * PEtab import (#403)
    * Hessian computation (#390)

* Improve hdf5 error output (#409)
* Outlaw large new files in GitHub commits (#388)


0.1 series
..........


0.1.0 (2020-06-17)
------------------

Objective

* Write solver settings to stream to enable serialization for distributed
  systems (#308)

* Refactor objective function (#347)
    * Removes necessity for all of the nasty binding/undbinding in AmiciObjective
    * Substantially reduces the complexity of the AggregatedObjective class
    * Aggregation of functions with inconsistent sensi_order/mode support
    * Introduce ObjectiveBase as an abstract Objective class
    * Introduce FunctionObjective for objectives from functions

* Implement priors with gradients, integrate with PEtab (#357)
* Fix minus sign in AmiciObjective.get_error_output (#361)
* Implement a prior class, derivatives for standard models, interface with
  PEtab (#357)
* Use `amici.import_model_module` to resolve module loading failure (#384)

Problem

* Tidy up problem vectors using properties (#393)

Optimization

* Interface IpOpt optimizer (#373)

Profiles

* Tidy up profiles (#356)
* Refactor profiles; add locally approximated profiles (#369)
* Fix profiling and visualization with fixed parameters (#393)

Sampling

* Geweke test for sampling convergence (#339)
* Implement basic Pymc3 sampler (#351)
* Make theano for pymc3 an optional dependency (allows using pypesto without
  pymc3) (#356)
* Progress bar for MCMC sampling (#366)
* Fix Geweke test crash for small sample sizes (#376)
* In parallel tempering, allow to only temperate the likelihood, not the prior
  (#396)

History and storage

* Allow storing results in a pre-filled hdf5 file (#290)
* Various fixes of the history (reduced vs. full parameters, read-in from file,
  chi2 values) (#315)
* Fix proper dimensions in result for failed start (#317)
* Create required directories before creating hdf5 file (#326)
* Improve storage and docs documentation (#328)
* Fix storing x_free_indices in hdf5 result (#334)
* Fix problem hdf5 return format (#336)
* Implement partial trace extraction, simplify History API (#337)
* Save really all attributes of a Problem to hdf5 (#342)

Visualization

* Customizable xLabels and tight layout for profile plots (#331)
* Fix non-positive bottom ylim on a log-scale axis in waterfall plots (#348)
* Fix "palette list has the wrong number of colors" in sampling plots (#372)
* Allow to plot multiple profiles from one result (#399)

Logging

* Allow easier specification of only logging for submodules (#398)

Tests

* Speed up travis build (#329)
* Update travis test system to latest ubuntu and python 3.8 (#330)
* Additional code quality checks, minor simplifications (#395)


0.0 series
..........


0.0.13 (2020-05-03)
-------------------

* Tidy up and speed up tests (#265 and others).
* Basic self-implemented Adaptive Metropolis and Adaptive Parallel Tempering
  sampling routines (#268).
* Fix namespace sample -> sampling (#275).
* Fix covariance matrix regularization (#275).
* Fix circular dependency `PetabImporter` - `PetabAmiciObjective` via
  `AmiciObjectBuilder`, `PetabAmiciObjective` becomes obsolete (#274).
* Define `AmiciCalculator` to separate the AMICI call logic (required for
  hierarchical optimization) (#277).
* Define initialize function for resetting steady states in `AmiciObjective`
  (#281).
* Fix scipy least squares options (#283).
* Allow failed starts by default (#280).
* Always copy parameter vector in objective to avoid side effects (#291).
* Add Dockerfile (#288).
* Fix header names in CSV history (#299).

Documentation:

* Use imported members in autodoc (#270).
* Enable python syntax highlighting in notebooks (#271).


0.0.12 (2020-04-06)
-------------------

* Add typehints to global functions and classes.
* Add `PetabImporter.rdatas_to_simulation_df` function (all #235).
* Adapt y scale in waterfall plot if convergence was too good (#236).
* Clarify that `Objective` is of type negative log-posterior, for
  minimization (#243).
* Tidy up `AmiciObjective.parameter_mapping` as implemented in AMICI now
  (#247).
* Add `MultiThreadEngine` implementing multi-threading aside the
  `MultiProcessEngine` implementing multi-processing (#254).
* Fix copying and pickling of `AmiciObjective` (#252, #257).
* Remove circular dependence history-objective (#254).
* Fix problem of visualizing results with failed starts (#249).
* Rework history: make thread-safe, use factory methods, make context-specific
  (#256).
* Improve PEtab usage example (#258).
* Define history base contract, enabling different backends (#260).
* Store optimization results to HDF5 (#261).
* Simplify tests (#263).

Breaking changes:

* `HistoryOptions` passed to `pypesto.minimize` instead of `Objective` (#256).
* `GlobalOptimizer` renamed to `PyswarmOptimizer` (#235).


0.0.11 (2020-03-17)
-------------------

* Rewrite AmiciObjective and PetabAmiciObjective simulation routine to directly use
  amici.petab_objective routines (#209, #219, #225).
* Implement petab test suite checks (#228).
* Various error fixes, in particular regarding PEtab and visualization.
* Improve trace structure.
* Fix conversion between fval and chi2, fix FIM (all #223).



0.0.10 (2019-12-04)
-------------------

* Only compute FIM when sensitivities are available (#194).
* Fix documentation build (#197).
* Add support for pyswarm optimizer (#198).
* Run travis tests for documentation and notebooks only on pull requests (#199).


0.0.9 (2019-10-11)
------------------

* Update to AMICI 0.10.13, fix API changes (#185). 
* Start using PEtab import from AMICI to be able to import constant species (#184, #185)
* Require PEtab>=0.0.0a16 (#183)


0.0.8 (2019-09-01)
------------------

* Add logo (#178).
* Fix petab API changes (#179).
* Some minor bugfixes (#168).


0.0.7 (2019-03-21)
------------------

* Support noise models in Petab and Amici.
* Minor Petab update bug fixes.


0.0.6 (2019-03-13)
------------------

* Several minor error fixes, in particular on tests and steady state.


0.0.5 (2019-03-11)
------------------

* Introduce AggregatedObjective to use multiple objectives at once.
* Estimate steady state in AmiciObjective.
* Check amici model build version in PetabImporter.
* Use Amici multithreading in AmiciObjective.
* Allow to sort multistarts by initial value.
* Show usage of visualization routines in notebooks.
* Various fixes, in particular to visualization.


0.0.4 (2019-02-25)
------------------

* Implement multi process parallelization engine for optimization.
* Introduce PrePostProcessor to more reliably handle pre- and
  post-processing.
* Fix problems with simulating for multiple conditions.
* Add more visualization routines and options for those (colors, 
  reference points, plotting of lists of result obejcts)


0.0.3 (2019-01-30)
------------------

* Import amici models and the petab data format automatically using
  pypesto.PetabImporter.
* Basic profiling routines.


0.0.2 (2018-10-18)
------------------

* Fix parameter values
* Record trace of function values
* Amici objective to directly handle amici models


0.0.1 (2018-07-25)
------------------

* Basic framework and implementation of the optimization
