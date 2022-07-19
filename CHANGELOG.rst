Release notes
=============


0.2 series
..........


0.2.13 (2022-05-24)
-------------------

* Ensembles:
    * Added standard deviation to ensemble prediction plots (#853)
* Storage
    * Distinguish between scalar and vector values in Hdf5History._get_hdf5_entries (#856)
    * Fix hdf5 history overwrite (#861)
    * Updated optimization storage format. Made attributes explicit. (#863)
    * Added problem to result from read_results_from_file (#862)
* General
    * Various additions to Optimize(r)Result summary method (#859, #865, #866, #867)
    * Fixed optimizer history fval offset (#834)
    * Updated the profile, minimize, sample and added overwrite as argument. (#864)
    * Fixed y-labels in pypesto.visualize.optimizer_history (#869)
    * Created show_bounds, to display proper sampling scatter plots. (#868)
    * Enabled saving messages and exit flags in hdf5 history in case of finished run (#873)
    * Select: use objective function evaluation time as optimization time for models with no estimated parameters (#872)
    * removed checking for equality and checking for np.allclose in test_aesara (#877)


0.2.12 (2022-04-11)
-------------------

* AMICI:
    * Update to renamed steady state sensitivity modes (#843)
    * Set amici.Solver.setReturnDataReportingMode (#835)
    * Optimize `pypesto/objective/amici_util.py::par_index_slices` (#845)
    * Remove Solver.getPreequilibration (#830)
    * fix n_res size for error output with parameter dependent sigma (#812)
    * PetabImporter: Auto-regenerate AMICI models in case of version mismatch (#848)
* Pymc3
    * Disable Pymc3 Sampler tests (#831)
*  Visualizations:
    * Waterfall zoom (#808)
    * Reverse opacities of colors in prediction trajectories plots(#838)
    * Model fit plots (#850)
* OptimizeResult:
    * Summary method (#816)
    * Append method for OptimizeResult (#815)
    * added __getattr__ function to OptimizeResult (#802)
* General:
    * disable progress bar in tests (#799)
    * Make Fides work with objectives, that do not have a hessian (#807)
    * removed ftol in favor of tol (#803)
    * Fix pyPESTO Select test; Update to stable black version (#810)
    * Fix id assignment in case of large number of starts (#825)
    * Temporarily fix jinja2 version (#826)
    * Upgrade black to be compatible with latest click (#829)
    * Fix wrong link in doc/example/hdf5_storage.ipynb (#827)
    * Mark test/base/test_prior.py::test_mode as flaky (#833)
    * Custom methods for autosave filenames (#822)
    * fix saving ensemble predictions to hdf5 (#840)
    * Upgrade nbQA to 1.3.1 (#846)
    * Replaced constantParameters with constant_parameters in notebook (#852)


0.2.11 (2022-01-11)
-------------------

* Model selection (#397):
    * Automated model selection with forward/backward/brute force methods and
      AIC/AICc/BIC criteria
    * Much functionality (methods, criteria, model space, problem
      specification) via `PEtab Select <https://github.com/PEtab-dev/petab_select>`
    * Plotting routines
    * `Example notebook <https://github.com/ICB-DCM/pyPESTO/blob/main/doc/example/model_selection.ipynb>`
    * Model calibration postprocessors
    * Select first model that improves on predecessor model
    * Use previous MLE as startpoint
    * Tests

* AMICI:
    * Maintain model settings when pickling for multiprocessing (#747)

* General:
    * Apply nbqa black and isort to auto-format all notebooks via
      pre-commit hook (#794)
    * Apply black formatting via pre-commit hook (#796)
    * Require Python >= 3.8 (#795)
    * Fix various warnings (#778)
    * Minor fixes (#792)


0.2.10 (2022-01-06)
-------------------

* AMICI:
    * Make AMICI objective report only what is being asked for (#777)

* Optimization:
    * (Breaking) Refactor startpoint generation with clear assignments;
      allow checking gradients (#769)
    * (Breaking) Prioritize history vs optimize result (#775)

* Storage:
    * Fix loading empty history and result generation from multiple
      histories (#764)
    * Fix autosave function for single-core (#770)
    * Fix potential autosave overwriting and typehints (#772)
    * Allow loading of partial results from history file (#783)

* CI:
    * Compile AMICI models without gradients in test suite (#774)

* General:
    * (Breaking) Create result sub-module; shift storage+result related
      functionality (#784)
    * Fix finite difference constant mode (#786)
    * Refactor ensemble module (#788)
    * Introduce general C constants file (#788)
    * Apply isort for automatic imports formatting (#785)
    * Reduce run log output (#789)
    * Various minor fixes (#765, #766, #768, #771)


0.2.9 (2021-11-03)
------------------

* General:
    * Automatically save results (#749)
    * Update all docstrings to numpy standard (#750)
    * Add Google Colab and nbviewer links to all notebooks for online
      execution (#758)
    * Option to not save hess and sres in result (#760)
    * Set minimum supported python version to 3.7 (#755)

* Visualization:
    * Parameterize start index in optimized model fit (#744)


0.2.8 (2021-10-28)
------------------

* PEtab:
    * Use correct measurement column name in `rdatas_to_simulation_df` (#721)
    * Visualize optimized model fit via PEtab problem (#725)
    * Un-ignore observable scaling tests (#742)
    * New function to plot model trajectory with custom time points (#739)

* Optimization:
    * OOD Refactor startpoint generation (#732)
    * Update to fides 0.6.0 (#733)
    * Correctly report FVAL vs CHI2 values in fides (#741)

* Ensemble:
    * Option for using weighted ensemble means (#702)
    * Default names and bounds for `Ensemble.from_sample` (#730)

* Storage:
    * Load optimization result from HDF5 history (#726)

* General:
    * Enable use of priors with least squares optimizers (#745)
    * Add temporary CITATION.cff file (#734)
    * Regular scheduled CI runs (#754)
    * Allow to not copy objective in problem (#756)

* Fixes:
    * Fix non-exported visualization in notebook (#729)
    * Mark some more tests as flaky (#704)
    * Fix minor data type and OOD issues in parameter and waterfall plots
      (#731)


0.2.7 (2021-07-30)
------------------

* Finite Differences:
    * Adaptive finite differences (#671)
    * Add helper function for checking gradients of objectives (#690)
    * Small bug fixes (#711, #714)

* Storage:
    * Store representation of the objective (#669)
    * Minor fixes in HDF5 history (#679)
    * HDF5 reader for ensemble predictions (#681)
    * Update storage demo jupyter notebook (#699)
    * Option to trim trace to be monotonically decreasing (#705)

* General:
    * Improved tests and bug fixes of validation intervals (#676, #685)
    * Add input file validation via PEtab linter for PEtab import (#678)
    * Remove default values from docstring (#680)
    * Minor fixes/improvements of ensembles (#687, #688)
    * Fix sorting of optimization values including `NaN` values (#691)
    * Specify axis limits for plotting (#693)
    * Minor fixes in visualization (#696)
    * Add installation option `all_optimizers` (#695)
    * Improve installation documentation (#689)
    * Update `pysb` and `BNG` version on GitHub Actions (#697)
    * Bug fix in steady state guesses (#715)


0.2.6 (2021-05-17)
------------------

* Objective:
    * Basic finite differences (#666)
    * Fix factor 2 in res/fval values (#619)

* Optimization:
    * Sort optimization results when appending (#668)
    * Read optimizer result from HDF5 (previously only CSV) (#663)

* Storage:
    * Load ensemble from HDF5 (#640)

* CI:
    * Add flake8 checks as pre-commit hook (#662)
    * Add efficient biological conversion reaction test model (#619)

* General:
    * No automatic import of the predict module (#657)
    * Assert unique problem parameter names (#665)
    * Load ensemble from optimization result with and without history usage
      (#640)
    * Calculate validation profile significance (#658)
    * Set pypesto screen logger to "INFO" by default (#667)

* Minor fixes:
    * Fix axis variable overwriting in `visualize.sampling_parameter_traces`
      (#665)


0.2.5 (2021-05-04)
------------------

* Objectives:
    * New Aesara objectve (#623, #629, #635)

* Sampling:
    * New Emcee sampler (#606)
    * Fix compatibility to new Theano version (#650)

* Storage:
    * Improve hdf5 storage documentation (#612)
    * Hdf5 history for MultiProcessEngine (#650)
    * Minor fixes (#637, #638, #645, #649)

* Visualization:
    * Fix bounds of parameter plots (#601)
    * Fix waterfall plots with multiple results (#611)

* CI:
    * Move CI tests on GitHub Actions to python 3.9 (#598)
    * Add issue template (#604)
    * Update BionetGen Link (#630)
    * Introduce project.toml (#634)

* General:
    * Introduce progress bar for optimization, profiles and ensembles (#641)
    * Extend gradient checking functionality (#644)

* Minor fixes:
    * Fix installation of ipopt (#599)
    * Fix Zenodo link (#601)
    * Fix duplicates in documentation (#603)
    * Fix least squares optimizers (#617 #631 #632)
    * Fix trust region options (#616)
    * Fix slicing for new AMICI release (#621)
    * Refactor and document latin hypercube sampling (#647)
    * Fix missing SBML name in PEtab import (#648)


0.2.4 (2021-03-12)
------------------

* Ensembles/Sampling:
    * General ensemble analysis, visualization, storage (#557, #565, #568)
    * Calculate and plot MCMC parameter and prediction CIs via ensemble
      definition, parallelize ensemble predictions (#490)

* Optimization:
    * New optimizer: SciPy Differential Evolution (#543)
    * Set fides default to hybrid (#578)

* AMICI:
    * Make `guess_steadystate` less restrictive (#561) and have a more
      intuitive default behavior (#562, #582)
    * Customize time points (#490)

* Storage:
    * Save HDF5 history with SingleCoreEngine (#564)
    * Add read/write function for whole results (#589)

* Engines:
    * MPI based distributed parallelization (#542)

* Visualization:
    * Speed up waterfall plots by resizing scales only once (#577)
    * Change waterfall default offset to 1 - minimum (#593)

* CI:
    * Move GHA CI tests to pull request level for better cooperability (#574)
    * Streamline test environments using tox and pre-commit hooks (#579)
    * Test profile and sampling storage (#585)
    * Update for Ubuntu 20.04, add rerun on failure (#587)

* Minor fixes (release notes #558, nlop tests #559, close files #495,
  visualization #554, deployment #560, flakiness #570,
  aggregated deepcopy #572, respect user-provided offsets #576,
  update to SWIG 4 #591, check overwrite in profile writing #566)


0.2.3 (2021-01-18)
------------------

* New optimizers:
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
