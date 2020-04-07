Release notes
=============


0.0 series
..........


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
