Log-normal relative measurements (PEtab v1)
============================================================

PEtab v1 observations with ``noiseDistribution: normal`` and
``observableTransformation: log`` or ``log10`` can use hierarchical
optimization for positive multiplicative scaling parameters and log-scale
noise parameters. The importer selects the log-normal inner solver
automatically; a custom objective is not required.

Specify the observation transformation and noise distribution in the PEtab
observable table. Mark the parameters to eliminate with ``parameterType``
equal to ``scaling`` or ``sigma`` in the parameter table, as for other
relative measurements. Then import the problem with hierarchical
optimization enabled:

.. code-block:: python

   import petab.v1 as petab
   from pypesto.petab import PetabImporter

   definition = petab.Problem.from_yaml("problem.yaml")
   problem = PetabImporter(definition, hierarchical=True).create_problem()

Scaling parameters multiply the model prediction in the original measurement
units. Estimated scaling parameters may use linear, log or log10 parameter
coordinates, with non-negative lower bounds and positive upper bounds. The
solution itself is strictly positive. Estimated sigma parameters use linear
parameter coordinates and bounds ``[0, infinity]``, following the existing
relative inner-parameter convention. Sigma is the standard deviation on the
chosen observation scale, rather than in the original measurement units.

The inner solve computes the weighted mean log scaling, applies its bounds
in log coordinates, and then computes each estimated sigma from the RMS
log-scale residual. The objective retains the original-measurement density,
including its change-of-variable Jacobian. The returned simulations remain
in their original units. Observations and predictions used in this solve
must be finite and strictly positive. A zero residual variance has no finite,
positive sigma maximum-likelihood estimate and raises an error.

Supported combinations
----------------------

* Separate experimental conditions, parameter overrides, replicate
  measurements and missing measurement entries.
* An estimated scale with fixed noise values, an estimated sigma with fixed
  scaling, or compatible scale and sigma parameters estimated together.
* Multiple scales sharing one estimated sigma, provided each scale's
  measurements all use that sigma.
* Separate quantitative measurements outside the relative inner problem.
* Objective values and first derivatives, using forward or adjoint
  sensitivities. Use an optimizer that requests these supported quantities.

All measured entries inside the relative inner problem must have ``log`` or
``log10`` observation transformations. The standard additive-Gaussian solver
continues to handle linear relative data.

Limitations
-----------

This implementation supports PEtab v1 only. It rejects mixed linear and log
relative inner data, inner additive offsets, nonmultiplicative scaling
formulas, priors on eliminated parameters, partially shared estimated sigma
parameters, and generalized bounds or transformed coordinates on estimated
sigmas. Residual mode and Hessian/Fisher-information requests are unsupported.

Choose solver tolerances and optimizer stopping settings for the model at
hand. In particular, adjoint validation may require explicit backward and
quadrature tolerances; this feature does not change the global solver defaults.

Validation
----------

The integration tests import PEtab definitions through the normal importer
and compare likelihoods, first derivatives and original-unit outputs with
the ordinary AMICI objective at the same conditional inner optimum. They
also cover central differences, scale bounds, unsupported inputs and a
synthetic L-BFGS-B optimization. In an environment with the hierarchical test
dependencies installed, run:

.. code-block:: bash

   python -m pytest test/hierarchical/test_lognormal.py -q

These checks establish implementation agreement for the tested cases; they
do not establish a global optimum, a speed improvement or biological validity.
