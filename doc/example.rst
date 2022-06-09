Examples
========

We provide a collection of example notebooks to get a better idea of how to
use pyPESTO, and illustrate core features.

The notebooks can be run locally with an installation of jupyter
(``pip install jupyter``), or online on Google Colab or nbviewer, following
the links at the top of each notebook.
At least an installation of pyPESTO is required, which can be performed by

.. code:: sh

   # install if not done yet
   !pip install pypesto --quiet

Potentially, further dependencies may be required.


Getting started
---------------

.. toctree::
   :maxdepth: 2

   example/rosenbrock.ipynb

PEtab and AMICI
---------------

.. toctree::
   :maxdepth: 2

   example/amici_import.ipynb
   example/petab_import.ipynb

Algorithms and features
-----------------------

.. toctree::
   :maxdepth: 2

   example/fixed_parameters.ipynb
   example/prior_definition.ipynb
   example/sampling_diagnostics.ipynb
   example/store.ipynb
   example/hdf5_storage.ipynb
   example/model_selection.ipynb

..
   temporarily disabled until pymc4 support is there https://github.com/ICB-DCM/pyPESTO/pull/818
   example/sampler_study.ipynb

Application examples
--------------------

.. toctree::
   :maxdepth: 2

   example/conversion_reaction.ipynb
   example/synthetic_data.ipynb
