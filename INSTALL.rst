Install and upgrade
===================


Requirements
------------

This package requires Python 3.8 or later (see :ref:`Python support`).
It is continuously tested on Linux, and most parts should also work on other
operating systems (MacOS, Windows).

I cannot use my system's Python distribution, what now?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several Python distributions can co-exist on a single system.
If you don't have access to a recent Python version via your
system's package manager (this might be the case for old
operating systems), it is recommended to install the latest
version of the
`Anaconda Python 3 distribution <https://www.continuum.io/downloads>`_.

Also, there is the possibility to use multiple virtual environments via::

    python3 -m virtualenv ENV_NAME
    source ENV_NAME/bin/activate

where ENV_NAME denotes an individual environment name,
if you do not want to mess up the system environment.


Install from PIP
----------------

The package can be installed from the Python Package Index PyPI
via pip::

    pip3 install pypesto


Install from GIT
----------------

If you want the bleeding edge version, install directly from github::

    pip3 install git+https://github.com/icb-dcm/pypesto.git

If you need to have access to the source code, you can download it via::

    git clone https://github.com/icb-dcm/pypesto.git

and then install from the local repository via::

    cd pypesto
    pip3 install .


Upgrade
-------

If you want to upgrade from an existing previous version, replace
``install`` by ``ìnstall --upgrade`` in the above commands.


Install optional packages and external dependencies
---------------------------------------------------

* pyPESTO includes multiple convenience methods to simplify
  parameter estimation for models generated using the toolbox
  `AMICI <https://github.com/AMICI-dev/AMICI>`_.
  To use AMICI, install it via pip::

    pip3 install amici

  or, in case of problems, follow the full instructions from the
  `AMICI documentation <https://amici.readthedocs.io/en/latest/python_installation.html>`_.

* This package inherently supports optimization using the dlib toolbox.
  To use it, install dlib via::

   pip3 install dlib

* All external dependencies can be installed through
  `this shell script <https://github.com/ICB-DCM/pyPESTO/blob/main/.github/workflows/install_deps.sh>`_.

.. _Python Support:

Python support
--------------

We adopt the
`NEP 29 - Recommend Python and NumPy version support as a community policy standard <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.
That means, we adopt a time window based policy for support of Python (and NumPy) versions.
