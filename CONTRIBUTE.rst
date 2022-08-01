Contribute
==========

Workflow
--------

If you start working on a new feature or a fix, please create an issue on
GitHub briefly describing the issue and assign yourself.
Your startpoint should always be the ``develop`` branch, which contains the
latest updates.

Create an own branch or fork, on which you can implement your changes. To
get your work merged, please:

1. create a pull request to the ``develop`` branch with a meaningful summary,
2. check that code changes are covered by tests, and all tests pass,
3. check that the documentation is up-to-date,
4. request a code review from the main developers.

Environment
-----------

If you contribute to the development of pyPESTO, install developer requirements
via::

    pip install -r requirements-dev.txt

This installs the tools described below.

Pre-commit hooks
~~~~~~~~~~~~~~~~

Firstly, this installs a `pre-commit <https://pre-commit.com/>`_ tool.
To add those hooks to the `.git` folder of your local clone such that they are
run on every commit, run::

    pre-commit install

When adding new hooks, consider manually running ``pre-commit run --all-files``
once as usually only the diff is checked. The configuration is specified in
``.pre-commit-config.yaml``.

Should it be necessary to perform commits without pre-commit verification,
use ``git commit --no-verify`` or the shortform ``-n``.

Tox
~~~

Secondly, this installs the virtual testing tool
`tox <https://tox.readthedocs.io/en/latest/>`_, which we use for all tests,
format and quality checks. Its configuration is specified in ``tox.ini``.
To run it locally, simply execute::

    tox [-e flake8,doc]

with optional ``-e`` options specifying the environments to run, see
``tox.ini`` for details.

GitHub Actions
--------------

For automatic continuous integration testing, we use GitHub Actions. All tests
are run there on pull requests and are required to pass. The configuration is
specified in ``.github/workflows/ci.yml``.

Documentation
-------------

To make pyPESTO easily usable, we try to provide good documentation,
including code annotation and usage examples.
The documentation is hosted at
`pypesto.readthedocs.io <https://pypesto.readthedocs.io>`_
and updated automatically on merges to the main branches.
To create the documentation locally, first install the requirements via::

    pip install .[doc]

and then compile the documentation via::

    cd doc
    make html

The documentation is then under ``doc/_build``.

Alternatively, the documentation can be compiled and tested via a single line::

    tox -e doc

When adding code, all modules, classes, functions, parameters, code blocks
should be properly documented.

For docstrings, we follow the numpy docstring standard.
Check
`here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for a detailed explanation.

Unit tests
----------

Unit tests are located in the ``test`` folder. All files starting with
``test_`` contain tests and are automatically run on GitHub Actions.
Run them locally via e.g.::

    tox -e base

with ``base`` covering basic tests, but some parts (``optimize,petab,...``)
being in separate subfolders. This boils mostly down to e.g.::

    pytest test/base

You can also run only specific tests.

Unit tests can be written with `pytest <https://docs.pytest.org/en/latest/>`_
or `unittest <https://docs.python.org/3/library/unittest.html>`_.

Code changes should always be covered by unit tests.
It might not always be easy to test code which is based on random sampling,
but we still encourage general sanity and integration tests.
We highly encourage a
`test-driven development <http://en.wikipedia.org/wiki/Test-driven_development>`_
style.

PEP8
----

We try to respect the `PEP8 <https://www.python.org/dev/peps/pep-0008>`_
coding standards. We run `flake8 <https://flake8.pycqa.org>`_ as part of the
tests. The flake8 plugins used are specified in ``tox.ini`` and the flake8
configuration is given in ``.flake8``. You can run the checks locally via::

    tox -e flake8
