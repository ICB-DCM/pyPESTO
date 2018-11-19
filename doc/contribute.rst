Contribute
==========


Contribute documentation
------------------------

To make pypesto easily usable, we are committed to documenting extensively.
This involves in particular documenting the functionality of methods and
classes, the purpose of single lines of code, and giving usage examples.
The documentation is hosted on 
`pypesto.readthedocs.io <https://pyabc.readthedocs.io>`_
and updated automatically every time the master branch on 
`github.com/icb-dcm/pypesto <https://github.com/icb-dcm/pypesto>`_ is updated.
To compile the documentation locally, use::

    cd doc
    make html


Contribute tests
----------------

Tests are located in the ``test`` folder. All files starting with ``test_``
contain tests and are automatically run on Travis CI. To run them manually,
type::

    python3 -m pytest test

or alternatively::

    python3 -m unittest test

You can also run specific tests.

Tests can be written with `pytest <https://docs.pytest.org/en/latest/>`_
or the `unittest <https://docs.python.org/3/library/unittest.html>`_ module.


PEP8
~~~~

We try to respect the `PEP8 <https://www.python.org/dev/peps/pep-0008>`_
coding standards. We run `flake8 <https://flake8.pycqa.org>`_ as part of the
tests. If flake8 complains, the tests won't pass. You can run it via::

    ./run_flake8.sh

in Linux from the base directory, or directly from python. More, you can use
the tool `autopep8 <https://pypi.org/project/autopep8>`_ to automatically
fix various coding issues.


Contribute code
---------------

* Internally, we use ``numpy`` for arrays. In particular, vectors are
  represented as arrays of shape (n,).
