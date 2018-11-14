Deploy
======


Versioning scheme
-----------------

For version numbers, we use ``A.B.C``, where

* ``C`` is increased for bug fixes,
* ``B`` is increased for new features,
* ``A`` is increased for major or API breaking changes.


Deploy a new release
--------------------

When you are done with the changes on your git branch, proceed as follows
to deploy a new release.


Merge into master
~~~~~~~~~~~~~~~~~

First, you need to merge into the master:

1. check that all tests on travis pass
2. check that the documentation is up-to-date
3. adapt the version number in the file pesto/version.py
4. update the release notes in doc/releasenotes.rst
5. merge into the origin master branch

To be able to actualize perform the merge, sufficient rights may be
required. Also, at least one review is required.


Upload to PyPI
~~~~~~~~~~~~~~

After a successful merge, you need to update also the package on PyPI:

5. create a so-called "wheel" via

   ::
     
       python setup.py bdist_wheel

   A wheel is essentially a zip archive which contains the source code
   and the binaries (if any).
6. upload the archive to PyPI using twine via

   ::

       twine upload dist/pypesto-x.y.z-py3-none-any.whl

   replacing x.y.z by the latest version number.

The last step will only be possible if you have sufficient rights.
