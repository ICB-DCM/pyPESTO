Deploy
======

New features and bug fixes are continuously added to the develop branch. On
every merge to master, the version number in ``pypesto/version.py``  should
be incremented as described below.

Versioning scheme
-----------------

For version numbers, we use ``A.B.C``, where

* ``C`` is increased for bug fixes,
* ``B`` is increased for new features and minor API breaking changes,
* ``A`` is increased for major API breaking changes.


Creating a new release
----------------------

After new commits have been added to the develop branch, changes can be merged
to master and a new version of pyPESTO can be released. Every merge to master
should coincide with an incremented version number and a git tag on the
respective merge commit.


Merge into master
~~~~~~~~~~~~~~~~~

1. create a pull request from develop to master
2. check that all tests on travis pass
3. check that the documentation is up-to-date
4. adapt the version number in the file ``pesto/version.py`` (see above)
5. update the release notes in ``doc/releasenotes.rst``
6. request a code review
7. merge into the origin master branch

To be able to actually perform the merge, sufficient rights may be
required. Also, at least one review is required.


Creating a release on github
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After merging into master, create a new release on Github.
In the release form:

* specify a tag with the new version as specified in ``pesto/version.py``,
  prefixed with ``v`` (e.g. ``v0.0.1``)
* include the latest additions to ``doc/releasenotes.rst`` in the release
  description

Tagging the release commit will automatically trigger deployment of the new
version to pypi.
