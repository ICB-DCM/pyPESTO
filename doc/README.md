# Documentation

The `doc/` folder contains the files for building the pyPESTO documentation.


## Requirements

The documentation is based on sphinx. Install via

```
pip3 install sphinx
```

Furthermore, the files specified in `../.rtd_pip_reqs.txt` and `../.rtd_apt_reqs.txt` are required. Install via

```
pip3 install --upgrade -r ../.rtd_pip_reqs.txt
```

and

```
cat ../.rtd_apt_reqs.txt | xargs sudo apt install -y
```

respectively.


## Build the documentation

The documentation can be built in different formats, e.g. in html via

```
make html
```

The built documentation can then be found locally in the `_build`
sub-directory.

The documentation is built and published automatically on readthedocs.io
every time the master branch on github.com is changed. It is recommended
to compile and check the documentation manually beforehand.
