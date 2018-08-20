# pyPESTO Documentation

This folder contains files for building the documentation of this package.


## Requirements

The documentation is based on sphinx. Install via

```
pip3 install sphinx
```

Furthermore, the files specified in `../.rst_pip_reqs.txt` and `../.rst_apt_reqs.txt` are required. Install via

```
pip3 install --update -r ../.rst_pip_reqs.txt
```

and

```
cat ../.rst_apt_reqs.txt | xargs sudo apt install
```

respectively.


## Buid the documentation

The documentation can be built in different formats, e.g. in html (to be then found in the `_build` sub-directory) via

```
make html
```
