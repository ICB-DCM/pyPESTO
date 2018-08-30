# pyPESTO Documentation

This folder contains files for building the documentation of this package.


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

The documentation can be built in different formats, e.g. in html (to be then found in the `_build` sub-directory) via

```
make html
```
