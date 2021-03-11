# AMICI & pyPESTO with Docker

## Create image

In the AMICI base directory run:

```bash
git archive -o <path to pypesto base directory>/docker/amici.tar.gz --format=tar.gz HEAD
cd <path to pypesto base directory>/docker && docker build -t $USER/amici_pypesto:latest .
```

To install pyPESTO from a particular branch, e.g. develop, use th following
line in the Dockerfile

```
RUN pip3 install git+https://github.com/ICB-DCM/pyPESTO.git@develop#egg=pypesto
```

environment file can be used with `--set-env` option of `ch-run` command. From
charliecloud documentation:

"
The purpose of `--set-env=FILE` is to set environment variables that cannot be
inherited from the host shell, e.g. Dockerfile ENV directives or other
build-time configuration
"
