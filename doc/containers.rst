.. _containers:

Containers
==========

We provide a pyPESTO docker image in OCI format through `docker.io/stephanmg/pypesto`.

Develop versions are pushed as tag ``latest``.

Docker image can also be build manually by the users using the Dockerfile: https://github.com/ICB-DCM/pyPESTO/tree/main/docker

If you manually build your container and need to convert to Apptainer, first export your docker image to `.tar` by:
`docker save -o pypesto.tar pypesto:latest` and pull the image before from docker.io by `docker pull docker.io/stephanmg/pypesto:latest`.

Then execute for generating a `.sif` file:
``apptainer build pypesto.sif docker-archive://pypesto.tar``

Likewise for singularity:
``singularity build pypesto.sif docker-archive://pypesto.tar``


Docker
------

To download / pull image: `docker pull docker.io/stephanmg/pypesto`.

To run: `docker run --rm pypesto python3 -c 'import pypesto; print(f"PyPESTO version: {pypesto.__version__}")'`.


Apptainer
---------

Same principle as for Docker, but replace with `apptainer` commands:

To download / pull image: `apptainer pull pypesto.sif docker://docker.io/stephanmg/pypesto:latest`.

To run: `apptainer exec --contain --cleanenv pypesto.sif python3 -c 'import pypesto; print(f"PyPESTO version: {pypesto.__version__}")'`

Note that we need to request a clean and contained environment with `--contain --cleanenv` for apptainer, otherwise some paths like
`/usr/`, `/lib`, `/home`, etc. will be inherited and lead to execution on the host system (e.g. pip packages and other dependencies
will be searched from the host)

Singularity
-----------

Same, but replace `apptainer` with `singularity`.
