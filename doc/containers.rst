.. _containers:

Containers
==========

We provide a pyPESTO docker image in OCI format through the Docker.io registry `docker.io/stephanmg/pypesto:latest`.

A docker image build is triggered on changes/commits in the `develop` branch of pyPESTO. The container is build and pushed via a GHA to the Docker.io registry with tag ``latest``.

The image can be used by using a pull from Docker.io registry: `docker pull docker.io/stephanmg/pypesto:latest`

Docker image can also be build manually by the users using the Dockerfile found here: https://github.com/ICB-DCM/pyPESTO/tree/main/docker

If you require to transfer the docker image manually, build the container using the Dockerfile, then export your docker image to `.tar` by:
`docker save -o pypesto.tar pypesto:latest`. The `.tar` file can then be transfered (e.g. with scp) to a remote location.


Run Docker image
----------------

To download / pull image: `docker pull docker.io/stephanmg/pypesto`.

To run: `docker run --rm pypesto python3 -c 'import pypesto; print(f"PyPESTO version: {pypesto.__version__}")'`.

Similar instructions for the Docker drop-in replacements Podman and Singularity/Apptainer.
