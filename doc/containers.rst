We provide a ``pyPESTO OCI image <https://hub.docker.com/r/stephanmg/pypesto>``__ through the Docker.io registry ``docker.io/stephanmg/pypesto:latest``.

A docker image build is triggered on changes/commits in the ``develop`` branch of pyPESTO. The container is build and pushed via a GHA to the Docker.io registry with tag ``latest``.

The image can be used by using a pull from Docker.io registry: ``docker pull docker.io/stephanmg/pypesto:latest``

The Docker image can also be built locally using the Dockerfile found here: https://github.com/ICB-DCM/pyPESTO/tree/main/docker

If you require to transfer the docker image manually, build the container using the Dockerfile, then export your docker image to ``.tar`` by:
``docker save -o pypesto.tar pypesto:latest``. The ``.tar`` file can then be transfered (e.g. with scp) to a remote location.

To download / pull image, use: ``docker pull docker.io/stephanmg/pypesto``.

To run container: ``docker run --rm pypesto python3 -c 'import pypesto; print(f"PyPESTO version: {pypesto.__version__}")'``.
