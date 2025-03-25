We provide a pyPESTO OCI image through the Docker.io registry.

A docker image build is triggered on changes/commits in the ``develop`` branch of pyPESTO. The container is built and pushed via a GHA to the Docker.io registry with the tag corresponding to ``latest``.

The image can be obtained by a pull from the Docker.io registry: ``docker pull docker.io/stephanmg/pypesto:latest``

The Docker image can also be built locally using the Dockerfile found here: https://github.com/ICB-DCM/pyPESTO/tree/main/docker

If you require to transfer the docker image manually, build the container using the Dockerfile, then export your docker image to ``.tar`` by:
``docker save -o pypesto.tar pypesto:latest``. The ``.tar`` file can then be transfered (e.g. with scp) to a remote location.

To run the pyPESTO container and print the current pyPESTO version: ``docker run --rm pypesto python3 -c 'import pypesto; print(f"pyPESTO version: {pypesto.__version__}")'``.
