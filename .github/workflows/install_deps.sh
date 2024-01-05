#!/bin/sh

# Install CI dependencies, arguments specify what is required

# Base packages
pip install --upgrade pip
pip install wheel setuptools

# Used to create local test environments
pip install tox

# Update package lists
if [ "$(uname)" = "Darwin" ]; then
  # MacOS
  :
else
  # Linux
  sudo apt-get update
fi

# Check arguments
for par in "$@"; do
  case $par in
    doc)
      # documentation
      sudo apt-get install pandoc
    ;;

    amici)
      # for amici
      if [ "$(uname)" = "Darwin" ]; then
        brew install swig hdf5 libomp
      else
        sudo apt-get install \
          clang libatlas-base-dev libhdf5-serial-dev libomp-dev swig
      fi
    ;;

    ipopt)
      # for ipopt
      sudo apt-get install \
	build-essential \
        coinor-libipopt1v5 coinor-libipopt-dev \
        gfortran lcov pkg-config python3-dev zlib1g-dev
    ;;

    pysb)
      # bionetgen
      wget -q -O bionetgen.tar \
        https://github.com/RuleWorld/bionetgen/releases/download/BioNetGen-2.8.5/BioNetGen-2.8.5-linux.tar.gz
      tar -xf bionetgen.tar
    ;;

    mpi)
      # mpi
      sudo apt-get install libopenmpi-dev
    ;;

    *)
      echo "Unknown argument: $par" >&2
      exit 1
    ;;
  esac
done
