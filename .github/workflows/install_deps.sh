#!/bin/sh

# Install CI dependencies, arguments specify what is required

# Base packages
pip install --upgrade pip
pip install wheel setuptools

# Needed for some dependencies
pip install cython

# Used to create local test environments
pip install cython tox


for par in "$@"; do
  case $par in
    doc)
      # documentation
      sudo apt-get install pandoc
      pip install -r .rtd_pip_reqs.txt
    ;;

    petab)
      # for amici
      sudo apt-get install \
        swig3.0 libatlas-base-dev libhdf5-serial-dev
      if [ ! -e /usr/bin/swig ]; then
        sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
      fi
    ;;

    ipopt)
      # for ipopt
      sudo apt-get install \
	build-essential \
        coinor-libipopt1v5 coinor-libipopt-dev \
        gfortran lcov pkg-config python-dev zlib1g-dev
      # ipopt does stuff during pip install otherwise
      pip install numpy six
    ;;

    pysb)
      # pysb
      pip install --force-reinstall \
        git+https://github.com/pysb/pysb.git@c434f6ab98301beee1bf9d2a5093f0c79da78824#egg=pysb

      # bionetgen
      wget -q -O bionetgen.tar.gz \
        https://bintray.com/jczech/bionetgen/download_file?file_path=BioNetGen-2.3.2-linux.tar.gz
      tar -xzf bionetgen.tar.gz
    ;;
    
    mpi)
      # mpi
      sudo apt install libopenmpi-dev
    ;;

    *)
      echo "Unknown argument" >&2
      exit 1
    ;;
  esac
done
