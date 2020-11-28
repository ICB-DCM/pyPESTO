#!/bin/sh

# Base packages that don't hurt
pip install wheel cython numpy

for par in "$@"; do
  case $par in
    docs)
      # documentation
      sudo apt-get install pandoc
      pip install -r .rtd_pip_reqs.txt
    ;;

    petab)
      # for amici
      sudo apt-get install \
        swig3.0 libatlas-base-dev libhdf5-serial-dev
      sudo ln -s /usr/bin/Swig3.0 /usr/bin/swig

      # amici dev
      pip install \
        git+https://github.com/amici-dev/amici.git@develop#egg=amici\&subdirectory=python/sdist

      # petab dev
      pip install https://github.com/petab-dev/petab/archive/develop.zip

      # petabtests
      pip install petabtests>=0.0.0a6
    ;;

    ipopt)
      # for ipopt
      sudo apt-get install \
	build-essential \
        coinor-libipopt1v5 coinor-libipopt-dev \
        gfortran lcov pkg-config python-dev zlib1g-dev
      pip install six
    ;;

    pysb)
      # pysb
      pip install \
        git+https://github.com/pysb/pysb.git@c434f6ab98301beee1bf9d2a5093f0c79da78824#egg=pysb

      # bionetgen
      wget -q -O bionetgen.tar.gz \
        https://bintray.com/jczech/bionetgen/download_file?file_path=BioNetGen-2.3.2-linux.tar.gz
      tar -xzf bionetgen.tar.gz
    ;;

    *)
      echo "Unknown argument" >&2
      exit 1
    ;;
  esac
done
