FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y \
    cmake \
    g++ \
    git \
    libatlas-base-dev \
    libhdf5-serial-dev \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    swig \
    && ln -sf /usr/bin/swig4.0 /usr/bin/swig


ENV AMICI_CXXFLAGS -fopenmp
ENV AMICI_LDFLAGS -fopenmp

RUN pip3 install -U --upgrade pip wheel \
    && pip3 install python-libsbml>=5.17.0 \
    && pip3 install -v amici[petab,pysb] \
    && python3 -m amici

# RUN pip3 install git+https://github.com/ICB-DCM/pyPESTO.git@develop#egg=pypesto
RUN pip3 install pyPESTO jupyter pyswarm dlib
