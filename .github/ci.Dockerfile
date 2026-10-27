# Toolchain and prebuilt dependencies for the Linux CI job in cmake-multi-platform.yml.
FROM ubuntu:26.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake curl git ninja-build zstd \
        libopenblas-dev libopenmpi-dev openmpi-bin \
        python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

ARG BOOST_VERSION=1.90.0
RUN curl -fsSL "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION//./_}.tar.gz" -o /tmp/boost.tgz \
    && tar -xzf /tmp/boost.tgz -C /tmp \
    && cd /tmp/boost_${BOOST_VERSION//./_} \
    && ./bootstrap.sh --prefix=/opt/boost \
    && ./b2 -j"$(nproc)" --with-program_options --with-url --with-container --with-json --with-serialization install \
    && rm -rf /tmp/boost.tgz /tmp/boost_*

ARG EIGEN_VERSION=5.0.0
RUN curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" \
    | tar -xz -C /opt

RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir pytest numpy

ENV PATH=/opt/venv/bin:$PATH \
    BOOST_ROOT=/opt/boost \
    EIGEN5_INCLUDE_DIR=/opt/eigen-${EIGEN_VERSION}/ \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
