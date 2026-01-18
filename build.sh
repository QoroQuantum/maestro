#!/bin/bash
set -e

if [ ! -d build ]
then
    mkdir -p build
fi
cd build

if [ ! -d eigen ]
then
    curl https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz --output eigen.tgz
    tar -xzvf eigen.tgz
    rm eigen.tgz
fi

if [ ! -d boost_1_89_0 ]
then
    curl https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz --output boost.tgz
    tar -xzvf boost.tgz
    rm boost.tgz
fi
# Skip building Boost if it's already present to save time.
if [ ! -d boost_1_89_0/lib ]
then
    cd boost_1_89_0
    ./bootstrap.sh --prefix=.
    # Use -j2 to limit parallel jobs, preventing memory exhaustion (OOM) 
    # during intensive Boost compilation.
    ./b2 -j2 --with-program_options --with-json --with-container --with-serialization --prefix=. install
    cd ..
fi

if [ ! -d QCSim ]
then
     git clone https://github.com/aromanro/QCSim.git
fi

if [ -z "${NO_QISKIT_AER}" ]; then
	if [ ! -d json ]
	then
		git clone https://github.com/nlohmann/json.git
	fi


	if [ ! -d qiskit-aer ]
	then
		git clone https://github.com/InvictusWingsSRL/qiskit-aer.git
	fi
fi

export EIGEN5_INCLUDE_DIR=$PWD/eigen-5.0.0
export BOOST_ROOT=$PWD/boost_1_89_0
export QCSIM_INCLUDE_DIR=$PWD/QCSim/QCSim
export JSON_INCLUDE_DIR=$PWD/json/single_include
export AER_INCLUDE_DIR=$PWD/qiskit-aer/src

cmake ..
make
make doc
