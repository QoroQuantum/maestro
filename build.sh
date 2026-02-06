#!/bin/bash
set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
    NPROC=$(sysctl -n hw.logicalcpu)
else
    if command -v nproc > /dev/null; then
        NPROC=$(nproc)
    else
        NPROC=1
    fi
fi
echo "Building with ${NPROC} parallel jobs."

check_lblas() {
    printf "int main() { return 0; }" | cc -x c - -lblas -o /dev/null 2>/dev/null
}

install_blas() {
    echo "BLAS not found. Attempting to install..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null; then
            echo "Installing openblas via Homebrew..."
            brew install openblas
            export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig:$PKG_CONFIG_PATH"
        else
            echo "Homebrew not found. Please install BLAS manually."
            return 1
        fi
    elif [ -f /etc/debian_version ]; then
        echo "Installing libopenblas-dev via apt..."
        sudo apt-get update && sudo apt-get install -y libopenblas-dev
    elif [ -f /etc/redhat-release ]; then
        echo "Installing openblas-devel via dnf..."
        sudo dnf install -y openblas-devel
    else
        echo "Could not detect package manager. Please install BLAS manually."
        return 1
    fi
}

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
cd boost_1_89_0
./bootstrap.sh --prefix=.
./b2 -j${NPROC} --with-program_options --with-json --with-container --with-serialization --prefix=. install
cd ..

if [ ! -d QCSim ]
then
     git clone https://github.com/aromanro/QCSim.git
fi

if [ -z "${NO_QISKIT_AER}" ]; then
    if ! check_lblas; then
        install_blas
    fi

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
make -j${NPROC}
make doc
