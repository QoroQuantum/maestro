---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_gpu_access)=
# Get GPU access

Run Maestro on NVIDIA GPUs: pick one of three ways to get access, then point
`SimulatorConfig` at the GPU.

The pip package simulates on the CPU only. GPU simulation (`SimulatorType.Gpu`,
`DistributedGpu` and `DistributedMpiGpu`) runs through a separate GPU plugin
that needs one of these:

## 1. Run on GPUs in our cloud (Solo)

Sign up at [dash.qoroquantum.net](https://dash.qoroquantum.net) and get 1,000
tokens free, with no credit card. Your Maestro code runs unchanged on our GPUs.
Tokens are spent at rates that depend on the backend and GPU type; the
dashboard lists them.

## 2. Free academic licence

Students, and researchers affiliated with a university or research institute,
can get a free licence to run the GPU plugin on their own machines and
clusters. Apply by emailing [team@qoroquantum.de](mailto:team@qoroquantum.de).

To activate the licence on cluster nodes without internet access, contact us at
[team@qoroquantum.de](mailto:team@qoroquantum.de).

## 3. On-premise for organizations

For a site licence on your own infrastructure, including air-gapped licensing,
see [Maestro Enterprise](https://maestro.qoroquantum.net).

(py_gpu_local)=
## Running GPU locally

Once you have the plugin and a licence, this snippet picks the best backend that
is available and falls back to the CPU otherwise. It runs here on the CPU
because the docs build has no GPU:

```{code-cell} python
import maestro
from maestro.circuits import QuantumCircuit

# is_distributed_gpu_available exists in Linux builds only.
if getattr(maestro, "is_distributed_gpu_available", lambda: False)():
    backend = maestro.SimulatorType.DistributedGpu
elif maestro.is_gpu_available() and maestro.init_gpu():
    backend = maestro.SimulatorType.Gpu
else:
    backend = maestro.SimulatorType.QCSim

bell = QuantumCircuit()
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

result = bell.execute(shots=1000, config=maestro.SimulatorConfig(simulator_type=backend))
print("ran on:", result["simulator"], result["counts"])
print("CUDA devices visible:", maestro.get_gpu_device_count())
```

(py_gpu_avail)=
### Set up the plugin

You need a Linux machine with an NVIDIA GPU. The plugin doesn't run natively on
macOS or Windows; on Windows, use WSL2 with Ubuntu.

| Requirement | Version |
|-------------|---------|
| OS | Linux on x86-64, Ubuntu 22.04 or newer recommended, or WSL2 |
| GPU | NVIDIA, Ampere (A100) or newer recommended; 8 GB of memory at minimum, 40 GB or more for large circuits |
| Driver and CUDA Toolkit | Compatible with CUDA 13.0 or newer; check with `nvidia-smi` |
| cuQuantum | 26.01 |

The easiest way to get cuQuantum is Conda:

```bash
conda create -n maestro_env python=3.11 -y
conda activate maestro_env
conda install -c conda-forge cuquantum openmpi -y
pip install qoro-maestro
```

The GPU release bundle, `maestro-gpu-simulators-linux-x64-v*`, contains the
plugin, `libmaestro_gpu_simulators.so`, and a `setup_gpu.sh` script that
automates these steps. Copy the library next to Conda's other libraries, point
the environment at CUDA and cuQuantum, and set the licence key Qoro gave you:

```bash
cp libmaestro_gpu_simulators.so $CONDA_PREFIX/lib/
export CUQUANTUM_ROOT=$CONDA_PREFIX
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"
```

Add the `export` lines to `~/.bashrc` to keep them across sessions. Maestro
activates the licence when it loads the plugin. The first activation needs an
internet connection, and the licence is then cached on that machine. By default,
it keeps working offline for 30 days before it needs a brief online check.

Licences are bound to a machine and user. On a cluster, every node that runs
Maestro needs its own activation; a shared home directory isn't enough. For
nodes without internet access, see route 2 above.

% TODO(engineering): say where the GPU release bundle is downloaded from. The
% plugin repo's own docs disagree on the oldest supported GPU: its README says
% Pascal (SM 6.0) and its INSTALL guide says Turing (SM 7.5). Confirm which, and
% replace "Ampere or newer recommended" with the minimum.

`maestro.is_gpu_available()` loads the plugin on first use and returns `False`
when it can't be loaded or no GPU is visible.

(py_gpu_usage)=
### Run on the GPU

Select the GPU with `simulator_type`. The circuit code doesn't change:

```python
# Not executed in the docs build: requires GPU access.
gpu_statevector = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.Statevector,
)
print(bell.execute(shots=1000, config=gpu_statevector)["counts"])

gpu_mps = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
print(bell.execute(shots=1000, config=gpu_mps)["counts"])
```

To choose a device, set `gpu_device` on the config, or call
`maestro.select_gpu_device(index)` to change the default for later simulators.
The GPU backend supports statevector, MPS, MPO, tensor-network, Pauli-propagation
and density-matrix simulation; see the {ref}`compatibility table <py_backend_table>`.

## Next steps

- To spread one statevector over several GPUs or nodes, see {doc}`/user-guide/scaling`.
- To tune MPS runs on the GPU, see {ref}`py_mps_gpu_svd`.
