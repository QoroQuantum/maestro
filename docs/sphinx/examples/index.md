# Examples

Each example is a notebook that runs when the docs are built, so the code and
its output always match the current release. Open one to read it, or copy the
cells into your own notebook.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} One circuit, three simulation methods
:link: ghz-across-methods
:link-type: doc

Run a GHZ state on a statevector, an MPS and a stabilizer simulator, from 20 to
1,000 qubits.
:::

:::{grid-item-card} Analytical and Monte Carlo noise
:link: noise-modes
:link-type: doc

Sweep a depolarizing rate and see why the two noise modes disagree on deeper
circuits.
:::
::::

:::{toctree}
:hidden:

ghz-across-methods
noise-modes
:::
