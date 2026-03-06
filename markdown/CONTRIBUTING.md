# Contributing to Maestro

Thank you for your interest in Maestro.

At this stage, Maestro is distributed under a dual-licensing model (GPL-3.0 for open use, and a separate commercial license). To keep the copyright and licensing model simple while the project is still evolving, we currently do not accept direct contributions into this repository.

That means:

- No pull requests with code or documentation changes will be merged.
- No external maintainers or collaborators are added to the repo at this time.

However, we very much welcome your feedback and ideas.

## How you can contribute right now

Even though we do not accept direct code contributions, you can still help Maestro improve in the following ways:

### 1. Feature requests

If you have ideas for new features, integrations, or performance improvements:

- Open a GitHub issue with the type `[Feature Request]`.
- Clearly describe:
  - The use case or problem you are trying to solve.
  - The type of simulator or workload (statevector, MPS, tensor network, DQC, etc.).
  - Any constraints (HPC environment, GPU availability, runtime limits, etc.).
- If helpful, include pseudocode or high-level API sketches.

We will triage and prioritise feature requests based on usefulness, complexity, and alignment with our roadmap.

### 2. Bug reports

If you find a bug:

- Open a GitHub issue with the type `[Bug]`.
- Include:
  - Maestro version and commit (if known).
  - Platform and compiler (for example: Ubuntu 22.04, GCC 12, CUDA 12.3).
  - Exact steps to reproduce the issue.
  - Minimal example circuit or configuration, if possible.
  - Expected behaviour versus what actually happens.
  - Relevant logs, error messages, or stack traces.

Minimal, reproducible examples are extremely valuable and help us fix issues faster.

### 3. Performance feedback

Maestro is heavily focused on performance and scaling in HPC and hybrid QC settings.

If you run benchmarks or experiments:

- Share your observations in an issue or discussion:
  - Circuits or workloads used.
  - Hardware configuration (CPUs, GPUs, memory).
  - Backends used (statevector, MPS, GPU, p-block, etc.).
  - Any comparisons to other simulators.
- If you can share plots, tables, or aggregated metrics (without revealing confidential data), that is very helpful.

---

## About code snippets in issues

Because we are not accepting direct pull requests, we sometimes implement features or fixes internally based on ideas or snippets shared by users.

By posting code snippets, pseudocode, or algorithms in issues or discussions, you agree that:

- You have the right to share that material.
- Qoro Quantum may freely use, modify, adapt, and redistribute that material as part of Maestro under its existing licenses (including GPL-3.0 and any commercial licenses we may offer), without any additional obligations or attribution requirements beyond what the GPL-3.0 license itself mandates.

If you are not comfortable with this, please avoid posting non-trivial code and instead describe behaviour at a higher level (for example, expected API shape, performance goals, or algorithmic requirements).

---

## Why we do not accept pull requests (for now)

Maestro is part of a broader software stack and business model that uses a dual-licensing approach. To avoid:

- complex copyright ownership questions,
- the need for contributor license agreements (CLAs), and
- potential conflicts between open-source and commercial licenses,

we currently keep all copyright in the codebase with Qoro Quantum and its employees/contractors.

Once the licensing and commercial structure is more mature, we may revisit the contribution model and allow direct contributions under a clear contributor agreement.

---

## Contact

If you would like to:

- discuss commercial licensing,
- explore collaborations or pilots,
- or talk about integrating Maestro into an HPC or quantum stack,

please reach out to Qoro Quantum directly via the contact details provided in the main README or on our website.

Thank you again for your interest in Maestro.
