# GPU library licensing

Licensed builds validate at library loading, initialization, and public simulator
creation/clone entry points. Simulator objects do not carry license state. Gates,
observations, state allocation/reset and cleanup do not consult licensing.
Shared libraries export only the C API declared in their public headers.

## Implementation plan

1. Validate each plugin when loading it. Keep development/test builds and the
   existing deployment configuration unchanged.
2. Have `InitLib()` verify admission and cached activation before creating its
   context. Return `nullptr` on rejection, with the actual licensing diagnostic.
3. Remove the simulator handle registry, distributed per-state license flag, and
   operation guards. Validate cached activation in every public simulator factory
   and clone, returning `nullptr` with a licensing diagnostic on rejection. Compile
   these checks out when licensing is disabled.
4. Make Maestro discovery and factories treat a rejected plugin as unavailable,
   following the same selection/fallback path as a missing plugin. Preserve
   the licensing reason in logs and explicit loader errors.
5. Restrict shared-library exports to an allowlist derived from the public API
   headers, including the MPI API only in the MPI variant.
6. Test loading/initialization failures, repeated discovery, creation/clone
   rejection after expiry, uninterrupted existing objects, MPI rank agreement,
   and complete export sets with licensing enabled and disabled.

## Runtime behavior

Maestro validates `MAESTRO_LICENSE_KEY` once when loading a local plugin, or
passes null to validate cached activation. Library initialization runs once;
its second check uses the cache and does not reactivate a key. A failed load or
initialization stays unavailable for the process lifetime, avoiding repeated
activation requests during fallback probes. Correct deployment settings before
restarting Maestro.

The simulator plugin's library initialization creates its host context; GPU
resources remain lazy until device selection/use. Discovery returns zero for a
rejected library, acquisition returns null, and both shared/unique simulator
factories return null. The distributed plugin likewise validates and initializes
before discovery reports availability or its factories return a simulator.

MPI loading resolves symbols first because validation requires the application's
communicator. At the first collective library initialization Maestro validates
all ranks and calls `InitLib` once. Each rank enters the collective state factory,
even if its library context is null, so invalid initialization handles are
rejected before vendor collectives. Subsequent states reuse the context, with
cached license validation inside the factory. MPI clones validate collectively
on the source state's communicator before duplication. One rank's rejection
returns `nullptr` on every rank. Explicit MPI requests retain their unavailable
backend error behavior and collective ordering requirements.

An existing simulator remains usable after later expiry. New simulators and
clones require a currently valid cached license, even with a previously valid
library handle. This includes the regular library's stabilizer and Pauli
propagator, whose factories do not take a library handle. There is no mid-job
revocation of existing objects. The C export `CheckLicense(void*)` remains only
for older binaries:
it returns 1 for a non-null handle, 0 for null, without licensing or MPI work.
Maestro no longer resolves it or exposes a simulator checkpoint method.

The backend `LICENSING.md` files describe build settings, cache/offline activation,
and deployment options. These mechanisms are unchanged.

## Regression coverage

`gpu_licensing_tests` covers successful, load-rejected, and init-rejected plugins
for the regular, distributed, and MPI loaders without GPU hardware. It checks
factory availability, diagnostics, and that repeated discovery/construction
does not repeat license validation or library initialization.

The backend licensed-path tests use a deterministic SDK substitute to check
invalid activation, cached initialization failure, recovery, rejection of new
objects/clones after expiry, and execution of existing objects after rejection.
SDK counters must remain unchanged during gates, observations and cleanup.
The regular plugin covers all seven simulator families; the distributed plugin
covers conventional and Ex local backends, placement policies, and MPI admission.
MPI factory rejection tests cover one invalid rank through both native and runtime
entry points without requiring CUDA-aware MPI. Production export tests compare
the entire defined dynamic symbol set with the declarations in the public headers.

Validation on 2026-09-21 (RTX 5090, one GPU):

- Simulator C API: 115 passed. Licensing/threaded/export/service CTest checks:
  10 passed. Maestro's registry also passed against the rebuilt real plugin.
- Maestro loader licensing checks: 9 passed. Both local distributed integration
  cases passed against the rebuilt plugin.
- Distributed licensing/local API/export/MPI admission/ABI checks: 17 passed,
  1 skipped. Full licensed MPI execution and clone coverage was skipped because
  the installed MPI is not CUDA-aware. Multi-GPU execution was not exercised.
- All three production shared libraries also built with licensing ON and the
  real SDK. All 6 loading/export checks passed. With licensing ON or OFF, exports
  are exactly 455 regular, 89 distributed local, and 97 distributed MPI API names.
  Invalid test product metadata was rejected by the real SDK; all seven regular
  factories returned null before GPU access.

The local MPI launcher needed `HWLOC_COMPONENTS=linux,stop` for these tests:
optional hardware discovery otherwise stalled on a display connection before
starting ranks. This setting was used only for test commands, with no system
configuration changes. Accepted/expired/revoked licenses were exercised through
the deterministic SDK substitute; real-SDK rejection used deliberately invalid
test product metadata. No real customer activation was performed.
