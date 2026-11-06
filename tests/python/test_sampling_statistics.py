"""The statistical comparison must not replay native frame batches."""

import importlib.util
from pathlib import Path


def _native_batch_seed(seed, batch):
    """Utils::RandomStream::Derive uses this SplitMix64 transform."""
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (batch + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def test_statistical_call_seeds_do_not_alias_native_frame_batches():
    script = Path(__file__).resolve().parents[1] / "sampling_statistics.py"
    spec = importlib.util.spec_from_file_location("_maestro_sampling_statistics_seed_schedule", script)
    statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistics)

    # The default 400k-shot comparison has twenty calls, each containing twenty
    # native 1024-shot batches. Native Derive mixes the call seed plus the same
    # golden-ratio stride; an unmixed outer stride aliases along outer+inner.
    base = 12345
    identities = {_native_batch_seed(statistics.call_seed(base, outer), inner)
                  for outer in range(20) for inner in range(20)}
    assert len(identities) == 400

    # Demonstrate that this grid exposes the old schedule's exact collisions.
    mask = (1 << 64) - 1
    old_identities = {_native_batch_seed((base + 0x9E3779B97F4A7C15 * (outer + 1)) & mask, inner)
                      for outer in range(20) for inner in range(20)}
    assert len(old_identities) == 39
