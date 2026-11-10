"""Exact Windows seeded histograms from an independently built 0.3.13 source.

This is a source-compatibility fixture using the recorded local dependencies,
not a claim about an independently downloaded published wheel. Other standard
libraries may implement random distributions differently, so it is Windows-only.
"""

import json
from pathlib import Path
import sys

import pytest

import maestro


FIXTURE_PATH = Path(__file__).with_name("data") / "sampling_counts_windows_0_3_13.json"
REFERENCE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Exact MSVC standard-library seeded fixture")


def _circuit(spec):
    if "qasm" in spec:
        return maestro.QasmToCirc().parse_and_translate(spec["qasm"])
    circuit = maestro.circuits.QuantumCircuit()
    for name, arguments in spec["operations"]:
        getattr(circuit, name)(*arguments)
    return circuit


def _run_case(case, policy):
    config_values = dict(case["config"])
    config_values["simulation_type"] = getattr(maestro.SimulationType, config_values["simulation_type"])
    config_values["counts_policy"] = policy
    config = maestro.SimulatorConfig(**config_values)
    circuit = _circuit(case["circuit"])
    noise = maestro.NoiseModel()
    for name, arguments in case.get("noise", []):
        getattr(noise, name)(*arguments)
    arguments = dict(case["arguments"])
    if case["api"] == "execute":
        return circuit.execute(config=config, **arguments)
    if case["api"] == "execute_suffix":
        checkpoint = maestro.PrefixCheckpointedSimulator(_circuit(case["prefix"]), case["num_qubits"], config)
        return checkpoint.execute_suffix(circuit, noise_model=noise, **arguments)
    return getattr(circuit, case["api"])(noise, config=config, **arguments)


def test_seeded_fixture_records_independent_source_provenance():
    provenance = REFERENCE["provenance"]
    assert provenance["commit"] == "1abd852ddc69d0f46e4a80d330c664bfa9e716af"
    assert provenance["version"] == "0.3.13"
    assert provenance["source_kind"] == "git archive HEAD; fresh isolated native build"
    assert "not a published wheel" in provenance["scope"]
    for key in ["archive_sha256", "source_tree_sha256", "case_specifications_sha256"]:
        assert len(provenance[key]) == 64
    assert len(provenance["dependencies"]["qcsim"]["headers_sha256"]) == 64
    assert len(REFERENCE["cases"]) >= 10


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["id"])
def test_legacy_counts_match_prechange_source_fixture(case):
    result = _run_case(case, "legacy")
    assert "samples" not in result
    assert result["counts"] == case["counts"]
    assert sum(result["counts"].values()) == case["arguments"]["shots"]
    assert result.get("sampling_strategy") != "reference_frame_v1"


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["id"])
def test_auto_replays_same_version_counts_for_fixture_inputs(case):
    counts = _run_case(case, "auto")["counts"]
    assert counts == _run_case(case, "auto")["counts"]
    assert sum(counts.values()) == case["arguments"]["shots"]
