"""Validation of Maestro classical measurement histograms."""
import numbers
import numpy as np


def validate_shots(shots):
    if isinstance(shots, bool) or not isinstance(shots, numbers.Integral) or shots <= 0:
        raise ValueError('shots must be a positive integer')
    return int(shots)


def counts_to_measurements(counts, num_measurements, shots, *, storage_width=None):
    """Strict Maestro bit order: character k is classical bit k."""
    if not counts:
        raise ValueError('Maestro returned an empty measurement histogram')
    rows, weights = [], []
    for bits, count in counts.items():
        bits = bits.replace(' ', '')
        if storage_width is not None and len(bits) == storage_width and storage_width > num_measurements:
            if set(bits[num_measurements:]) - {'0'}:
                raise ValueError('Nonzero data outside the declared measurement record')
            bits = bits[:num_measurements]
        if len(bits) != num_measurements or set(bits) - {'0', '1'}:
            raise ValueError(f'Maestro measurement width/content mismatch: expected {num_measurements}')
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
            raise ValueError('Maestro counts must be nonnegative integers')
        rows.append([b == '1' for b in bits])
        weights.append(count)
    if sum(weights) != shots:
        raise ValueError(f'Maestro returned {sum(weights)} shots, expected {shots}')
    return np.repeat(np.asarray(rows, dtype=bool).reshape(len(rows), num_measurements), weights, axis=0)
