import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis import count_spikes


def test_count_spikes_simple() -> None:
    dff = np.array(
        [
            [
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0.5,
                0,
            ],
        ]
    )

    expected_n_spikes = [0, 1]  # Example expected values
    expected_spike_lengths = [[], [1]]  # Example expected lengths

    n_spikes, spike_lengths = count_spikes(dff, 0.3)

    assert n_spikes == expected_n_spikes, "Number of spikes does not match"
    assert all(
        np.array_equal(a, b) for a, b in zip(spike_lengths, expected_spike_lengths)
    ), "Spike lengths do not match"


def test_count_spikes_simple_change_threshold() -> None:
    dff = np.array(
        [
            [
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0.5,
                0,
            ],
        ]
    )

    expected_n_spikes = [0, 0]  # Example expected values
    expected_spike_lengths = [[], []]  # Example expected lengths

    n_spikes, spike_lengths = count_spikes(dff, 100)

    assert n_spikes == expected_n_spikes, "Number of spikes does not match"
    assert all(
        np.array_equal(a, b) for a, b in zip(spike_lengths, expected_spike_lengths)
    ), "Spike lengths do not match"


def test_count_spikes_multiple_spikes() -> None:
    dff = np.array(
        [
            [0, 0.1, 0.5, 0.4, 0.2, 0.6, 0.7, 0.2, 0],
            [0, 0.2, 0.2, 0.8, 0.6, 0.1, 0.4, 0.2, 0],
        ]
    )

    expected_n_spikes = [2, 2]  # Example expected values
    expected_spike_lengths = [[2, 2], [2, 1]]  # Example expected lengths

    n_spikes, spike_lengths = count_spikes(dff, 0.3)

    assert n_spikes == expected_n_spikes, "Number of spikes does not match"
    assert all(
        np.array_equal(a, b) for a, b in zip(spike_lengths, expected_spike_lengths)
    ), "Spike lengths do not match"


def test_count_spikes_spike_does_not_end() -> None:
    dff = np.array(
        [
            [0, 0, 0.5, 0.5, 0.5],
        ]
    )

    expected_n_spikes = [1]  # Example expected values
    expected_spike_lengths = [[3]]  # Example expected lengths

    n_spikes, spike_lengths = count_spikes(dff, 0.3)

    assert len(spike_lengths) == 1
    assert n_spikes == expected_n_spikes, "Number of spikes does not match"
    assert all(
        np.array_equal(a, b) for a, b in zip(spike_lengths, expected_spike_lengths)
    ), "Spike lengths do not match"


def test_count_spikes_spike_multiple_spike_with_non_ending_spikes() -> None:
    dff = np.array(
        [
            [0, 0, 0.5, 0.2, 0.5, 0.5],
            [0, 0, 100, 0, 0, 0],
            [0, 100, 0, 0, 0, 100],
        ]
    )

    expected_n_spikes = [2, 1, 2]  # Example expected values
    expected_spike_lengths = [[1, 2], [1], [1, 1]]  # Example expected lengths

    n_spikes, spike_lengths = count_spikes(dff, 0.3)

    assert len(spike_lengths) == 3
    assert n_spikes == expected_n_spikes, "Number of spikes does not match"
    assert all(
        np.array_equal(a, b) for a, b in zip(spike_lengths, expected_spike_lengths)
    ), "Spike lengths do not match"
