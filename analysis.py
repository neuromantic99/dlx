from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


from utils import (
    load_image,
    select_elipse_from_stack,
    threshold_detect,
    threshold_detect_falling_edge,
    compute_dff_single_cell,
    threshold_detect,
    threshold_detect_falling_edge,
)
from consts import UMBRELLA, coord_path


def load_recording(
    rna: str, well: str | int, experiment: Path
) -> Tuple[np.ndarray, np.ndarray]:
    file_name = f"{rna}_{well}_{experiment.name.strip('.czi')}.npy"
    image = load_image(str(experiment))
    assert image is not None
    stack = image.get_image_data().squeeze()
    file_name = f"{rna}_{well}_{experiment.name.strip('.czi')}.npy"
    cell_coords = np.load(coord_path / file_name)
    return stack, cell_coords


def extract_cells(
    stack: np.ndarray, cell_coords: np.ndarray, do_dff: bool
) -> np.ndarray:

    dff = []
    for cell_coord in cell_coords:
        cell_stack = select_elipse_from_stack(cell_coord, stack)
        trace = np.nanmean(cell_stack, axis=(1, 2))
        if do_dff:
            dff.append(compute_dff_single_cell(trace))
        else:
            dff.append(trace)
    return np.array(dff)


def show_rois(stack: np.ndarray, cell_coords: np.ndarray) -> None:
    vmin = 0
    vmax = np.max(stack) // 10
    plt.figure()
    plt.imshow(np.mean(stack, 0), cmap="gray", vmin=vmin, vmax=vmax)
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(alpha=0)

    plt.figure()
    plt.imshow(np.mean(stack, 0), cmap="gray", vmin=vmin, vmax=vmax)
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(alpha=0)

    for cell_coord in cell_coords:
        cell_stack = np.nanmean(select_elipse_from_stack(cell_coord, stack), 0)
        cell_stack[~np.isnan(cell_stack)] = vmax
        plt.imshow(cell_stack, alpha=0.5, cmap="gray", vmin=vmin, vmax=vmax)

    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(alpha=0)
    plt.colorbar()
    plt.show()


def count_spikes(
    dff: np.ndarray, threshold: float
) -> Tuple[List[int], List[np.ndarray]]:
    n_spikes = []
    spike_lengths: List[np.ndarray] = []
    for trace in dff:
        spike_onsets = threshold_detect(trace, threshold)
        spike_offsets = threshold_detect_falling_edge(trace, threshold)

        n_spikes.append(len(spike_onsets))
        if len(spike_onsets) == 0:
            spike_lengths.append(np.array([]))
            continue

        if len(spike_onsets) == len(spike_offsets):
            spike_lengths.append(spike_offsets - spike_onsets)
        elif len(spike_onsets) - 1 == len(spike_offsets):
            spike_lengths_cell = spike_offsets - spike_onsets[:-1]
            spike_lengths_cell = np.append(
                spike_lengths_cell, len(trace) - spike_onsets[-1]
            )
            spike_lengths.append(spike_lengths_cell)

    return n_spikes, spike_lengths


def plot_dff(dff: np.ndarray, threshold: float) -> None:
    plt.figure()
    x = np.linspace(0, 180, dff.shape[1])
    for idx in range(dff.shape[0]):
        plt.figure()
        trace = dff[idx]
        spike_onsets = threshold_detect(trace, threshold)
        spike_offsets = threshold_detect_falling_edge(trace, threshold)
        plt.plot(dff[idx], "r")
        plt.plot(
            spike_onsets, np.ones(len(spike_onsets)) * threshold, ".", color="green"
        )
        plt.plot(
            spike_offsets, np.ones(len(spike_offsets)) * threshold, ".", color="blue"
        )

    plt.show()


if __name__ == "__main__":

    # experiment = Path("/Users/jamesrowland/Code/dlx/Experiment-1401.czi")
    HERE = Path(__file__).parent
    test_data = True

    if not test_data:
        rna = "91"
        assert rna in {"90", "91", "scrambled"}
        well = 5
        experiments = list((UMBRELLA / rna / str(well)).glob("*.czi"))
        experiment = experiments[3]
        stack, cell_coords = load_recording(rna, well, experiment)
        # show_rois(stack, cell_coords)
        dff = extract_cells(stack, cell_coords, do_dff=True)
        trace = extract_cells(stack, cell_coords, do_dff=False)
        np.save(HERE / "dff.npy", dff)
    else:
        dff = np.load(HERE / "dff.npy")

    threshold = 0.3
    plot_dff(dff, threshold=threshold)
    # count_spikes(dff, threshold=threshold)
