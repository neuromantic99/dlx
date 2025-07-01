from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tifffile

import pandas as pd
from scipy import stats
import seaborn as sns
from dask.distributed import Client

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

import statsmodels.formula.api as smf


HERE = Path(__file__).parent
DOWNSAMPLED_PATH = Path("/Volumes/hard_drive/rikesh_tauko/downsampled_videos")


def load_recording(
    rna: str, well: str | int, experiment: Path
) -> Tuple[np.ndarray, np.ndarray]:
    file_name = f"{rna}_{well}_{experiment.name.strip('.czi')}.npy"
    print(file_name)
    image = load_image(str(experiment))
    assert image is not None
    stack = image.get_image_data().squeeze()
    recording_coord_path = coord_path / file_name
    print(recording_coord_path)
    print("\n")
    cell_coords = np.load(recording_coord_path)
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


def stack_dff_chunks(dff: np.ndarray) -> np.ndarray:
    """shape (n_chunks, n_cells, chunk_size) to shape (n_cells, n_chunks * chunk_size)
    TODO: test this
    """
    transposed = np.transpose(dff, (1, 0, 2))
    return transposed.reshape(dff.shape[1], -1)


def extract_cells_chunked(experiment: Path, cell_coords: np.ndarray) -> np.ndarray:

    image = load_image(str(experiment))
    n_frames = image.shape[0]
    chunk_size = 50
    dff = []

    x = 0
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        print(f"Processing frames {start} to {end}")
        chunk = image.get_image_dask_data(
            "TYX", T=range(start, end), C=0, Z=0
        ).compute()

        dff.append(extract_cells(chunk, cell_coords, do_dff=False))
        del chunk
        x += 1
        if x > 20:
            break

    return stack_dff_chunks(np.array(dff))


def full_downsampled_video(experiment: Path, downsample: int = 2) -> np.ndarray:
    """
    Load a full video and downsample it by a factor of `downsample`.
    """
    image = load_image(str(experiment))
    n_frames = image.shape[0]
    chunk_size = 50
    dff = []
    # x = 0

    for start in range(0, (n_frames // chunk_size) * chunk_size, chunk_size):

        end = min(start + chunk_size, n_frames)
        print(f"Processing frames {start} to {end}")
        chunk = image.get_image_dask_data(
            "TYX", T=range(start, end), C=0, Z=0
        ).compute()
        plt.imshow(np.mean(chunk, 0))
        1 / 0

        dff.append(chunk[:, ::downsample, ::downsample])
        # x += 1
        # if x > 2:
        #     break

        del chunk

    dff = np.array(dff)
    return dff.reshape(-1, dff.shape[2], dff.shape[3])


def main() -> None:

    session = "24-11-21 - Tau KO neurons"
    indicator = "Cal520"
    genotype = "KOLF"
    experiments = list((UMBRELLA / session / indicator / genotype).glob("*.czi"))
    if not experiments:
        raise ValueError(
            f"No experiments found in {UMBRELLA / session / indicator / genotype}"
        )
    for idx, experiment in enumerate(reversed(experiments)):
        if "CarrierOverview" in experiment.name:
            print(f"Skipping {experiment.name}")
            continue

        save_path = (
            DOWNSAMPLED_PATH
            / f"downsampled_video_{session}_{indicator}_{genotype}_{experiment.name.strip('.czi')}.npy"
        )
        # if save_path.exists():
        #     print(f"File {save_path} already exists, skipping.")
        #     continue

        print(experiment)

        video = full_downsampled_video(experiment, 5)
        np.save(
            save_path,
            video,
        )

    # file_name = f'{session}_{indicator}_{genotype}_{experiment.name.strip(".czi")}.npy'

    # cell_coords = np.load(coord_path / file_name)
    # dff = extract_cells_chunked(experiment, cell_coords)
    # for idx, cell in enumerate(dff):
    #     plt.plot(cell)

    # plt.show()

    # 1 / 0


def npy_to_tiff() -> None:

    if not DOWNSAMPLED_PATH.exists():
        raise ValueError(f"Path {DOWNSAMPLED_PATH} does not exist")
    for file in DOWNSAMPLED_PATH.glob("*.npy"):
        array = np.load(file)

        save_path = file.with_suffix(".tif")
        if save_path.exists():
            print(f"File {save_path} already exists, skipping.")
            continue
        tifffile.imwrite(
            save_path,
            array,
            imagej=True,
            photometric="minisblack",
        )


if __name__ == "__main__":
    main()
