from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd
from scipy import stats
import seaborn as sns

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


def test_analysis() -> None:

    redo = True

    if redo:
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
    count_spikes(dff, threshold=threshold)


def process_all_recordings() -> None:

    for rna in ["90", "91", "scrambled"]:
        wells = [
            folder.name
            for folder in list((UMBRELLA / rna).glob("*"))
            if folder.name in {"1", "2", "3", "4", "5"}
        ]
        for well in wells:
            recordings = list((UMBRELLA / rna / well).glob("*.czi"))
            for recording in recordings:

                save_path = HERE / "results" / f"{rna}_{well}_{recording.name}.npy"
                if save_path.exists():
                    print(f"Already done {rna} {well} {recording.name}")
                    continue

                try:
                    stack, cell_coords = load_recording(rna, well, recording)
                except FileNotFoundError:
                    print(f"File not found for {rna} {well} {recording.name}")
                    continue

                dff = extract_cells(stack, cell_coords, do_dff=True)
                n_spikes, spike_lengths = count_spikes(dff, threshold=0.3)
                np.save(
                    save_path,
                    {"n_spikes": n_spikes, "spike_lengths": spike_lengths},
                )


def plot_results() -> None:
    result_files = list(HERE.glob("results/*.npy"))
    to_plot = {"scrambled": [], "NPTX2": []}
    n_scrambled = 0
    n_nptx2 = 0
    for result in result_files:
        data = np.load(result, allow_pickle=True).item()
        n_spikes = data["n_spikes"]
        spike_lengths = [
            np.mean(spike_lengths) for spike_lengths in data["spike_lengths"]
        ]

        if "scrambled" in result.name:
            to_plot["scrambled"].extend(n_spikes)
            n_scrambled += 1
        else:
            to_plot["NPTX2"].extend(n_spikes)
            n_nptx2 += 1

    to_plot["scrambled"] = np.array(to_plot["scrambled"]) / 180
    to_plot["NPTX2"] = np.array(to_plot["NPTX2"]) / 180

    print(f"Number of recordings: {len(result_files)}")
    print(f"Number of recording scrambled: {n_scrambled}")
    print(f"Number of recording NPTX2: {n_nptx2}")
    print(f"Number of cells scrambled: {len(to_plot['scrambled'])}")
    print(f"Number of cells NPTX2: {len(to_plot['NPTX2'])}")

    sns.kdeplot(to_plot, fill=True, common_norm=False)
    plt.xlim(0, None)
    plt.xlabel("Number of transients / second")
    plt.show()


def all_cells_stats(df: pd.DataFrame) -> None:

    assert ((df["subject"] == "scrambled") == (df["condition"] == "scrambled")).all()

    model = smf.mixedlm(
        formula="n_spikes ~ condition",  # Fixed effect
        data=df,
        groups=df["subject"],  # Higher-level grouping factor (subjects)
        re_formula="1",  # Random intercepts
        vc_formula={"well": "0 + C(well)"},  # Well-level random intercepts
    )

    result = model.fit(reml=True)

    sns.stripplot(data=df, x="condition", y="n_spikes", fc="black", s=3, alpha=0.5)
    sns.boxplot(data=df, x="condition", y="n_spikes", showfliers=False)
    plt.title(f"p = {round(result.pvalues['condition[T.scrambled]'], 3)}")
    plt.ylabel("Number of transients")
    plt.show()


def build_all_cells_df() -> pd.DataFrame:
    result_files = list(HERE.glob("results/*.npy"))

    df_dict = {"condition": [], "n_spikes": [], "well": [], "subject": []}

    for result in result_files:
        data = np.load(result, allow_pickle=True).item()
        subject = result.name.split("_")[0]
        assert subject in {"90", "91", "scrambled"}
        well = result.name.split("_")[1]
        assert well in {"1", "2", "3", "4", "5"}
        well = f"{well}_{subject}"
        n_spikes = data["n_spikes"]

        if "scrambled" in result.name:
            condition = "scrambled"
        elif "90" in result.name or "91" in result.name:
            condition = "NPTX2"
        else:
            raise ValueError("Unknown condition")

        df_dict["condition"].extend([condition] * len(n_spikes))
        df_dict["well"].extend([well] * len(n_spikes))
        df_dict["subject"].extend([subject] * len(n_spikes))
        df_dict["n_spikes"].extend(n_spikes)

    df = pd.DataFrame(df_dict)
    return df


def plot_well_means(all_cells_df: pd.DataFrame) -> None:

    meaned_results = {"90": [], "91": [], "scrambled": []}
    for well in all_cells_df["well"].unique():
        meaned = all_cells_df[all_cells_df["well"] == well]["n_spikes"].mean()
        meaned_results[well.split("_")[1]].append(meaned)

    df = pd.DataFrame(
        {
            "dna": ["90"] * 5 + ["91"] * 5 + ["scrambled"] * 5,
            "condition": ["NPTX2"] * 10 + ["scrambled"] * 5,
            "n_spikes": meaned_results["90"]
            + meaned_results["91"]
            + meaned_results["scrambled"],
        }
    )

    sns.boxplot(data=df, x="condition", y="n_spikes")

    sns.stripplot(
        data=df,
        x="condition",
        y="n_spikes",
        hue="dna",
        palette=[
            sns.color_palette()[2],
            sns.color_palette()[3],
            sns.color_palette()[4],
        ],
    )

    # Fit Linear Mixed Model with dna as random effect
    model = smf.mixedlm("n_spikes ~ condition", df, groups=df["dna"])
    result = model.fit()
    plt.ylabel("Mean number of transients in well")
    print(result.summary())
    plt.title(f"p = {round(result.pvalues["condition[T.scrambled]"], 3)}")
    plt.show()


if __name__ == "__main__":

    # process_all_recordings()
    df = build_all_cells_df()
    # all_cells_stats(df)
    plot_well_means(df)
