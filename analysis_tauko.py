from pathlib import Path
import numpy as np
from consts import UMBRELLA
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main() -> None:

    session = "24-11-20 - Tau KO neurons"
    indicator = "Cal520"
    genotype = "KOLF"
    experiments = UMBRELLA / session / indicator / genotype
    folders = [f for f in experiments.iterdir() if f.is_dir()]
    for folder in folders:
        if folder.name != "6.2":
            continue
        if "CarrierOverview" in folder.name:
            print(f"Skipping {folder.name}")
            continue

        # stat = np.load(s2p / "stat.npy", allow_pickle=True)
        # skewness = [s["skew"] for s in stat]
        # skewness = np.array(skewness)

        s2p = folder / "suite2p" / "plane0"
        f = np.load(s2p / "F.npy")
        spks = np.load(s2p / "cascade_results.npy")
        dff = np.load(s2p / "cascade_preprocessed.npy")

        # spks[spks < 0.2] = 0
        sorted_order = np.argsort(np.nansum(spks, 1))

        for active in [False, True]:

            id = 0
            example_cells_idx = sorted_order[-4:] if active else sorted_order[0:4]
            # for idx in sorted_order[-4:]:
            for idx in example_cells_idx:
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
                ax2 = ax1.twinx()

                ax1.plot(dff[idx, :], color="red")
                ax1.set_ylabel("dF/F", color="red")

                ax2.plot(spks[idx, :], color="blue")
                ax2.set_ylabel("Spikes", color="blue")
                if np.nanmax(spks[idx, :]) < 1:
                    ax2.set_ylim(None, 1)

                t = f"{"Not " if not active else ""}Active neuron {id}. Kolf. Recording {folder.name}"
                plt.xlabel("Time (frames)")
                plt.title(t)
                plt.tight_layout()
                plt.savefig(HERE.parent / "figures" / "tau_ko_examples" / (t + ".png"))

                id += 1

        plt.show()


if __name__ == "__main__":
    main()
