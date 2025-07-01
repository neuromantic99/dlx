from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import load_image


path = Path(
    "/Volumes/hard_drive/rikesh_tauko/downsampled_videos/downsampled_video_24-11-20 - Tau KO neurons_Cal520_KOLF_4.1.npy"
)
stack = np.load(path)

# experiment = Path(
#     "/Volumes/MarcBusche/Rikesh/CD7/24-11-20 - Tau KO neurons/Cal520/KOLF/4.1.czi"
# )
# stack = load_image(str(experiment))


frame_mean = np.mean(stack, (1, 2))

normalized_stack = stack - frame_mean[:, np.newaxis, np.newaxis]


std_projection = np.std(normalized_stack, 0)


std_projection[std_projection < np.percentile(std_projection, 99)] = 0
# Normalise each frame by the mean to remove drift

plt.imshow(std_projection)

1 / 0
