import aicsimageio
import xml.etree.ElementTree as ET
import numpy as np
import xmltodict


def get_image_metadata_czi(AICSImage_object: aicsimageio.aics_image.AICSImage) -> dict:
    metadata = AICSImage_object.metadata
    metadata_string = ET.tostring(element=metadata, encoding="unicode")
    return xmltodict.parse(metadata_string)


def load_image(image_directory_path: str) -> aicsimageio.aics_image.AICSImage:
    """Return AICSImage object"""
    # Selects the first scene found
    return aicsimageio.AICSImage(image_directory_path)


def select_elipse_from_stack(shape_data: np.ndarray, stack: np.ndarray) -> np.ndarray:

    if shape_data.shape == (4, 3):
        coords = shape_data[:, 1:3]
        assert stack.ndim == 3
    elif shape_data.shape == (4, 2):
        coords = shape_data
    else:
        raise ValueError("shape_data must have shape (4, 3) or (4, 2)")

    center = np.mean(coords, axis=0)
    y_center, x_center = center

    # Compute the semi-axes (half-width and half-height)
    b = (np.max(coords[:, 0]) - np.min(coords[:, 0])) / 2  # Semi-minor axis (height)
    a = (np.max(coords[:, 1]) - np.min(coords[:, 1])) / 2  # Semi-major axis (width)

    # Create an empty mask with the same size as the matrix
    mask_shape = (stack.shape[1], stack.shape[2])
    mask = np.zeros(mask_shape, dtype=bool)

    # Generate a grid of (y, x) coordinates (corrected for NumPy's indexing)
    y_indices, x_indices = np.indices(mask_shape)  # Shape (rows, cols)

    # Apply the ellipse equation
    ellipse_mask = (
        ((x_indices - x_center) ** 2) / a**2 + ((y_indices - y_center) ** 2) / b**2
    ) <= 1
    mask[ellipse_mask] = True  # Set pixels inside the ellipse to True

    masked_matrix = np.where(mask[None, :, :], stack, np.nan)

    return masked_matrix


def compute_dff_single_cell(f: np.ndarray) -> np.ndarray:
    f0 = np.percentile(f, 10)
    return (f - f0) / f0


def threshold_detect(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Returns the indices where signal crosses the threshold"""
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


def threshold_detect_falling_edge(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Returns the indices where signal falls below the threshold"""
    thresh_signal = signal > threshold
    falling_edges = (
        np.where(thresh_signal[:-1] & ~thresh_signal[1:])[0] + 1
    )  # Detect falling transitions
    return falling_edges
