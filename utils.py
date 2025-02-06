import aicsimageio
import xml.etree.ElementTree as ET
import numpy as np
import xmltodict


def get_image_metadata_czi(AICSImage_object: aicsimageio.aics_image.AICSImage) -> dict:
    metadata = AICSImage_object.metadata
    metadata_string = ET.tostring(element=metadata, encoding="unicode")
    return xmltodict.parse(metadata_string)


def load_image(image_directory_path: str) -> aicsimageio.aics_image.AICSImage | None:
    """Return AICSImage object"""
    # Selects the first scene found
    try:
        # AICSImageIo is lazy loading the image, hence the file is opened twice here
        return aicsimageio.AICSImage(image_directory_path)
    except FileNotFoundError as e:
        print(f"ERROR: There was no image found at '{image_directory_path}'.")
        return None


def select_elipse_from_stack(shape_data: np.ndarray, stack: np.ndarray) -> np.ndarray:

    assert shape_data.shape == (4, 3)
    assert stack.ndim == 3

    coords = shape_data[:, 1:3]

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
