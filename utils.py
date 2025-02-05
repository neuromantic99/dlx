import aicsimageio
from pathlib import Path
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
