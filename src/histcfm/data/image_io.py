"""Image loading used by the formal cell-level HistCFM data path.

Derived from ``load_image`` in SydneyBioX/GHIST ``dataio/utils.py``:
https://github.com/SydneyBioX/GHIST

Modified for HistCFM on 2026-08-12 to use package-local typing and delayed
backend imports. The array shape/channel behavior is retained. Distributed
under GNU GPL version 3 only; see the repository ``LICENSE`` file.
"""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def load_image(path: PathLike):
    """Load a TIFF or conventional image with the audited GHIST shape rules.

    TIFF-family paths use ``tifffile``. Other formats use ``imageio``. A
    four-dimensional array loses its final singleton/page channel exactly as
    in the audited loader, and channel-first arrays with three leading
    channels are moved to channel-last layout.

    This function is derived from GHIST ``dataio/utils.py::load_image``.
    """

    image_path = Path(path)
    if image_path.suffix.lower() in (".tif", ".ome.tif"):
        import tifffile

        image = tifffile.imread(str(image_path))
    else:
        import imageio.v2 as imageio

        image = imageio.imread(str(image_path))

    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    if image.shape[0] == 3:
        import numpy as np

        image = np.moveaxis(image, 0, -1)

    return image
