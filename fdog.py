from skimage.filters import gaussian
from skimage.util import img_as_float
import numpy as np


def difference_of_gaussians(
    image,
    low_sigma,
    high_sigma=None,
    *,
    mode="nearest",
    cval=0,
    channel_axis=None,
    truncate=4.0,
    p=None,
):
    """
    Stolen from scikit_image.
    """
    image = img_as_float(image)
    low_sigma = np.array(low_sigma, dtype="float", ndmin=1)
    if high_sigma is None:
        high_sigma = low_sigma * 1.6
    else:
        high_sigma = np.array(high_sigma, dtype="float", ndmin=1)

    if channel_axis is not None:
        spatial_dims = image.ndim - 1
    else:
        spatial_dims = image.ndim

    if len(low_sigma) != 1 and len(low_sigma) != spatial_dims:
        raise ValueError(
            "low_sigma must have length equal to number of" " spatial dimensions of input"
        )
    if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
        raise ValueError(
            "high_sigma must have length equal to number of" " spatial dimensions of input"
        )

    low_sigma = low_sigma * np.ones(spatial_dims)
    high_sigma = high_sigma * np.ones(spatial_dims)

    if any(high_sigma < low_sigma):
        raise ValueError("high_sigma must be equal to or larger than" "low_sigma for all axes")

    im1 = gaussian(
        image,
        low_sigma,
        mode=mode,
        cval=cval,
        channel_axis=channel_axis,
        truncate=truncate,
        preserve_range=False,
    )

    im2 = gaussian(
        image,
        high_sigma,
        mode=mode,
        cval=cval,
        channel_axis=channel_axis,
        truncate=truncate,
        preserve_range=False,
    )

    if p is not None and p != 0:
        return (1 + p) * im1 - p * im2
    else:
        return im1 - im2
