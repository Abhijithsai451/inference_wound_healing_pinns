import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray
from scipy.ndimage import gaussian_filter

# Default Parameters configuration

DEFAULT_ROLLING_WINDOW = 5
DEFAULT_GAUSSIAN_SIGMA = 2.0

def apply_rolling_mean(series: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """
    Applies the central Rolling Mean Filter to a single data series.

    Central Rolling Mean Filter:
    The Central Rolling Mean (often called a Moving Average) is the simplest and most intuitive smoothing technique.
    It is a filter where the output at any point is the average of the data within a specified window centered at that point.
    Key Characteristics:
    Window Size ($W$): The critical parameter. A larger window results in stronger smoothing but also causes more
    distortion of sharp features in the original signal.
    Weighting: All points within the window are given equal weight (1/W)
    Effect: This method is effective, but because of the uniform weighting, it can introduce slight "jumps" or
    discontinuities in the smoothed curve, especially when the window is large.
    """
    if window % 2 == 0:
        raise ValueError("Rolling window must be an odd number for a central mean.")

    return series.rolling(
        window = window,
        center = True,
        min_periods = 1
    ).mean()

def apply_gaussian_smoothing(series: pd.Series, sigma: float = DEFAULT_GAUSSIAN_SIGMA) -> ndarray:
    """
    Gaussian SmoothingGaussian smoothing is a more sophisticated and generally preferred method for reducing noise.
    It is based on convolution with a Gaussian kernel (a bell-shaped function).

    The Mathematics:
    Instead of assigning equal weight to all neighboring points, Gaussian smoothing assigns weights based on the
    Gaussian probability distribution (the "bell curve"). Points closer to the central point receive a much higher
    weight, and the influence decreases rapidly as distance increases.
    """
    # Gaussian filter works on Numpy Array
    return gaussian_filter(series.values,
                           sigma= sigma,
                           mode = 'reflect'
                           )





