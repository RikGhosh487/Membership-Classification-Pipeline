#!/usr/bin/env python

"""
    Gaussian Distribution based two-tail error data reduction data-processing pipeline for dataframes
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"

def chop_tails(data, param, bins=25, factor=1, verbose=0):
    """
    Selects a certain parameter from a `pandas` DataFrame and creates
    a Matplotlib histogram for the given data. Then it fits a `Gaussian
    Distribution` to perform statistical error reduction.
    \n
    Args:
        data (pandas.core.frame.DataFrame): The `Pandas` dataframe that contains the dataset
        param (str): An attribute present in the `data` parameter that will be processed in the
                     algorithm.
        bins (int, optional): Number of bins for the histogram. Defaults to 25.
        factor (float, optional): Scaling factor for standard deviation for Gaussian fitting. Must
                                be a positive value. Defaults to 1.
        verbose (int, optional): Verbosity for the algorithm. Must be in the range [0,2].
                                 0 -> No histogram or debug statements, data returned post processeing
                                 1 -> Histogram displayed and data returned post processing
                                 2 -> Histogram displayed, debug statements printed, and data returned
                                 Defaults to 0.
    \n
    Returns:
        [pandas.core.frame.Dataframe]: the same dataframe `data` after the required restrictions have
                                       been enforced via the Gaussian Distribution and `pthresh`.
    \n
    Raises:
        KeyError: invalid parameter `param`
        ValueError: bins < 0
        ValueError: factor < 0.
        ValueError: verbosity != 0 or 1 or 2
    """
    # scopt constants
    raw_array = None
    acceptable_verbose = [0, 1, 2]

    # prerequirement checks
    try:
        raw_array = np.array(data[param])
    except KeyError:
        print(f"The parameter {param} is not present in the dataframe. Please provide a valid parameter")
    if bins < 1:
        raise ValueError(f"Must have at least 1 bin.\nValue provided: {bins}")
    if factor < 0.:
        raise ValueError(f"factor must be a positive number.\nValue provided: {factor}")
    if verbose not in acceptable_verbose:
        raise ValueError(f"Verbosity can only be 0, 1, or 2.\nValue provided: {verbose}")
    
    mean = np.mean(raw_array)   # mean
    std = np.std(raw_array)     # standard deviation
    if verbose > 0:     # histogram and fitting
        hist, bins, _ = plt.hist(x=raw_array, bins=bins, density=True, histtype="step") # histogram
        if verbose == 2:
            print(f"\u03bc: {mean}, \u03c3: {std}")
        p = stats.norm.pdf(bins, mean, std)
        plt.plot(bins, p, color="crimson", linestyle="-", label="Gaussian Fit")
        plt.title(f"{param.upper()} Standardized Distribution with Gaussian Fit")
        plt.axvline(mean + factor * std, color="chocolate", linestyle="-.", label="Max Threshold")
        plt.axvline(mean - factor * std, color="navy", linestyle="-.", label="Min Threshold")
        plt.legend(loc="best")
        plt.xlabel(param)
        plt.ylabel("counts")
        plt.show()
    
    # only retain data within thresholds
    data = data[(data[param] <= mean + factor * std) & (data[param] >= mean - factor * std)]
    return data
