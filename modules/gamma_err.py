#!/usr/bin/env python

"""
    Gamma Distribution based single-tail error data reduction data-processing pipeline for dataframes
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

def remove_error(data, param, bins=25, pthresh=.5, verbose=0):
    """
    Selects a certain parameter from a `pandas` DataFrame and creates
    a Matplotlib histogram for the given data. Then it fits a `Gamma
    Distribution` to perform statistical error reduction.
    \n
    Args:
        data (pandas.core.frame.DataFrame): The `Pandas` dataframe that contains the dataset
        param (str): An attribute present in the `data` parameter that will be processed in the
                     algorithm. This `param` needs to be `x >= 0 for x in data[param]` in order
                     to fit a `Gamma Distribution`
        bins (int, optional): Number of bins for the histogram. Defaults to 25.
        pthresh (float, optional): Maximum threshold for selecting datapoints. Must be in the range
                                   [0., 1.]. Defaults to .85.
        verbose (int, optional): Verbosity for the algorithm. Must be in the range [0,2].
                                 0 -> No histogram or debug statements, data returned post processeing
                                 1 -> Histogram displayed and data returned post processing
                                 2 -> Histogram displayed, debug statements printed, and data returned
                                 Defaults to 0.
    \n
    Returns:
        [pandas.core.frame.Dataframe]: the same dataframe `data` after the required restrictions have
                                       been enforced via the Gamma Distribution and `pthresh`.
    \n
    Raises:
        KeyError: invalid parameter `param`
        ValueError: data for parameter contains negatives
        ValueError: bins < 0
        ValueError: pthresh > 1. or pthresh < 0.
        ValueError: verbosity != 0 or 1 or 2
    """
    # scope constants
    raw_array = None
    acceptable_verbose = [0, 1, 2]

    # prerequirement checks
    try:
        raw_array = np.array(data[param])
    except KeyError:
        print(f"The parameter {param} is not present in the dataframe. Please provide a valid parameter")
    if sum(1 for number in raw_array if number < 0) != 0:
        raise ValueError("Selected parameter must only contain positive numbers.\nNegatives found")
    if bins < 1:
        raise ValueError(f"Must have at least 1 bin.\nValue provided: {bins}")
    if pthresh > 1. or pthresh < 0.:
        raise ValueError("pthresh must be within the following range: 0. < pthresh < 1." 
                + f"\nValue provided: {pthresh}")
    if verbose not in acceptable_verbose:
        raise ValueError(f"Verbosity can only be 0, 1, or 2.\nValue provided: {verbose}")
    
    alpha = math.pow(np.mean(raw_array), 2) / np.var(raw_array) # alpha calculation
    if verbose > 0:     # histogram and fitting
        plt.hist(x=raw_array, bins=bins, density=True, histtype="step") # histogram
        beta = np.var(raw_array) / np.mean(raw_array)               # beta calculation
        if verbose == 2:
            print(f"\u03b1: {alpha}, \u03b2: {beta}")
        x = np.linspace(0, max(raw_array), 1000)
        plt.plot(x, stats.gamma.pdf(x, alpha, scale=beta), color="crimson", linestyle="-", label="Gamma Fit")
        plt.title(f"{param.upper()} Standardized Distribution with Gamma Fit")
        plt.axvline(pthresh * max(raw_array), color="navy", linestyle="-.", label="Max Threshold")
        plt.legend(loc="best")
        plt.xlabel(param)
        plt.ylabel("counts")
        plt.show()
    
    # only retain data below max threshold
    data = data[data[param] <= pthresh * max(raw_array)]
    # removing error column after reduction
    data = data.drop([param], axis="columns")
    return data
