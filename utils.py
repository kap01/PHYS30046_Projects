import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import iminuit
import mplhep as hep
from scipy.integrate import quad
from functools import partial
plt.style.use(hep.style.LHCb2)

''' 
Imports required for the likelihood minimisation algorithm that we will use to fit the
generated data in the above cell in order to determine gZ
'''
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit

# we also need a cost function to fit and import the Binned Log Likelihood function
from iminuit.cost import BinnedNLL

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)


import numpy as np
from scipy.interpolate import interp1d

def gen_data(n_events, func, x_min, x_max, npoints=100_000, *args, **kwargs):
    """
    Generate random samples distributed according to an arbitrary
    non‑negative function using inverse transform sampling.

    Parameters
    ----------
    n_events : int
        Number of samples to draw.
    func : callable
        The target function. It must accept the sample variable as its first
        argument. Additional parameters for func can be supplied via *args
        and **kwargs.
    x_min, x_max : float
        Domain over which to sample.
    npoints : int, optional
        Number of grid points used to build the CDF (higher values give
        a better approximation of the integral).
    *args, **kwargs :
        Additional arguments passed to func.

    Returns
    -------
    ndarray
        Array of samples distributed according to func on [x_min, x_max].
    """
    # Create a grid on the desired domain
    x_grid = np.linspace(x_min, x_max, npoints)

    # Evaluate the function on the grid.  Try vectorised evaluation,
    # otherwise fall back to a loop.
    try:
        f_vals = func(x_grid, *args, **kwargs)
    except Exception:
        f_vals = np.array([func(x, *args, **kwargs) for x in x_grid])

    # Ensure the function is non‑negative; if it can be negative, clip to zero
    f_vals = np.clip(f_vals, a_min=0.0, a_max=None)

    # Construct the CDF by cumulative summation and normalise it
    cdf = np.cumsum(f_vals)
    cdf /= cdf[-1]  # now cdf[-1] == 1

    # Create the inverse CDF via interpolation
    inv_cdf = interp1d(cdf, x_grid, bounds_error=False,
                       fill_value=(x_min, x_max))

    # Sample uniformly on [0,1] and map through the inverse CDF
    random_probs = np.random.rand(n_events)
    sampled_x = inv_cdf(random_probs)

    return sampled_x


'''
The function below defines the cumulative distribution function of the cross-section expression
that we will feed into the minimisation algorithm
'''
def func_cdf(func, bin_edges,*params):
    xmin, xmax = bin_edges[0], bin_edges[-1]

    
    def indef_integr(xx):
        val, _ = quad(func, xmin, xx,args=params)
        return val

    norm = indef_integr(xmax)

    # evaluate for each bin edge
    integr = np.array([indef_integr(xx) for xx in np.atleast_1d(bin_edges)])
    return (integr - indef_integr(xmin)) / norm
