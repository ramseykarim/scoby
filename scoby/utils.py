"""
Helper functions for the catalog creation / reduction routines
A lot of stuff that was piled into g0_stars.py should go into here
Created: May 5, 2020 (deeper into quarantine)

Updated April 3, 2022
Removing filepath specifications (moving to config.py) and removing definitions
that were specific to my projects.
"""
__author__ = "Ramsey Karim"

import os
import numpy as np
import matplotlib.pyplot as plt

import socket
import pwd

from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay  # Hello, old friend

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

"""
Debug/plotting utilities
"""


def plot_coordinates(data_gen_f, coords, setup=True, show=True, subplot=(111,)):
    """
    Quickly throw up some coordinates onto an image
    :param data_gen_f: some function that returns an image and a WCS object
        Can be None if setup is False; in that case, plt.gca() should be a
        WCS projection
    :param coords: SkyCoord array
    :param setup: open the data? make a suplot?
    :param show: plt.show()?
    :param subplot: subplot number to pass to plt.subplot if setup==True.
        will be unpacked if it's a tuple
    """
    if setup:
        img, w = data_gen_f()
        if isinstance(subplot, tuple):
            plt.subplot(*subplot, projection=w)
        else:
            plt.subplot(subplot, projection=w)
        plt.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(1), vmax=np.arcsinh(80))
    plt.scatter(coords.ra.deg, coords.dec.deg, transform=get_transform(),
                color='red', s=2)
    if show:
        plt.show()


def get_transform():
    """
    Shortcut for plotting
    """
    return plt.gca().get_transform('world')


"""
Saving catalog (debug, but also just useful)
"""


def save_df_html(df, na_rep='--'):
    """
    Quickly save the argument dataframe to a test.html file, and print its length
    """
    print(len(df))
    df.to_html("~/Downloads/test.html", na_rep=na_rep)



"""
Everything below is a utility for this project
"""


def coordinate_midpoint(coord1, coord2):
    """
    Return the SkyCoord midpoint between two coordinates.
    :param coord1: SkyCoord (or castable as SkyCoord)
    :param coord2: SkyCoord (or castable as SkyCoord)
    :returns: SkyCoord midpoint between the two arguments
    """
    coord1, coord2 = SkyCoord(coord1), SkyCoord(coord2)
    position_angle = coord1.position_angle(coord2)
    length = coord1.separation(coord2)
    return coord1.directional_offset_by(position_angle, length / 2.)


"""
TWO METHODS FOR CALCULATING SEPARATION FROM A SET OF POINTS
These make arrays of distances from an arbitrary SkyCoord given an existing WCS
(The older functions that did distance_from_center are in < May 2020 git commits)

pixelgrid is faster (not sure about scaling difference but it's better than linear)
wcssep is more accurate across larger areas, since it deals in great circles instead of planes
Depending on the approximation being made, it may not matter so much
May 5, 2020: I remember testing this, but don't remember how big these differences were
"""


def distance_from_point_pixelgrid(point_coord, w, distance_los):
    """
    Faster distance grid method
    :param point_coord: a SkyCoord object or array
    :param w: a WCS object
    :param distance_los: a Quantity or float distance
        The resulting grid comes out in these units.
    """
    # grid_shape from w.array_shape
    # Remember that world_to_array_index returns integers and world_to_pixel returns floats
    grid_shape = w.array_shape
    ref_pixel = w.world_to_pixel(point_coord)[::-1]
    # Get physical separation per pixel along each axis at 0,0 (assume they do not change -- this should be ok for
    # small regions)
    ds_di = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0] + 1, ref_pixel[1])).to(
        'rad').to_value() * distance_los
    ds_dj = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0], ref_pixel[1] + 1)).to(
        'rad').to_value() * distance_los
    grid = np.sqrt((ds_di * (np.arange(grid_shape[0]) - ref_pixel[0]))[:, np.newaxis] ** 2 + (ds_dj * (
            np.arange(grid_shape[1]) - ref_pixel[1]))[np.newaxis, :] ** 2)
    return grid


def distance_from_point_wcssep(point_coord, w, distance_los):
    """
    Again, this is way slower (probably grid.size**2)
    :param point_coord: a SkyCoord object or array
    :param w: a WCS object
    :param distance_los: a Quantity or float distance
        The resulting grid comes out in these units.
    """
    grid = np.full(w.array_shape, np.nan)
    ij_arrays = tuple(idx_grid.ravel() for idx_grid in np.mgrid[tuple(slice(0, shape_i) for shape_i in w.array_shape)])
    grid[ij_arrays] = w.array_index_to_world(*ij_arrays).separation(point_coord).to('rad').to_value() * distance_los
    return grid


"""
INTERPOLATION
Originally from LeithererTable, but now used in PoWRGrid too
"""


def delaunay_triangulate(x, y):
    """
    Get the Delaunay triangulation of the x and y grid
    """
    xy = np.array([x, y]).T
    return Delaunay(xy)


def fit_characteristic(xy_delaunay, z):
    """
    Return an interpolation function.
    This CloughTocher2DInterpolator function is pretty smart, so I don't have
        to do much else with it.
    :param xy_delaunay: should be the Delaunay triangulation of the XY grid
    :param z: z array with same shape as x or y
    """
    interp_function = CloughTocher2DInterpolator(xy_delaunay, z, fill_value=np.nan)
    return interp_function


"""
Really fundamental stuff
The polynomial and flquantiles functions are little convenience functions I wrote a while back and just import from
other code. Since that clearly won't work in this case (and I haven't packaged them) I'll just include them here.
"""


def polynomial(x, fit):
    """
    Polynomial given x array/scalar and coefficients
    TODO: there is a numpy.polynomial library that might be "safer" to use
    What I'm doing here is pretty simple, but it's still nice to use (or at least be aware of) existing tools that do
    the same thing.
    :param x: x array or scalar
    :param fit: coefficient sequence. Assumes "np.polyfit" coefficient ordering, so 0th coeff is for highest order
    :return: x-shaped array (or scalar float) of polynomial y values
    """
    deg = len(fit) - 1
    if hasattr(x, 'ndim') and x.ndim > 0:
        solution = np.zeros(x.shape)
    else:
        solution = 0.
    for i, coeff in enumerate(fit):
        solution += coeff * x ** (deg - i)
    return solution


def flquantiles(x, q):
    """
    Get values of first and last q-quantiles of x values.
    (In the name, FL = First and Last quantiles)
    If x is multi-D, only works if first axis (0) is sample value axis.
    :param x: sample values
    :param q: number of quantiles. Should be >2.
    :return: tuple(first, last) where first and last have dtype of x[i]
    """
    sorted_x = np.sort(x, axis=0)
    first_quant = sorted_x[len(x) // q]
    last_quant = sorted_x[(q - 1) * len(x) // q]
    return first_quant, last_quant
