"""
Helper functions for the catalog creation / reduction routines
A lot of stuff that was piled into g0_stars.py should go into here
Created: May 5, 2020 (deeper into quarantine)
"""
__author__ = "Ramsey Karim"

import os
import numpy as np
import matplotlib.pyplot as plt

import socket
import pwd

from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay # Hello, old friend

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

"""
RCW49 specific things
"""
wd2_center_coord = SkyCoord("10 23 58.1 -57 45 49", unit=(u.hourangle, u.deg), frame='fk5')
wd2_cluster_center_coord = SkyCoord("10:24:01.7533 -57:45:33.193", unit=(u.hourangle, u.deg)) # ~11 arcsec error

"""
Project-specific path definitions and functions
"""
feedback_path = "/home/rkarim/Research/Feedback/"
if not os.path.isdir(feedback_path):
    feedback_path = "/home/ramsey/Documents/Research/Feedback/"

ancillary_data_path = f"{feedback_path}rcw49_data/"
cii_path = f"{ancillary_data_path}sofia/"
irac_path = f"{ancillary_data_path}spitzer/irac/"

m16_data_path = f"{feedback_path}m16_data/"

misc_data_path = f"{feedback_path}misc_data/"

figures_path = f"{feedback_path}feedback_code/figures/"

# Similar to the PATH shell variable. Where to look for data, in this order
data_paths = ["", ancillary_data_path, m16_data_path]

def search_for_file(filename):
    """
    Look for the given filename in the specified data_paths, in order.
    Return the first valid one.
    Raise an error if none are valid
    :param filename: string filename
    """
    for dp in data_paths:
        potential_filename = dp + filename
        if os.path.exists(potential_filename):
            # print("HEY ", os.path.abspath(potential_filename))
            return os.path.abspath(potential_filename)
    raise RuntimeError(f"Invalid data filename: \"{filename}\"")


def load_irac(n=1, header=True):
    """
    Loads whatever IRAC band I have on my laptop
    Used to be called "irac_data", but that was not descriptive so I renamed
        this to load_irac on May 22, 2020
    If header keyword, returns image, header. If not, returns image, WCS
    """
    fn = f"30002561.30002561-28687.IRAC.{n}.median_mosaic.fits"
    img, hdr = fits.getdata(irac_path+fn, header=True)
    if header:
        return img, hdr
    else:
        return img, WCS(hdr)


def irac_data():
    """
    For backwards compatibility; I renamed the function on May 22, 2020
    """
    print("==\n\tWARNING: irac_data is deprecated; rename this function call to load_irac(n: 1-4)\n==")
    return load_irac()


def load_cii(n=0):
    """
    Easily load in a [CII] moment 0 image
    n is 0 for the full-range, 1 for [-12, -8], and 2 for [-8, -4]
    """
    if n == 0:
        stub = "fullrange"
    elif n == 1:
        stub = "-12to-8"
    elif n == 2:
        stub = "-8to-4"
    elif n == 3:
        stub = "-25to0"
    img, hdr = fits.getdata(f"{cii_path}mom0_{stub}.fits", header=True)
    return img, WCS(hdr)


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
This is a personal utility, I should save this in some package I use for every project
"""
def create_png_metadata(title=None, file=None, func=None, **extra_metadata_kwargs):
    """
    January 12, 2022
    Create metadata for PNGs. Include author (my name), title if provided,
    and the responsible Python filename for the source and function name if
    provided
    :param title: string title for image. Short but descriptive.
    :param file: you must pass `__file__` to this argument. I could do some
        stack inspection to back up, but if I ever use a decorator I'm not
        guaranteed one step up in the stack, so I'll be conservative and do this
    :param func: string name of current function
    :param extra_metadata_kwargs: any extra key/val pairs to pass into
        the PNG metadata. Uses the 'update' dict method, so you could use this
        to overload with a custom author or source
    :returns: dict appropriate to pass to the 'metadata' kwarg in matplotlib's
        savefig method
    """
    source = []
    if file is not None:
        source.append(os.path.basename(file).replace('.py', ''))
    if func is not None:
        source.append(func)
    source = '.'.join(source)
    if not source:
        source = 'feedback_code, unspecified location'
    source = f'({pwd.getpwuid(os.getuid())[0]}@{socket.gethostname()}) {source}'
    metadata = {'Author': __author__, 'Source': source}
    if title is not None:
        metadata['Title'] = title
    metadata.update(extra_metadata_kwargs)
    return metadata


"""
Something related to point separation
"""
"""
Everything below is a utility for this project
"""
def coordinate_midpoint(coord1, coord2):
    """
    Return the SkyCoord midpoint between two coordinates.
    :param coord1: SkyCoord (or castable)
    :param coord2: SkyCoord (or castable)
    :returns: SkyCoord midpoint between the two arguments
    """
    coord1, coord2 = SkyCoord(coord1), SkyCoord(coord2)
    position_angle = coord1.position_angle(coord2)
    length = coord1.separation(coord2)
    return coord1.directional_offset_by(position_angle, length/2.)



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
    # Get physical separation per pixel along each axis at 0,0 (assume they do not change -- this should be ok for small regions)
    ds_di = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0]+1, ref_pixel[1])).to('rad').to_value() * distance_los
    ds_dj = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0], ref_pixel[1]+1)).to('rad').to_value() * distance_los
    grid = np.sqrt((ds_di*(np.arange(grid_shape[0]) - ref_pixel[0]))[:, np.newaxis]**2 + (ds_dj*(np.arange(grid_shape[1]) - ref_pixel[1]))[np.newaxis, :]**2)
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
