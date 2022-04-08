"""
Configuration file
Put the file paths for the models this code needs to run.
You have to download the PoWR models yourself since it's a lot of data.

############################################
You have to set `powr_path` yourself
Scroll down to set `powr_path`
############################################

You can also change `temp_path`, which is towards the bottom of this file.

Created: April 3, 2022
"""
__author__ = "Ramsey Karim"

import importlib.resources
import os
from pathlib import Path

import pandas as pd

# Data for this project that I am including in the Github repo will go in the `data` package
# You could put PoWR models here too, I have scoby_data/PoWR/ in the .gitignore

from .scoby_data import test_data

"""
PoWR model directory
This path should point to a folder somewhere on your machine which contains each grid of PoWR models you will use.
Example directory structure:

your_powr_model_directory/
    OB/
        modelparameters.txt (this needs to be in each grid directory, should come with the code)
        ob-i_15-20_sed.txt
        ob-i_15-22_sed.txt
        ob-i_15-24_sed.txt
        ob-i_16-20_sed.txt
        (and so on, like 200-ish files)
    WNL-H50/
        modelparameters.txt
        wnl-h50_01-04_sed.txt
        wnl-h50_01-05_sed.txt
        wnl-h50_01-06_sed.txt
        wnl-h50_02-04_sed.txt
        (and so on, like 160-ish files)

If this is your directory, you will set powr_path = "your_powr_model_directory" and that's all you have to worry about.

PoWR models are available for download here: https://www.astro.physik.uni-potsdam.de/~wrh/PoWR/powrgrid1.php
You should download the full spectrum for every model in each grid.
When enter a model grid, you will see "Grid Download" instructions on the page, which instruct you to pick a random
model and proceed. On the specific model's page, you will see a "Start grid download" button with instructions.
You will need to download the "Line spectrum (calibrated)" version for the entire grid. The wavelength range is up to
you, so long as it includes what you will need. A wider wavelength range is safer.

This will require a long download and create ~0.1 GB of data in total for maybe 4-5 grids.
(at least when I downloaded them in 2020; I am not sure what has changed, but I know they have updated things).

The models are organized into grids for each category of star ("OB" for early type main sequence, "WNE" for early
type WN, etc).
When you download (and unzip) a given grid of models, you will end up with a directory full of text files.
Each text file contains the spectrum for one model and the directory represents that whole grid.
Leave the text filenames unchanged (they should look something like "wnl-h50_09-04_sed.txt" and so on).

!!!
IMPORTANT:
Name the grid directory to match the name of the grid.
I hardcoded these directory names in, so the OB grid needs to be in the directory "OB" and so on. The names I have
hardcoded are:
OB  WC  WNE  WNL  WNL-H50

!!!!!!!!!!!!!!!!!!!!! EVEN MORE IMPORTANT!
If you need to use more grids than this, you will have to do some editing.
Honestly if you have to use any WR stars at all, you will need to edit. I only had 1 WR star and I hardcoded its
parameters because WRs are tricky. You do not need to deal with WR stars to run this code; you can do all OBs and it
will be much easier. But you can handle WRs if you want to put in a little extra work. There is already a lot of code
for handling them and their uncertainties ready to go in here.
!!!!!!!!!!!!!!!!!!!!!


Let's say your folder is "/home/user/PoWR"
That folder, PoWR/, should contain the subfolders: OB, WNE, WNL-H50, etc (see above)
At minimum, you need the OB grid (OB-I on PoWR, but name it "OB" in your directory structure). You don't need the WR
grids if you aren't using WR stars.
"""
# PoWR Path below!!!
#
#
#
#
# This is my PoWR path! Change it to yours!!!!!
powr_path = "/home/ramsey/Documents/Research/Feedback/misc_data/SpectralTypes/PoWR"
#
#
#
#
#
#
# PoWR Path ^^^^^^^^^
# Catch missing powr_path
if not os.path.isdir(powr_path):
    raise FileNotFoundError(f"Could not find the PoWR grid path <{powr_path}>. You will need to download some PoWR "
                            f"models in order to run scoby. Check the documentation on GitHub for instructions. "
                            f"Once you have downloaded them, you'll need to specify the path to them in config.py")

"""
Provide some directory where it is safe to make random temporary files.
This directory will be used if you identify a catalog reduction process as involving "large" amounts of data.
For example, if you have like >10 stars in your catalog and you ask for a 2D spatially-resolved G0 map that's 100 pixels
on a side, the code will need to make and operate on an array of 10x100x100 = 10^5 elements at some point during the
reduction. BUT, then you have the spectral axis while you're integrating the spectra into G0, so you can see how memory
can quickly get out of hand.

I work around this by using numpy memory maps and operating on the array in sections to save your memory and allow
work on arrays that are too large to fit into memory (or will slow down your computer). In order to do this, I need to
save a temporary file somewhere.

One advantage of this: if you don't delete the temporary files (they're probably not that large), one of them stores
the cube of "distances from each star", which takes some time to calculate but can be reused if you're not changing your
selection of stars or your map grid (WCS for desired output map).
TODO: I could probably create some way to reuse it even if you DO change your selection of stars.
"""
temp_dir = "/home/ramsey/Downloads"  # This is mine, you can change it or let the code automatically make you one
if not os.path.isdir(temp_dir):
    # Try to make a directory in Downloads (or see if it's been done already)
    # I chose Downloads because it always already looks like a junk directory on my machines, so it seems like an ok
    # place to dump temporary files (but not so hard to find as true "tmp")
    downloads_path = os.path.join(Path.home(), "Downloads")
    # Try to make a temp directory and, in the process, see if it's already there (or if you can even find Downloads)
    proposed_temp_dir = os.path.join(downloads_path, "tmp_scoby_software")
    try:
        os.mkdir(proposed_temp_dir)
        # Success! Leave a note in case someone comes across that directory
        print(f"Created directory for temporary files in {proposed_temp_dir}. See more info about this in config.py, "
              "where you can also change the path.")
        temp_dir = proposed_temp_dir
        with open(os.path.join(temp_dir, 'readme.txt'), 'w') as f:
            f.write(f'Folder created by {__file__} in the scoby package (synthesizing cluster-wide '
                    f'characteristics from catalogs of OB stars).')
    except FileNotFoundError as e:
        # Can't find your Downloads folder where I'd expect, so you'll have to set the path yourself in this file
        raise RuntimeError(f"Can't locate your Downloads directory in order to set a tmp file path. "
                           f"I tried <{downloads_path}>. You'll have to define a tmp file path in config.py.") from e
    except FileExistsError:
        # This already exists! Done, just set temp_dir to the found directory
        temp_dir = proposed_temp_dir

# The Leitherer, Martins, and Sternberg tables have already been included in this module in the "scoby_data" folder

test_catalog_fn = "OBradec.csv"


def load_test_data():
    """
    For testing: open an example catalog as a pandas DataFrame
    :returns: pandas DataFrame with RAdeg, DEdeg, and SpectralType columns
        There are some other columns but you can get rid of them if you want
    """
    with importlib.resources.open_text(test_data, test_catalog_fn) as f:
        df = pd.read_csv(f)
    return df
