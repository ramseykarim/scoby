"""
General-purpose / example functions for wrangling star catalogs into usable DataFrames.
For example, I have some functions in here that deal with turning string coordinates
into SkyCoord objects.

You could write your own functions into here too. Or model stuff after these functions.

Created: April 7, 2022
"""

__author__ = "Ramsey Karim"

import pandas as pd

from astropy.coordinates import SkyCoord, FK5
from astropy import units as u


def convert_hhmmss(hhmmss_hhmmss: str) -> tuple[str, str]:
    """
    Split a single ra,dec string without spaces into separate RA and Dec strings
    Usage example:
    Where you could ordinarily do this if you had RA, Dec already separated:
    >>> ra, dec = 155.979910, -57.757480
    >>> coord_1 = SkyCoord(ra, dec, unit=u.deg)
    You can use this function to make a SkyCoord out of this type of string
    >>> coord_hhmmss_string = '102400.52-574444.6'
    >>> coord_2 = SkyCoord(*convert_hhmmss(coord_hhmmss_string), unit=(u.hourangle, u.deg))
    Doesn't seem like unique functionality, but I'll delete it when it's
    clear I won't use it

    :param hhmmss_hhmmss: a string with unseparated hours, minutes, and seconds
        (see the example above). The string will have both RA and Dec, which
        will be separated by a plus or minus sign (you could modify this)
    :returns: tuple of RA, Dec which are expressed as hh:mm:ss.s(s) strings
        '102400.52-574444.6' -> "10:24:00.52", "57:44:44.6"
        '55555+30201.23' -> "5:55:55", "+3:02:01.23"
        These strings can be passed to SkyCoord with no problem
    """
    char_list = list(hhmmss_hhmmss)
    coord, second_coord = [], None
    count = 1
    while char_list:
        item = char_list.pop()
        if item.isnumeric():
            coord.append(item)
            if '.' in coord:
                if not count % 2:
                    coord.append(':')
                count += 1
        elif item == '.':
            coord.append(item)
        elif item in ['+', '-']:
            if coord[-1] == ':':
                coord.pop()
            coord.append(item)
            second_coord = coord
            coord = []
            count = 1
    ra, dec = coord, second_coord
    ra.reverse(), dec.reverse()
    if ra[0] == ':':
        ra.pop(0)
    return "".join(ra), "".join(dec)


def coords_from_hhmmss(df: pd.DataFrame, frame=FK5):
    """
    Cobble together the HMS coordinates into RAdeg, DEdeg
    This is from the original catalog reduction, made for the Ascenso catalog,
        and I edited it to be more general and stop at the SkyCoord step
        (4/16/2020)
    :param  df: a pandas DataFrame with columns RAh, RAm, RAs, DEd, DEm, DEs
        Some tables do this! This function is only useful if they do
        This DataFrame will be modified by this function; a "SkyCoord" column
        will be added which has the astropy SkyCoord representation of the
        coordinates described by the 6 columns listed above
    :param frame: The coordinate frame of the 6 column RA/Dec coordinates.
        Once the SkyCoord is made, you can convert to other frames.
        Something like ICRS vs FK5 probably won't introduce huge inaccuracies
        but it's nice to have it correct.
    """
    RAstr = df.apply(lambda row: ":".join(row[x] for x in ('RAh', 'RAm', 'RAs')), axis=1)
    DEstr = df.apply(lambda row: row['DE-'] + ":".join(row[x] for x in ('DEd', 'DEm', 'DEs')), axis=1)
    RADEstr = RAstr + " " + DEstr
    df['SkyCoord'] = RADEstr.apply(lambda radec_string: SkyCoord(radec_string, unit=(u.hourangle, u.deg), frame=frame))


def read_table_format(file_handle, n_cols):
    """
    Read one of those standard-format table descriptors
    Example of one of these: (This is just an example! It can read ANY header of this format!)
        Byte-by-byte Description of file: table4.dat (not sure where this is from)
        --------------------------------------------------------------------------------
           Bytes Format Units   Label     Explanations
        --------------------------------------------------------------------------------
           1-  2  I2    h       RAh       Right Ascension J2000 (hours)
           4-  5  I2    min     RAm       Right Ascension J2000 (minutes)
           7- 10  F4.1  s       RAs       Right Ascension J2000 (seconds)
              12  A1    ---     DE-       Declination J2000 (sign)
          13- 14  I2    deg     DEd       Declination J2000 (degrees)
          16- 17  I2    arcmin  DEm       Declination J2000 (minutes)
          19- 22  F4.1  arcsec  DEs       Declination J2000 (seconds)
          24- 29  F6.3  mag     Vmag      Mean observed Johnson V magnitude
          31- 35  F5.3  mag   e_Vmag      Uncertainty on the V magnitude
          37- 41  F5.3  mag     B-V       Mean observed Johnson B-V colour
          43- 47  F5.3  mag   e_B-V       Uncertainty on the B-V colour
        --------------------------------------------------------------------------------
    :param file_handle: should be open and currently at the first column line.
    :param n_cols: tells how many columns are present, and thus how many rows
        this program should read.
    :returns: the column byte intervals and the column names. These are ready
        to be passed directly to pandas.read_fwf as the colspecs and names
        keyword arguments.
    """
    col_intervals, col_labels = [], []
    # slices for the byte format beginning/end integers
    sl0, sl1 = slice(1, 4), slice(5, 8)
    # slice for the column name
    sln = slice(21, 32)
    for i in range(n_cols):
        line = file_handle.readline()
        if line[sl0].isspace() and line[sl1].isspace():
            continue
        start = int(line[sl0]) - 1 if not line[sl0].isspace() else int(line[sl1]) - 1
        end = int(line[sl1])
        col_intervals.append((start, end))
        label = max(line[sln].split(), key=len)
        col_labels.append(label)
    return col_intervals, col_labels
