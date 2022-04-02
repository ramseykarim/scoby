"""
script to read/create star catalogs with pandas
file used to be called readcat.py
created: October 21, 2019

Added some tables from different authors and standardized coordinates into
SkyCoords earlier in the process. Added a lot of documentation.
updated: April 14 - May 5, 2020
"""
__author__ = "Ramsey Karim"

import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5, ICRS

from . import utils


catalog_directory = utils.ancillary_data_path + "catalogs/"

def convert_hhmmss(hhmmss_hhmmss):
    """
    Split a single ra,dec string without spaces into separate RA and Dec strings
    Usage example:
    # tft_name = '102400.52-574444.6'
    # vphasra, vphasdec = 155.979910, -57.757480
    #
    # vphas_coord = SkyCoord(vphasra, vphasdec, unit=u.deg)
    # tft_coord = SkyCoord(*convert_hhmmss(tft_name), unit=(u.hourangle, u.deg))
    Doesn't seem like unique functionality, but I'll delete it when it's
    clear I won't use it
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


def coords_from_hhmmss(df, frame=FK5):
    """
    Cobble together the HMS coordinates into RAdeg, DEdeg
    This is from the original catalog reduction, made for the Ascenso catalog,
        and I edited it to be more general and stop at the SkyCoord step
        (4/16/2020)
    """
    RAstr = df.apply(lambda row: ":".join(row[x] for x in ('RAh', 'RAm', 'RAs')), axis=1)
    DEstr = df.apply(lambda row: row['DE-'] + ":".join(row[x] for x in ('DEd', 'DEm', 'DEs')), axis=1)
    RADEstr = RAstr + " " + DEstr
    df['SkyCoord'] = RADEstr.apply(lambda radec_string: SkyCoord(radec_string, unit=(u.hourangle, u.deg), frame=frame))


def read_table_format(file_handle, n_cols):
    """
    Read one of those standard-format table descriptors
    Example of one of these:
        Byte-by-byte Description of file: table4.dat
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


def skiplines(file_handle, n_lines):
    """
    skips ahead n_lines in the already-opened file_handle
    :param file_handle: file handle that is open
    :param n_lines: number of lines to skip
    """
    for i in range(n_lines):
        file_handle.readline()


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Tsujimoto %&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""

def openTFT_single(filenumber):
    """
    Open a Tsujimoto 2007 table (there are 3)
    Helper function for openTFT_complete
    :param filenumber: integer 1, 2 or 3 indicating table number
    """
    filename = catalog_directory+"Tsujimoto2007/tbl{:d}".format(filenumber)
    TFT_tblinfo = {1: (9, 22, 24), 2: (9, 20, 20), 3: (9, 14, 14)}
    with open(filename) as f:
        skip1, n_cols, skip2 = TFT_tblinfo[filenumber]
        skiplines(f, skip1)
        col_intervals, col_labels = read_table_format(f, n_cols)
        skiplines(f, skip2)
        df = pd.read_fwf(f, colspecs=col_intervals, names=col_labels, index_col='Num')
    return df


def openTFT_complete():
    """
    Open all 3 TFT tables and combine them
    Tables are from Tsujimoto et al. 2007 (ApJ 665:719-735)
    Go in here and print df.columns if you want to know what's in there
    TFT Specifics:
        Includes original SIRIUS NIR data from this project.
            Tsujimoto states that the SIRIUS data is a larger FOV than the
            similar Ascenso et al. 2007 data.
            Both this SIRIUS data and Ascenso are J,H,Ks bands.
        Cross-matches to NOMAD, 2MASS, (SIRIUS,) and GLIMPSE
        NOMAD: Naval Observatory Merged Astrometric Dataset, "simple merge of data from the Hipparcos, Tycho-2, UCAC-2 and USNO-B1 catalogues,
            supplemented by photometric information from the 2MASS final release point source catalogue"
        SIRIUS: IR instrument on the IRSF, a 55in reflector telescope at the
            SAAO, the South African Astronomical Observatory
        The flag in the "NIR" column is blank if no NIR data, T if data from 2MASS,
            and S if data from SIRIUS
        IDs in the NOMAD, 2MASS, and GLIMPSE catalogs are given.
        For the RAdeg, DEdeg given in table 1, separations compared to the IAU
        name (ICRS frame, presumably) suggest that these are FK5
    :returns: pandas dataframe
    """
    # The first is like RA,Dec and stuff (mostly unimportant)
    df_TFT = openTFT_single(1).filter(['RAdeg', 'DEdeg', 'PosErr'])
    # Second is X ray stuff
    df_TFT_X = openTFT_single(2)
    # Third is cross matching in IR
    df_TFT_cross = openTFT_single(3)
    for colname in df_TFT_cross:
        df_TFT[colname] = df_TFT_cross[colname]
    for colname in df_TFT_X:
        df_TFT[colname] = df_TFT_X[colname]
    del df_TFT_cross, df_TFT_X
    # These need to be strings so I can look for 'ET' and they come out mixed string / float (NaNs)
    df_TFT['F-ID'].replace(np.nan, '', regex=True, inplace=True)
    # df_TFT['F-ID'] = df_TFT['F-ID'].astype(str)
    # Convert RAdeg and DEdeg to SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RAdeg'], row['DEdeg'], unit=u.deg, frame=FK5)
    df_TFT['SkyCoord'] = df_TFT.apply(make_skycoords, axis=1)
    return df_TFT


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% VPHAS+ &%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openVPHAS_single(tablenumber):
    """
    Open one of the two VPHAS+ tables
    Helper function for openVPHAS_complete
    :param tablenumber: integer, either 5 or 6, for Table 5 or 6
    :returns: pandas dataframe
    """
    with open(catalog_directory+f"VPHAS/table{tablenumber}.dat") as f:
        # VPHAS .dat tables
        header = f.readline().strip('#').split()
        df_VPHAS = pd.read_table(f, comment='#', names=header, delim_whitespace=True,
            dtype={**{x:int for x in ('ID', 'VPHAS_ID')},
                **{x:str for x in ('MSP_ID', 'SIMBAD_ID', 'notes')},
                **{x:"Int64" for x in {'VA_ID', 'TFT_ID'}}
                },
            index_col=('ID' if tablenumber == 5 else 'VPHAS_ID'))
    return df_VPHAS


def openVPHAS_complete():
    """
    Open both VPHAS+ tables and return a combined table
    Both tables from Mohr-Smith et al. 2015 (MNRAS 450,3855–3873).
    This is an optical study done with the VST.
    The full table is 1073 items and is much more than we need
    The estimates for parameters Teff and DM are poorly constrained, according
        to the authors. A0 and Rv are well constrained and informative.
        Teff could still be useful if no other information is available for that
        source.
    They did some cross-matching with Tsujimoto et al. 2007 and
        Vargas Alvarez et al. 2013, though this notably will not catch cross-
        matches between VA13 and TFT if there was no VPHAS detection.
    They include JHKs NIR photometry from Ascenso et al. 2007 when possible,
        and 2MASS when not. There could still be cross-matches between Ascenso
        and Tsuimoto (though these would be NIR-redundant...)
    :returns: pandas dataframe
    """
    # General info, photometry, cross-matching with TFT and VA
    df_VPHAS_info = openVPHAS_single(5)
    df_VPHAS_info.index.name = 'VPHAS_ID'
    # Fitted parameters, including effective temperature and reddening
    df_VPHAS_params = openVPHAS_single(6)
    for colname in df_VPHAS_params:
        df_VPHAS_info[colname] = df_VPHAS_params[colname]
    del df_VPHAS_params
    # Make SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RA'], row['DEC'], unit=u.deg, frame=FK5)
    df_VPHAS_info['SkyCoord'] = df_VPHAS_info.apply(make_skycoords, axis=1)
    return df_VPHAS_info


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&% Vargas Alvarez &%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openVA_simplecatalog():
    """
    Open the main catalog table, Table 2, from the HST survey
    :returns: pandas dataframe
    """
    with open(catalog_directory+"VargasAlvarez2013/tbl2") as f:
        skiplines(f, 10)
        col_intervals, col_labels = read_table_format(f, 18)
        skiplines(f, 6)
        df_VA = pd.read_fwf(f, colspecs=col_intervals, names=col_labels, na_values=("99.999", "0.000"),
            index_col='ID')
    return df_VA


def openVA_ET():
    """
    Open the spectral type table, Table 6, from the HST survey
    This table has spectroscopically derived types as well as some distance
        estimates (that probably shouldn't be trusted)
    :returns: pandas dataframe
    """
    with open(catalog_directory+"VargasAlvarez2013/tbl6") as f:
        skiplines(f, 3)
        header = f.readline().split()[:3]
        header.append("subtype")
        skiplines(f, 1)
        df_VA_ET = pd.read_table(f, delim_whitespace=True, names=header,
            skipfooter=2, engine='python', usecols=[0, 1, 2, 3], index_col='ID')
    return df_VA_ET


def openVA_ST_helper(file_handle, stop_condition, handle_uncertainty=True, final_filter=None):
    rows = []
    continuing = True
    while continuing:
        row = file_handle.readline().strip().split("\t")
        # Get rid of their machine-unfriendly superscripts in the index column
        row[0] = row[0].replace("^", "").replace("a", "").replace("b", "").replace(",", "")
        # Check to see if we hit the stop condition
        if stop_condition(row):
            continuing = False
            continue
        if handle_uncertainty:
            for item in list(row): # Avoid infinite loop
                # Split values and uncertainties
                if '+or-' in item:
                    val, err = item.split(" +or- ")
                    # Use index bc we are modifying the list
                    idx = row.index(item)
                    row[idx] = val
                    row.insert(idx + 1, err)
        if final_filter is not None:
            row = final_filter(row)
        rows.append(row)
    return rows



def openVA_ST():
    """
    Open the OB table, Table 2, from the HST survey
    This table has some late O / early B candidates not included in Table 6
    This table also has temperatures for some of the stars included in Table 6
    It's kind of a messy (machine un-readable) table, so it will be tricky to
        read it in
    :returns pandas dataframe:
    """
    with open(catalog_directory+"VargasAlvarez2013/tbl3") as f:
        skiplines(f, 3)
        header = f.readline().strip().split("\t")
        # They put errors in the same column as the values, so we need to separate them
        for colname in list(header): # Avoid infinite loop
            # These are the columns with uncertainties
            if "EW" in colname or colname == 'T':
                # Put in error columns; use index bc we are modifying the list
                header.insert(header.index(colname) + 1, colname+"_ERR")
        skiplines(f, 2)
        first_rows = openVA_ST_helper(f, lambda r: not r[0].isnumeric())
        second_rows = openVA_ST_helper(f, lambda r: not r[0].isnumeric(),
            handle_uncertainty=False, final_filter=lambda r: r[:4])
    # Make and combine dataframes
    df_VA_ST = pd.DataFrame(first_rows, columns=header).set_index('ID')
    df_temp = pd.DataFrame(second_rows, columns=header[:4]).set_index('ID')
    df_temp['Spectral'] = 'ET'
    df_VA_ST = df_VA_ST.append(df_temp)
    del df_temp
    # Fix a bunch of things
    temp_mask = df_VA_ST['MSP91'].apply(lambda x: '...' not in x)
    df_VA_ST['MSP91'].where(temp_mask, inplace=True)
    # Set types
    df_VA_ST.index = df_VA_ST.index.astype("int64")
    df_VA_ST = df_VA_ST.astype({'T': float, 'T_ERR': float, **{x: str for x in ('MSP91', 'V', 'B - V', 'Spectral')}})
    # Reduce down to certain columns
    wanted_columns = ['MSP91', 'V', 'B - V', 'T', 'T_ERR', 'Spectral']
    df_VA_ST = df_VA_ST[wanted_columns]
    # Now remove duplicate indices (it's VA#664, binary, ST is fine in VPHAS)
    df_VA_ST = df_VA_ST.loc[~df_VA_ST.index.duplicated()]
    # This was messy but it worked!
    return df_VA_ST


def openVA_complete():
    """
    Opens the full catalog from the HST survey and adds in spectral types
        from the spectroscopically analyzed subset of the catalog
    Catalog from Vargas Alvarez et al 2013 (AJ 145:125)
    :returns: pandas dataframe
    """
    # Get Table 5, the full catalog
    df_VA = openVA_simplecatalog()
    # Get Table 6, with spectroscopically determined types
    df_VA_ET = openVA_ET()
    for colname in ['Spectral', 'subtype']:
        df_VA[colname] = df_VA_ET[colname]
    del df_VA_ET
    # Get Table 3, with some extra info like temperatures
    # Also has some stars that aren't in Table 6 because spectral type isn't precisely known
    df_VA_ST = openVA_ST()
    # Update existing unknown spectral types & stuff
    df_VA.update(df_VA_ST, overwrite=False)
    for colname in df_VA_ST.columns:
        if colname not in df_VA.columns:
            df_VA[colname] = df_VA_ST[colname]
    # Make SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RAdeg'], row['DEdeg'], unit=u.deg, frame=FK5)
    df_VA['SkyCoord'] = df_VA.apply(make_skycoords, axis=1)
    return df_VA



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Ascenso %&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


# OPEN ASCENSO
def openAscenso_simplecatalog():
    """
    Open the Ascenso catalog, Table 2
    :returns: pandas dataframe
    """
    # Column names and data are in 2 different files
    with open(catalog_directory+"Ascenso2007/ReadMe") as f:
        skiplines(f, 32)
        col_intervals, col_labels = read_table_format(f, 14)
    with open(catalog_directory+"Ascenso2007/w2phot.dat") as f:
        df_Ascenso = pd.read_fwf(f, colspecs=col_intervals, names=col_labels,
            na_values=("99.99", "9.999"), index_col='Seq',
            dtype={x:str for x in ('RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs')})
    return df_Ascenso


def openAscenso_complete():
    """
    Open the NIR catalog from Ascenso et al. 2007 (A&A 466, 137–149)
    The catalog contains J,H,Ks band photometry from SOFI instrument on the
        ESO New Technology Telescope (NTT) in Chile.
    Vargas Alvarez et al. 2013 and Mohr-Smith et al. 2015 (VPHAS+) both use
        Ascenso's JHKs photometry. VPHAS+ gives JHKs from Ascenso where
        possible, and 2MASS otherwise, though they don't specify which (like
        Tsujimoto does with SIRIUS vs 2MASS).
        VPHAS+ including Ascenso photometry, again, does not preclude
        cross-matches between Ascenso and Tsujimoto, or Ascenso and VA (which
        where not given in the paper), or Ascenso and Rauw(?)
    :returns: pandas dataframe
    """
    df_Ascenso = openAscenso_simplecatalog()
    coords_from_hhmmss(df_Ascenso)
    return df_Ascenso


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Rauw &%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openRauw():
    """
    Open the BV catalog from Rauw et al. 2007 (A&A 463, 981–991) made using
        ANDICAM on CTIO
    The catalog only gives BV photometry.
    Rauw et al. 2007 and 2011 both give some spectral types, but need to be
        put in manually since they never made good tables. There are Rauw 2011
        Objects A B C D .. etc whose coordinates are only given in text, not
        a table.
    """
    with open(catalog_directory+"/Rauw/ReadMe") as f:
        skiplines(f, 92)
        col_intervals, col_labels = read_table_format(f, 11)
    with open(catalog_directory+"/Rauw/table4.dat") as f:
        df_Rauw = pd.read_fwf(f, colspecs=col_intervals, names=col_labels,
            dtype={x:str for x in ('RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs')})
    # Re-index to 1-indexed
    df_Rauw.index = list(x+1 for x in range(len(df_Rauw)))
    coords_from_hhmmss(df_Rauw)
    return df_Rauw


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%& Cross-matching %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

This one monolothic function will do the catalog comparing + crossmatching
It will return all 3 catalog dataframes as well as a new index catalog with all
the matching
"""

def build_catalog():
    """
    Opens VPHAS, TFT, and VA catalogs
    Compile a simple catalog containing the IDs of the stars in other
        catalogs
    Other properties can/will be added later
    This catalog will not be limited to any particular region; later, we will
        make a distance-from-Wd2-center cut and can offer two different catalogs
    This function does most of the heavy lifting with matching catalog indices
        and cross-matching the VA to TFT catalogs. Past this function, you have
        all your indices figured out
    Returns a new DataFrame with a new index, plus the 3 original
        DataFrames (just in case copies were made somewhere in here) with
        indices into the new DataFrame where applicable.
        The new DataFrame is sorted by RA (which I think is the usual practice)
    """
    # Unindexed marker is 0; needs next index >0
    UNINDEXED = 0
    # Indices start from 1 (makes more sense in catalog context)
    # This variable should keep changing
    next_index = 1

    """
    VPHAS only
    """
    # Start with the VPHAS catalog
    df_VPHAS = openVPHAS_complete()
    # Make an ID column that contains the "global catalog" ID, common to all three catalogs
    df_VPHAS['tempID'] = pd.Series([None]*len(df_VPHAS), dtype="Int64") # Does not force float, like np.nan does
    # Mark the known ST objects
    df_VPHAS.loc[df_VPHAS['ST'].notnull(), 'tempID'] = UNINDEXED # Mark as unindexed
    # Mark the "WD2" OB candidates (selected by similar reddening to known Wd2 objects)
    df_VPHAS.loc[df_VPHAS['notes'].apply(lambda x: ("WD2" in x) if isinstance(x, str) else False), 'tempID'] = UNINDEXED
    # Count up the VPHAS candidates
    # notnull works fine with pd.NA
    n_candidates = len(df_VPHAS.loc[df_VPHAS['tempID'].notnull()])
    print("valid VPHAS, [", n_candidates, ']\n')
    # We have a handfull of candidates already

    # We need to start assigning these unique (temporary) IDs, and we will start
    #  from 1 (0 is for unindexed)
    # Assign all n_candidates stars unique IDs
    df_VPHAS.loc[df_VPHAS['tempID'].notnull(), 'tempID'] = list(range(next_index, next_index+n_candidates))
    # Update the next available index
    next_index += n_candidates


    """
    TFT (x VPHAS)
    """
    # Open TFT
    df_TFT = openTFT_complete()
    df_TFT['tempID'] = pd.Series([None]*len(df_TFT), dtype="Int64")
    # Mark the ETs with the "unindexed" value
    df_TFT.loc[df_TFT['F-ID'].apply(lambda row: 'ET' in row), 'tempID'] = UNINDEXED
    print("TFT ETs", len(df_TFT.loc[df_TFT['tempID'].notnull()]))

    # Before indexing the TFT candidates, check for existing VPHAS candidate matches
    # These are the *indexed* VPHAS rows that have TFT IDs
    VPHAS_has_TFT_match = df_VPHAS['TFT_ID'].notnull()
    df_TFTmatched_VPHAScand = df_VPHAS.loc[df_VPHAS['tempID'].notnull() & VPHAS_has_TFT_match]
    print("VPHAS cand X TFT", len(df_TFTmatched_VPHAScand))
    # Assign to TFT the tempIDs already given to VPHAS objects matched with these TFT objects
    df_TFT.loc[df_TFTmatched_VPHAScand['TFT_ID'], 'tempID'] = df_TFTmatched_VPHAScand['tempID'].values
    # This amounts to assigning 22 IDs, 21 of which are to "ET"s and 1 to an unmarked TFT star
    print("non-ET TFTs ID'd as OB in VPHAS", len(df_TFT.loc[(df_TFT['tempID'].notnull()) & (df_TFT['F-ID'].apply(lambda x: 'ET' not in x))]))

    # This leaves 22 unique TFT candidates (22 + 21(ETs in VPHAS) = 43 TFT ETs)
    # Look for these in non-VPHAS flagged objects (see if TFT_ID)
    # Search for thus-far unindexed (TFT ET w/o VPHAS ET) TFT ET objects that are indexed in VPHAS
    """
    FUN FACT: this apparently doesn't create a new DF object, it uses copies somewhere
    That means I can't change values in this unless I do that ".copy()" thing at the end
    See: https://stackoverflow.com/a/49523122
    """
    df_TFTmatched_VPHAS_all_TFTET = df_TFT.loc[(df_TFT['tempID'] == UNINDEXED) & df_TFT.index.isin(df_VPHAS.loc[VPHAS_has_TFT_match, 'TFT_ID'])].copy()
    # Gotta join them to get all those nice columns
    df_TFTmatched_VPHAS_all_TFTET = df_TFTmatched_VPHAS_all_TFTET.join(df_VPHAS.loc[VPHAS_has_TFT_match].reset_index().set_index('TFT_ID'), how='inner', rsuffix='_VPHAS')
    # WOOO THIS WORKED, there are 4 TFT ET candidates with VPHAS matches that were not flagged as ET by VPHAS
    # 3 of these have VA IDs too (see below; 3 of the 4 VA-unknown but VPHAS+TFT id'd ETs)
    # Now give these candidates some IDs before assigning the TFT-unique IDs
    n_candidates = len(df_TFTmatched_VPHAS_all_TFTET) # there are 4
    print("TFT ETs matched with VPHAS but not VPHAS flagged [", n_candidates, "]")
    df_TFTmatched_VPHAS_all_TFTET['tempID'] = list(range(next_index, next_index+n_candidates))
    next_index += n_candidates
    df_TFT.update(df_TFTmatched_VPHAS_all_TFTET['tempID'])
    df_TFTmatched_VPHAS_all_TFTET = df_TFTmatched_VPHAS_all_TFTET.reset_index().set_index('VPHAS_ID')
    df_VPHAS.update(df_TFTmatched_VPHAS_all_TFTET['tempID'])

    # Find the number of unique TFT candidates (not in VPHAS at all)
    n_candidates = len(df_TFT.loc[df_TFT['tempID'] == UNINDEXED])
    print("unique TFT [", n_candidates, "]")
    # Assign unused indexes to these unique candidates
    df_TFT.loc[(df_TFT['tempID'] == UNINDEXED), 'tempID'] = list(range(next_index, next_index+n_candidates))
    # Update the next available index
    next_index += n_candidates
    print("total candidates VPHAS+TFT", next_index-1, '\n')

    """
    VA (x VPHAS)
    """
    # Open VA and get things with spectral types (this seems to be the ET indicator)
    df_VA = openVA_complete()
    df_VA['tempID'] = pd.Series([None]*len(df_VA), dtype="Int64")
    df_VA.loc[df_VA['Spectral'].notnull(), 'tempID'] = UNINDEXED
    print("VA ETs", len(df_VA.loc[df_VA['tempID'].notnull()]))

    # Check for VPHAS candidate matches
    VPHAS_has_VA_match = df_VPHAS['VA_ID'].notnull()
    df_VAmatched_VPHAScand = df_VPHAS.loc[df_VPHAS['tempID'].notnull() & VPHAS_has_VA_match]
    print("VPHAS cand X VA", len(df_VAmatched_VPHAScand))
    print("same^ but also ST known (should be 24 according to paper): ", len(df_VAmatched_VPHAScand.loc[df_VAmatched_VPHAScand['ST'].notnull()]))
    # Assign the existing tempIDs to the VA rows
    df_VA.loc[df_VAmatched_VPHAScand['VA_ID'], 'tempID'] = df_VAmatched_VPHAScand['tempID'].values

    # Figure out what objects were flagged as ET by VPHAS but not known ST in VA
    ## 4 total; 3 are TFT-ETs and 1 seems like a VPHAS literature find (has SIMBAD ID: V* V712 Car)
    print("VPHAS known ST (ET) but VA unknown ST / not ET", len(df_VPHAS.loc[df_VPHAS['VA_ID'].notnull() & df_VPHAS['tempID'].notnull()].reset_index().set_index('VA_ID').loc[df_VA['Spectral'].isnull()].reset_index().set_index('VPHAS_ID')))
    ## not a necessary step in the reduction, just good to know

    # Check for VA candidate matches in VPHAS (should be the "ETs" from Table 3 mainly)
    df_VAmatched_VPHAS_all_VAET = df_VA.loc[(df_VA['tempID'] == UNINDEXED) & df_VA.index.isin(df_VPHAS.loc[VPHAS_has_VA_match, 'VA_ID'])].copy()
    # Join the VPHAS table into these rows
    df_VAmatched_VPHAS_all_VAET = df_VAmatched_VPHAS_all_VAET.join(df_VPHAS.loc[VPHAS_has_VA_match].reset_index().set_index('VA_ID'), how='inner', rsuffix='_VPHAS')
    # This worked, there are 3 and they're all "ETs" from Table 3, so makes sense VPHAS wouldn't have mentioned them
    """
    One has a TFT index
    This means that the "18 unique TFT objects" are actually 17
    I don't think this is the same "17" as the line below, from VPHAS+ (M-S 2015) section 5.1.1:
        "They [TFT] identified 17 new X-rayemitting OB candidates in this larger region"
    I am not sure what to think here
    """
    n_candidates = len(df_VAmatched_VPHAS_all_VAET)
    print("VA ETs matched w/ VPHAS but not VPHAS flagged [", n_candidates, "]")
    df_VAmatched_VPHAS_all_VAET['tempID'] = list(range(next_index, next_index+n_candidates))
    next_index += n_candidates
    # Update the VA, VPHAS, and TFT catalogs with these indices
    df_VA.update(df_VAmatched_VPHAS_all_VAET['tempID'])
    df_VPHAS.update(df_VAmatched_VPHAS_all_VAET.reset_index().set_index('VPHAS_ID')['tempID'])
    df_TFT.update(df_VAmatched_VPHAS_all_VAET.loc[df_VAmatched_VPHAS_all_VAET['TFT_ID'].notnull()].reset_index().set_index('TFT_ID'))

    # Leave the VA-unique matches unindexed until we finish TFT crossmatches

    """
    VA x TFT (crossmatch)
    Here's the next hard step.
    We need sublists of TFT and VA candidates that are not in VPHAS at all
    Then we need to search the entire catalog of one for each item in the other's sublist
    For each separation "min" below a certain threshold
        I used < 0.5 arcsec in my previous reduction
        VPHAS+ says they use < 1 arcsec in their SIMBAD crossmatch
        TFT gives position errors in column "PosErr", which are mostly around 0.5 anyway
    """
    # Unique TFT (no VPHAS)
    df_TFT_ET_notVPHAS = df_TFT.loc[df_TFT['tempID'].notnull() & ~df_TFT.index.isin(df_VPHAS.loc[VPHAS_has_TFT_match, 'TFT_ID'])].copy()
    # Unique VA (no VPHAS)
    df_VA_ET_notVPHAS = df_VA.loc[df_VA['tempID'].notnull() & ~df_VA.index.isin(df_VPHAS.loc[VPHAS_has_VA_match, 'VA_ID'])].copy()
    # If there is no VPHAS ID, there is no chance of a pre-existing crossmatch between the two

    # OK NOW DO THE CROSSMATCH
    TFT_ET_coords = SkyCoord(df_TFT_ET_notVPHAS['SkyCoord'].values)
    TFT_coords = SkyCoord(df_TFT['SkyCoord'].values)
    VA_ET_coords = SkyCoord(df_VA_ET_notVPHAS['SkyCoord'].values)
    VA_coords = SkyCoord(df_VA['SkyCoord'].values)
    # Match the ET subsets to the full catalogs
    # alhamdulillah there is an astropy function for doing most of this
    # tmp is a variable that we aren't using
    TFT_idx, TFT_to_VA_sep, tmp = VA_ET_coords.match_to_catalog_sky(TFT_coords)
    VA_idx, VA_to_TFT_sep, tmp = TFT_ET_coords.match_to_catalog_sky(VA_coords)
    df_VA_ET_notVPHAS['other_min'] = df_TFT.index.values[TFT_idx]
    df_VA_ET_notVPHAS['other_min_sep_as'] = TFT_to_VA_sep.arcsec
    df_TFT_ET_notVPHAS['other_min'] = df_VA.index.values[VA_idx]
    df_TFT_ET_notVPHAS['other_min_sep_as'] = VA_to_TFT_sep.arcsec

    # Now go through and vet these matches
    # Make sure the object claiming the match is also the claimed match's match
    def make_vetting_function(this_catalog, these_coords, other_ET_catalog, other_catalog):
        def vet_match(row):
            """
            Check a match with the object from another catalog
            This match was from this object crossmatched against the whole other catalog
            """
            # Check if this is an ET-ET match
            if (row['other_min'] in other_ET_catalog.index.values):
                # It is an ET-ET match, so get the match's matched index
                other_match_idx = other_ET_catalog.loc[row['other_min'], 'other_min']
            else:
                # other item isn't ET; check it against this whole catalog
                coord = other_catalog.loc[row['other_min'], 'SkyCoord']
                other_match_iloc = coord.match_to_catalog_sky(these_coords)[0]
                other_match_idx = this_catalog.index.values[other_match_iloc]
            return (other_match_idx == row.name)
        return vet_match

    vet_TFT_match = make_vetting_function(df_TFT, TFT_coords, df_VA_ET_notVPHAS, df_VA)
    vet_VA_match = make_vetting_function(df_VA, VA_coords, df_TFT_ET_notVPHAS, df_TFT)
    df_TFT_ET_notVPHAS['vetted'] = df_TFT_ET_notVPHAS.apply(vet_TFT_match, axis=1) & (df_TFT_ET_notVPHAS['other_min_sep_as'] < 2*df_TFT_ET_notVPHAS['PosErr']) # Use stated TFT (Chandra) position errors
    df_VA_ET_notVPHAS['vetted'] = df_VA_ET_notVPHAS.apply(vet_VA_match, axis=1) & (df_VA_ET_notVPHAS['other_min_sep_as'] < 0.5) # Use that 0.5 arcsec cutoff again
    # We have crossmatches! At least a couple for each ET set!
    print("TFT ETs crossmatched with VA", len(df_TFT_ET_notVPHAS.loc[df_TFT_ET_notVPHAS['vetted']]))
    # Now, "update" (w/o overwrite) VA's tempID using TFT (so that ET TFTs with indices are included but VA 0s aren't deleted)
    df_VA.update(df_TFT_ET_notVPHAS.loc[df_TFT_ET_notVPHAS['vetted']].reset_index().set_index('other_min')['tempID'], overwrite=True)
    # Check if any VA-to-TFT matches are unresolved (I think they are done now)
    unresolved_matches = df_VA.loc[df_VA_ET_notVPHAS['vetted'] & (df_VA['tempID'].notnull())]
    print("VA ETs crossmatched with TFT: ", len(unresolved_matches), ", but matched with non-ET TFT?", len(unresolved_matches.loc[unresolved_matches['tempID'] == UNINDEXED]))
    # Both of these have indices, so we're finished with the TFT/VA crossmatches
    # Now assign indices to the unindexed VA ETs

    # Now find the VA ET candidates that neither VPHAS nor TFT catalogged at all
    n_candidates = len(df_VA.loc[df_VA['tempID'] == UNINDEXED])
    print("unique VA [", n_candidates, "]")
    df_VA.loc[(df_VA['tempID'] == UNINDEXED), 'tempID'] = list(range(next_index, next_index+n_candidates))
    next_index += n_candidates
    print("total candidates VPHAS+TFT+VA", next_index-1, '\n')

    # Make "ET masks" for each catalog, now that they're not changing
    def make_ET_mask(df):
        df['ET_MASK'] = df['tempID'].notnull()
    make_ET_mask(df_VPHAS), make_ET_mask(df_TFT), make_ET_mask(df_VA)


    """
    Build the index catalog and reindex it to be sorted by RA
    """
    # Set up the new DataFrame
    new_index_catalog = pd.DataFrame(pd.Series(range(1, next_index), dtype="Int64", name='tempID')).set_index('tempID')
    # Set up a function to do this
    def set_index_column_from(df, index_column_name):
        """
        Return a tempID-indexed column of indices from this DataFrame
        """
        return df.loc[df['ET_MASK']].reset_index().set_index('tempID')[index_column_name].astype(pd.Int64Dtype())
    # Apply it to all three catalogs
    new_index_catalog['VPHAS_ID'] = set_index_column_from(df_VPHAS, 'VPHAS_ID')
    new_index_catalog['VA_ID'] = set_index_column_from(df_VA, 'ID')
    new_index_catalog['TFT_ID'] = set_index_column_from(df_TFT, 'Num')
    # Set up a coordinate-getting function
    def get_coordinates_from(df):
        """
        Return a tempID-indexed column of SkyCoords
        """
        return df.loc[df['ET_MASK']].reset_index().set_index('tempID')['SkyCoord']
    # Reverse order because Series.update is overwrite=True by default
    #   (and if I use DataFrame.update, it affects the dtype of the _ID columns)
    # As a last resort, use TFT if no other is present
    new_index_catalog['SkyCoord'] = get_coordinates_from(df_TFT)
    # Then use VPHAS, since it's optical
    new_index_catalog['SkyCoord'].update(get_coordinates_from(df_VPHAS))
    # If possible, use VA since HST has the highest spatial resolution
    new_index_catalog['SkyCoord'].update(get_coordinates_from(df_VA))
    # Make temporary RA column to sort by
    new_index_catalog['RA'] = new_index_catalog['SkyCoord'].apply(lambda x: x.ra.deg)
    # Sort by RA and then drop that temporary column
    new_index_catalog.sort_values(by='RA', inplace=True)
    new_index_catalog.drop(columns='RA', inplace=True)
    # Reset the index so we don't have trouble with this next step
    new_index_catalog = new_index_catalog.reset_index()
    # Make a new column called FB_ID (FEEDBACK_ID) for these final IDs
    new_index_catalog['FB_ID'] = pd.Series(range(1, next_index), dtype="Int64", name='FB_ID')

    # Now go through and update the catalog DataFrames with the final FB_ID
    # Make a function to make this easier
    def set_FB_ID_by_index(index_column_name):
        """
        :param index_column_name: the name of the column containing the IDs for a given catalog
        """
        return new_index_catalog.loc[new_index_catalog[index_column_name].notnull()].set_index(index_column_name)['FB_ID']

    df_VPHAS['FB_ID'] = set_FB_ID_by_index('VPHAS_ID')
    df_TFT['FB_ID'] = set_FB_ID_by_index('TFT_ID')
    df_VA['FB_ID'] = set_FB_ID_by_index('VA_ID')
    # Drop the temporary ID column from all catalogs to avoid confusion
    df_VPHAS.drop(columns='tempID', inplace=True)
    df_TFT.drop(columns='tempID', inplace=True)
    df_VA.drop(columns='tempID', inplace=True)
    new_index_catalog.drop(columns='tempID', inplace=True)
    # Set the new catalog index to be the final index, FB_ID
    new_index_catalog = new_index_catalog.set_index('FB_ID')
    return df_VPHAS, df_TFT, df_VA, new_index_catalog


def save_indexed_dfs_as_pickle():
    raise RuntimeError("I've already run these on May 5, 2020")
    df_VPHAS, df_TFT, df_VA, new_catalog = build_catalog()
    df_VPHAS.to_pickle(catalog_directory+"VPHAS/VPHAS_DataFrame_indexed.pkl")
    df_TFT.to_pickle(catalog_directory+"Tsujimoto2007/TFT_DataFrame_indexed.pkl")
    df_VA.to_pickle(catalog_directory+"VargasAlvarez2013/VA_DataFrame_indexed.pkl")
    new_catalog.to_pickle(catalog_directory+"Ramsey/index_DataFrame.pkl")


def load_indexed_dfs():
    df_VPHAS = pd.read_pickle(catalog_directory+"VPHAS/VPHAS_DataFrame_indexed.pkl")
    df_TFT = pd.read_pickle(catalog_directory+"Tsujimoto2007/TFT_DataFrame_indexed.pkl")
    df_VA = pd.read_pickle(catalog_directory+"VargasAlvarez2013/VA_DataFrame_indexed.pkl")
    new_catalog = pd.read_pickle(catalog_directory+"Ramsey/index_DataFrame.pkl")
    return df_VPHAS, df_TFT, df_VA, new_catalog


def load_final_catalog_df():
    return pd.read_pickle(catalog_directory+"Ramsey/catalog_may5_2020.pkl")



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&% Main function %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""



def main():
    return catalog_reduction_v2()


def catalog_reduction_v2():
    """
    Updated on 4/14/20
    Indexing finished on 5/5/2020
    """
    df_VPHAS, df_TFT, df_VA, new_catalog = build_catalog()
    raise RuntimeError("I've already run this on May 5, 2020 and saved the pkl")
    df_VPHAS, df_TFT, df_VA, new_catalog = load_indexed_dfs()

    # Nullify "ETs" in spectral types
    df_VA['Spectral'] = df_VA['Spectral'].where(df_VA['Spectral'].astype(str) != 'ET', other="")
    # and change NaNs to empty string bc they map to "nan" strings :/
    df_VA.loc[:, ['Spectral', 'subtype']] = df_VA[['Spectral', 'subtype']].where(df_VA[['Spectral', 'subtype']].notnull(), other="")
    df_VA[['Spectral', 'subtype']] = df_VA[['Spectral', 'subtype']].astype(str)
    # Load in the spectral types from VA and VPHAS, with VPHAS preferred
    new_catalog['Spectral'] = (lambda d: d['Spectral'] + d['subtype'])(df_VA.loc[df_VA['ET_MASK']].reset_index().set_index('FB_ID'))
    new_catalog['Spectral'].update(df_VPHAS.loc[df_VPHAS['ET_MASK']].reset_index().set_index('FB_ID')['ST'].rename('Spectral'))
    # Fill NaNs with O8/B1.5 (late O, early B)
    new_catalog['Spectral'] = new_catalog['Spectral'].where(new_catalog['Spectral'].notnull() & (new_catalog['Spectral'] != ""), other='O8/B1.5')

    df = new_catalog
    # Saves a DataFrame with important "SkyCoord" and "Spectral" fields
    # df.to_pickle(catalog_directory+"Ramsey/catalog_may5_2020.pkl")
    df.to_html("/home/ramsey/Downloads/test.html")
    return


    # return df
    # # print(df_TFT.columns)
    # return

    # Debug plot
    star_coords = SkyCoord(df.SkyCoord.values)
    catalog_utils.plot_coordinates(catalog_utils.irac_data, star_coords)


"""
I removed the original catalog reduction from this file, since this new one
    seems to work. Check earlier git commits for the previous one
    Check at least before April 2020 (file used to be called readcat.py)
"""

if __name__ == "__main__":
    args = main()
