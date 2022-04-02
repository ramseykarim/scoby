"""
Implementation of the Martins 2005 spectral type calibrations
Martins et al 2005 used CMFGEN to re-do the Vacca 1996 calibrations with
a)  full NLTE treatment, b) winds, and c) metals & line blanketing
Vacca's work didn't have winds or metals, according to Martins.

Martins also used observational data to compare to the theoretical calibrations.
I am using the theoretical tables because that seems more self-consistent.
See Figures 2-4 in Martins et al 2005.

These tables have a lot of the same info as Sternberg et al 2003, but Martins
only covers O stars, where Sternberg covers early B.

I could use these for O stars and Sternberg for B.
Issue: Sternberg's B stars are hotter than Martins' late O stars... Oh well.
I'll use Vacca for B star temperature & gravity; I can use Sternberg if I really
need total luminosity or something, but I should be able to get ionizing flux
or photon numbers from the PoWR spectra. Winds will come from Leitherer.
I will need luminosity to get winds from Leitherer.... I can extrapolate from
late O stars...

Created: June 2, 2020
"""

import pandas as pd

from . import parse_sptype
from .. import utils

martins_path = f"{utils.misc_data_path}SpectralTypes/Martins/"

table_info = {
    # Some nice, clean hardcoding
    # (skiplines, lumclass)
    1: (8, 'V'),
    2: (7, 'III'),
    3: (7, 'I'),
}

replace_key = {
    # Specific to this LaTeX table we have to deal with.
    '\\': '', '$': '',
    'rm ': '',
    '{': '', '}': '',
    '_': ' ',
    '[': '', ']': '',
    '^': '',
    'rsun': 'Rsun', 'msun': 'Msun', 'lL': 'log L',
    ' spec': '', 'teff': 'Teff', 'mv': 'Mv',
    'log ': 'log_',
}

def split_line_and_apply_replace_key(line):
    line = line.strip() # Get rid of the newline
    for k in replace_key:
        line = line.replace(k, replace_key[k]) # LaTeX-specific hardcode-y replacement
    # Remove surrounding whitespace again and replace spaces with underscores
    return [x.strip() for x in line.split('&')][:-1] # Address dangling ampersand


def load_table_df(n, skiplines, just_units=False):
    """
    Load one of the Martins et al 2005 Tables 1-3 as a pandas DataFrame.
    These are LaTeX tables, unfortunately the only machine-readable table
    available from the online article. They need a lot of hardcoding.
    :param n: table number, must be 1, 2, or 3
    """
    with open(f"{martins_path}table{n}.tex", 'r') as f:
        for i in range(skiplines):
            f.readline() # Skip table setup
        colnames = split_line_and_apply_replace_key(f.readline())
        units = split_line_and_apply_replace_key(f.readline())
        if just_units:
            # Same files/operations for units, so option to stop here
            return pd.DataFrame(units, index=colnames, columns=['Units'])
        f.readline() # Skip
        df_list = []
        for i in range(12):
            # These lines are pretty well-behaved, but still need some fixing
            line = [float(x.strip()) for x in f.readline().replace('--', '-').replace('~', '').split('&')[:-1]]
            # Convert spectral type number (e.g. 3) to letter/number combo (e.g. O3)
            line[0] = "".join(parse_sptype.number_to_st(line[0]))
            df_list.append(line)
    return pd.DataFrame(df_list, columns=colnames).set_index('ST')


def load_tables_df():
    """
    Load all the Martins tables and also the units. All tables have the same indices,
    column names, and column units.
    This is the same return format as sternberg.load_tables_df.
    Returns: dict(str(lum_class), DataFrame(values)), DataFrame(units)
    """
    df_dict = {table_info[x][1]: load_table_df(x, table_info[x][0]) for x in table_info}
    unit_df = load_table_df(1, table_info[1][0], just_units=True)
    # Just one hardcoded edit
    unit_df.loc['log_L', 'Units'] = 'Lsun'
    return df_dict, unit_df


"""
Sternberg-like functions
"""
