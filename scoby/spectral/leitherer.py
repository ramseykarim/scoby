"""
Open and use Tables 1 and 2 from Leitherer et al 2010 (ApJSup 189:309–335)
Leitherer et al use WM-BASIC (Pauldrach et al 2001) to generate vinf and Mdot for
a grid of L and Teff spanning early types.

The Sternberg models don't take clumping into account, so their mass loss rates
are likely overestimated (Puls et al 2008).
Leitherer et al are responsible for Starburst 99.

Created: June 8, 2020
"""
__author__ = "Ramsey Karim"


import numpy as np
import pandas as pd
from collections.abc import Sequence # abc = Abstract Base Class

from . import sttable
from .. import utils

leitherer_path = f"{utils.misc_data_path}SpectralTypes/Leitherer/"


def open_single_table(n, just_units=False):
    """
    Open the table as a pandas DataFrame
    :param n: table number from the Leitherer 2010 paper; must be 1 or 2.
    :param just_units: if True, returns only a DataFrame with the units.
        Units are astropy.units.Unit readable
    :returns: pandas DataFrame
    """
    fn = f"{leitherer_path}tbl{n}.dat"
    with open(fn, 'r') as f:
        for i in range(3):
            # Skip first 3 lines
            f.readline()
        colnames = f.readline().strip().split('\t')
        # Extract units and clean up column names
        units = []
        # Also set up dtypes
        dtypes = {}
        for i in range(len(colnames)):
            c = colnames[i]
            if '(' in c:
                if ')' not in c:
                    raise RuntimeError(f"Malformed column name: {c}")
                unit_str = c[c.index('(')+1 : c.index(')')].replace('^', '').replace('_', '')
                new_c = c[:c.index('(')].strip().replace(' ', '_')
                colnames[i] = new_c
                units.append(unit_str)
            else:
                new_c = c
                units.append('')
            if new_c in ['Sp. Type', 'T_eff']: # T_eff has commas; fix later
                dtypes[new_c] = str
            elif new_c == 'Model':
                dtypes[new_c] = int
            else:
                dtypes[new_c] = float
        # Get rid of index column; pandas "ignores" it; column names shift
        # colnames = colnames[1:]
        # units = units[1:]
        if just_units:
            # Return early with just the units DataFrame
            return pd.DataFrame(units, index=colnames, columns=['Units'])
        # Read the rest of the file as a CSV
        # index_col=False solves issue of delimiters at the end of lines (not sure precisely why)
        df = pd.read_csv(f, sep='\t', names=colnames, index_col=False, dtype=dtypes).set_index('Model')
    return df


def open_tables():
    """
    Open and combine the Leitherer tables
    :returns: pandas DataFrame
    """
    tbl1 = open_single_table(1)
    tbl1['T_eff'] = tbl1['T_eff'].apply(lambda s: s.replace(',', '')).astype(float)
    tbl2 = open_single_table(2)
    units_df = open_single_table(1, just_units=True).append(open_single_table(2, just_units=True))
    units_df = units_df.loc[~units_df.index.duplicated(keep='first')]
    for c in tbl2.columns:
        tbl1[c] = tbl2[c]
    return tbl1, units_df


class LeithererTable:
    """
    Very similar to STTable, but handles the Leitherer tables.
    These are 2D grids in L and T, so the STTable 1D interpolation won't work.
    """

    def __init__(self):
        tbl, units_df = open_tables()
        self.table = tbl
        self.column_units = units_df
        # Make the XY Delaunay for quicker interpolation. Work in log T
        self.xy_delaunay = utils.delaunay_triangulate(np.log10(self.table['T_eff']),
            self.table['log_L'])
        self.memoized_interpolations = {}

    @sttable.sanitize_characteristic
    def lookup_characteristic(self, characteristic, T, logL):
        """
        :param characteristic: a recognized column name from the Leitherer
            tables
        :param T: linear effective temperature in Kelvins
        :param logL: log luminosity in solLum
        :returns: the value of this characteristic, reasonably interpolated.
            NaN if unreasonable.
        """
        # Check if we have an existing interpolation for this characteristic
        if characteristic in self.memoized_interpolations:
            interp_function = self.memoized_interpolations[characteristic]
        else:
            # Make one
            z = self.table[characteristic]
            interp_function = utils.fit_characteristic(self.xy_delaunay, z)
            self.memoized_interpolations[characteristic] = interp_function
        # Work in log T
        result = interp_function(np.log10(T), logL)
        # Need to fix the numpy scalar issue
        if hasattr(result, 'ndim') and result.ndim == 0:
            # This is likely a numpy scalar; return a float
            return float(result)
        else:
            # This might be an array; just return it
            # Or it doesn't have ndim so just return it and see what happens
            return result
