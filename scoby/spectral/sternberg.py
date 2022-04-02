"""
Script to read the Sternberg et al spectral types tables

Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020

Created: June 2, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import parse_sptype

from .. import utils

sternberg_path = f"{utils.misc_data_path}SpectralTypes/Sternberg/"
spectypes_table = f"{sternberg_path}spectypes.txt"
column_name_table = f"{sternberg_path}colnames.txt"


def table_name(spectral_subtype):
    """
    Filenames of Sternberg tables
    """
    # Spectral subtypes V, III, and I are available
    return f"{sternberg_path}class{spectral_subtype}.txt"


def load_tables_df():
    """
    Load Sternberg tables as DataFrames into a dictionary
    """
    # Load column names and units
    with open(column_name_table, 'r') as f:
        colnames = f.readline().split()
        units = f.readline().replace('1/cm2s', 'cm-2s-1').split()
    # Load spectral types;;;; DO WE USE THESE EVER??
    # Editorial note (April 29, 2020) it seems these are just a list of spectral types, which doesn't seem all that special
    with open(spectypes_table, 'r') as f:
        spectypes = f.readline().split()
    # Create units table
    col_units = pd.DataFrame(units, index=colnames, columns=['Units'])
    # Create star tables
    lc_dfs = {lc: pd.read_table(table_name(lc), index_col=0, names=colnames) for lc in parse_sptype.luminosity_classes}
    # Fix the Teff column comma issue
    for lc in parse_sptype.luminosity_classes:
        lc_dfs[lc]['Teff'] = lc_dfs[lc].apply(lambda row: float(row['Teff'].replace(',', '')), axis=1)
    return lc_dfs, col_units


def fit_characteristic(df_subtype, characteristic):
    # Get characteristic interp (i.e. "Teff")
    # from df_subtype (i.e. spectral_type_df_dict["III"])
    independent, dependent = np.array([st_to_number(x) for x in df_subtype.index]), df_subtype[characteristic]
    interp_from_number = interp1d(independent, dependent, kind='linear')
    def interp_function(spectral_type):
        try:
            return interp_from_number(st_to_number(spectral_type))
        except:
            return np.nan
    return interp_function


class S03_OBTables:
    """
    This is a wrapper class for the Sternberg tables (Sternberg et al. 2003)
    You should be able to give the "spectral type tuple", which is
        something like ("O", "7.5", "V"), and you can call for characteristics
        in the Sternberg tables.
    The characteristic lookup will first try to match your spectral type
        exactly, and next will interpolate using the "spectral type to number"
        map that Vacca used. This is NOT Vacca's Teff, log_g calibration (though
        Sternberg says they use Vacca's calibrations for some things.)
    """
    def __init__(self):
        self.star_tables, self.column_units = load_tables_df()
        self.memoized_interpolations = {}
        self.memoized_type_names = {}

    def lookup_characteristic(self, spectral_type_tuple, characteristic):
        # Spectral type tuple can be 3 or 4 elements (4th is peculiarity, ignored)
        # Characteristic is a valid column name, i.e. Teff or log_L
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        # Discard peculiarity and sanitize input
        spectral_type_tuple = spectral_type_tuple[:3]
        spectral_type_tuple = parse_sptype.sanitize_tuple(spectral_type_tuple)
        if not spectral_type_tuple:
            return 0
        lettertype, subtype, lumclass = spectral_type_tuple
        # Get subtype DataFrame
        df_lumclass = self.star_tables[lumclass]
        if lettertype+subtype in df_lumclass.index:
            # If the spectral type is in this table, return value for characteristic
            return df_lumclass[characteristic].loc[lettertype+subtype]
        else:
            # If type is not in table, interpolate between types using number approximation
            if spectral_type_tuple in self.memoized_interpolations:
                # Check if there is a saved interpolation for this type/subtype
                interp_function = self.memoized_interpolations[spectral_type_tuple]
            else:
                # If we don't, make one
                interp_function = fit_characteristic(df_lumclass, characteristic)
                self.memoized_interpolations[spectral_type_tuple] = interp_function
            # Use interpolation to return a value for the characteristic
            return interp_function(spectral_type_tuple)

    # self.sanitize_tuple used to be here but was moved to parse_sptype.py

    def lookup_units(self, characteristic):
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        return self.column_units['Units'].loc[characteristic]


def get_catalog_properties_sternberg(cat, characteristic):
    """
    DataFrame-friendly version of S03_OBTables.lookup_characteristic
    These are the Sternberg tables
    cat is a pandas DataFrame with a SpectralType_ReducedTuple column
    June 2, 2020: Can probably also delete this function
    TODO: delet this
    """
    sternberg_tables = S03_OBTables()
    cat[characteristic+'_S03'] = cat.SpectralType_ReducedTuple.apply(sternberg_tables.lookup_characteristic, args=(characteristic,))
