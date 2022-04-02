"""
Spectral Type Table: STTable
Common wrapper class for Sternberg and Martins tables, since they're very similar

Created: June 8, 2020
"""
__author__ = "Ramsey Karim"


import numpy as np
from scipy.interpolate import interp1d

from ... import misc_utils

from .. import utils

from . import parse_sptype


INVALID_SPTYPE_PROPERTY_FLAG = np.nan


def sanitize_characteristic(function_to_decorate):
    """
    Decorator function to check the characteristic argument against the
    available characteristics
    The "characteristic" parameter must be the first* argument
    The real first argument is the instance of STTable
    """
    def decorated_function(*args):
        self = args[0]
        characteristic = args[1]
        if characteristic in self.column_units.index:
            return function_to_decorate(*args)
        else:
            msg = f"Try one of: {self.column_units.index.values}"
            raise RuntimeError(f"{characteristic} is not a recognized column name.\n{msg}")

    return decorated_function


def fit_characteristic(df_lumclass, characteristic):
    """
    Get interpolation function for a certain "characteristic" (column) for the
    given table as a function of the index (transformed into a number)
    The interpolation is somewhat smart; it will interpolate if possible,
    extrapolate if reasonable, and return INVALID_SPTYPE_PROPERTY_FLAG if
    unreasonable.
    """
    independent_variable = np.array([parse_sptype.st_to_number(x) for x in df_lumclass.index])
    dependent_variable = df_lumclass[characteristic]
    interp_from_number = interp1d(independent_variable, dependent_variable, kind='linear')
    # Get the limits in which interpolation is valid
    lo, hi = independent_variable.min(), independent_variable.max()
    # Prepare a linear extrapolation based on the last three points
    lastN = 3
    extrap_fit = np.polyfit(independent_variable[-lastN:], dependent_variable[-lastN:], deg=1)
    def interp_function(spectral_type):
        spectral_type_number = parse_sptype.st_to_number(spectral_type)
        # Check if it's past the interp limits and polyfit if it is, up to a
        # certain point (hardcoded), else return INVALID_SPTYPE_PROPERTY_FLAG
        # Note: "high" limit is "late-type" limit, "low" limit is "early-type"
        if lo <= spectral_type_number <= hi:
            # Within interpolation bounds; interpolate
            return float(interp_from_number(spectral_type_number))
        elif hi < spectral_type_number <= hi + 3:
            # Extrapolate, but not too far from high limit
            return misc_utils.polynomial(spectral_type_number, extrap_fit)
        else:
            # Either less than low limit (unlikely) or too far past high limit
            return INVALID_SPTYPE_PROPERTY_FLAG
    return interp_function


class STTable:
    """
    Spectral Type Table
    Wrapper class for Sternberg and Martins tables.
    This class appeared previously as S03_OBTables in sternberg.py. Since
    reading in the Martins tables is a nearly identical task, I have generalized
    this class.

    The main class method here is lookup_characteristic.
    You should be able to give the "spectral type tuple", which is
    something like ("O", "7.5", "V"), and you can call for characteristics
    in the table.
    The characteristic lookup will first try to match your spectral type
    exactly, and next will interpolate using the "spectral type to number"
    map that Vacca used.
    """
    def __init__(self, table_dict, col_units):
        self.table_dict = table_dict
        self.column_units = col_units
        # Implement some basic memoization, since we know a lot of types
        # will be redundant
        self.memoized_interpolations = {}

    @sanitize_characteristic
    def lookup_characteristic(self, characteristic, spectral_type_tuple):
        """
        The primary method of this class, a lookup function for spectral
        type characteristics.
        Note that the argument order was swapped around from the original
        Sternberg-only version of this function.

        Updated August 4, 2020:
        If the luminosity class is II, change it to III.
        II is a bright giant, III is a normal giant (according to Wikipedia,
        Stellar_classification). II seems closer to III than I (supergiants).

        :param characteristic: valid column name of table
        :param spectral_type_tuple: can be 3 or 4 elements
            (4th is ignored). Tuple gets sanitized by parse_sptype.sanitize_tuple.
            Should be in the form recognized by the parse_sptype module
        """
        # Discard peculiarity and sanitize input
        spectral_type_tuple = parse_sptype.sanitize_tuple(spectral_type_tuple[:3])
        if not spectral_type_tuple: # Handle sanitize_tuple: False
            # Need a good flag for nonstandard spectral types, etc
            return INVALID_SPTYPE_PROPERTY_FLAG
        # Split up tuple
        lettertype, subtype, lumclass = spectral_type_tuple
        # If luminosity class is II, use III and print a message
        if lumclass == 'II':
            # print(f"{__name__}:\nLuminosity class II (bright giant) encountered; defaulting to III (normal giant) for Star: {parse_sptype.st_tuple_to_string(spectral_type_tuple)}, Charateristic: {characteristic}")
            lumclass = 'III'
        # Get luminosity class DataFrame from self.table_dict
        df_lumclass = self.table_dict[lumclass]
        if lettertype+subtype in df_lumclass.index:
            # If spectral type is in this table, just return the characteristic
            return df_lumclass[characteristic].loc[lettertype+subtype]
        else:
            # If type not in table, interpolate between types using sptype number.
            # "characteristic-lumclass tuple" uniquely selects a curve
            cltup = (characteristic, lumclass)
            if cltup in self.memoized_interpolations:
                # Check if there is a saved interpolation for this type
                interp_function = self.memoized_interpolations[cltup]
            else:
                # If there isn't, make one and save it for next time too
                interp_function = fit_characteristic(df_lumclass, characteristic)
                self.memoized_interpolations[cltup] = interp_function
            # Use interpolation to return a value for the characteristic
            return interp_function(spectral_type_tuple)

    @sanitize_characteristic
    def lookup_units(self, characteristic):
        """
        Find the units for the given characteristic. These units should be
        able to be passed directly into astropy.units.Unit and recognized.
        :param characteristic: valid column name of table
        """
        return self.column_units['Units'].loc[characteristic]
