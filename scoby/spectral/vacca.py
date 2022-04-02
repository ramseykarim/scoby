"""
Implementation of Vacca 1996 spectral type calibrations

Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020

Created: June 2, 2020
"""
__author__ = "Ramsey Karim"

from . import parse_sptype

def vacca_calibration(spectral_type_tuple, characteristic):
    """
    Coding up Vacca 1996 Table 4
    Created: November 2019
    Rewritten: April 29, 2020
    Replacing the old "vacca_calibration" function that has a very different
        call signature than S03_OBTables.lookup_characteristic.
    :param spectral_type_tuple: spectral type expressed as a tuple
    :param characteristic: string characteristic descriptor. For Vacca,
        this would be "Teff" or "log_g"
    """
    spectral_type_number = parse_sptype.st_to_number(spectral_type_tuple)
    luminosity_class_number = parse_sptype.lc_to_number(spectral_type_tuple[2])
    # Original vacca_calibration code below here
    if luminosity_class_number == 1 and spectral_type_number > 9.5:
        raise RuntimeWarning('LC I(a) stars above O9.5 are not supported by this calibration.')
    coefficients = {
        'Teff': (59.85, -3.10, -0.19, 0.11),
        'log_g': (4.429, -0.140, -0.039, 0.022), # evolved g
    }
    multiplier = {'Teff': 1e3, 'log_g': 1}
    S, L = spectral_type_number, luminosity_class_number
    A, B, C, D = coefficients[characteristic]
    return (A + (B*S) + (C*L) + (D*S*L)) * multiplier[characteristic]


def get_catalog_properties_vacca(cat, characteristic):
    """
    DataFrame-friendly version of vacca_calibration
    cat is a pandas DataFrame with a SpectralType_ReducedTuple column
    """
    cat[characteristic+"_V96"] = cat.SpectralType_ReducedTuple.apply(vacca_calibration)
