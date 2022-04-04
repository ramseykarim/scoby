"""
Testing the Martins, Leitherer, etc. table loading
This involves (very basic) testing the filepath configuration in config.py and then checking that loading works
and that the tables make sense.

I'm using Python's unittest module to do this.
I am copying over some tests from sptype_tests.py (which may or may not exist when I'm done reorganizing) since that
was where I dumped all tests before trying to organize this better.

Created: April 4, 2022
"""
__author__ = "Ramsey"

import unittest
import os

import numpy as np
import matplotlib.pyplot as plt

# Hopefully this works!
import scoby


class TestConfig(unittest.TestCase):

    def test_run_config(self):
        # Config has a little bit of code in it, so I'm checking to see if it imports alright
        from scoby import config
        self.assertTrue(os.path.isdir(config.temp_dir))
        self.assertTrue(os.path.isdir(config.powr_path))

    def test_plot_sptype_calibration_stuff(self):
        dfs, col_units = scoby.spectral.sternberg.load_tables_df()
        dfs2, col_units2 = scoby.spectral.martins.load_tables_df()
        colors = {'I': 'blue', 'III': 'green', 'V': 'red'}
        plt.figure(figsize=(14, 9))
        Teff, log_g = 'Teff', 'log_g'
        # The characteristics to go on each axis
        char_x, char_y = Teff, "log_L"
        for lc in scoby.spectral.parse_sptype.luminosity_classes:
            st_sb03 = np.array([scoby.spectral.parse_sptype.st_to_number(i) for i in dfs[lc].index])
            st_m05 = np.array([scoby.spectral.parse_sptype.st_to_number(i) for i in dfs2[lc].index])
            independent, dependent = dfs[lc][char_x], dfs[lc][char_y]
            ind2, dep2 = dfs2[lc]['Teff'], dfs2[lc][char_y]
            plt.plot(independent, dependent, 'x', color=colors[lc], label='S03')
            plt.plot(ind2, dep2, '.', color=colors[lc], label='M05')
            fit = scoby.spectral.sternberg.interp1d(independent, dependent, kind='linear')
            x = np.linspace(independent.min(), independent.max(), 50)
            plt.plot(x, fit(x), '--', label=f'fit to Sternberg+2003 class {lc}', color=colors[lc])

        plt.legend()
        plt.ylabel(char_y), plt.xlabel(char_x)
        plt.gca().invert_xaxis()
        plt.show()
