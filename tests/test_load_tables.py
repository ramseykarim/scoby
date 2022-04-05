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

from astropy import units as u

import scoby


class TestConfig(unittest.TestCase):
    """
    File path configuration and loading files and making sure tables look ok (have numbers in them)
    """

    def test_run_config(self):
        # Config has a little bit of code in it, so I'm checking to see if it imports alright
        from scoby import config
        self.assertTrue(os.path.isdir(config.temp_dir))
        self.assertTrue(os.path.isdir(config.powr_path))

    def test_martins_calibration_load(self):
        df1, u1 = scoby.spectral.martins.load_tables_df()
        df2, u2 = scoby.spectral.sternberg.load_tables_df()
        print("\n<MARTINS load>")
        print(u2.index)
        for i in u2.Units:
            print(i, u.Unit(i))
        print(u1.index)
        for i in u1.Units:
            print(i, u.Unit(i))
        print("</MARTINS load>\n")

    def test_martins_calibration(self):
        print("\n<MARTINS calib>")
        df1, u1 = scoby.spectral.martins.load_tables_df()
        print(df1['V'])
        df2, u2 = scoby.spectral.sternberg.load_tables_df()
        print(df2['V'])
        print('-----')
        print(u1)
        print(u2)
        print("</MARTINS calib>\n")

    def test_sttables(self):
        """
        I used this to confirm that STTable gives good looking results
        for both Sternberg and Martins
        """
        df1, u1 = scoby.spectral.martins.load_tables_df()
        df2, u2 = scoby.spectral.sternberg.load_tables_df()
        stt1 = scoby.spectral.sttable.STTable(df1, u1)
        stt2 = scoby.spectral.sttable.STTable(df2, u2)

    def test_leitherer_open(self):
        """
        open_tables works, as far as I can tell
        """
        df1, u1 = scoby.spectral.leitherer.open_tables()
        print("\n<LEITHERER>")
        print(df1)
        print(u1)
        print("</LEITHERER>\n")
        return df1, u1
