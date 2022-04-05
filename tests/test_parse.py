"""
Test the string spectral type parsing functions

These took me so long to write lol, and these tests should help debug anything that can go wrong

Copied tests from sptype_tests.py (where I dumped all tests before reorganizing code)

Created: April 5, 2022
"""
__author__ = "Ramsey Karim"

import unittest

import os
import pandas as pd

from scoby import spectral, config

test_catalog_fn = os.path.join(config.test_data_path, "OBradec.csv")


def load_test_data():
    """
    Helper function to load the test catalog for these tests
    :returns: pandas DataFrame
    """
    return pd.read_csv(test_catalog_fn)


class TestLoadData(unittest.TestCase):
    """
    Make sure that test data shows up (and do I need to store the pkl file on github or can I store the CSV which is
    like 1/10th the size?)
    Result: I don't need the SkyCoords from the pkl file, so I will only store the CSV which has the spectral types I
    need for this test.
    """

    def test_load_csv(self):
        print(load_test_data())


class TestParse(unittest.TestCase):
    """
    There are only three tests, so here they are
    """

    def test_st_parse_slashdash(self):
        """
        This tests using "uncertainty" in spectral type when the uncertainty was specified in the stated type.
        Catalogs use types like "O4-5" or "O8III/V" or "O9/B0", and while we could just pick one, we don't know the
        author's intention. So instead of picking one ourselves, we'll store both and randomly select when we need to
        "realize" the cluster.

        Tested the new subtype dash behavior, it looks like it works! (this is an old comment)
        """
        cat = load_test_data()
        tests = ['O8/B1.5V', 'O8-B1.5V', cat.SpectralType[19], cat.SpectralType[5], cat.SpectralType[7], 'B5/6.5III',
                 'O4II/III', cat.SpectralType[26], cat.SpectralType[27], 'O4-5.5V/III*', "O5:V"]

        for t in tests:
            l = spectral.parse_sptype.st_parse_slashdash(t)
            print(t, '\t', l)
            for x in l:
                print('\t', spectral.parse_sptype.st_parse_type(x))
            print()

    def test_st_adjacent(self):
        """
        Test grabbing the adjacent spectral types (like a half step away in either direction)
        """
        tests = ['O8', 'O4.5V', 'O2If', 'B2.5', 'O9.5']
        print("Finding adjacent stellar types for \"inherent\" uncertainty")
        for t in tests:
            print(t, end=': ')
            t = spectral.parse_sptype.st_parse_type(t)
            t = tuple(x for x in t if x)
            print(t, end=' --> ')
            print(spectral.parse_sptype.st_adjacent(t))

    def test_full_st_parse(self):
        """
        This is no longer the "full" parse (? so where is it? that was an old comment, context lost to history)

        Revisiting on April 5, 2022:
        Well, this definitely tests something, I'm just not sure what I meant by that
        I think what I meant up there was that the "full" parse is part of STResolver now, which is tested separately
        TODO I think what I will do is delete this function since it seems pretty specific to my project
        """
        cat = load_test_data()
        count = 0
        for st in cat.SpectralType:
            assert isinstance(st, str)
            print(f"{st:30s}", end=": ")
            if st == 'ET':
                st = 'O9.5V'
                count += 1
            stars = [[spectral.parse_sptype.st_parse_type(x) for x in spectral.parse_sptype.st_parse_slashdash(y)] for y
                     in spectral.parse_sptype.st_parse_binary(st)]
            types = [x[0][0:2] for x in stars]
            if types and all(types[0]):
                print([i[0] + i[1] for i in types])
            else:
                print(stars)
        print(f"There are {count} ET stars")
