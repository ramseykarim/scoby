"""
Tests that check parsing_utils.py

Created: April 7, 2022
"""
__author__ = "Ramsey Karim"

import unittest

from scoby import parsing_utils

from astropy.coordinates import SkyCoord


class TestParsing(unittest.TestCase):

    def test_hhmmss(self):
        input_coordinate = "102400.52-574444.6"
        desired_output = ("10:24:00.52", "-57:44:44.6")
        real_output = parsing_utils.convert_hhmmss(input_coordinate)
        self.assertEqual(desired_output, real_output)
        input_coordinate = '55555.0+30201.23'
        desired_output = ("5:55:55.0", "+3:02:01.23")
        real_output = parsing_utils.convert_hhmmss(input_coordinate)
        self.assertEqual(desired_output, real_output)
        with self.assertRaises(ValueError):
            # original input raises error, no good for SkyCoord
            SkyCoord(input_coordinate, unit=('hourangle', 'deg'))
        # function output is ok for SkyCoord, no error
        SkyCoord(*real_output, unit=('hourangle', 'deg'))
