"""
Tests that check the STResolver

Copied these tests from sptype_tests.py

Created: April 5, 2022
"""
__author__ = "Ramsey Karim"

import unittest

import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

from scoby import spectral

from .utils import create_debug_img_metadata, debug_img_path, DEBUG_SHOW


class TestSTResolver(unittest.TestCase):
    """
    Tests that check the STResolver object
    """

    def test_WR(self):
        print("<test_WR>")
        spectral.stresolver.random.seed(1312)
        np.random.seed(1312)
        powr_grids = {x: spectral.powr.PoWRGrid(x) for x in ('OB', "WNE", "WNL", "WNL-H50")}
        cal_tables = spectral.sttable.STTable(*spectral.sternberg.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        s = spectral.stresolver.STResolver("WN6ha")
        s.link_calibration_table(cal_tables)
        s.link_leitherer_table(ltables)
        s.link_powr_grids(powr_grids)

        st = s.spectral_types['WN6ha'][0]
        print(st)
        print("Rtrans: ", np.log10(spectral.powr.PoWRGrid.calculate_Rt(*st[5:9])))
        s.populate_all()
        # for pm in s.powr_models['WN6ha']:
        #     print(pm)
        print(s.get_mass_loss_rate())
        print("</test_WR>\n")

    def test_Catalog(self):
        print("<test_Catalog>")
        # def f(x):
        #     # function to pass NaNs and print MODEL from DataFrames
        #     try:
        #         return x['MODEL']
        #     except:
        #         return "[NO MODEL]"
        spectral.stresolver.random.seed(1312)
        np.random.seed(1312)
        powr_grids = {x: spectral.powr.PoWRGrid(x) for x in spectral.powr.AVAILABLE_POWR_GRIDS}
        cal_tables = spectral.sttable.STTable(*spectral.sternberg.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()
        # cat = pd.read_pickle(f"{parse.catalog_directory}Ramsey/catalog_may5_2020.pkl")
        # print(cat.columns)
        # tests = cat.Spectral.values
        # print(tests)
        tests = ['O5+O7+B1', 'O5-6.5III+O3V', 'O4I/III', 'B1-2', 'C*', 'O4V+PMS']

        spectral.stresolver.UNCERTAINTY = False  # toggle the half-type/sampling
        catr = spectral.stresolver.CatalogResolver(tests,
                                                   calibration_table=cal_tables, leitherer_table=ltables,
                                                   powr_dict=powr_grids)
        print(catr)
        for s in catr.star_list:
            print(s.spectral_types)
        # mdot, mdot_e = catr.get_mass_loss_rate(nsamples=10)
        # print(mdot)
        # print(mdot_e)
        fluxes = catr.get_array_FUV_flux()
        fluxes = catr.get_array_ionizing_flux()
        for f in fluxes:
            print(f"{f[0]:.2E}, ({f[1][0]:.2E}, {f[1][1]:.2E})")

        # count = 0
        # for t in tests:
        #     if t == "ET":
        #         t = "O7.5/B1"
        #     s = spectral.stresolver.STResolver(t)
        #     print(t, '\n\t', s)
        #     print('\t', s.__repr__())
        #     s.link_calibration_table(cal_tables)
        #     s.link_leitherer_table(ltables)
        #     s.link_powr_grids(powr_grids)
        #     s.populate_mass_loss_rate()
        #     print(s.get_mass_loss_rate())
        #     # s.rollcall(dictionary=s.mdot)
        #     # print(s.mdot)
        #     # print(f">>>>>>>> {s.random_possibility(s.mdot):.3E}")

        # count += 1
        # if count > 15:
        #     break
        # return s
        print("</test_Catalog>\n")

    def test_L(self):
        print("<test_L>")
        spectral.stresolver.random.seed(1312)
        np.random.seed(1312)
        powr_grids = {x: spectral.powr.PoWRGrid(x) for x in spectral.powr.AVAILABLE_POWR_GRIDS}
        cal_tables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        tests = ['O6', 'O6', 'O6']
        TLs = [(30000., 5.2), (32050., 4.9), (28001., 5.1)]

        spectral.stresolver.UNCERTAINTY = False  # toggle the half-type/sampling
        catr = spectral.stresolver.CatalogResolver(tests,
                                                   calibration_table=cal_tables, leitherer_table=ltables,
                                                   powr_dict=powr_grids)
        print("Intrinsic half-type uncertainty is turned ", end="")
        if spectral.stresolver.UNCERTAINTY:
            print("ON, so there should be nonzero uncertainties (-, +) in the tuple after the median measurement")
        else:
            print("OFF, so the uncertainties should be zero.")

        for x in catr.get_array_FUV_flux():
            print(x)
        for s in catr:
            print(s.powr_models)
        catr.populate_FUV_flux()  # needs to refresh the FUV flux array
        for x in catr.get_array_FUV_flux():
            print(x)
        print("</test_L>\n")


    def test_Q(self):
        spectral.stresolver.random.seed(1312)
        np.random.seed(1312)
        powr_grids = {x: spectral.powr.PoWRGrid(x) for x in spectral.powr.AVAILABLE_POWR_GRIDS}
        cal_tables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        st_numbers = np.arange(3., 13., 0.5)
        st_tuples = [spectral.parse_sptype.sanitize_tuple(spectral.parse_sptype.number_to_st(x)) for x in st_numbers]
        tests = ["".join(x) for x in st_tuples]

        spectral.stresolver.UNCERTAINTY = False  # toggle the half-type/sampling
        catr = spectral.stresolver.CatalogResolver(tests,
                                                   calibration_table=cal_tables, leitherer_table=ltables,
                                                   powr_dict=powr_grids)
        Q = catr.get_array_ionizing_flux()
        Q_array = np.log10(np.array([f[0].to(1 / u.s).to_value() for f in Q]))

        Q_martins_array = np.array([cal_tables.lookup_characteristic('Qo', st_tuple) for st_tuple in st_tuples])

        plt.subplot(121)
        plt.plot(st_numbers, 10 ** (Q_martins_array - Q_array))
        plt.xlabel("Spectral Type")
        plt.ylabel("M05/PoWR")

        plt.subplot(122)
        plt.plot(st_numbers, Q_martins_array, label='M05')
        plt.plot(st_numbers, Q_array, label='PoWR')
        plt.xlabel("Spectral Type")
        plt.ylabel("$Q_0$")
        plt.legend()
        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_Q.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_Q"))
