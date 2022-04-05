"""
Test PoWR model grid stuff

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


class TestPoWR(unittest.TestCase):
    """
    Test PoWR
    """

    def test_powr_totalL_accuracy(self):
        """
        This checks the accuracy of integrating the PoWR model spectrum against the log L model parameter for that model

        Obviously this will be larger for log L values that don't perfectly align with a gridpoint's model log L value
        But it shouldn't be a huge deal..... right? (it looks like it could be a factor of 2, which means I probably
        could normalize)

        TODO This sounds like an additional test: convert the whole-number main sequence O type stars to models and
            check the luminosity from the Leitherer table against the integrated model spectrum value
            If there's a significant difference, then I should consider normalizing the spectrum by the Leitherer logL

        Also: this doesn't check WR stars, those are a whole can of worms. See the other test suite below this one.
        """
        tbl = spectral.powr.PoWRGrid('OB')
        count = 0
        for model_info in tbl.iter_models():
            count += 1
            wl, flux = tbl.get_model_spectrum(model_info)
            lum = np.trapz(flux, x=wl).to('solLum')
            print(f"Model {model_info.MODEL} (Teff {model_info.T_EFF}, log_g {model_info.LOG_G}):")
            print(f" -> Integrated luminosity (numpy): {lum.to_string()}")
            print(f" -> Difference (%): {((lum.to_value() / (10 ** model_info.LOG_L)) - 1) * 100.:.5}")
            print()
            if count > 20:
                break

    def test_powr_plot(self):
        # Plotting stuff
        kwargs = {}
        if DEBUG_SHOW:
            kwargs['show'] = True
        else:
            kwargs['show'] = False
            kwargs['savename'] = os.path.join(debug_img_path, "test_powr_plot.png")
            kwargs['metadata'] = create_debug_img_metadata(file=__file__, func_name="test_powr_plot")
        # tbl.plot_grid_space(tbl.grid_info.LOG_L, "log L")
        tbl = spectral.powr.PoWRGrid('WNE')
        tbl.plot_grid_space(c='blue', show=False, alpha=0.3)
        tbl = spectral.powr.PoWRGrid('WNL')
        tbl.plot_grid_space(c='red', setup=False, alpha=0.3, **kwargs)

    def test_all_PoWR(self):
        # Plotting stuff
        kwargs = {}
        if DEBUG_SHOW:
            kwargs['show'] = True
        else:
            kwargs['show'] = False
            kwargs['savename'] = os.path.join(debug_img_path, "test_all_PoWR.png")
            kwargs['metadata'] = create_debug_img_metadata(file=__file__, func_name="test_all_PoWR")
        tbl = spectral.powr.PoWRGrid('WNL')
        wf1 = tbl.get_model_spectrum(4.4, 1.7)
        wf2 = tbl.get_model_spectrum(5.0, 0.0)
        tbl.plot_spectrum(*wf1, show=False, xunit=u.eV, fuv=True, ylog=False)
        tbl_ob = spectral.powr.PoWRGrid('OB')
        wf3 = tbl_ob.get_model_spectrum(43000, 3.8)
        tbl.plot_spectrum(*wf2, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
        wf4 = tbl_ob.get_model_spectrum(16000, 3.0)
        tbl.plot_spectrum(*wf3, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
        tbl.plot_spectrum(*wf4, setup=False, xunit=u.eV, fuv=True, ylog=False, **kwargs)

    def test_powr_3param(self):
        """
        Check powr logL vs log g vs Teff

        I move forward with this exploration in test_L_vs_T_vs_g
        """
        tbl = spectral.powr.PoWRGrid('OB')
        print(tbl.grid_info)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Teff = tbl.grid_info['T_EFF'] / 1000.
        logL = tbl.grid_info['LOG_L']
        log_g = tbl.grid_info['LOG_G']
        ax.scatter(Teff, logL, log_g, c='k', marker='o')
        ax.set_xlabel('Teff')
        ax.set_ylabel('log L')
        ax.set_zlabel('log g')
        plt.title("Checking log L vs log g vs Teff smoothness")
        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_powr_3param.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_powr_3param"))

    def test_powr_retrieval_by_L(self):
        tbl = spectral.powr.PoWRGrid('OB')
        print(tbl.get_model_info(33040., 3.6))
        print()
        # This works!
        print(tbl.get_model_info(33020., 5.2, 'L'))


class TestWRCalibration(unittest.TestCase):
    """
    There was a bug in my older code before I fixed it on February 16, 2022, according to my commit history
    The PoWR models give calibrated OB spectra, so when you integrate them, they match the model's log L parameter

    But WR stars are both gridded differently and also invoke some sort of scale-invariance such that the PoWR models
    normalize the spectra to a log L of 5.3, so when you load a model you have to re-normalize it to your log L.
    I wasn't doing this until Feb 2022, when I fixed this bug. I had to implement the fix in STResolver, since PoWRGrid
    doesn't know about the WR's measured log L (nor should it)
    """

    def confirm_that_WR_needs_calibration(self):
        """
        Confirm that model spectra integrate to logL=5.3 solLum
        """
        spectral.stresolver.random.seed(1312)

        tbl = spectral.powr.PoWRGrid('WNL-H50')
        powr_grids = {'WNL-H50': tbl}
        cal_tables = spectral.sttable.STTable(*spectral.sternberg.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        s = spectral.stresolver.STResolver('WN6ha')
        s.link_calibration_table(cal_tables)
        s.link_leitherer_table(ltables)
        s.link_powr_grids(powr_grids)

        # st = s.spectral_types['WN6ha'][0]
        lum_list = np.array([spectral.stresolver.STResolver.get_WR_luminosity(st) for st in s.spectral_types['WN6ha']])
        lum = np.mean(lum_list)
        mdls = {m['MODEL']: m for m in s.powr_models['WN6ha']}

        wl_spectra_tuples = {m_name: mdls[m_name]['grid'].get_model_spectrum(mdls[m_name]) for m_name in mdls.keys()}

        print("If these are all 5.3, then WRs do indeed need calibration")
        for m_name in wl_spectra_tuples.keys():
            wl, flux = wl_spectra_tuples[m_name]
            print(np.log10(np.trapz(flux, x=wl).to(u.solLum).to_value()))

    def test_WR_calibration(self):
        """
        Plot the spectrum, check what it integrates to, see if I can normalize it
        by hand
        """
        spectral.stresolver.random.seed(1312)

        tbl = spectral.powr.PoWRGrid('WNL-H50')
        powr_grids = {'WNL-H50': tbl}
        cal_tables = spectral.sttable.STTable(*spectral.sternberg.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        s = spectral.stresolver.STResolver('WN6ha')
        s.link_calibration_table(cal_tables)
        s.link_leitherer_table(ltables)
        s.link_powr_grids(powr_grids)

        # st = s.spectral_types['WN6ha'][0]
        lum_list = np.log10(
            np.array([spectral.stresolver.STResolver.get_WR_luminosity(st) for st in s.spectral_types['WN6ha']]))
        integrated_spec_list = []
        scaled_integrated_spec_list = []
        count = 0
        for st, model_info in zip(s.spectral_types['WN6ha'], s.powr_models['WN6ha']):
            wl, flux = model_info['grid'].get_model_spectrum(model_info)
            lum = np.log10(np.trapz(flux, x=wl).to(u.solLum).to_value())
            integrated_spec_list.append(lum)
            wl, flux = spectral.stresolver.STResolver.get_model_spectrum(st, model_info)
            lum = np.log10(np.trapz(flux, x=wl).to(u.solLum).to_value())
            scaled_integrated_spec_list.append(lum)
        plt.plot(lum_list, integrated_spec_list, '.', label='raw')
        plt.plot(lum_list, scaled_integrated_spec_list, '.', label='scaled')
        plt.legend()
        plt.xlim([5.25, 6.4])
        plt.ylim([5.25, 6.4])
        plt.plot([5.25, 6.4], [5.25, 6.4], color='k', linestyle='--', alpha=0.5, zorder=0)
        plt.title("the scaled points should fall along x=y line, and the raw points should all have the same y value",
                  fontsize=8)
        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_WR_calibration.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_WR_calibration"))
        ######### Can set this back up somehow, this just plots the spectra
        #     tbl.plot_spectrum(*wl_spectra_tuples[m_name], label=m_name, setup=setup, xunit=u.Angstrom, ylog=False, show=False, linewidth=0.7)
        #     if setup:
        #         setup = False
        # plt.xlim((2e2, 5e3))
        # plt.show()

    def test_WR_calibration_2(self):
        """
        Check that the FUV and ionizing fluxes change when I implement the bug fix
        """
        spectral.stresolver.random.seed(1312)

        tbl = spectral.powr.PoWRGrid('WNL-H50')
        powr_grids = {'WNL-H50': tbl}
        cal_tables = spectral.sttable.STTable(*spectral.sternberg.load_tables_df())
        ltables = spectral.leitherer.LeithererTable()

        s = spectral.stresolver.STResolver('WN6ha')
        s.link_calibration_table(cal_tables)
        s.link_leitherer_table(ltables)
        s.link_powr_grids(powr_grids)

        s.populate_FUV_flux()
        s.populate_ionizing_flux()

        plt.subplot(121)
        plt.title("(WR calib fix test) FUV")
        plt.hist([x.to_value() for x in s.fuv['WN6ha']])
        plt.subplot(122)
        plt.title("ionizing")
        plt.hist([x.to_value() for x in s.ionizing['WN6ha']])

        """
        if I put in the old line:
            wlflux = model_info['grid'].get_model_spectrum(model_info)
        then FUV is ~8e4 to 1e5 and ionizing is 0.8 to 1.4 e49
        if I put in the new line:
            wlflux = STResolver.get_model_spectrum(st_tuple, model_info)
        then FUV is ~3 to 8 e5 and ionizing is 4 to 9 e49
        Reasonable difference!
        """

        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_WR_calibration_2.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_WR_calibration_2"))
