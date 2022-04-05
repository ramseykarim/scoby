"""
Testing spectral type calibration, which involves mapping from log L and Teff to log g and then connecting
those to mass loss rate and terminal wind velocity and stuff like that

I am copying these tests from sptype_tests.py, which was where I dumped all my tests before I reorganized this code

Created: April 5, 2022
"""
__author__ = "Ramsey Karim"

import unittest

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import scoby

from .utils import create_debug_img_metadata, debug_img_path, DEBUG_SHOW


class TestCalibration(unittest.TestCase):
    """
    Plotting Log L against Teff and things like that, just checking on the relations between stellar properties
    """

    def test_plot_sptype_calibration(self):
        """
        Plot Log L vs Teff for Martins and Sternberg tables
        (This used to plot log g vs Teff and included Vacca, but Vacca doesn't do log L)
        TODO: Include Leitherer?
        """
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
            plt.plot(independent, dependent, 'x', color=colors[lc], label=f'Sternberg+2003 (S03) {lc}')
            plt.plot(ind2, dep2, '.', color=colors[lc], label=f'Martins+2005 {lc}')
            fit = scoby.spectral.sternberg.interp1d(independent, dependent, kind='linear')
            x = np.linspace(independent.min(), independent.max(), 50)
            plt.plot(x, fit(x), '--', label=f'fit to S03 class {lc}', color=colors[lc])

        plt.legend()
        plt.ylabel(char_y), plt.xlabel(char_x)
        plt.gca().invert_xaxis()
        plt.savefig(os.path.join(debug_img_path, "test_plot_sptype_calibration_stuff.png"),
                    metadata=create_debug_img_metadata(file=__file__, func_name="test_plot_sptype_calibration_stuff"))

    def test_L_vs_T_vs_g(self):
        """
        I want to see if I can use logL and T to map cleanly to log g
        (Sept 23, 2020 for use with Cyg OB2)

        This actually tests PoWR and the fancy interpolation method I use (see powr.py)
        """
        df1, u1 = scoby.spectral.martins.load_tables_df()  # dictionary of dfs
        fig = plt.figure()
        colors = {'I': 'blue', 'III': 'green', 'V': 'red'}
        ax = fig.add_subplot(111, projection='3d')
        all_T, all_L, all_g = [], [], []
        for lc in scoby.spectral.parse_sptype.luminosity_classes:
            Teff = df1[lc]['Teff'] / 1000
            all_T.extend(list(Teff))
            logL = df1[lc]['log_L']
            all_L.extend(list(logL))
            log_g = df1[lc]['log_g']
            all_g.extend(list(log_g))
            ax.scatter(Teff, logL, log_g, c=colors[lc], marker='o')

        tbl = scoby.spectral.powr.PoWRGrid('OB')
        # Teff = tbl.grid_info['T_EFF']/1000.
        # logL = tbl.grid_info['LOG_L']
        # log_g = tbl.grid_info['LOG_G']
        # ax.scatter(Teff, logL, log_g, c='k', marker='o')

        Teff = np.arange(20., 50., 1.5) * 1000
        logL = np.linspace(4.4, 6.2, Teff.size)
        tbl.interp_g(Teff[0], logL[0])
        Teff, logL = np.meshgrid(Teff, logL, indexing='xy')
        TL_grid = np.stack([np.log10(Teff), logL], axis=-1)
        log_g = tbl.TL_interp(TL_grid)
        Teff /= 1000
        ax.plot_surface(Teff, logL, log_g, color='orange', alpha=0.3)
        """
        This interpolation is better than the plane fit below. It covers a much
        larger area and reflects some of the curvature of the log_g surface.
        But both are roughly consistent with each other!
        """

        ax.set_xlabel('Teff')
        ax.set_ylabel('log L')
        ax.set_zlabel('log g')

        # From stackoverflow: https://stackoverflow.com/a/20700063
        def plane(x, y, params):
            a, b, d = params[:3]
            z = a * x + b * y + d
            return z

        points = np.array([all_T, all_L, all_g])

        def fit_plane(params):
            residuals = points[2, :] - plane(points[0, :], points[1, :], params)
            return np.sum(residuals ** 2)

        res = minimize(fit_plane, [1, 1, 1])
        print(res.x)
        """
        THE FIT IS:
        [ 0.05727171 -0.65728093  5.20380702]
        We only need to run this once!!
        This is a fit to T/1000, logL for log_g
        """

        xx, yy = np.meshgrid(np.array([27, 45]), np.array([4.5, 6.1]))
        zz = plane(xx, yy, res.x)

        ax.plot_surface(xx, yy, zz, alpha=0.3)

        plt.savefig(os.path.join(debug_img_path, "test_L_vs_T_vs_g.png"),
                    metadata=create_debug_img_metadata(file=__file__, func_name="test_L_vs_T_vs_g"))


class TestLeitherer(unittest.TestCase):
    """
    There are a lot of tests specific to the Leitherer 2010 tables so I'm putting them in their own test suite
    """

    def test_leitherer_grid_smoothness(self):
        """
        Check interpolation via CloughTocher2DInterpolator (and inherent smoothness
        of these characteristics across the T, L model grid)

        Results: Mdot is smooth, interpolation looks good. scipy.interpolate.interp2d
        is much worse than this CloughTocher2DInterpolator I am now using, which
        looks pretty good.
        v_inf is a little rockier, looks like there are some jumps (we know about these)
        The interp still deals with them reasonably well. Nothing is that extreme.
        Mass is also a little rocky, similar valleys to v_inf, but same deal.
        Nothing too bad.
        Radius is remarkably smooth and well behaved.

        The output plots are in pairs, with the left side showing only the table data and the right side showing
        table data overlaid on the interpolation. You should use the right side to look for color discontinuity between
        the interpolation (image) and table data (scattered points)
        """
        df1, u1 = scoby.spectral.leitherer.open_tables()
        T = np.log10(df1['T_eff'])
        Tlabel = f"log T_eff ({u1.loc['T_eff', 'Units']})"
        L = df1['log_L']
        Llabel = f"log L ({u1.loc['log_L', 'Units']})"

        available_characteristics = {'R': 'radius', 'M': 'mass', 'v_inf': 'terminal wind velocity',
                                     'log_Mdot': 'mass loss rate'}

        # Two rows (so two characteristics per row) and 4 columns (two characteristics x two plots per column)
        grid_shape = (2, 4)

        for idx, selected_characteristic_key in enumerate(available_characteristics):
            z = df1[selected_characteristic_key]
            characteristic_label = available_characteristics[selected_characteristic_key]

            TL = np.array([T.values, L.values]).T
            interp = scoby.utils.CloughTocher2DInterpolator(TL, z, fill_value=np.nan)
            print(u1)

            plt.subplot2grid(grid_shape, (idx//2, (idx % 2)*2))

            plt.scatter(T, L, marker='o', c=z, vmin=np.min(z), vmax=np.max(z))
            # plt.colorbar()
            plt.xlabel(Tlabel), plt.ylabel(Llabel)
            plt.gca().invert_xaxis()
            aspect = plt.gca().get_aspect()
            xlim, ylim = (T.min(), T.max()), (L.min(), L.max())
            plt.title(characteristic_label + " tbl data")

            plt.subplot2grid(grid_shape, (idx//2, (idx % 2)*2 + 1))

            zz = interp(T, L)
            print(T.shape, L.shape, zz.shape)
            print(zz.shape)
            plt.scatter(T, L, c=zz, vmin=np.min(z), vmax=np.max(z))
            xx, yy = np.meshgrid(np.linspace(*xlim, 50), np.linspace(*ylim, 50))
            zz = interp(xx, yy)
            plt.imshow(zz, origin='lower', vmin=np.min(z), vmax=np.max(z),
                       extent=[*xlim, *ylim])
            plt.title(characteristic_label + " tbl+interp")
            plt.xlabel(Tlabel)
            plt.colorbar()
            plt.gca().invert_xaxis()
            plt.gca().set_aspect(aspect)

        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_leitherer_grid_smoothness.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_leitherer_grid_smoothness"))

    def test_leitherer_sptypes(self):
        """
        This test shows that I can connect the Martins (or Sternberg) tables to
        the Leitherer tables and thus reliably map spectral type to mass loss rate
        via T & L. I can also get vinf, so I can get momentum flux.
        """
        ltables = scoby.spectral.leitherer.LeithererTable()
        mtables = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())
        sptypes_n = np.arange(3, 13., 0.5)
        print(mtables.column_units)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        for lc in scoby.spectral.parse_sptype.luminosity_classes:
            sptypes = [scoby.spectral.parse_sptype.number_to_st(x) + (lc,) for x in sptypes_n]
            T_arr = [mtables.lookup_characteristic('Teff', x) for x in sptypes]
            logL_arr = [mtables.lookup_characteristic('log_L', x) for x in sptypes]
            axes[0].plot(np.log10(T_arr), logL_arr, '--', label=lc)
            mdot_arr = ltables.lookup_characteristic('log_Mdot', T_arr, logL_arr)
            vinf_arr = ltables.lookup_characteristic('v_inf', T_arr, logL_arr)
            c0 = axes[0].scatter(np.log10(T_arr), logL_arr, marker='o', c=mdot_arr, vmin=-9, vmax=-5)
            pdot_arr = np.log10((10. ** np.array(mdot_arr)) * (1.988e33 / 3.154e7) * np.array(vinf_arr) * 1e5)
            print(pdot_arr)
            c1 = axes[1].scatter(np.log10(T_arr), logL_arr, marker='o', c=pdot_arr, vmin=24, vmax=29)
        for i in range(2):
            plt.sca(axes[i])
            plt.xlabel("log T"), plt.ylabel("log L")
            plt.gca().invert_xaxis()
            if not i:
                plt.legend()
                plt.colorbar(c0, label='log $\dot M$ ($M_{\\odot} yr-1$)')
                plt.title("O-type L vs T and mass loss rate")
            else:
                plt.colorbar(c1, label='log $\dot p$ (dyne)')
                plt.title("momentum transfer rate")
        plt.tight_layout()
        if DEBUG_SHOW:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_img_path, "test_leitherer_sptypes.png"),
                        metadata=create_debug_img_metadata(file=__file__, func_name="test_leitherer_sptypes"))

    def test_leitherer_individuals(self):
        ltables = scoby.spectral.leitherer.LeithererTable()
        mtables = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())
        sptypes = ["O3V", "O5III"]
        for s in sptypes:
            print(s)
            T = mtables.lookup_characteristic('Teff', s)
            logL = mtables.lookup_characteristic('log_L', s)
            print(T, logL)
            print(type(T), type(logL))
            logmdot = ltables.lookup_characteristic('log_Mdot', T, logL)
            print(logmdot, type(logmdot))

