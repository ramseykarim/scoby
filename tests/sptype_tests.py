"""
Tests for the spectral module, dealing with spectral type reading and stellar
parameter assignment.

Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020

Created: June 2, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize
from astropy import units as u

from .. import utils
from .. import parse
from .. import spectral


def main():
    """
    Easier to have this at the top, never have to scroll down.
    "args" variable will contain any return values
    """
    return test_WR_calibration_2()


def plot_sptype_calibration_stuff():
    dfs, col_units = spectral.sternberg.load_tables_df()
    dfs2, col_units2 = spectral.martins.load_tables_df()
    colors = {'I':'blue','III':'green','V':'red'}
    plt.figure(figsize=(14, 9))
    for lc in spectral.parse_sptype.luminosity_classes:
        Teff, log_g = 'Teff', 'log_g'
        charX, charY = Teff, "log_L"
        st_sb03 = np.array([spectral.parse_sptype.st_to_number(i) for i in  dfs[lc].index])
        st_m05 = np.array([spectral.parse_sptype.st_to_number(i) for i in  dfs2[lc].index])
        independent, dependent = dfs[lc][charX], dfs[lc][charY]
        ind2, dep2 = dfs2[lc]['teff'], dfs2[lc][charY]
        plt.plot(independent, dependent, 'x', color=colors[lc], label='SB03')
        plt.plot(ind2, dep2, '.', color=colors[lc], label='M05')
        fit = spectral.sternberg.interp1d(independent, dependent, kind='linear')
        x = np.linspace(independent.min(), independent.max(), 50)
        plt.plot(x, fit(x), '--', label=f'fit to Sternberg+2003 class {lc}', color=colors[lc])
        # plt.plot(x, [spectral.vacca.vacca_calibration((*spectral.parse_sptype.number_to_st(i), lc), charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
        # plt.plot(x, [vacca_calibration(i, {'I':1,'III':3,'V':5}[lc], charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
    # plt.legend()
    plt.ylabel(charY), plt.xlabel(charX)
    plt.gca().invert_xaxis()
    plt.show()


def test_martins_calibration_load():
    df1, u1 = spectral.martins.load_tables_df()
    df2, u2 = spectral.sternberg.load_tables_df()
    print(u2.index)
    for i in u2.Units:
        print(i, u.Unit(i))
    print(u1.index)
    for i in u1.Units:
        print(i, u.Unit(i))


def test_martins_calibration():
    df1, u1 = spectral.martins.load_tables_df()
    print(df1['V'])
    df2, u2 = spectral.sternberg.load_tables_df()
    print(df2['V'])
    print('-----')
    print(u1)
    print(u2)


def test_sttables():
    """
    I used this to confirm that STTable gives good looking results
    for both Sternberg and Martins
    """
    df1, u1 = spectral.martins.load_tables_df()
    df2, u2 = spectral.sternberg.load_tables_df()
    stt1 = spectral.sttable.STTable(df1, u1)
    stt2 = spectral.sttable.STTable(df2, u2)
    return stt1, stt2


def test_L_vs_T_vs_g():
    """
    I want to see if I can use logL and T to map cleanly to log g
    (Sept 23, 2020 for use with Cyg OB2)
    """
    df1, u1 = spectral.martins.load_tables_df() # dictionary of dfs
    fig = plt.figure()
    colors = {'I':'blue','III':'green','V':'red'}
    ax = fig.add_subplot(111, projection='3d')
    all_T, all_L, all_g = [], [], []
    for lc in spectral.parse_sptype.luminosity_classes:
        Teff = df1[lc]['Teff']/1000
        all_T.extend(list(Teff))
        logL = df1[lc]['log_L']
        all_L.extend(list(logL))
        log_g = df1[lc]['log_g']
        all_g.extend(list(log_g))
        ax.scatter(Teff, logL, log_g, c=colors[lc], marker='o')

    tbl = spectral.powr.PoWRGrid('OB')
    # Teff = tbl.grid_info['T_EFF']/1000.
    # logL = tbl.grid_info['LOG_L']
    # log_g = tbl.grid_info['LOG_G']
    # ax.scatter(Teff, logL, log_g, c='k', marker='o')

    Teff = np.arange(20., 50., 1.5)*1000
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
        z = a*x + b*y + d
        return z

    points = np.array([all_T, all_L, all_g])
    def fit_plane(params):
        residuals = points[2, :] - plane(points[0, :], points[1, :], params)
        return np.sum(residuals**2)

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

    plt.show()


def test_leitherer_open():
    """
    open_tables works, as far as I can tell
    """
    df1, u1 = spectral.leitherer.open_tables()
    # print(df1)
    print(u1)
    return df1, u1


def test_leitherer_grid_smoothness():
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
    """
    df1, u1 = spectral.leitherer.open_tables()
    T = np.log10(df1['T_eff'])
    Tlabel = f"log T_eff ({u1.loc['T_eff', 'Units']})"
    L = df1['log_L']
    Llabel = f"log L ({u1.loc['log_L', 'Units']})"
    z = df1['R']
    TL = np.array([T.values, L.values]).T
    interp = utils.CloughTocher2DInterpolator(TL, z, fill_value=np.nan)
    print(u1)
    plt.subplot(121)
    plt.scatter(T, L, marker='o', c=z, vmin=np.min(z), vmax=np.max(z))
    plt.colorbar()
    plt.xlabel(Tlabel), plt.ylabel(Llabel)
    plt.gca().invert_xaxis()
    aspect = plt.gca().get_aspect()
    xlim, ylim = (T.min(), T.max()), (L.min(), L.max())
    plt.subplot(122)
    zz = interp(T, L)
    print(T.shape, L.shape, zz.shape)
    print(zz.shape)
    plt.scatter(T, L, c=zz, vmin=np.min(z), vmax=np.max(z))
    xx, yy = np.meshgrid(np.linspace(*xlim, 50), np.linspace(*ylim, 50))
    zz = interp(xx, yy)
    plt.imshow(zz, origin='lower', vmin=np.min(z), vmax=np.max(z),
        extent=[*xlim, *ylim])
    plt.colorbar()
    plt.gca().invert_xaxis()
    plt.gca().set_aspect(aspect)
    plt.show()


def test_leitherer_sptypes():
    """
    This test shows that I can connect the Martins (or Sternberg) tables to
    the Leitherer tables and thus reliably map spectral type to mass loss rate
    via T & L. I can also get vinf, so I can get momentum flux.
    """
    ltables = spectral.leitherer.LeithererTable()
    mtables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
    sptypes_n = np.arange(3, 13., 0.5)
    print(mtables.column_units)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for lc in spectral.parse_sptype.luminosity_classes:
        sptypes = [spectral.parse_sptype.number_to_st(x)+(lc,) for x in sptypes_n]
        T_arr = [mtables.lookup_characteristic('Teff', x) for x in sptypes]
        logL_arr = [mtables.lookup_characteristic('log_L', x) for x in sptypes]
        axes[0].plot(np.log10(T_arr), logL_arr, '--', label=lc)
        mdot_arr = ltables.lookup_characteristic('log_Mdot', T_arr, logL_arr)
        vinf_arr = ltables.lookup_characteristic('v_inf', T_arr, logL_arr)
        c0 = axes[0].scatter(np.log10(T_arr), logL_arr, marker='o', c=mdot_arr, vmin=-9, vmax=-5)
        pdot_arr = np.log10((10.**np.array(mdot_arr))*(1.988e33/3.154e7) * np.array(vinf_arr) * 1e5)
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
    plt.show()


def test_leitherer_individuals():
    ltables = spectral.leitherer.LeithererTable()
    mtables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
    sptypes = ["O3V", "O5III"]
    for s in sptypes:
        print(s)
        T = mtables.lookup_characteristic('Teff', s)
        logL = mtables.lookup_characteristic('log_L', s)
        print(T, logL)
        print(type(T), type(logL))
        logmdot = ltables.lookup_characteristic('log_Mdot', T, logL)
        print(logmdot, type(logmdot))



old_catalog_fn = f"{utils.ancillary_data_path}catalogs/Ramsey/old_catalogs/OBradec.pkl"

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Older tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def test_st_parse_slashdash():
    """
    Tested the new subtype dash behavior, it looks like it works!
    """
    cat = pd.read_pickle(old_catalog_fn)
    tests = ['O8/B1.5V', 'O8-B1.5V', cat.SpectralType[19], cat.SpectralType[5], cat.SpectralType[7], 'B5/6.5III', 'O4II/III', cat.SpectralType[26], cat.SpectralType[27], 'O4-5.5V/III*', "O5:V"]
    for t in tests:
        l = spectral.parse_sptype.st_parse_slashdash(t)
        print(t, '\t', l)
        print('\t', [spectral.parse_sptype.st_parse_type(x) for x in l])
        print()


def test_st_adjacent():
    tests = ['O8', 'O4.5V', 'O2If', 'B2.5', 'O9.5']
    for t in tests:
        print(t, end=': ')
        t = spectral.parse_sptype.st_parse_type(t)
        t = tuple(x for x in t if x)
        print(t, end=' --> ')
        print(spectral.parse_sptype.st_adjacent(t))



def test_full_st_parse():
    """
    This is no longer the "full" parse
    """
    cat = pd.read_pickle(old_catalog_fn)
    count = 0
    for st in cat.SpectralType:
        assert isinstance(st, str)
        print(f"{st:30s}", end=": ")
        if st == 'ET':
            st = 'O9.5V'
            count += 1
        stars = [[spectral.parse_sptype.st_parse_type(x) for x in spectral.parse_sptype.st_parse_slashdash(y)] for y in spectral.parse_sptype.st_parse_binary(st)]
        types = [x[0][0:2] for x in stars]
        if types and all(types[0]):
            print([i[0]+i[1] for i in types])
        else:
            print(stars)
    print(f"There are {count} ET stars")


def test_powr_totalL_accuracy():
    tbl = spectral.powr.PoWRGrid('OB')
    count = 0
    for model_info in tbl.iter_models():
        count += 1
        wl, flux = tbl.get_model_spectrum(model_info)
        lum = np.trapz(flux, x=wl).to('solLum')
        print(f"Model {model_info.MODEL} (Teff{model_info.T_EFF}/log_g{model_info.LOG_G}:")
        print(f" -> Integrated luminosity (numpy): {lum.to_string()}")
        print(f" -> Difference (%): {((lum.to_value() / (10**model_info.LOG_L))-1)*100.:.5}")
        print()
        if count > 20:
            break


def test_powr_plot():
    # tbl.plot_grid_space(tbl.grid_info.LOG_L, "log L")
    tbl = spectral.powr.PoWRGrid('WNE')
    tbl.plot_grid_space(c='blue', show=False, alpha=0.3)
    tbl = spectral.powr.PoWRGrid('WNL')
    tbl.plot_grid_space(c='red', setup=False, alpha=0.3)


def testSuite_PoWR():
    tbl = spectral.powr.PoWRGrid('WNL')
    wf1 = tbl.get_model_spectrum(4.4, 1.7)
    wf2 = tbl.get_model_spectrum(5.0, 0.0)
    tbl.plot_spectrum(*wf1, show=False, xunit=u.eV, fuv=True, ylog=False)
    tblob = spectral.powr.PoWRGrid('OB')
    wf3 = tblob.get_model_spectrum(43000, 3.8)
    tbl.plot_spectrum(*wf2, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    wf4 = tblob.get_model_spectrum(16000, 3.0)
    tbl.plot_spectrum(*wf3, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    tbl.plot_spectrum(*wf4, setup=False, xunit=u.eV, fuv=True, ylog=False)


def test_powr_3param():
    """
    Check powr logL vs log g vs Teff

    I move forward with this exploration in test_L_vs_T_vs_g
    """
    tbl = spectral.powr.PoWRGrid('OB')
    print(tbl.grid_info)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Teff = tbl.grid_info['T_EFF']/1000.
    logL = tbl.grid_info['LOG_L']
    log_g = tbl.grid_info['LOG_G']
    ax.scatter(Teff, logL, log_g, c='k', marker='o')
    ax.set_xlabel('Teff')
    ax.set_ylabel('log L')
    ax.set_zlabel('log g')
    plt.show()


def test_powr_retrieval_by_L():
    tbl = spectral.powr.PoWRGrid('OB')
    print(tbl.get_model_info(33040., 3.6))
    print()
    # This works!
    print(tbl.get_model_info(33020., 5.2, 'L'))


def test_STResolver_WR():
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



def test_STResolver():
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

    spectral.stresolver.UNCERTAINTY = False # toggle the half-type/sampling
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

def test_STResolver_L():
    spectral.stresolver.random.seed(1312)
    np.random.seed(1312)
    powr_grids = {x: spectral.powr.PoWRGrid(x) for x in spectral.powr.AVAILABLE_POWR_GRIDS}
    cal_tables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
    ltables = spectral.leitherer.LeithererTable()

    tests = ['O6', 'O6', 'O6']
    TLs = [(30000., 5.2), (32050., 4.9), (28001., 5.1)]

    spectral.stresolver.UNCERTAINTY = False # toggle the half-type/sampling
    catr = spectral.stresolver.CatalogResolver(tests,
        calibration_table=cal_tables, leitherer_table=ltables,
        powr_dict=powr_grids)
    for x in catr.get_array_FUV_flux():
        print(x)
    for x in catr.link_powr_grids(powr_grids, listof_TL_pair=TLs):
        print(x)
    catr.populate_FUV_flux() # needs to refresh the FUV flux array
    for x in catr.get_array_FUV_flux():
        print(x)


def test_STResolver_Q():
    spectral.stresolver.random.seed(1312)
    np.random.seed(1312)
    powr_grids = {x: spectral.powr.PoWRGrid(x) for x in spectral.powr.AVAILABLE_POWR_GRIDS}
    cal_tables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
    ltables = spectral.leitherer.LeithererTable()

    st_numbers = np.arange(3., 13., 0.5)
    st_tuples = [spectral.parse_sptype.sanitize_tuple(spectral.parse_sptype.number_to_st(x)) for x in st_numbers]
    tests = ["".join(x) for x in st_tuples]

    spectral.stresolver.UNCERTAINTY = False # toggle the half-type/sampling
    catr = spectral.stresolver.CatalogResolver(tests,
        calibration_table=cal_tables, leitherer_table=ltables,
        powr_dict=powr_grids)
    Q = catr.get_array_ionizing_flux()
    Q_array = np.log10(np.array([f[0].to(1/u.s).to_value() for f in Q]))

    Q_martins_array = np.array([cal_tables.lookup_characteristic('Qo', st_tuple) for st_tuple in st_tuples])

    plt.subplot(121)
    plt.plot(st_numbers, 10**(Q_martins_array - Q_array))
    plt.xlabel("Spectral Type")
    plt.ylabel("M05/PoWR")

    plt.subplot(122)
    plt.plot(st_numbers, Q_martins_array, label='M05')
    plt.plot(st_numbers, Q_array, label='PoWR')
    plt.xlabel("Spectral Type")
    plt.ylabel("$Q_0$")
    plt.legend()
    plt.show()

def test_catalog():
    df = parse.load_final_catalog_df()
    df['RA'] = df['SkyCoord'].apply(lambda x: x.ra.deg)
    df['DEC'] = df['SkyCoord'].apply(lambda x: x.dec.deg)
    df = df.drop('SkyCoord', 1)
    fn = f"{utils.ancillary_data_path}catalogs/Ramsey/catalog_may5_2020.csv"
    df.to_csv(fn)
    # df = pd.read_csv(fn)
    return df


"""
A few tests to check the WR luminosity calibration issue
"""
def WR_tests():
    print("test_powr_plot")
    test_powr_plot()
    print("testSuite_PoWR")
    testSuite_PoWR()
    print("test_STResolver_WR")
    test_STResolver_WR()


def confirm_that_WR_needs_calibration():
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

    setup = True
    for m_name in wl_spectra_tuples.keys():
        wl, flux = wl_spectra_tuples[m_name]
        print(np.log10(np.trapz(flux, x=wl).to(u.solLum).to_value()))



def test_WR_calibration():
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
    lum_list = np.log10(np.array([spectral.stresolver.STResolver.get_WR_luminosity(st) for st in s.spectral_types['WN6ha']]))
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
    plt.show()
    ######### Can set this back up somehow, this just plots the spectra
    #     tbl.plot_spectrum(*wl_spectra_tuples[m_name], label=m_name, setup=setup, xunit=u.Angstrom, ylog=False, show=False, linewidth=0.7)
    #     if setup:
    #         setup = False
    # plt.xlim((2e2, 5e3))
    # plt.show()



def test_WR_calibration_2():
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
    plt.title("FUV")
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

    plt.show()



if __name__ == "__main__":
    args = main()
