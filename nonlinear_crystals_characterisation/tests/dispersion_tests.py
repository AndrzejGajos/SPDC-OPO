from nonlinear_crystals_characterisation import *
from numpy import pi
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Refractive index test
    nd_yag = 1.064
    nd_ylf = 1.0475
    period1 = 19.48
    period2 = 18
    pp_vector = [1, 0, 0]
    PPLNgayer2008.plot_ref_indexs(RangeLimits(min=0.5, max=4), 25.0, RangeLimits(min=0.0, max=100.0), 1.0)
     # SHG tests
    ppln = PPLNgayer2008(period1, pp_vector, pp_sign=Sign.PLUS)
    speeds = SHGspeeds(pump=Speeds.FAST, shg=Speeds.FAST)
    temperatures = RangeLimits(max=100, min=50)
    ppln.plot_temp_shg(speeds, temperatures, pi/2, 0, 1.1)
    polling = RangeLimits(max=19.5, min=18.5)
    ppln.plot_ppol_shg(speeds, polling, pi/2, 0, 1.1)
    plt.show()
    ppln = PPLNgayer2008(31, pp_vector, pp_sign=Sign.MINUS)
    # OPO plot test
    speeds = [Speeds.FAST, Speeds.FAST, Speeds.FAST]
    ppln.plot_wave_opo(speeds, RangeLimits(min=nd_ylf, max=nd_yag), pi/2, 0, 25)
    ppln.plot_temp_opo(speeds, RangeLimits(min=25, max=200), nd_ylf, pi/2, 0)
    ppln.plot_periodicpol_opo(speeds, RangeLimits(min=30.5, max=32.5), nd_yag, pi/2, 0, 25)
    plt.show()
    # OPO find setups
    ppln = PPLNgayer2008(31, pp_vector, pp_sign=Sign.MINUS)
    lista = ppln.find_opo(0.775, 25, quiet=False)
    ppln.find_crystal_period(0.775, 1.55, OpoPol.HETERO, quiet=False)
    ppln.find_crystal_period(0.775, 1.55, OpoPol.HOMO, quiet=False)
    ppln.find_crystal_temperature(0.775, 1.550, OpoPol.HETERO, quiet=False)
    ppln.find_collinear_opo(Mode.CRYSTAL_PERIOD, 0.775, 1.55, OpoPol.HOMO, quiet=False)
    ppln.find_collinear_opo(Mode.TEMPERATURE, 0.775, 1.55, OpoPol.HETERO, quiet=False, x0 = 9.1)
    setups = ppln.find_collinear_opo(Mode.TEMP_AND_PERIOD, 0.775, 1.55, OpoPol.HOMO, quiet=False)
    setup = lista[2]
    ppln.plot_temperature(setup, RangeLimits(min=25, max=200))
    ppln.plot_wavelength(setup, RangeLimits(0.6, 0.85))
    ppln.plot_periodicpolling(setup, RangeLimits(25, 35))
    plt.show()


