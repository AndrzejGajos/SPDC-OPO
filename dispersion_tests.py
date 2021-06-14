from Dispersion import *
from numpy import pi

if __name__ == '__main__':
    nd_yag = 1.064
    nd_ylf = 1.0475
    period1 = 19.48 #18 #19.48
    period2 = 18
    pp_vector = [1, 0, 0]
    PPLNgayer2008.plot_ref_indexs(RangeLimits(min=0.5, max=4), 25.0, RangeLimits(min=0.0, max=100.0), 1.0)
    ppln = PPLNgayer2008(period1, pp_vector, pp_sign=Sign.PLUS)
    #
    speeds = SHGspeeds(pump=Speeds.FAST, shg=Speeds.FAST)
    temperatures = RangeLimits(max=100, min=50)
    ppln.plot_temp_shg(speeds, temperatures, pi/2, 0)
    polling = RangeLimits(max=19.5, min=18.5)
    ppln.plot_ppol_shg(speeds, polling, pi/2, 0)
    ppln = PPLNgayer2008(31, pp_vector, pp_sign=Sign.MINUS)
    ppln.plot_temp_opo([Speeds.FAST, Speeds.FAST, Speeds.FAST], RangeLimits(min=25, max=200), nd_ylf, pi/2, 0)
    ppln.plot_periodicpol_opo([Speeds.FAST, Speeds.FAST, Speeds.FAST], RangeLimits(min=30.5, max=32.5), nd_yag, pi/2, 0, 25)
    lista = ppln.find_opo(0.775, 25)

