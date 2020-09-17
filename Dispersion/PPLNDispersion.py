from .CrystalDispersion import CrystalDispersion
import numpy as np

class PPLNgayer2008(CrystalDispersion):
#5% MgO-doped congruent LiNbO3 (CLN),
    SellmeierCoeff = {"xcoeff": np.array([5.653, 0.1185, 0.2091, 89.61, 10.85, 0.0197]),
                      "ycoeff": np.array([5.653, 0.1185, 0.2091, 89.61, 10.85, 0.0197]),
                      "zcoeff": np.array([5.756, 0.0983, 0.2020, 189.32, 12.52, 0.0132])}

    TempCoeff = {"thermalExp": {"alpha": 1.54 * 10**-5, "beta": 5 * 10**-9, "RoomTemp": 25},
                 "tempX": np.array([7.941 * 10**-7, 3.134 * 10**-8, -4.641 * 10**-9, -2.188 * 10**-6]),
                 "tempY": np.array([7.941 * 10**-7, 3.134 * 10**-8, -4.641 * 10**-9, -2.188 * 10**-6]),
                 "tempZ": np.array([2.86 * 10**-6, 4.7 * 10**-8, 6.113 * 10**-8, 1.516*10**-4])}

    def __init__(self, CrystalPeriod, PPVec, PPSign = 1):
        super().__init__(CrystalPeriod, PPVec, self.SellmeierCoeff, self.TempCoeff, PPSign)

    def __ns(self, lam, T, Sell, TempCoeff):
        f = (T - 24.5) * (T + 570.82)
        lam2 = lam**2
        dispersion = np.abs(Sell[0] + TempCoeff[0] * f + (Sell[1] + TempCoeff[1] * f)/(lam2 - (Sell[2] + TempCoeff[2] * f)**2) + (Sell[3] + TempCoeff[3] * f)/ (lam2 - Sell[4]**2) - Sell[5] * lam2)**0.5
        return dispersion

    def nx(self, Wavelength, Temperature):
        return self.__ns(Wavelength, Temperature, self.SellmeierCoeff["xcoeff"], self.TempCoeff["tempX"])

    def ny(self, Wavelength, Temperature):
        return self.__ns(Wavelength, Temperature, self.SellmeierCoeff["ycoeff"], self.TempCoeff["tempY"])

    def nz(self, Wavelength, Temperature):
        return self.__ns(Wavelength, Temperature, self.SellmeierCoeff["zcoeff"], self.TempCoeff["tempZ"])
