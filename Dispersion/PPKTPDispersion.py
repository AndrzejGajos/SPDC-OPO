from .CrystalDispersion import CrystalDispersion

class PPKTPMisiaszek(CrystalDispersion):

    SellmeierCoeff = {"xcoeff": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
                      "ycoeff": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
                      "zcoeff": [4.59423, 0.06272, 0.04814, 110.80672, 86.12171]}

    TempCoeff = {"thermalExp": {"alpha": 6.7 * 10**-6, "beta": 1.1 * 10**-8, "RoomTemp": 25},
                 "tempX": [0.1717, -0.5353, 0.8416, 0.1627, 20, 10**-5],
                 "tempY": [0.1997, -0.4063, 0.5154, 0.5425, 20, 10**-5],
                 "tempZ": [0.9221, -2.922, 3.6677, -0.1897, 20, 10**-5]}

    def __init__(self, CrystalPeriod, PPVec):
        super().__init__ (CrystalPeriod, PPVec, self.SellmeierCoeff, self.TempCoeff)

    def ns(self, lam, T, Sell, TempCoeff):
        dispersion = abs(Sell[0] + Sell[1]/(lam**2 - Sell[2]) + Sell[3]/ (lam**2 - Sell[4]))**0.5
        tempInflu  = (TempCoeff[0]/lam**3 + TempCoeff[1]/lam**2 + TempCoeff[2]/lam + TempCoeff[3]) * (T - TempCoeff[4]) * TempCoeff[5]
        return dispersion + tempInflu

    def nx(self, Wavelength, Temperature):
        return self.ns(Wavelength, Temperature, self.SellmeierCoeff["xcoeff"], self.TempCoeff["tempX"])

    def ny(self, Wavelength, Temperature):
        return self.ns(Wavelength, Temperature, self.SellmeierCoeff["ycoeff"], self.TempCoeff["tempY"])

    def nz(self, Wavelength, Temperature):
        return self.ns(Wavelength, Temperature, self.SellmeierCoeff["zcoeff"], self.TempCoeff["tempZ"])
