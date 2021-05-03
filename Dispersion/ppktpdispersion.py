from .crystaldispersion import CrystalDispersion
from .oposetup import Sign


class PPKTPMisiaszek(CrystalDispersion):
    _SELLMEIER_COEFF = {"xcoeff": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
                       "ycoeff": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
                       "zcoeff": [4.59423, 0.06272, 0.04814, 110.80672, 86.12171]}

    _TEMP_COEFF = {"thermal_expansion": {"alpha": 6.7 * 10 ** -6, "beta": 1.1 * 10 ** -8, "room_temperature": 25},
                  "tempX": [0.1717, -0.5353, 0.8416, 0.1627, 20, 10 ** -5],
                  "tempY": [0.1997, -0.4063, 0.5154, 0.5425, 20, 10 ** -5],
                  "tempZ": [0.9221, -2.922, 3.6677, -0.1897, 20, 10 ** -5]}

    def __init__(self, crystal_period: float, pp_vec: list, pp_sign=Sign.MINUS) -> None:
        super().__init__(crystal_period, pp_vec, pp_sign)

    @staticmethod
    def _ns(lam, temperature, sell_coeff, temp_coeff) -> float:
        dispersion = abs(sell_coeff[0] + sell_coeff[1] / (lam ** 2 - sell_coeff[2])
                         + sell_coeff[3] / (lam ** 2 - sell_coeff[4])) ** 0.5
        temp_influ = (temp_coeff[0] / lam ** 3 + temp_coeff[1] / lam ** 2 + temp_coeff[2] / lam + temp_coeff[3]) * (
                temperature - temp_coeff[4]) * temp_coeff[5]
        return dispersion + temp_influ

    def nx(self, wavelength: float, temperature: float) -> float:
        return self._ns(wavelength, temperature, self._SELLMEIER_COEFF["xcoeff"], self._TEMP_COEFF["tempX"])

    def ny(self, wavelength: float, temperature: float) -> float:
        return self._ns(wavelength, temperature, self._SELLMEIER_COEFF["ycoeff"], self._TEMP_COEFF["tempY"])

    def nz(self, wavelength: float, temperature: float) -> float:
        return self._ns(wavelength, temperature, self._SELLMEIER_COEFF["zcoeff"], self._TEMP_COEFF["tempZ"])

# TODO tests -> __name__ == main
