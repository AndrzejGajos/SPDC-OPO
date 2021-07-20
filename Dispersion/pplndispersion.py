from .crystaldispersion import CrystalDispersion
from .oposetup import Sign


class PPLNgayer2008(CrystalDispersion):
    """5% MgO-doped congruent LiNbO3 (CLN)
    DOI: 10.1007/s00340-008-2998-2
    """
    _SELLMEIER_COEFF = {"xcoeff": [5.653, 0.1185, 0.2091, 89.61, 10.85, 0.0197],
                        "ycoeff": [5.653, 0.1185, 0.2091, 89.61, 10.85, 0.0197],
                        "zcoeff": [5.756, 0.0983, 0.2020, 189.32, 12.52, 0.0132]}

    _TEMP_COEFF = {"thermal_expansion": {"alpha": 1.54 * 10 ** -5, "beta": 5 * 10 ** -9, "room_temperature": 25},
                   "tempX": [7.941 * 10 ** -7, 3.134 * 10 ** -8, -4.641 * 10 ** -9, -2.188 * 10 ** -6],
                   "tempY": [7.941 * 10 ** -7, 3.134 * 10 ** -8, -4.641 * 10 ** -9, -2.188 * 10 ** -6],
                   "tempZ": [2.86 * 10 ** -6, 4.7 * 10 ** -8, 6.113 * 10 ** -8, 1.516 * 10 ** -4]}

    def __init__(self, crystal_period: float, pp_vec: list, pp_sign=Sign.MINUS) -> None:
        super().__init__(crystal_period, pp_vec, pp_sign)

    @staticmethod
    def _ns(lam, temperature, sell_coeff, temp_coeff) -> float:
        f = (temperature - 24.5) * (temperature + 570.82)
        lam2 = lam ** 2
        dispersion = abs(sell_coeff[0] + temp_coeff[0] * f
                         + (sell_coeff[1] + temp_coeff[1] * f) / (lam2 - (sell_coeff[2] + temp_coeff[2] * f) ** 2)
                         + (sell_coeff[3] + temp_coeff[3] * f) / (lam2 - sell_coeff[4] ** 2)
                         - sell_coeff[5] * lam2) ** 0.5
        return dispersion

    @classmethod
    def nx(cls, wavelength: float, temperature: float) -> float:
        return cls._ns(wavelength, temperature, cls._SELLMEIER_COEFF["xcoeff"], cls._TEMP_COEFF["tempX"])

    @classmethod
    def ny(cls, wavelength: float, temperature: float) -> float:
        return cls._ns(wavelength, temperature, cls._SELLMEIER_COEFF["ycoeff"], cls._TEMP_COEFF["tempY"])

    @classmethod
    def nz(cls, wavelength: float, temperature: float) -> float:
        return cls._ns(wavelength, temperature, cls._SELLMEIER_COEFF["zcoeff"], cls._TEMP_COEFF["tempZ"])

