from abc import ABCMeta, abstractmethod
from numpy import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, minimize_scalar
from .oposetup import OpoSetup, Speeds, Sign, RangeLimits, BeamParams, SHGspeeds, Mode, OpoPol
from itertools import product

ROOM_TEMP = 25


class ResultIsNaNError(Exception):
    def __init__(self): super().__init__('Optimization did not terminate successfully.')


class WaveVectorMismatchTooBig(Exception):
    def __init__(self, mismatch: float): super().__init__('Wavevector mismatch (' + str(mismatch) + ') is too large.')


class CrystalDispersion(metaclass=ABCMeta):
    """Abstract class containing all the relevant physics and methods for SHG and OPO calculations.

    CrystalDispersion contains methods for calculating refractive index for an arbitrary direction of propagation for
    for beams in of two polarization modes - fast or slow mode. Together with methods for calculating quasi wavevector
    of periodically polled crystals it allows for calculation of phase mismatch for any three beams propagating inside
    a crystal.

    Class contains methods for finding and plotting solutions of minimization of wavevector mismatch.
    """

    _STEP_NUM = 100
    _MAX_ANGLE = 0.2
    _MAX_WAVELENGTH = 5.0
    _MAX_TEMPERATURE = 200.0
    _MIN_TEMPERATURE = -20.0
    _MAX_PERIOD = 100.0
    _MIN_PERIOD = 1.0
    _MAX_WAVE_MISMATCH = 1e-8
    _DEL_LAMBDA = 0.03

    def __init__(self, crystal_period: float, pp_vec: list, pp_sign: Sign) -> None:
        """
        Parameters
        ----------
        crystal_period : float
            Length in micrometers of crystal period.
        pp_vec : list
            Direction of periodic polling. Vector must have a norm equal to 1.
        pp_sign: Sign
            Sign a periodic polling: is specifies if quasi wavevector is to be added or subtracted from phase mismatch.
        """
        self._crystal_period = crystal_period
        self._pp_vec = np.array(pp_vec)
        self._temperature = 0
        self._pp_sign = pp_sign

    def expansion(self, temperature: float) -> float:
        """:returns thermal expansion coefficient for a given temperature.

        Parameters
        ----------
        temperature : float
            Crystal temperature in Celsius.
        """
        exc = self._TEMP_COEFF["thermal_expansion"]
        return 1 + exc["alpha"] * (temperature - exc["room_temperature"]) \
                 + exc["beta"] * (temperature - exc["room_temperature"]) ** 2

    def get_period_vec(self, temperature: float) -> np.array:
        """:returns quasi pseudovector for a given temperature.

        Parameters
        ----------
        temperature : float
            Crystal temperature in Celsius.
        """
        if self._crystal_period == 0:
            vec = np.array([0, 0, 0])
        else:
            vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) * self._pp_vec * self._pp_sign.value
        return vec

    @classmethod
    @abstractmethod
    def nx(cls, wavelength: float, temperature: float) -> float:
        """:returns  refractive index for x polarised beam.

        Parameters
        ----------
        wavelength : float
            Beam wavelength in micrometers
        temperature : float
            Crystal temperature in Celsius.
        """

    @classmethod
    @abstractmethod
    def ny(cls, wavelength: float, temperature: float) -> float:
        """:returns  refractive index for y polarised beam.

        Parameters
        ----------
        wavelength : float
            Beam wavelength in micrometers
        temperature : float
            Crystal temperature in Celsius.
        """

    @classmethod
    @abstractmethod
    def nz(cls, wavelength: float, temperature: float) -> float:
        """:returns  refractive index for z polarised beam.

        Parameters
        ----------
        wavelength : float
            Beam wavelength in micrometers
        temperature : float
            Crystal temperature in Celsius.
        """

    @classmethod
    def plot_ref_indexs(cls, wavelength: RangeLimits, temperature0: float, temperature: RangeLimits, wavelength0: float,
                        yaxis: RangeLimits = None, stepnum: int = None) -> None:
        """Plots refractive index dependence on wavelength and temperature.

        Method designed for creating refractive index plots to compare them with the ones from literature.
        Result of this method is a plot consisting of two subplots - one depicting dependence of refractive index
        on wavelength but for constant temperature and the other with changing temperature but set wavelength.

        Parameters
        ----------
        wavelength : RangeLimits
            Range of subplot showing dependence of refractive index on wavelength. Range should be given in
            micrometres.
        temperature0 : float
            Crystal temperature in Celsius for subplot showing dependence of refractive index on wavelength.
        temperature : RangeLimits
            Range of subplot showing dependence of refractive index on temperature. Range should be given in
            Celsius.
        wavelength0 : float
            Wavelength in micrometres for showing dependence of refractive index on temperature.
        yaxis : RangeLimits, optional
            Range of y-axis.
        stepnum : int, optional
            Number of points in both subplots.
        """
        if stepnum is None:
            stepnum = cls._STEP_NUM
        del_lam = (wavelength.max - wavelength.min) / stepnum
        wavelength_array = np.arange(wavelength.min, wavelength.max, del_lam)
        del_temp = (temperature.max - temperature.min) / stepnum
        temperature_array = np.arange(temperature.min, temperature.max, del_temp)
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
        if yaxis is not None:
            ax1.set_ylim(yaxis.min, yaxis.max)
        ax1.set_xlim(wavelength.min, wavelength.max)
        ax1.plot(wavelength_array, cls.nx(wavelength_array, temperature0))
        ax1.plot(wavelength_array, cls.ny(wavelength_array, temperature0))
        ax1.plot(wavelength_array, cls.nz(wavelength_array, temperature0))
        ax1.set_xlabel('Wavelength [um]')
        ax1.set_ylabel('Ref. index')
        ax2.plot(temperature_array, cls.nx(wavelength0, temperature_array))
        ax2.plot(temperature_array, cls.ny(wavelength0, temperature_array))
        ax2.plot(temperature_array, cls.nz(wavelength0, temperature_array))
        ax2.set_xlabel('Temperature [C]')
        ax2.set_ylabel('Ref. index')
        fig.show()

    @staticmethod
    def _trans(theta: float, phi: float) -> np.array:
        return np.array([[cos(theta) * cos(phi), - sin(phi), sin(theta) * cos(phi)],
                         [cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi)], [-sin(theta), 0, cos(theta)]])

    @staticmethod
    def _dir(theta: float, phi: float) -> np.array:
        return np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])

    @staticmethod
    def _eff_ref_indx(speed: Speeds, s: np.array, n: np.array) -> float:
        [sx, sy, sz] = s
        [nx, ny, nz] = n
        b = sx ** 2 * (1 / ny ** 2 + 1 / nz ** 2) + sy ** 2 * (1 / nx ** 2 + 1 / nz ** 2) + sz ** 2 * (
                1 / nx ** 2 + 1 / ny ** 2)
        c = sx ** 2 * (1 / ny ** 2 * 1 / nz ** 2) + sy ** 2 * (1 / nx ** 2 * 1 / nz ** 2) + sz ** 2 * (
                1 / nx ** 2 * 1 / ny ** 2)
        if speed == Speeds.SLOW:
            return (abs(2 / (b - (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        elif speed == Speeds.FAST:
            return (abs(2 / (b + (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        else:
            raise TypeError('Not a speed')

    def _ref_indx(self, speed: Speeds, theta_pump: float, phi_pump: float, beam: BeamParams,
                  temperature: float) -> float:
        lam = beam.wavelength
        theta = beam.theta
        phi = beam.phi
        s = np.dot(self._trans(theta_pump, phi_pump), self._dir(theta, phi))
        n = np.array([self.nx(lam, temperature), self.ny(lam, temperature), self.nz(lam, temperature)])
        return self._eff_ref_indx(speed, s, n)

    def kvec(self, speed: Speeds, beam: BeamParams, theta_pump: float, phi_pump: float, temperature: float) -> np.array:
        """:returns wavevector for a given beam in crystal frame of reference.

        Parameters
        ----------
        speed : Speeds
            Polarization of beam.
        beam : BeamParams
            Beam parameters specified in pumping beam frame of reference.
        theta_pump, phi_pump : float
            Angles specifying pumping beam direction of propagation (eq theta_pump = 0, phi_pump = 0 -> pumping beam
            propagates along z-axis)
        temperature : float
            Crystal temperature in Celcius.
        """
        lam = beam.wavelength
        theta = beam.theta
        phi = beam.phi
        k0 = 2 * np.pi / lam
        n = self._ref_indx(speed, theta_pump, phi_pump, beam, temperature)
        return n * k0 * np.dot(self._trans(theta_pump, phi_pump), self._dir(theta, phi))

    def kpvec(self, speed: Speeds, pump: BeamParams, temperature: float) -> np.array:
        """:returns pumping beam wavevector in crystal frame of reference.

        Parameters
        ----------
        speed : Speeds
            Polarization of pumping beam.
        pump : BeamParams
            Beam parameters specified in pumping beam frame of reference.
        temperature : float
            Crystal temperature in Celcius.
        """
        beam = BeamParams(pump.wavelength, 0, 0)
        theta_pump = pump.theta
        phi_pump = pump.phi
        return self.kvec(speed, beam, theta_pump, phi_pump, temperature)

    ####################################################################################################################
    #                                                   SHG METHODS
    ####################################################################################################################

    def _shg_deltak(self, speeds: SHGspeeds, pump_beam: BeamParams, temperature: float) -> float:
        if self._crystal_period == 0:
            pp_vec = np.array([0, 0, 0])
        else:
            pp_vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) \
                     * self._pp_vec * self._pp_sign.value
        pump_speed = speeds.pump
        shg_speed = speeds.shg
        lam_shg = pump_beam.wavelength / 2
        shg_beam = BeamParams(lam_shg, pump_beam.theta, pump_beam.phi)
        kp = self.kpvec(pump_speed, pump_beam, temperature)
        kout = self.kpvec(shg_speed, shg_beam, temperature)
        return np.linalg.norm(2 * kp - kout + pp_vec)

    def _plot_shg(self, mode: Mode, arguments: RangeLimits, params: list, stepnum: int) -> plt:

        def fun(lam: float, y: float) -> float:
            pump_beam = BeamParams(lam, params[1][0], params[1][1])
            if mode == Mode.TEMPERATURE:
                temperature = y
                return self._shg_deltak(params[0], pump_beam, temperature)
            elif mode == Mode.CRYSTAL_PERIOD:
                periodic_polling = y
                temporary = self._crystal_period
                self._crystal_period = periodic_polling
                result = self._shg_deltak(params[0], pump_beam, params[3])
                self._crystal_period = temporary
                return result
            else:
                return np.nan

        def shg(x: np.array, centarg: np.array) -> float:
            sol = fsolve(fun, centarg, x, full_output=True)
            if sol[2] == 1:
                return sol[0][0]
            else:
                return np.nan

        v_shg = np.vectorize(shg)
        delx = (arguments.max - arguments.min) / stepnum
        arg = np.arange(arguments.min, arguments.max, delx)
        fig, ax = plt.subplots(1, 1, constrained_layout=True, sharey=True)
        ax.plot(arg, v_shg(arg, params[2]))
        return fig, ax

    def plot_temp_shg(self, speeds: SHGspeeds, temperatures: RangeLimits, theta_pump: float, phi_pump: float,
                      starting_lam: float = 1.1, stepnum: int = None) -> plt:
        """Plots SHG dependence on temperature.

        Temperature changes refractive index and crystal period. These changes result in shifting pumping wavelength for
        which SHG (in a given direction and polarization combination) is efficient.

        Parameters
        ----------
        speeds : SHGspeeds
            Pumping beam and SHG beam polarizations.
        temperatures : RangeLimits
            Temperature range in Celcius for plotting SHG wavelength dependence on temperature.
        theta_pump, phi_pump : float
            Angles specifying pumping beam direction of propagation (eq theta_pump = 0, phi_pump = 0 -> pumping beam
            propagates along z-axis)
        starting_lam : float, optional
             The starting estimate for the roots of SHG phase mismatch.
        stepnum : int, optional
            Number of points in both subplots
        """
        params = [speeds, [theta_pump, phi_pump], starting_lam]
        if stepnum is None:
            stepnum = self._STEP_NUM
        fig, ax = self._plot_shg(Mode.TEMPERATURE, temperatures, params, stepnum)
        ax.set_ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        ax.set_xlabel('Temperature [C]')
        fig.suptitle('Dependence of SHG on temperature')
        fig.show()

    def plot_ppol_shg(self, speeds: SHGspeeds, polling: RangeLimits, theta_pump: float, phi_pump: float,
                      starting_lam: float = 1.1, temperature: float = ROOM_TEMP, stepnum: int = None) -> plt:
        """Plots SHG dependence on periodic polling.

        Changing periodic polling result in shifting pumping wavelength for
        which SHG (in a given direction and polarization combination) is efficient.

        Parameters
        ----------
        speeds : SHGspeeds
            Pumping beam and SHG beam polarizations.
        polling : RangeLimits
            Periodic polling range in micrometers for plotting SHG wavelength dependence on a crystal period.
        theta_pump, phi_pump : float
            Angles specifying pumping beam direction of propagation (eq theta_pump = 0, phi_pump = 0 -> pumping beam
            propagates along z-axis).
        starting_lam : float, optional
             The starting estimate for the roots of SHG phase mismatch.
        temperature : float, optional
              Crystal temperature in Celcius.
        stepnum : int, optional
            Number of points in both subplots.
        """
        params = [speeds, [theta_pump, phi_pump], starting_lam, temperature]
        if stepnum is None:
            stepnum = self._STEP_NUM
        fig, ax = self._plot_shg(Mode.CRYSTAL_PERIOD, polling, params, stepnum)
        ax.set_ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        ax.set_xlabel('Periodic polling [' + '\u03BC' + 'm]')
        fig.suptitle('Dependence of SHG on periodic polling')
        fig.show()

    ####################################################################################################################
    #                                                   OPO METHODS: General methods
    ####################################################################################################################

    def _theta_idler(self, pump_speed: Speeds, pump_beam: BeamParams, signal_speed: Speeds, signal_beam: BeamParams,
                     temperature: float) -> float:
        if self._crystal_period == 0:
            pp_vec = 0
        else:
            pp_vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) * self._pp_sign.value
        propdir = self._dir(pump_beam.theta, pump_beam.phi)
        nsignal = self._ref_indx(signal_speed, pump_beam.theta, pump_beam.phi, signal_beam, temperature)
        kp = np.linalg.norm(self.kpvec(pump_speed, pump_beam, temperature))
        k_parallel = np.dot(self._pp_vec, propdir) * pp_vec
        ks = nsignal * 2 * np.pi / signal_beam.wavelength
        ks_perpendicular = ks * sin(signal_beam.theta)
        ks_parallel = ks * cos(signal_beam.theta)
        return np.arcsin(ks_perpendicular / ((kp - ks_parallel + k_parallel) ** 2 + ks_perpendicular ** 2) ** 0.5)

    def _deltak0vec(self, speeds: Speeds, pump_beam: BeamParams, signal_beam: BeamParams,
                    temperature: float) -> np.array:
        if self._crystal_period == 0:
            pp_vec = np.array([0, 0, 0])
        else:
            pp_vec = 2 * np.pi / (
                    self._crystal_period * self.expansion(temperature)) * self._pp_vec * self._pp_sign.value
        [pump_speed, signal_speed, idler_speed] = speeds
        lam_idler = 1 / (1 / pump_beam.wavelength - 1 / signal_beam.wavelength)
        theta_idler = self._theta_idler(pump_speed, pump_beam, signal_speed, signal_beam, temperature)
        idler_beam = BeamParams(wavelength=lam_idler, theta=theta_idler, phi=signal_beam.phi + np.pi)
        kp = self.kpvec(pump_speed, pump_beam, temperature)
        ks = self.kvec(signal_speed, signal_beam, pump_beam.theta, pump_beam.phi, temperature)
        ki = self.kvec(idler_speed, idler_beam, pump_beam.theta, pump_beam.phi, temperature)
        vec = np.dot(pp_vec, self._dir(pump_beam.theta, pump_beam.phi)) * self._dir(pump_beam.theta, pump_beam.phi)
        return kp - ks - ki + vec

    def _find_signal_beam(self, speedlist: np.array, pump_beam: BeamParams, temperature: float, lam_signal0=1.5,
                          ang_signal0=1e-6) -> np.array:

        def find_signal(x):
            signal = BeamParams(wavelength=x[0], theta=x[1], phi=0)
            dk0 = self._deltak0vec(speedlist, pump_beam, signal, temperature)
            return np.linalg.norm(dk0)

        sol = minimize(find_signal, np.array([lam_signal0, ang_signal0]),
                       bounds=((pump_beam.wavelength + self._DEL_LAMBDA, pump_beam.wavelength * 2),
                               (0, self._MAX_ANGLE)), method='L-BFGS-B', options={'ftol': self._MAX_WAVE_MISMATCH / 10})
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        deltak = sol.fun
        signal_beam = BeamParams(wavelength=sol.x[0], theta=sol.x[1], phi=0)
        return np.array([signal_beam, deltak, sol.success], dtype=object)

    def _plot_opo(self, mode: Mode, lim: RangeLimits, params: list, stepnum: int) -> list:

        def opo(x):
            set_period = self._crystal_period
            try:
                if mode == Mode.TEMPERATURE:
                    temperature = x
                    speeds = params[0]
                    pump_beam = BeamParams(wavelength=params[1], theta=params[2], phi=params[3])
                    [signal_beam, delk, _] = self._find_signal_beam(speeds, pump_beam, temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], pump_beam, speeds[1], signal_beam, temperature)
                    return np.array([signal_beam.wavelength, signal_beam.theta, signal_beam.phi, theta_idl, delk])
                elif mode == Mode.WAVELENGTH:
                    speeds = params[0]
                    temperature = params[3]
                    pump_beam = BeamParams(wavelength=x, theta=params[1], phi=params[2])
                    [signal_beam, delk, _] = self._find_signal_beam(speeds, pump_beam, temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], pump_beam, speeds[1], signal_beam, temperature)
                    return np.array([signal_beam.wavelength, signal_beam.theta, signal_beam.phi, theta_idl, delk])
                elif mode == Mode.CRYSTAL_PERIOD:
                    period = x
                    self._crystal_period = period
                    speeds = params[0]
                    temperature = params[4]
                    pump_beam = BeamParams(wavelength=params[1], theta=params[2], phi=params[3])
                    [signal_beam, delk, _] = self._find_signal_beam(speeds, pump_beam, temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], pump_beam, speeds[1], signal_beam, temperature)
                    return np.array([signal_beam.wavelength, signal_beam.theta, signal_beam.phi, theta_idl, delk])
                else:
                    return None
            except Exception as exc:
                print(exc)
                return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
            finally:
                self._crystal_period = set_period

        vopo = np.vectorize(opo, otypes=[np.ndarray])
        delx = (lim.max - lim.min) / stepnum
        arg = np.arange(lim.min, lim.max, delx)
        result = np.vstack(vopo(arg))
        [lam_signal, theta_signal, theta_idler, deltak] = [result[:, 0], result[:, 1], result[:, 3], result[:, 4]]
        if mode != Mode.WAVELENGTH:
            lam_idler = 1 / (1 / params[1] - 1 / lam_signal)
        else:
            lam_idler = 1 / (1 / arg - 1 / lam_signal)
        [fig, (ax1, ax2, ax3)] = plt.subplots(3)
        ax1.plot(arg, lam_signal)
        ax1.plot(arg, lam_idler)
        ax1.set(ylabel='Wavelength [' + '\u03BC' + 'm]')
        ax2.plot(arg, theta_signal)
        ax2.plot(arg, theta_idler)
        ax2.set(ylabel='Opening angle [rad]')
        ax3.scatter(arg, np.log10(deltak + 10 ** -20))
        ax3.set(ylabel='log($\Delta k$) [log(1/' + '\u03BC' + 'm )]')
        return [fig, (ax1, ax2, ax3)]

    def plot_wave_opo(self, speeds: list, pump: RangeLimits, theta_pump: float, phi_pump: float,
                      temperature: float = ROOM_TEMP, stepnum: int = None) -> None:
        """Plots OPO dependence on pumping wavelength.

        Plots idler and signal wavelength dependence on pumping wavelength.
        Temperature and periodic polling are constant.

        Parameters
        ----------
        speeds : list
            Polarization configuration defined as: [pump_polarization, signal_polarization, idler_polarization].
        pump : RangeLimits
            Pumping wavelength range in micrometres for plotting OPO wavelengths dependence on pumping wavelength.
        theta_pump, phi_pump : float
            Angles specifying pumping beam direction of propagation (e.g theta_pump = 0, phi_pump = 0 -> pumping beam
            propagates along z-axis).
        temperature : float, optional
              Crystal temperature in Celcius.
        stepnum : int, optional
            Number of points in both subplots.
        """
        params = [speeds, theta_pump, phi_pump, temperature]
        if stepnum is None:
            stepnum = self._STEP_NUM
        [fig, (ax1, ax2, ax3)] = self._plot_opo(Mode.WAVELENGTH, pump, params, stepnum)
        ax3.set_xlabel('Pump wavelength [' + '\u03BC' + 'm]')
        fig.suptitle('Dependence of OPO wavelengths on pumping wavelength')
        fig.show()

    def plot_wavelength(self, setup: OpoSetup, pump_wavelength: RangeLimits, stepnum: int = None) -> None:
        """Plots OPO dependence on pumping wavelength.

        Plots idler and signal wavelength dependence on pumping wavelength.
        Temperature and periodic polling are constant.

        Parameters
        ----------
        setup : OpoSetup
            Instance of OpoSetup. Instance contains information about polarizations, wavelengths etc.
        pump_wavelength : RangeLimits
            Pumping wavelength range in micrometres for plotting OPO wavelengths dependence on pumping wavelength.
        stepnum : int, optional
            Number of points in both subplots.
        """
        speeds = [setup.pol_pump, setup.pol_signal, setup.pol_idler]
        theta_pump = setup.theta_pump
        phi_pump = setup.phi_pump
        temperature = setup.temperature
        temp = [self._crystal_period, self._pp_sign, np.copy(self._pp_vec)]
        self._crystal_period = setup.crystal_period
        self._pp_sign = setup.sign
        self._pp_vec = setup.pp_dir
        self.plot_wave_opo(speeds, pump_wavelength, theta_pump, phi_pump, temperature, stepnum)
        self._crystal_period = temp[0]
        self._pp_sign = temp[1]
        self._pp_vec = temp[2]

    def plot_temp_opo(self, speeds: list, temperature: RangeLimits, lam_pump: float, theta_pump: float,
                      phi_pump: float, stepnum: int = None) -> None:
        """Plots OPO dependence on crystal temperature.

        Plots idler and signal wavelength dependence on crystal temperature.
        Pumping wavelength and periodic polling are constant.

        Parameters
        ----------
        speeds : list
            Polarization configuration defined as: [pump_polarization, signal_polarization, idler_polarization].
        temperature : RangeLimits
            Temperature range in Celsius for plotting OPO wavelengths dependence on temperature.
        theta_pump, phi_pump : float
            Angles specifying pumping beam direction of propagation (e.g theta_pump = 0, phi_pump = 0 -> pumping beam
            propagates along z-axis).
        lam_pump : float, optional
              Pumping wavelength in micrometers.
        stepnum : int, optional
            Number of points in both subplots.
        """
        params = [speeds, lam_pump, theta_pump, phi_pump]
        if stepnum is None:
            stepnum = self._STEP_NUM
        [fig, (ax1, ax2, ax3)] = self._plot_opo(Mode.TEMPERATURE, temperature, params, stepnum)
        ax3.set(xlabel='Temperature [C]')
        fig.suptitle('Dependence of OPO on temperature')
        fig.show()

    def plot_temperature(self, setup: OpoSetup, temperature: RangeLimits, stepnum: int = None) -> None:
        """Plots OPO dependence on crystal temperature.

        Plots idler and signal wavelength dependence on crystal temperature for a given OPO setup.

        Parameters
        ----------
        setup : OpoSetup
            Instance of OpoSetup. Instance contains information about polarizations, wavelengths etc.
        temperature : RangeLimits
            Temperature range in Celsius for plotting OPO wavelengths dependence on temperature.
        stepnum : int, optional
            Number of points in both subplots.
        """
        speeds = [setup.pol_pump, setup.pol_signal, setup.pol_idler]
        lam_pump = setup.lam_pump
        theta_pump = setup.theta_pump
        phi_pump = setup.phi_pump
        temp = [self._crystal_period, self._pp_sign, np.copy(self._pp_vec)]
        self._crystal_period = setup.crystal_period
        self._pp_sign = setup.sign
        self._pp_vec = setup.pp_dir
        self.plot_temp_opo(speeds, temperature, lam_pump, theta_pump, phi_pump, stepnum)
        self._crystal_period = temp[0]
        self._pp_sign = temp[1]
        self._pp_vec = temp[2]

    def plot_periodicpol_opo(self, speeds: list, period: RangeLimits, lam_pump: float,
                             theta_pump: float, phi_pump: float, temperature: float, stepnum: int = None) -> None:
        """Plots OPO dependence on periodic polling length.

         Plots idler and signal wavelength dependence on periodic polling length.
         Temperature and pumping wavelength are constant.

         Parameters
         ----------
         speeds : list
             Polarization configuration defined as: [pump_polarization, signal_polarization, idler_polarization].
         period : RangeLimits
            Length range in micrometers for plotting OPO wavelengths dependence periodic polling length.
         temperature : float
             Crystal temperature in Celsius.
         theta_pump, phi_pump : float
             Angles specifying pumping beam direction of propagation (e.g theta_pump = 0, phi_pump = 0 -> pumping beam
             propagates along z-axis).
         lam_pump : float, optional
               Pumping wavelength in micrometers.
         stepnum : int, optional
             Number of points in both subplots.
         """
        params = [speeds, lam_pump, theta_pump, phi_pump, temperature]
        if stepnum is None:
            stepnum = self._STEP_NUM
        [fig, (ax1, ax2, ax3)] = self._plot_opo(Mode.CRYSTAL_PERIOD, period, params, stepnum)
        ax3.set_xlabel('Periodic polling [' + '\u03BC' + 'm]')
        fig.suptitle('Dependence of OPO wavelengths on periodic polling')
        fig.show()

    def plot_periodicpolling(self, setup: OpoSetup, period: RangeLimits, stepnum=None) -> None:
        """Plots OPO dependence on pumping wavelength.

        Plots idler and signal wavelength dependence on pumping wavelength for a given OPO setup.

        Parameters
        ----------
        setup : OpoSetup
            Instance of OpoSetup. Instance contains information about polarizations, wavelengths etc.
        period : RangeLimits
            Length range in micrometers for plotting OPO wavelengths dependence periodic polling length.
        stepnum : int, optional
            Number of points in both subplots.
        """
        speeds = [setup.pol_pump, setup.pol_signal, setup.pol_idler]
        lam_pump = setup.lam_pump
        theta_pump = setup.theta_pump
        phi_pump = setup.phi_pump
        temperature = setup.temperature
        temp = [self._crystal_period, self._pp_sign, np.copy(self._pp_vec)]
        self._pp_sign = setup.sign
        self._pp_vec = setup.pp_dir
        self.plot_periodicpol_opo(speeds, period, lam_pump, theta_pump, phi_pump, temperature, stepnum)
        self._crystal_period = temp[0]
        self._pp_sign = temp[1]
        self._pp_vec = temp[2]

    ####################################################################################################################
    #                       OPO METHODS: Find OPO for crystal with arbitrary T but with set PP and pumping wavelength
    ####################################################################################################################

    def find_opo(self, lam_pump: float, temperature: float, quiet=True, look_along_polling=True) -> list:
        """:returns list of OpoSetup

        For a crystal defined in __init__ find for a given temperature and pumping wavelength all possible (efficent)
        OPO processes. Function goes over all possible combinations of polarization, sign of pseudovector and - if
        look_along_polling is False - all possible direction of pumping. For each combination function tries to find set
        of opening angles and wavelengths of idler and signal beams so that they minimize wavector mismatch.

        Parameters
        ----------
        lam_pump : float
            Pumping wavelength in micrometers.
        temperature : float
            Crystal temperature.
        quiet : bool, optional
            If True, function prints error and messages.
        look_along_polling : bool, optional
            If True function assumes that pumping happens along periodic polling direction.
        """
        print('findOPO: starting search...')
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystal_dir = [theta, phi]
        if look_along_polling:
            pump_dirs = [crystal_dir]
        else:
            pump_dirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        crystal_sign = self._pp_sign
        for direction in pump_dirs:
            pump_beam = BeamParams(wavelength=lam_pump, theta=direction[0], phi=direction[1])
            if direction == crystal_dir:
                signs = [1, -1]
            else:
                signs = [1]
            for speeds in product(Speeds, repeat=3):
                pump_speed = speeds[0]
                signal_speed = speeds[1]
                idler_speed = speeds[2]
                speed_com = np.array([pump_speed, signal_speed, idler_speed])
                for sign in signs:
                    self._pp_sign = Sign(sign)
                    try:
                        [signal_beam, deltak, issuccess] = self._find_signal_beam(speed_com, pump_beam, temperature)
                        lam_idler = 1 / (1 / lam_pump - 1 / signal_beam.wavelength)
                        theta_idler = self._theta_idler(pump_speed, pump_beam, signal_speed, signal_beam, temperature)
                        idler_beam = BeamParams(wavelength=lam_idler, theta=theta_idler, phi=np.pi)
                        instance = OpoSetup(bool(issuccess), deltak, pump_beam, signal_beam, idler_beam, temperature,
                                            self._crystal_period, self._pp_sign, self._pp_vec, pump_speed, idler_speed,
                                            signal_speed)
                        instance.display()
                        setups.append(instance)
                    except Exception as inst:
                        if not quiet:
                            print(inst.args)

        self._pp_sign = crystal_sign
        print('findOPO: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups

    ####################################################################################################################
    #                       OPO METHODS: Find PP or T, but with set OPO and pumping direction
    ####################################################################################################################

    def _find_setup_parameters(self, mode: Mode, speeds: np.array, pump_beam: BeamParams, lam_signal: float,
                               pp0: float, temperature0: float, ang_signal0: float) -> np.array:
        if pp0 is None:
            pp0 = self._crystal_period
        if mode == Mode.TEMPERATURE:
            self._crystal_period = pp0
            (x0, xmin, xmax) = (temperature0, self._MIN_TEMPERATURE, self._MAX_TEMPERATURE)

            def find_signal(x):
                _temperature = x[1]
                signal_beam = BeamParams(wavelength=lam_signal, theta=x[0], phi=0)
                dk0 = self._deltak0vec(speeds, pump_beam, signal_beam, _temperature)
                return np.linalg.norm(dk0)
        elif mode == Mode.CRYSTAL_PERIOD:
            (x0, xmin, xmax) = (pp0, self._MIN_PERIOD, self._MAX_PERIOD)

            def find_signal(x):
                self._crystal_period = x[1]
                signal_beam = BeamParams(wavelength=lam_signal, theta=x[0], phi=np.pi)
                dk0 = self._deltak0vec(speeds, pump_beam, signal_beam, temperature0)
                return np.linalg.norm(dk0)
        else:
            raise TypeError('Incorrect mode.')
        sol = minimize(find_signal, np.array([ang_signal0, x0]), bounds=((0, self._MAX_ANGLE), (xmin, xmax)),
                       method='L-BFGS-B', options={'ftol': self._MAX_WAVE_MISMATCH / 10})
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        theta_signal = sol.x[0]
        deltak = sol.fun
        if mode == Mode.TEMPERATURE:
            (temperature, period) = (sol.x[1], self._crystal_period)
        else:
            (temperature, period) = (temperature0, sol.x[1])
        return np.array([theta_signal, temperature, period, deltak, sol.success])

    def _find_crystal_setup(self, mode: Mode, lam_pump: float, lam_signal: float, pol_mode: OpoPol, temperature0: float,
                            pp_guess: float, ang_signal0: float, quiet: bool, look_along_polling) -> list:
        temp = self._crystal_period
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystal_dir = [theta, phi]
        if look_along_polling:
            pump_dirs = [crystal_dir]
        else:
            pump_dirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        crystal_sign = self._pp_sign
        for direction in pump_dirs:
            if direction == crystal_dir:
                signs = [1, -1]
            else:
                signs = [1]
            pump_beam = BeamParams(wavelength=lam_pump, theta=direction[0], phi=direction[1])
            for speeds in product(Speeds, repeat=2):
                pump_speed = speeds[0]
                signal_speed = speeds[1]
                if pol_mode == OpoPol.HETERO:
                    idler_speed = Speeds((signal_speed.value + 1) % 2)
                elif pol_mode == OpoPol.HOMO:
                    idler_speed = signal_speed
                else:
                    raise TypeError('Incorrect polarization mode.')
                speed_list = np.array([pump_speed, signal_speed, idler_speed])
                for sign in signs:
                    self._pp_sign = Sign(sign)
                    try:
                        parameters = self._find_setup_parameters(mode, speed_list, pump_beam, lam_signal, pp_guess,
                                                                 temperature0, ang_signal0)
                        [theta_signal, temperature, period, deltak, issuccess] = parameters
                        lam_idler = 1 / (1 / lam_pump - 1 / lam_signal)
                        signal_beam = BeamParams(wavelength=lam_signal, theta=theta_signal, phi=0)
                        theta_idler = self._theta_idler(pump_speed, pump_beam, signal_speed, signal_beam, temperature)
                        idler_beam = BeamParams(wavelength=lam_idler, theta=theta_idler, phi=np.pi)
                        instance = OpoSetup(bool(issuccess), deltak, pump_beam, signal_beam, idler_beam, temperature,
                                            period, self._pp_sign, self._pp_vec, pump_speed, idler_speed, signal_speed)
                        instance.display()
                        setups.append(instance)
                    except Exception as inst:
                        if not quiet:
                            print(inst.args)
                    finally:
                        self._crystal_period = temp
                        self._pp_sign = crystal_sign
        return setups

    def find_crystal_period(self, lam_pump: float, lam_signal: float, pol_mode: OpoPol, temperature=ROOM_TEMP,
                            pp_guess: float = None, ang_signal0: float = 0.01, quiet=True,
                            look_along_polling=True) -> list:
        """:returns list of OpoSetup

        For a given crystal method finds periodic polling length which allows for a desired OPO process to happen.
        Desired process is defined by parameters.

        Parameters
        ----------
        lam_pump : float
            Pumping wavelength in micrometers.
        lam_signal : float
            Signal wavelength in micrometers
        pol_mode : OpoPol
            Specifies if OPO process should produce beams of the same (HOMO) or different polarizations (HETERO).
        temperature : float
            Crystal temperature.
        pp_guess : float, optional
            Initial guess of periodic polling length for minimization function.
        ang_signal0 : float, optional
            Initial guess of signal beam opening angle for minimization function.
        quiet : bool, optional
            If True, function prints error and messages.
        look_along_polling : bool, optional
            If True function assumes that pumping happens along periodic polling direction.
        """
        print('find_crystal_period: starting search...')
        setups = self._find_crystal_setup(Mode.CRYSTAL_PERIOD, lam_pump, lam_signal, pol_mode, temperature, pp_guess,
                                          ang_signal0, quiet, look_along_polling)
        print('find_crystal_period: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups

    def find_crystal_temperature(self, lam_pump: float, lam_signal: float, pol_mode: OpoPol,
                                 temperature_guess: float = ROOM_TEMP, ang_signal0: float = 0.01,
                                 quiet=True, look_along_polling=True) -> list:
        """:returns list of OpoSetup

        For a given crystal method finds temperature which allows for a desired OPO process to happen.
        Desired process is defined by parameters. Periodic polling length is defined at initialization process.

        Parameters
        ----------
        lam_pump : float
            Pumping wavelength in micrometers.
        lam_signal : float
            Signal wavelength in micrometers
        pol_mode : OpoPol
            Specifies if OPO process should produce beams of the same (HOMO) or different polarizations (HETERO).
        temperature_guess : float
            Initial guess of crystal temperature for minimization function.
        ang_signal0 : float, optional
            Initial guess of signal beam opening angle for minimization function.
        quiet : bool, optional
            If True, function prints error and messages.
        look_along_polling : bool, optional
            If True function assumes that pumping happens along periodic polling direction.
        """
        print('find_crystal_temperature: starting search...')
        setups = self._find_crystal_setup(Mode.TEMPERATURE, lam_pump, lam_signal, pol_mode, temperature_guess, None,
                                          ang_signal0, quiet, look_along_polling)
        print('find_crystal_temperature: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups

    ####################################################################################################################
    #                       OPO METHODS: Find setup for a collinear OPO
    ####################################################################################################################

    def _find_period(self, speeds: np.array, pump_beam: BeamParams, lam_signal: float,
                     temperature0: float) -> np.array:

        def find_period_length(x):
            self._crystal_period = x
            signal_beam = BeamParams(wavelength=lam_signal, theta=1e-12, phi=0)
            dk0 = self._deltak0vec(speeds, pump_beam, signal_beam, temperature0)
            return np.linalg.norm(dk0)

        sol = minimize_scalar(find_period_length, bounds=[self._MIN_PERIOD, self._MAX_PERIOD], method="bounded",
                              options={'xatol': 1e-16, 'maxiter': 1000, 'disp': 0})
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        period = sol.x
        deltak = sol.fun
        return np.array([temperature0, period, deltak, sol.success])

    def _find_temperature(self, speeds: np.array, pump_beam: BeamParams, lam_signal, pp0) -> np.array:
        if pp0 > 0:
            self._crystal_period = pp0

        def find_temp(x):
            _temperature = x
            signal_beam = BeamParams(wavelength=lam_signal, theta=1e-12, phi=0)
            dk0 = self._deltak0vec(speeds, pump_beam, signal_beam, _temperature)
            return np.linalg.norm(dk0)

        sol = minimize_scalar(find_temp, bounds=(self._MIN_TEMPERATURE, self._MAX_TEMPERATURE), method="bounded",
                              options={'xatol': 1e-16, 'maxiter': 1000, 'disp': 0})
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        temperature = sol.x
        deltak = sol.fun
        return np.array([temperature, self._crystal_period, deltak, sol.success])

    def _find_optimum(self, speeds: np.array, pump_beam: BeamParams, lam_signal: float,
                      temperature0, pp0) -> np.array:
        if pp0 < 0:
            pp0 = self._crystal_period

        def find_temp(x):
            _temperature = x[0]
            self._crystal_period = x[1]
            signal_beam = BeamParams(wavelength=lam_signal, theta=1e-12, phi=0)
            dk0 = self._deltak0vec(speeds, pump_beam, signal_beam, _temperature)
            return np.linalg.norm(dk0)

        sol = minimize(find_temp, np.array([temperature0, pp0]),
                       bounds=((self._MIN_TEMPERATURE, self._MAX_TEMPERATURE), (self._MIN_PERIOD, self._MAX_PERIOD)),
                       method='L-BFGS-B', options={'ftol': self._MAX_WAVE_MISMATCH / 10})
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        temperature = sol.x[0]
        deltak = sol.fun
        return np.array([temperature, self._crystal_period, deltak, sol.success])

    def find_collinear_opo(self, mode: Mode, lam_pump: float, lam_signal: float, pol_mode: OpoPol, x0=None, quiet=True,
                           look_along_polling=True) -> list:
        """:returns list of OpoSetup

        For a given crystal method finds temperature which allows for a desired OPO process to happen.
        Desired process is defined by parameters. Periodic polling length is defined at initialization process.

        Parameters
        ----------
        mode: Mode
            Specifies if method should look for optimal temperature (Mode.TEMPERATURE), optimal length of periodic
            polling (Mode.CRYSTAL_PERIOD) or optimal crystal period and temperature (Mode.TEMP_AND_PERIOD)
        lam_pump : float
            Pumping wavelength in micrometers.
        lam_signal : float
            Signal wavelength in micrometers
        pol_mode : OpoPol
            Specifies if OPO process should produce beams of the same (HOMO) or different polarizations (HETERO).
        x0 : float or list
            Initial guess of crystal temperature (float; Mode.TEMPERATURE) or periodic polling
            (float; Mode.CRYSTAL_PERIOD) or both (list -> [temperature, periodic_polling], Mode.TEMP_AND_PERIOD)
            for minimization function.
        quiet : bool, optional
            If True, function prints error and messages.
        look_along_polling : bool, optional
            If True function assumes that pumping happens along periodic polling direction.
        """
        temp = self._crystal_period
        print('findCollinearOPO: starting search...')
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystal_dir = [theta, phi]
        if look_along_polling:
            pump_dirs = [crystal_dir]
        else:
            pump_dirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        crystal_sign = self._pp_sign
        for direction in pump_dirs:
            if direction == crystal_dir:
                signs = [1, -1]
            else:
                signs = [1]
            pump_beam = BeamParams(wavelength=lam_pump, theta=direction[0], phi=direction[1])
            for sign in signs:
                self._pp_sign = Sign(sign)
                for speed_com in product(Speeds, repeat=2):
                    pump_speed = speed_com[0]
                    signal_speed = speed_com[1]
                    if pol_mode == OpoPol.HETERO:
                        idler_speed = Speeds((signal_speed.value + 1) % 2)
                    elif pol_mode == OpoPol.HOMO:
                        idler_speed = signal_speed
                    else:
                        raise TypeError('Incorrect polarization mode.')
                    speeds = np.array([pump_speed, signal_speed, idler_speed])
                    try:
                        if mode == Mode.TEMPERATURE:
                            if x0 is None:
                                pp0 = -1
                            else:
                                pp0 = x0
                            [temperature, period, deltak, issuccess] = self._find_temperature(speeds, pump_beam,
                                                                                              lam_signal, pp0)
                        elif mode == Mode.CRYSTAL_PERIOD:
                            if x0 is None:
                                temperature0 = ROOM_TEMP
                            else:
                                temperature0 = x0
                            [temperature, period, deltak, issuccess] = self._find_period(speeds, pump_beam, lam_signal,
                                                                                         temperature0)
                        elif mode == Mode.TEMP_AND_PERIOD:
                            if x0 is None:
                                temperature0 = ROOM_TEMP
                                pp0 = -1
                            else:
                                [temperature0, pp0] = x0
                            [temperature, period, deltak, issuccess] = self._find_optimum(speeds, pump_beam, lam_signal,
                                                                                          temperature0, pp0)
                        else:
                            raise Exception('Incorrect mode')
                        lam_idler = 1 / (1 / lam_pump - 1 / lam_signal)
                        signal_beam = BeamParams(wavelength=lam_signal, theta=1e-12, phi=0)
                        idler_beam = BeamParams(wavelength=lam_idler, theta=1e-12, phi=np.pi)
                        instance = OpoSetup(bool(issuccess), deltak, pump_beam, signal_beam, idler_beam, temperature,
                                            period, self._pp_sign, self._pp_vec, pump_speed, idler_speed, signal_speed)
                        instance.display()
                        setups.append(instance)
                    except Exception as exc:
                        if not quiet:
                            print(exc.args)
                    finally:
                        self._crystal_period = temp
                        self._pp_sign = crystal_sign
        print('findCollinearOPO: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups
