from abc import ABCMeta, abstractmethod
from numpy import cos, sin
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, minimize_scalar
from .oposetup import OpoSetup, Speeds, Sign

ROOM_TEMP = 25
C_CONST = 3 * 10 ** 14


class ResultIsNaNError(Exception):
    def __init__(self): super().__init__('Optimization did not terminate successfully.')


class WaveVectorMismatchTooBig(Exception):
    def __init__(self, mismatch: float): super().__init__('Wavevector mismatch (' + str(mismatch) + ') is too large.')


class OpoPol(Enum):
    HOMO = 0
    HETERO = 1


class Mode(Enum):
    TEMPERATURE = 1
    WAVELENGTH = 2
    CRYSTAL_PERIOD = 3
    TEMP_AND_PERIOD = 4


class CrystalDispersion(metaclass=ABCMeta):
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
        self._crystal_period = crystal_period
        self._pp_vec = np.array(pp_vec)
        self._temperature = 0
        self._pp_sign = pp_sign

    def expansion(self, temperature: float) -> float:
        exc = self._TEMP_COEFF["thermal_expansion"]
        return 1 + exc["alpha"] * (temperature - exc["room_temperature"]) \
               + exc["beta"] * (temperature - exc["room_temperature"]) ** 2

    def get_period_vec(self, temperature: float) -> np.array:
        if self._crystal_period == 0:
            vec = np.array([0, 0, 0])
        else:
            vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) * self._pp_vec * self._pp_sign.value
        return vec

    @abstractmethod
    def nx(self, wavelength: float, temperature: float) -> float:
        """returns  ref. index for x polarised wave"""

    @abstractmethod
    def ny(self, wavelength: float, temperature: float) -> float:
        """returns  ref. index for y polarised wave"""

    @abstractmethod
    def nz(self, wavelength: float, temperature: float) -> float:
        """returns  ref. index for z polarised wave"""

    def plot_ref_indexs(self, lambda_lim: list, temperature: float, temperature_lim: list, wavelength: float,
                        stepnum=None) -> None:
        if stepnum is not None:
            stepnum = self._STEP_NUM
        dellam = (lambda_lim[1] - lambda_lim[0]) / stepnum
        lam = np.arange(lambda_lim[0], lambda_lim[1], dellam)
        del_temp = (temperature_lim[1] - temperature_lim[0]) / stepnum
        temperatures = np.arange(temperature_lim[0], temperature_lim[1], del_temp)
        plt.subplot(121)
        plt.plot(self.nx(lam, temperature))
        plt.plot(self.ny(lam, temperature))
        plt.plot(self.nz(lam, temperature))
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Ref. index')
        plt.subplot(122)
        plt.plot(self.nx(wavelength, temperatures))
        plt.plot(self.ny(wavelength, temperatures))
        plt.plot(self.nz(wavelength, temperatures))
        plt.xlabel('Temperature [C]')
        plt.ylabel('Ref. index')
        plt.show()

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
        if speed == Speeds(0):
            return (abs(2 / (b - (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        elif speed == Speeds(1):
            return (abs(2 / (b + (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        else:
            raise TypeError('Not a speed')

    def _ref_indx(self, speed: Speeds, theta_pump: float, phi_ump: float, lam: float,
                  theta: float, phi: float, temperature: float) -> float:
        s = np.dot(self._trans(theta_pump, phi_ump), self._dir(theta, phi))
        n = [self.nx(lam, temperature), self.ny(lam, temperature), self.nz(lam, temperature)]
        return self._eff_ref_indx(speed, s, n)

    def _ref_indx_pump(self, speed: Speeds, lam: float, theta: float, phi: float, temperature: float) -> float:
        return self._ref_indx(speed, 0, 0, lam, theta, phi, temperature)

    def kvec(self, speed: Speeds, lam: float, theta_pump: float, phi_pump: float, theta: float,
             phi: float, temperature: float) -> np.array:
        k0 = 2 * np.pi / lam
        n = self._ref_indx(speed, theta_pump, phi_pump, lam, theta, phi, temperature)
        return n * k0 * np.dot(self._trans(theta_pump, phi_pump), self._dir(theta, phi))

    def kpvec(self, speed: Speeds, lam: float, theta_pump: float, phi_pump: float, temperature: float) -> np.array:
        return self.kvec(speed, lam, 0, 0, theta_pump, phi_pump, temperature)

    ####################################################################################################################
    #                                                   SHG METHODS
    ####################################################################################################################

    def _shg_deltak(self, speeds: Speeds, lam_pump: float, theta_pump: float,
                    phi_pump: float, temperature: float) -> float:
        if self._crystal_period == 0:
            pp_vec = np.array([0, 0, 0])
        else:
            pp_vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) \
                     * self._pp_vec * self._pp_sign.value
        [pump_speed, shg_speed] = speeds
        lam_shg = lam_pump / 2
        kp = self.kpvec(pump_speed, lam_pump, theta_pump, phi_pump, temperature)
        kout = self.kpvec(shg_speed, lam_shg, theta_pump, phi_pump, temperature)
        return np.linalg.norm(2 * kp - kout + pp_vec)

    def _plot_shg(self, mode: Mode, lam_lim: list, params: list, stepnum: int) -> plt:
        const = np.array(params)

        def fun(lam: float, y: float) -> float:
            if mode == Mode.TEMPERATURE:
                temperature = y
                return self._shg_deltak(const[0], lam, const[1][0], const[1][1], temperature)
            elif mode == Mode.CRYSTAL_PERIOD:
                periodic_polling = y
                temporary = self._crystal_period
                self._crystal_period = periodic_polling
                result = self._shg_deltak(const[0], lam, const[1][0], const[1][1], const[3])
                self._crystal_period = temporary
                return result
            else:
                return np.nan

        def shg(x: float, centarg: float) -> float:
            sol = fsolve(fun, centarg, x, full_output=True)
            if sol[2] == 1:
                return sol[0][0]
            else:
                return np.nan

        v_shg = np.vectorize(shg)
        delx = (lam_lim[1] - lam_lim[0]) / stepnum
        arg = np.arange(lam_lim[0], lam_lim[1], delx)
        plt.plot(arg, v_shg(arg, const[2]))
        return plt

    def plot_temp_shg(self, speeds: Speeds, temp_max: float, temp_min: float, theta_pump: float,
                      phi_pump: float, stepnum=None) -> plt:  # untested
        lim = [temp_max, temp_min]
        starting_lam = 1.1
        params = [speeds, [theta_pump, phi_pump], starting_lam]
        if stepnum is not None:
            stepnum = self._STEP_NUM
        plot = self._plot_shg(Mode.TEMPERATURE, lim, params, stepnum)
        plot.ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        plot.xlabel('Temperature [C]')
        plot.suptitle('Dependence of SHG on temperature')
        plot.show()
        return plot

    def plot_ppol_shg(self, speeds: Speeds, pp_max: float, pp_min: float, theta_pump: float, phi_pump: float,
                      temperature=ROOM_TEMP, stepnum=None) -> plt:  # untested
        lim = [pp_max, pp_min]
        starting_lam = 1.1
        params = [speeds, [theta_pump, phi_pump], starting_lam, temperature]
        if stepnum is not None:
            stepnum = self._STEP_NUM
        plot = self._plot_shg(Mode.CRYSTAL_PERIOD, lim, params, stepnum)
        plot.ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        plot.xlabel('Periodic polling [' + '\u03BC' + 'm]')
        plot.suptitle('Dependence of SHG on periodic polling')
        plot.show()
        return plot

    ####################################################################################################################
    #                                                   OPO METHODS: General methods
    ####################################################################################################################

    def _theta_idler(self, pump_speed: Speeds, lam_pump: float, theta_pump: float, phi_pump: float,
                     signal_speed: Speeds, lam_signal: float, theta_signal: float, phi_signal: float,
                     temperature: float) -> float:  # unsure about abs #not tested
        if self._crystal_period == 0:
            pp_vec = 0
        else:
            pp_vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) * self._pp_sign.value
        propdir = self._dir(theta_pump, phi_pump)
        nsignal = self._ref_indx(signal_speed, theta_pump, phi_pump, lam_signal, theta_signal, phi_signal, temperature)
        kp = np.linalg.norm(self.kpvec(pump_speed, lam_pump, theta_pump, phi_pump, temperature))
        k_parallel = np.dot(self._pp_vec, propdir) * pp_vec
        ks = nsignal * 2 * np.pi / lam_signal
        ks_perpendicular = ks * sin(theta_signal)
        ks_parallel = ks * cos(theta_signal)
        return np.arcsin(ks_perpendicular / ((kp - ks_parallel + k_parallel) ** 2 + ks_perpendicular ** 2) ** 0.5)

    def deltak0vec(self, speeds: Speeds, lam_pump: float, theta_pump: float, phi_pump: float, lam_signal: float, theta_signal: float, phi_signal: float, temperature: float) -> np.array:
        if self._crystal_period == 0:
            pp_vec = np.array([0, 0, 0])
        else:
            pp_vec = 2 * np.pi / (self._crystal_period * self.expansion(temperature)) * self._pp_vec * self._pp_sign.value
        [pump_speed, signal_speed, idler_speed] = speeds
        lam_idler = 1 / (1 / lam_pump - 1 / lam_signal)
        theta_idler = self._theta_idler(pump_speed, lam_pump, theta_pump, phi_pump, signal_speed, lam_signal,
                                        theta_signal,
                                        phi_signal, temperature)
        kp = self.kpvec(pump_speed, lam_pump, theta_pump, phi_pump, temperature)
        ks = self.kvec(signal_speed, lam_signal, theta_pump, phi_pump, theta_signal, phi_signal, temperature)
        ki = self.kvec(idler_speed, lam_idler, theta_pump, phi_pump, theta_idler, phi_signal + np.pi, temperature)
        return kp - ks - ki + np.dot(pp_vec, self._dir(theta_pump, phi_pump)) * self._dir(theta_pump, phi_pump)

    def _find_signal_beam(self, speedlist: np.array, lam_pump: float, theta_pump: float, phi_pump: float, temperature: float, lam_signal0=1.5, angSignal0=1e-6) -> np.array:
        phi_signal = 0

        def find_signal(x):
            lam_sig = x[0]
            theta_sig = x[1]
            return np.linalg.norm(self.deltak0vec(speedlist, lam_pump, theta_pump, phi_pump, lam_sig, theta_sig,
                                                  phi_signal, temperature))

        sol = minimize(find_signal, np.array([lam_signal0, angSignal0]),
                       bounds=((lam_pump + self._DEL_LAMBDA, lam_pump * 2),
                               (0, self._MAX_ANGLE)))
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        lam_signal = sol.x[0]
        theta_signal = sol.x[1]
        deltak = sol.fun
        return np.array([lam_signal, theta_signal, deltak, sol.success])

    def _plot_opo(self, mode: Mode, lim: list, params: list, stepnum: int) -> None:
        const = np.array(params)

        def opo(x):
            try:
                if mode == Mode.TEMPERATURE:
                    temperature = x
                    speeds = const[0]
                    lam_pump = const[1]
                    theta_pump = const[2]
                    phi_pump = const[3]
                    [lam_sig, theta_sig, delk] = self._find_signal_beam(speeds, lam_pump, theta_pump, phi_pump,
                                                                                temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], lam_pump, theta_pump, phi_pump, speeds[1], lam_sig,
                                                    theta_sig, 0, temperature)
                    return np.array([lam_sig, theta_sig, theta_idl, delk])
                elif mode == Mode.WAVELENGTH:
                    lam_pump = x
                    speeds = const[0]
                    theta_pump = const[1]
                    phi_pump = const[2]
                    temperature = const[3]
                    [lam_sig, theta_sig, delk] = self._find_signal_beam(speeds, lam_pump, theta_pump, phi_pump,
                                                                                temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], lam_pump, theta_pump, phi_pump, speeds[1], lam_sig,
                                                    theta_sig, 0, temperature)
                    return np.array([lam_sig, theta_sig, theta_idl, delk])
                elif mode == Mode.CRYSTAL_PERIOD:
                    period = x
                    self.Temp = self._crystal_period
                    self._crystal_period = period
                    speeds = const[0]
                    lam_pump = const[1]
                    theta_pump = const[2]
                    phi_pump = const[3]
                    temperature = const[4]
                    [lam_sig, theta_sig, delk] = self._find_signal_beam(speeds, lam_pump, theta_pump, phi_pump,
                                                                                temperature)[[0, 1, 2]]
                    theta_idl = self._theta_idler(speeds[0], lam_pump, theta_pump, phi_pump, speeds[1], lam_signal,
                                                    theta_sig, 0, temperature)
                    self._crystal_period = self.Temp
                    self.Temp = 0
                    return np.array([lam_sig, theta_sig, theta_idl, delk])
                else:
                    return None
            except Exception as exc:
                print(exc)
                return np.array([np.nan, np.nan, np.nan, np.nan])

        vopo = np.vectorize(opo, otypes=[np.ndarray])
        delx = (lim[1] - lim[0]) / stepnum
        arg = np.arange(lim[0], lim[1], delx)
        result = np.vstack(vopo(arg))
        [lam_signal, theta_signal, theta_idler, deltak] = [result[:, 0], result[:, 1], result[:, 2], result[:, 3]]
        if mode != Mode.WAVELENGTH:
            lam_idler = 1 / (1 / const[1] - 1 / lam_signal)
        else:
            lam_idler = 1 / (1 / arg - 1 / lam_signal)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(arg, lam_signal)
        ax1.plot(arg, lam_idler)
        ax1.set(ylabel='Wavelength [' + '\u03BC' + 'm]')
        ax2.plot(arg, theta_signal)
        ax2.plot(arg, theta_idler)
        ax2.set(ylabel='Opening angle [rad]')
        ax3.scatter(arg, np.log10(deltak))
        ax3.set(ylabel='log($\Delta k$) [log(1/' + '\u03BC' + 'm )]')
        #return plt

    def plot_wave_opo(self, speeds, LamPmax, LamPmin, theta_pump, phi_pump, temperature, stepnum=None) -> None:
        lim = [LamPmax, LamPmin]
        params = [speeds, theta_pump, phi_pump, temperature]
        if stepnum is not None:
            stepnum = self._STEP_NUM
        self._plot_opo(Mode.WAVELENGTH, lim, params, stepnum)
        plt.ylabel('Wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Pump wavelength [' + '\u03BC' + 'm]')
        plt.suptitle('Dependence of OPO wavelengths on pumping wavelength')
        plt.show()
        #return plot

    def plotTempOPO(self, speeds, Tmax, Tmin, lam_pump, theta_pump, phi_pump, stepnum=None) -> None:
        lim = [Tmax, Tmin]
        params = [speeds, lam_pump, theta_pump, phi_pump]
        if stepnum is not None:
            stepnum = self._STEP_NUM
        self._plot_opo(Mode.TEMPERATURE, lim, params, stepnum)
        plt.xlabel('Temperature [C]')
        plt.suptitle('Dependence of OPO on temperature')
        plt.show()
        #return plot

    def plotTemp(self, setup: OpoSetup, Tmax, Tmin, stepnum=None) -> None:
        speeds = [setup.pol_pump, setup.pol_signal, setup.pol_idler]
        lam_pump = setup.lam_pump
        theta_pump = setup.theta_pump
        phi_pump = setup.phi_pump
        return self.plotTempOPO(speeds, Tmax, Tmin, lam_pump, theta_pump, phi_pump, stepnum)

    def plotPPolOPO(self, speeds, PPmax, PPmin, lam_pump, theta_pump, phi_pump, T, stepnum=None) -> None:
        lim = [PPmax, PPmin]
        params = [speeds, lam_pump, theta_pump, phi_pump, T]
        if stepnum is not None:
            stepnum = self._STEP_NUM
        self._plot_opo(Mode.CRYSTAL_PERIOD, lim, params, stepnum)
        plt.ylabel('Wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Periodic polling [' + '\u03BC' + 'm]')
        plt.suptitle('Dependence of OPO wavelengths on periodic polling')
        plt.show()
        #return plot

    ####################################################################################################################
    #                       OPO METHODS: Find OPO for crystal with arbitrary T but with set PP and pumping wavelength
    ####################################################################################################################

    def findOPO(self, lam_pump, T, quiet=True, LookAlongPolling=True):
        print('findOPO: starting search...')
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystalDir = [theta, phi]
        if LookAlongPolling:
            pumpDirs = [crystalDir]
        else:
            pumpDirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        Crystalsign = self._pp_sign
        for direction in pumpDirs:
            [theta_pump, phi_pump] = direction
            if direction == crystalDir:
                signs = [1, -1]
            else:
                signs = [1]
            for pump_speed in Speeds:
                for signal_speed in Speeds:
                    for idlerSpeed in Speeds:
                        speedCom = np.array([pump_speed, signal_speed, idlerSpeed])
                        for sign in signs:
                            self._pp_sign = sign
                            try:
                                [lam_signal, theta_signal, deltak, issuccess] = self._find_signal_beam(speedCom,
                                                                                                       lam_pump,
                                                                                                       theta_pump,
                                                                                                       phi_pump, T)
                                lamIdler = 1 / (1 / lam_pump - 1 / lam_signal)
                                theta_idler = self._theta_idler(pump_speed, lam_pump, theta_pump, phi_pump,
                                                                signal_speed,
                                                                lam_signal, theta_signal, 0, T)
                                instance = OpoSetup(issuccess, deltak, lam_pump, lam_signal, lamIdler, theta_pump,
                                                    phi_pump, T, self._crystal_period, Sign.sign, pump_speed,
                                                    idlerSpeed,
                                                    signal_speed, theta_signal, theta_idler)
                                instance.display()
                                setups.append(instance)
                            except Exception as inst:
                                if not quiet:
                                    print(inst.args)

        self._pp_sign = Crystalsign
        print('findOPO: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups

    ####################################################################################################################
    #                       OPO METHODS: Find PP or T, but with set OPO and pumping direction
    ####################################################################################################################

    def _findSetupParameters(self, mode, speedCom, lam_pump, theta_pump, phi_pump, lam_signal, PP0, T0, angSignal0):
        phi_signal = 0
        if PP0 is not None:
            PP0 = self._crystal_period
        if mode == Mode.Temperature:
            self._crystal_period = PP0
            (X0, Xmin, Xmax) = (T0, self._MIN_TEMPERATURE, self._MAX_TEMPERATURE)

            def findSignal(x):
                theta_signal = x[0]
                T = x[1]
                return np.linalg.norm(
                    self.deltak0vec(speedCom, lam_pump, theta_pump, phi_pump, lam_signal, theta_signal, phi_signal, T))
        elif mode == Mode.CRYSTAL_PERIOD:
            (X0, Xmin, Xmax) = (PP0, self._MIN_PERIOD, self._MAX_PERIOD)

            def findSignal(x):
                theta_signal = x[0]
                self._crystal_period = x[1]
                return np.linalg.norm(
                    self.deltak0vec(speedCom, lam_pump, theta_pump, phi_pump, lam_signal, theta_signal, phi_signal, T0))
        else:
            raise TypeError('Incorrect mode.')
        sol = minimize(findSignal, np.array([angSignal0, X0]), bounds=((0, self._MAX_ANGLE), (Xmin, Xmax)))
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        theta_signal = sol.x[0]
        if mode == Mode.Temperature:
            (T, period) = (sol.x[1], self._crystal_period)
        elif mode == Mode.CRYSTAL_PERIOD:
            (T, period) = (T0, sol.x[1])
        deltak = sol.fun
        return np.array([theta_signal, T, period, deltak, sol.success])

    def _findCrystalSetup(self, mode, lam_pump, lam_signal, PolMode, T0, PPguess, angSignal0, quiet, LookAlongPolling):
        print('findCrystalSetup: starting search...')
        temp = self._crystal_period
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystalDir = [theta, phi]
        if LookAlongPolling:
            pumpDirs = [crystalDir]
        else:
            pumpDirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        Crystalsign = self._pp_sign
        for direction in pumpDirs:
            if direction == crystalDir:
                signs = [1, -1]
            else:
                signs = [1]
            [theta_pump, phi_pump] = direction
            for pump_speed in Speeds:
                for signal_speed in Speeds:
                    if PolMode == OpoPol.HETERO:
                        idlerSpeed = Speeds((signal_speed.value + 1) % 2)
                    if PolMode == OpoPol.HOMO:
                        idlerSpeed = signal_speed
                    speedCom = np.array([pump_speed, signal_speed, idlerSpeed])
                    for sign in signs:
                        self._pp_sign = sign
                        self._crystal_period = temp
                        try:
                            [theta_signal, T, period, deltak, issuccess] = self._findSetupParameters(mode, speedCom,
                                                                                                     lam_pump,
                                                                                                     theta_pump,
                                                                                                     phi_pump,
                                                                                                     lam_signal,
                                                                                                     PPguess, T0,
                                                                                                     angSignal0)
                            lamIdler = 1 / (1 / lam_pump - 1 / lam_signal)
                            theta_idler = self._theta_idler(idlerSpeed, theta_pump, phi_pump, lam_signal, theta_signal,
                                                            0,
                                                            T)
                            instance = OpoSetup(issuccess, deltak, lam_pump, lam_signal, lamIdler, theta_pump, phi_pump,
                                                T,
                                                period, Sign.sign, pump_speed, idlerSpeed, signal_speed, theta_signal,
                                                theta_idler)
                            instance.disp()
                            setups.append(instance)
                        except Exception as inst:
                            if not quiet:
                                print(inst.args)
                        finally:
                            self._crystal_period = temp
        self._pp_sign = Crystalsign
        print('findCrystalSetup: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups

    def findCrystalPeriod(self, lam_pump, lam_signal, PolMode, T0=ROOM_TEMP, PPguess=None, angSignal0=0.01, quiet=True,
                          LookAlongPolling=True):
        setups = self._findCrystalSetup(Mode.CRYSTAL_PERIOD, lam_pump, lam_signal, PolMode, T0, PPguess, angSignal0,
                                        quiet,
                                        LookAlongPolling)
        return setups

    def findCrystalTemperature(self, lam_pump, lam_signal, PolMode, PPguess=None, T0=ROOM_TEMP, angSignal0=0.01,
                               quiet=True, LookAlongPolling=True):
        setups = self._findCrystalSetup(Mode.Temperature, lam_pump, lam_signal, PolMode, T0, PPguess, angSignal0, quiet,
                                        LookAlongPolling)
        return setups

    ####################################################################################################################
    #                       OPO METHODS: Find setup for a colinear OPO
    ####################################################################################################################

    def _findPeriod(self, speedCom, lam_pump, theta_pump, phi_pump, lam_signal, T0):
        phi_signal = 0

        def findPeriod(x):
            self._crystal_period = x
            return np.linalg.norm(
                self.deltak0vec(speedCom, lam_pump, theta_pump, phi_pump, lam_signal, 1e-12, phi_signal, T0))

        sol = minimize_scalar(findPeriod, bounds=(self._MIN_PERIOD, self._MAX_PERIOD), method='bounded')
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        period = sol.x
        deltak = sol.fun
        return np.array([T0, period, deltak, sol.success])

    def _findTemp(self, speedCom, lam_pump, theta_pump, phi_pump, lam_signal, PP0):
        phi_signal = 0
        if PP0 > 0:
            self._crystal_period = PP0

        def findTemp(x):
            T = x
            return np.linalg.norm(
                self.deltak0vec(speedCom, lam_pump, theta_pump, phi_pump, lam_signal, 1e-12, phi_signal, T))

        sol = minimize_scalar(findTemp, bounds=(self._MIN_TEMPERATURE, self._MAX_TEMPERATURE), method='bounded')
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        T = sol.x
        deltak = sol.fun
        PP0 = self._crystal_period
        return np.array([T, PP0, deltak, sol.success])

    def _findOptimum(self, speedCom, lam_pump, theta_pump, phi_pump, lam_signal, T0, PP0):
        phi_signal = 0
        if PP0 < 0:
            PP0 = self._crystal_period

        def findTemp(x):
            T = x[0]
            self._crystal_period = x[1]
            return np.linalg.norm(
                self.deltak0vec(speedCom, lam_pump, theta_pump, phi_pump, lam_signal, 1e-12, phi_signal, T))

        sol = minimize(findTemp, np.array([T0, PP0]),
                       bounds=((self._MIN_TEMPERATURE, self._MAX_TEMPERATURE), (self._MIN_PERIOD, self._MAX_PERIOD)))
        if np.isnan(sol.fun):
            raise ResultIsNaNError()
        elif sol.fun > self._MAX_WAVE_MISMATCH:
            raise WaveVectorMismatchTooBig(sol.fun)
        T = sol.x[0]
        deltak = sol.fun
        PP0 = self._crystal_period
        return np.array([T, PP0, deltak, sol.success])

    def findCollinearOPO(self, mode, lam_pump, lam_signal, PolMode, X0=None, quiet=True, LookAlongPolling=True):
        temp = self._crystal_period
        print('findCollinearOPO: starting search...')
        setups = []
        theta = np.arccos(self._pp_vec[2])
        phi = np.arccos(self._pp_vec[0] / sin(theta))
        crystalDir = [theta, phi]
        if LookAlongPolling:
            pumpDirs = [crystalDir]
        else:
            pumpDirs = [[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]]
        Crystalsign = self._pp_sign
        for direction in pumpDirs:
            if direction == crystalDir:
                signs = [1, -1]
            else:
                signs = [1]
            [theta_pump, phi_pump] = direction
            for sign in signs:
                self._pp_sign = sign
                for pump_speed in Speeds:
                    for signal_speed in Speeds:
                        if PolMode == OpoPol.HETERO:
                            idlerSpeed = Speeds((signal_speed.value + 1) % 2)
                        if PolMode == OpoPol.HOMO:
                            idlerSpeed = signal_speed
                        speedCom = np.array([pump_speed, signal_speed, idlerSpeed])
                        self._crystal_period = temp
                        try:
                            if mode == Mode.Temperature:
                                if X0 is not None:
                                    PP0 = -1
                                else:
                                    PP0 = X0
                                [T, period, deltak, issuccess] = self._findTemp(speedCom, lam_pump, theta_pump,
                                                                                phi_pump,
                                                                                lam_signal, PP0)
                            elif mode == Mode.CRYSTAL_PERIOD:
                                if X0 is not None:
                                    T0 = ROOM_TEMP
                                else:
                                    T0 = X0
                                [T, period, deltak, issuccess] = self._findPeriod(speedCom, lam_pump, theta_pump,
                                                                                  phi_pump, lam_signal, T0)
                            elif mode == Mode.TempPeriod:
                                if X0 is not None:
                                    T0 = ROOM_TEMP
                                    PP0 = -1
                                else:
                                    [T0, PP0] = X0
                                [T, period, deltak, issuccess] = self._findOptimum(speedCom, lam_pump, theta_pump,
                                                                                   phi_pump, lam_signal, T0, PP0)
                            else:
                                raise Exception('Incorrect mode')
                            lamIdler = 1 / (1 / lam_pump - 1 / lam_signal)
                            instance = OpoSetup(issuccess, deltak, lam_pump, lam_signal, lamIdler, theta_pump, phi_pump,
                                                T,
                                                period, Sign.sign, pump_speed, idlerSpeed, signal_speed, 0, 0)
                            instance.disp()
                            setups.append(instance)
                        except Exception as exc:
                            if not quiet:
                                print(exc.args)
                        finally:
                            self._crystal_period = temp
        self._pp_sign = Crystalsign
        print('findCollinearOPO: search finished. ' + str(len(setups)) + ' setups found. \n')
        return setups
