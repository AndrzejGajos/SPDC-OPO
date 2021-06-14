import numpy as np
from enum import Enum
from typing import NamedTuple


class Speeds(Enum):
    SLOW = 0
    FAST = 1


class RangeLimits(NamedTuple):
    max: float
    min: float


class BeamParams(NamedTuple):
    wavelength: float
    theta: float
    phi: float


class OpoSpeeds(NamedTuple):
    pump: Speeds
    idler: Speeds
    signal: Speeds


class SHGspeeds(NamedTuple):
    pump: Speeds
    shg: Speeds


class OpoPol(Enum):
    HOMO = 0
    HETERO = 1


class Mode(Enum):
    TEMPERATURE = 1
    WAVELENGTH = 2
    CRYSTAL_PERIOD = 3
    TEMP_AND_PERIOD = 4


class Sign(Enum):
    MINUS = -1
    PLUS = 1


class OpoSetup:

    def __init__(self, issuccess: bool, delta_k: float, pump_beam: BeamParams, signal_beam: BeamParams,
                 idler_beam: BeamParams, temperature: float, crystal_period: float, pseudo_vector_sign: Sign,
                 pol_pump: Speeds, pol_idler: Speeds, pol_signal: Speeds) -> None:
        self._issuccess = issuccess
        self._delta_k = delta_k
        self._lam_pump = pump_beam.wavelength
        self._lam_signal = signal_beam.wavelength
        self._lam_idler = idler_beam.wavelength
        self._theta_pump = pump_beam.theta
        self._phi_pump = pump_beam.phi
        self._temperature = temperature
        self._crystal_period = crystal_period
        self._pseudo_vector_sign = pseudo_vector_sign
        self._pol_pump = pol_pump
        self._pol_idler = pol_idler
        self._pol_signal = pol_signal
        self._theta_signal = signal_beam.theta
        self._theta_idler = idler_beam.theta
        self._dir_pump = np.array([np.sin(pump_beam.theta) * np.cos(pump_beam.phi),
                                   np.sin(pump_beam.theta) * np.sin(pump_beam.phi),
                                   np.cos(pump_beam.theta)])

    @property
    def delta_k(self) -> float:
        return self._delta_k

    @property
    def lam_pump(self) -> float:
        return self._lam_pump

    @property
    def lam_signal(self) -> float:
        return self._lam_signal

    @property
    def lam_idler(self) -> float:
        return self._lam_idler

    @property
    def theta_pump(self) -> float:
        return self._theta_pump

    @property
    def phi_pump(self) -> float:
        return self._phi_pump

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def crystal_period(self) -> float:
        return self._crystal_period

    @property
    def sign(self) -> Sign:
        return self._pseudo_vector_sign

    @property
    def pol_pump(self) -> Speeds:
        return self._pol_pump

    @property
    def pol_idler(self) -> Speeds:
        return self._pol_idler

    @property
    def pol_signal(self) -> Speeds:
        return self._pol_signal

    @property
    def theta_signal(self) -> float:
        return self._theta_signal

    @property
    def theta_idler(self) -> float:
        return self._theta_idler

    @property
    def dir_pump(self) -> np.array:
        return self._dir_pump.copy()

    def display(self):
        print('\033[1m' + 'Is optimal?: ' + str(self._issuccess) + '\033[0m')
        print('Phase mismatch: ' + str(self.delta_k) + ' [1/um]')
        print('Pump and crystal setup: \t wavelength: ' + str(self.lam_pump) + ' [um]' + '\t Theta: ' + str(
            self.theta_pump) + ' [rad]' + '\t Phi: ' + str(self.phi_pump) + ' [rad]' + '\t Temperature: ' + str(
            self.temperature) + ' [C]')
        print('Crystal period: \t' + str(self.crystal_period) + ' [um]')
        if self.sign == Sign.PLUS:
            print('Sign of pseudo vector: +')
        else:
            print('Sign of pseudo vector: -')
        print('Polarization Setup: \t Pump: ' + str(self.pol_pump) + '\t Signal: ' + str(
            self.pol_signal) + '\t Idler: ' + str(self.pol_idler))
        print('\t Signal wavelength: ' + str(self.lam_signal) + ' [um]')
        print('\t Idler wavelength: ' + str(self.lam_idler) + ' [um]')
        print('\t Signal opening angle: ' + str(np.abs(self.theta_signal)) + ' [rad]')
        print('\t Idler opening angle: ' + str(np.abs(self.theta_idler)) + ' [rad]')
