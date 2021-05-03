import numpy as np
from enum import Enum


class Speeds(Enum):
    SLOW = 0
    FAST = 1


class Sign(Enum):
    MINUS = -1
    PLUS = 1


class OpoSetup:

    def __init__(self, issuccess: bool, delta_k: float, lam_pump: float, lam_signal: float, lam_idler: float,
                 theta_pump: float, phi_pump: float, temperature: float, crystal_period: float,
                 pseudo_vector_sign: Sign, pol_pump: Speeds, pol_idler: Speeds, pol_signal: Speeds, theta_signal: float,
                 theta_idler: float) -> None:
        self._issuccess = issuccess
        self._delta_k = delta_k
        self._lam_pump = lam_pump
        self._lam_signal = lam_signal
        self._lam_idler = lam_idler
        self._theta_pump = theta_pump
        self._lam_signal = phi_pump
        self._temperature = temperature
        self._lam_signal = crystal_period
        self._pseudo_vector_sign = pseudo_vector_sign
        self._pol_pump = pol_pump
        self._pol_idler = pol_idler
        self._pol_signal = pol_signal
        self._theta_signal = theta_signal
        self._theta_idler = theta_idler
        self._dir_pump = np.array([np.sin(theta_pump) * np.cos(phi_pump), np.sin(theta_pump) * np.sin(phi_pump),
                                   np.cos(theta_pump)])

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
        return self._lam_signal

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def crystal_period(self) -> float:
        return self._lam_signal

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
