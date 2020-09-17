import abc
from numpy import cos
from numpy import sin
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from .OPOSetup import OpoSetup


class Speeds(Enum):
    SLOW = 0
    FAST = 1

class OpoPol(Enum):
    HOMO = 0
    HETERO = 1

class Mode(Enum):
    Temperature = 1
    Wavelength  = 2
    CrystalPeriod = 3
    TempPeriod = 4

RoomTemp = 25

class CrystalDispersion (metaclass=abc.ABCMeta):

    __c_const = 3 * 10**14
    __StepNum = 100
    __maxAngle = 0.2
    __maxWavelength = 5
    __maxTemp = 200
    __minTemp = -20
    __maxPeriod = 100
    __minPeriod = 1
    __maxWaveMismatch = 1e-6
    __delLam = 0.03

    def __init__(self, crystalPeriod, pPVec, sellmeierCoeff, tempCoeff, PPsign):
        self.CrystalPeriod = crystalPeriod
        self.PPvec = np.array(pPVec)
        self.SellmeierCoeff = sellmeierCoeff
        self.TempCoeff = tempCoeff
        self.Temp = 0
        self.PPsign = PPsign

    def __expansion(self, T):
        exc = self.TempCoeff["thermalExp"]
        return 1 + exc["alpha"] * (T - exc["RoomTemp"]) + exc["beta"] * (T - exc["RoomTemp"])**2

    def GetPeriod(self, T):
        if self.CrystalPeriod == 0:
            K = np.array([0, 0, 0])
        else:
            K = 2 * np.pi / (self.CrystalPeriod * self.__expansion(T)) * self.PPvec * self.PPsign
        return K

    @abc.abstractmethod
    def nx(self, wavelength, temperature):
        """returns  ref. index of x polarised wave"""

    @abc.abstractmethod
    def ny(self, wavelength, temperature):
        """returns  ref. index of y polarised wave"""

    @abc.abstractmethod
    def nz(self, wavelength, temperature):
        """returns  ref. index of z polarised wave"""

    def plotRefIndxs(self, lamLim, lamT, TLim, Tlam, StepNum = np.nan):
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        dellam = (lamLim[1] - lamLim[0]) / StepNum
        lam = np.arange(lamLim[0], lamLim[1], dellam)
        delT = (TLim[1] - TLim[0]) / StepNum
        T = np.arange(TLim[0], TLim[1], delT)
        plt.subplot(121)
        plt.plot(self.nx(lam, lamT))
        plt.plot(self.ny(lam, lamT))
        plt.plot(self.nz(lam, lamT))
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Ref. index')
        plt.subplot(122)
        plt.plot(self.nx(Tlam, T))
        plt.plot(self.ny(Tlam, T))
        plt.plot(self.nz(Tlam, T))
        plt.xlabel('Temperature [C]')
        plt.ylabel('Ref. index')
        plt.show()

    def __trans(self, theta, phi):
         return np.array([[cos(theta) * cos(phi), - sin(phi), sin(theta) * cos(phi)], [cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi)], [-sin(theta), 0, cos(theta)]])

    def __dir(self, theta, phi):
        return np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])

    def __effRefIndx(self, speed, s, n):
        [sx, sy, sz] = s
        [nx, ny, nz] = n
        b = sx ** 2 * (1 / ny ** 2 + 1 / nz ** 2) + sy ** 2 * (1 / nx ** 2 + 1 / nz ** 2) + sz ** 2 * (1 / nx ** 2 + 1 / ny ** 2)
        c = sx ** 2 * (1 / ny ** 2 * 1 / nz ** 2) + sy ** 2 * (1 / nx ** 2 * 1 / nz ** 2) + sz ** 2 * (1 / nx ** 2 * 1 / ny ** 2)
        if speed == Speeds(0):
            return (abs(2 / (b - (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        elif speed == Speeds(1):
            return (abs(2 / (b + (b ** 2 - 4 * c) ** 0.5))) ** 0.5
        else:
            raise TypeError('Not a speed')

    def __refIndx(self, speed, thetaPump, phiPump, lam, theta, phi, T):
        s = np.dot(self.__trans(thetaPump, phiPump), self.__dir(theta, phi))
        n = [self.nx(lam, T), self.ny(lam, T), self.nz(lam, T)]
        return self.__effRefIndx(speed, s, n)

    def __refIndxPump(self, speed, lam, theta, phi, T):
        return self.__refIndx(speed, 0, 0, lam, theta, phi, T)

    def __kvec(self, speed, lam, thetaPump, phiPump, theta, phi, T):
        k0 = 2 * np.pi / lam
        n = self.__refIndx(speed, thetaPump, phiPump, lam, theta, phi, T)
        return n * k0 * np.dot(self.__trans(thetaPump, phiPump), self.__dir(theta, phi))

    def __kpvec(self, speed, lam, thetaPump, phiPump, T):
        return self.__kvec(speed, lam, 0, 0, thetaPump, phiPump, T)

    ####################################################################################################################
    #                                                   SHG METHODS
    ####################################################################################################################

    def __SHGdeltaK(self, speeds, lamPump, thetaPump, phiPump, T):
        if self.CrystalPeriod == 0:
            K = np.array([0, 0, 0])
        else:
            K = 2 * np.pi / (self.CrystalPeriod * self.__expansion(T)) * self.PPvec * self.PPsign
        [pumpSpeed, shgSpeed] = speeds
        lamSHG = lamPump / 2
        kp = self.__kpvec(pumpSpeed, lamPump, thetaPump, phiPump, T)
        kout = self.__kpvec(shgSpeed, lamSHG, thetaPump, phiPump, T)
        return np.linalg.norm(2 * kp - kout + K)


    def __plotSHG(self, mode, lim, params, StepNum):
        const = np.array(params)
        def fun(x, y):
            if mode == Mode(1): # Temperature characteristic
                T = y
                return self.__SHGdeltaK(const[0], x, const[1][0], const[1][1], T)
            elif mode == Mode(3): # Periodic polling characteristic
                PP = y
                self.Temp = self.CrystalPeriod
                self.CrystalPeriod = PP
                result = self.__SHGdeltaK(const[0], x, const[1][0], const[1][1], const[3])
                self.CrystalPeriod = self.Temp
                self.Temp = 0
                return result
        def SHG(x, centarg):
            sol = fsolve(fun, centarg, x, full_output=True)
            if sol[2]==1:
                return sol[0]
            else:
                return np.nan
        vSHG = np.vectorize(SHG)
        delx = (lim[1] - lim[0]) / StepNum
        arg = np.arange(lim[0], lim[1], delx)
        plt.plot(arg, vSHG(arg, const[2]))
        return plt


    def plotTempSHG(self, speeds, Tmax, Tmin, thetaPump, phiPump, StepNum = np.nan): #untested
        lim = [Tmax, Tmin]
        StartLam = 1.1
        params = [speeds, [thetaPump, phiPump], StartLam]
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        plt = self.__plotSHG(Mode(1), lim, params, StepNum)
        plt.ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Temperature [C]')
        plt.suptitle('Dependence of SHG on temperature')
        plt.show()
        return plt

    def plotPPolSHG(self, speeds, PPmax, PPmin, thetaPump, phiPump, T = RoomTemp, StepNum = np.nan): #untested
        lim = [PPmax, PPmin]
        StartLam = 1.1
        params = [speeds, [thetaPump, phiPump], StartLam, T]
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        plt = self.__plotSHG(Mode(3), lim, params, StepNum)
        plt.ylabel('SHG wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Periodic polling [' + '\u03BC' + 'm]')
        plt.suptitle('Dependence of SHG on periodic polling')
        plt.show()
        return plt


    ####################################################################################################################
    #                                                   OPO METHODS: General methods
    ####################################################################################################################

    def __thetaIdler(self, pumpSpeed, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T): # unsure about abs #not tested
        if self.CrystalPeriod == 0:
            K = 0
        else:
            K = 2 * np.pi / (self.CrystalPeriod * self.__expansion(T)) * self.PPsign
        propdir = self.__dir(thetaPump, phiPump)
        nsignal = self.__refIndx(pumpSpeed, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T)
        KT = abs(np.linalg.norm(np.cross(self.PPvec, propdir))) * K
        KII = np.dot(self.PPvec, propdir) * K
        ks = nsignal * 2 * np.pi / lamSignal
        ksT = ks * sin(thetaSignal)
        ksII = ks * cos(thetaSignal)
        return np.arcsin ((ksT - KT)/((ksT- KT)**2 + (ksII - KII)**2)**0.5)

    def __deltak0vec(self, speeds, lamPump, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T):
        if self.CrystalPeriod == 0:
            K = np.array([0, 0, 0])
        else:
            K = 2 * np.pi / (self.CrystalPeriod * self.__expansion(T)) * self.PPvec * self.PPsign
        [pumpSpeed, signalSpeed, idlerSpeed] = speeds
        lamIdler = 1 / (1/lamPump - 1/lamSignal)
        thetaIdler = self.__thetaIdler(idlerSpeed, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T)
        kp = self.__kpvec(pumpSpeed, lamPump, thetaPump, phiPump, T)
        ks = self.__kvec(signalSpeed, lamSignal, thetaPump, phiPump, thetaSignal, phiSignal, T)
        ki = self.__kvec(idlerSpeed, lamIdler, thetaPump, phiPump, thetaIdler, phiSignal + np.pi, T)
        return kp - ks - ki - np.dot(K, self.__dir(thetaPump, phiPump)) * self.__dir(thetaPump, phiPump)

    def __findSignalBeam(self, speeds, lamPump, thetaPump, phiPump, T, lamSignal0 = 1.5, angSignal0 = 1e-6):
        phiSignal = 0
        def findSignal(x):
            lamSignal = x[0]
            thetaSignal = x[1]
            return np.linalg.norm(self.__deltak0vec(speeds, lamPump, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T))
        sol = minimize(findSignal, np.array([lamSignal0, angSignal0]), bounds=((lamPump + self.__delLam, lamPump *2), (0, self.__maxAngle))) #sol = minimize(findSignal, np.array([lamSignal0, angSignal0]), bounds=((lamPump, lamPump * 2), (0, self.__maxAngle)), tol=1e-7)
        if np.isnan(sol.fun):
            raise Exception('Optimization did not terminate successfully.')
        elif sol.fun > self.__maxWaveMismatch:
            raise Exception('Wavevector mismatch is bigger than ' + str(self.__maxWaveMismatch))
        lamSignal = sol.x[0]
        thetaSignal = sol.x[1]
        deltak = sol.fun
        return np.array([lamSignal, thetaSignal, deltak, sol.success])

    def __plotOPO(self, mode, lim, params, StepNum):
        const = np.array(params)
        def OPO(x):
            try:
                if mode == Mode(1): # Temperature characteristic
                    T = x
                    speeds = const[0]
                    lamPump = const[1]
                    thetaPump = const[2]
                    phiPump = const[3]
                    lamSignal = self.__findSignalBeam(speeds, lamPump, thetaPump, phiPump, T)[0]
                    return lamSignal
                elif mode == Mode(2): # Wavelength characteristic
                    lamPump = x
                    speeds = const[0]
                    thetaPump = const[1]
                    phiPump = const[2]
                    T = const[3]
                    lamSignal = self.__findSignalBeam(speeds, lamPump, thetaPump, phiPump, T)[0]
                    return lamSignal
                elif mode == Mode(3): # Periodic polling characteristic
                    PP = x
                    self.Temp = self.CrystalPeriod
                    self.CrystalPeriod = PP
                    speeds = const[0]
                    lamPump = const[1]
                    thetaPump = const[2]
                    phiPump = const[3]
                    T = const[4]
                    lamSignal = self.__findSignalBeam(speeds, lamPump, thetaPump, phiPump, T)[0]
                    self.CrystalPeriod = self.Temp
                    self.Temp = 0
                    return lamSignal
            except Exception as inst:
                print(inst)
                return np.nan
        vOPO = np.vectorize(OPO)
        delx = (lim[1] - lim[0]) / StepNum
        arg = np.arange(lim[0], lim[1], delx)
        lamSignal = vOPO(arg)
        if mode != Mode(2):
            lamIdler = 1 / (1 / const[1] - 1 / lamSignal)
        else:
            lamIdler = 1 / (1 / arg - 1 / lamSignal)
        plt.plot(arg, lamSignal)
        plt.plot(arg, lamIdler)
        return plt


    def plotWaveOPO(self, speeds, LamPmax, LamPmin, thetaPump, phiPump, T, StepNum = np.nan):
        lim = [LamPmax, LamPmin]
        params = [speeds, thetaPump, phiPump, T]
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        plt = self.__plotOPO(Mode(2), lim, params, StepNum)
        plt.ylabel('Wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Pump wavelength [' + '\u03BC' + 'm]')
        plt.suptitle('Dependence of OPO wavelengths on pumping wavelength')
        plt.show()
        return plt

    def plotTempOPO(self, speeds, Tmax, Tmin, lamPump, thetaPump, phiPump, StepNum = np.nan):
        lim = [Tmax, Tmin]
        params = [speeds, lamPump, thetaPump, phiPump]
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        plt = self.__plotOPO(Mode(1), lim, params, StepNum)
        plt.ylabel('Wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Temperature [C]')
        plt.suptitle('Dependence of OPO wavelengths on temperature')
        plt.show()
        return plt

    def plotPPolOPO(self, speeds, PPmax, PPmin, lamPump, thetaPump, phiPump, T, StepNum = np.nan):
        lim = [PPmax, PPmin]
        params = [speeds, lamPump, thetaPump, phiPump, T]
        if np.isnan(StepNum):
            StepNum = self.__StepNum
        plt = self.__plotOPO(Mode(3), lim, params, StepNum)
        plt.ylabel('Wavelength [' + '\u03BC' + 'm]')
        plt.xlabel('Periodic polling [' + '\u03BC' + 'm]')
        plt.suptitle('Dependence of OPO wavelengths on periodic polling')
        plt.show()
        return plt


    ####################################################################################################################
    #                       OPO METHODS: Find OPO for crystal with arbitrary T but with set PP and pumping wavelength
    ####################################################################################################################

    def findOPO(self, lamPump, T, quiet = True):
        print('findOPO: starting search...')
        list = []
        pumpDirs = [[0, 0], [np.pi/2, 0], [np.pi/2, np.pi/2]]
        Crystalsign = self.PPsign
        for dir in pumpDirs:
            [thetaPump, phiPump] = dir
            for pumpSpeed in Speeds:
                for signalSpeed in Speeds:
                    for idlerSpeed in Speeds:
                        speedCom = np.array([pumpSpeed, signalSpeed, idlerSpeed])
                        for sign in [1, -1]:
                            self.PPsign = sign
                            try:
                                [lamSignal, thetaSignal, deltak, issuccess] = self.__findSignalBeam(speedCom, lamPump, thetaPump, phiPump, T)
                                lamIdler = 1 / (1 / lamPump - 1 / lamSignal)
                                thetaIdler = self.__thetaIdler(idlerSpeed, thetaPump, phiPump, lamSignal, thetaSignal, 0, T)
                                instance = OpoSetup(issuccess, deltak, lamPump, lamSignal, lamIdler, thetaPump, phiPump, T, self.CrystalPeriod, sign, pumpSpeed, idlerSpeed, signalSpeed, thetaSignal, thetaIdler)
                                instance.disp()
                                list.append(instance)
                            except Exception as inst:
                                if not quiet:
                                    print(inst.args)

        self.PPsign = Crystalsign
        print('findOPO: search finished. ' + str(len(list)) + ' setups found. \n')
        return list

    ####################################################################################################################
    #                       OPO METHODS: Find PP or T, but with set OPO and pumping direction
    ####################################################################################################################

    def __findSetupParameters(self, mode, speedCom, lamPump, thetaPump, phiPump, lamSignal, PP0, T0, angSignal0):
        phiSignal = 0
        if np.isnan(PP0):
            PP0 = self.CrystalPeriod
        if mode == Mode.Temperature:
            self.CrystalPeriod = PP0
            (X0, Xmin, Xmax) = (T0, self.__minTemp, self.__maxTemp)
            def findSignal(x):
                thetaSignal = x[0]
                T = x[1]
                return np.linalg.norm(self.__deltak0vec(speedCom, lamPump, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T))
        elif mode == Mode.CrystalPeriod:
            (X0, Xmin, Xmax) = (PP0, self.__minPeriod, self.__maxPeriod)
            def findSignal(x):
                thetaSignal = x[0]
                self.CrystalPeriod = x[1]
                return np.linalg.norm(self.__deltak0vec(speedCom, lamPump, thetaPump, phiPump, lamSignal, thetaSignal, phiSignal, T0))
        else:
            raise Exception('Incorrect mode.')
        sol = minimize(findSignal, np.array([angSignal0, X0]), bounds=((0, self.__maxAngle), (Xmin, Xmax)))
        if np.isnan(sol.fun):
            raise Exception('Optimization did not terminate successfully.')
        elif sol.fun > self.__maxWaveMismatch:
            raise Exception('Wavevector mismatch is bigger than ' + str(self.__maxWaveMismatch))
        thetaSignal = sol.x[0]
        if mode == Mode.Temperature:
            (T, period) = (sol.x[1], self.CrystalPeriod)
        elif mode == Mode.CrystalPeriod:
            (T, period) = (T0, sol.x[1])
        deltak = sol.fun
        return np.array([thetaSignal, T, period, deltak, sol.success])

    def __findCrystalSetup(self, mode, lamPump, lamSignal, PolMode, T0, PPguess, angSignal0, quiet):
        print('findCrystalSetup: starting search...')
        temp = self.CrystalPeriod
        list = []
        pumpDirs = np.array([[0, 0], [np.pi/2, 0], [np.pi/2, np.pi/2]])
        Crystalsign = self.PPsign
        for dir in pumpDirs:
            [thetaPump, phiPump] = dir
            for pumpSpeed in Speeds:
                for signalSpeed in Speeds:
                    if PolMode == OpoPol.HETERO:
                        idlerSpeed = Speeds((signalSpeed.value + 1) % 2)
                    if PolMode == OpoPol.HOMO:
                        idlerSpeed = signalSpeed
                    speedCom = np.array([pumpSpeed, signalSpeed, idlerSpeed])
                    for sign in [1, -1]:
                        self.PPsign = sign
                        self.CrystalPeriod = temp
                        try:
                            [thetaSignal, T, period, deltak, issuccess] = self.__findSetupParameters(mode, speedCom, lamPump, thetaPump, phiPump, lamSignal, PPguess, T0, angSignal0)
                            lamIdler = 1 / (1 / lamPump - 1 / lamSignal)
                            thetaIdler = self.__thetaIdler(idlerSpeed, thetaPump, phiPump, lamSignal, thetaSignal, 0, T)
                            instance = OpoSetup(issuccess, deltak, lamPump, lamSignal, lamIdler, thetaPump, phiPump, T, period, sign, pumpSpeed, idlerSpeed, signalSpeed, thetaSignal, thetaIdler)
                            instance.disp()
                            list.append(instance)
                        except Exception as inst:
                            if not quiet:
                                print(inst.args)
                        finally:
                            self.CrystalPeriod = temp
        self.PPsign = Crystalsign
        print('findCrystalSetup: search finished. ' + str(len(list)) + ' setups found. \n')
        return list

    def findCrystalPeriod(self, lamPump, lamSignal, PolMode, T0 = RoomTemp, PPguess = np.nan, angSignal0 = 0.01, quiet = True):
        list = self.__findCrystalSetup(Mode.CrystalPeriod, lamPump, lamSignal, PolMode, T0, PPguess, angSignal0, quiet)
        return list

    def findCrystalTemperature(self, lamPump, lamSignal, PolMode, PPguess = np.nan, T0 = RoomTemp, angSignal0 = 0.01, quiet = True):
        list = self.__findCrystalSetup(Mode.Temperature, lamPump, lamSignal, PolMode, T0, PPguess, angSignal0, quiet)
        return list

    ####################################################################################################################
    #                       OPO METHODS: Find setup for a colinear OPO
    ####################################################################################################################

    def __findPeriod(self, speedCom, lamPump, thetaPump, phiPump, lamSignal, T0):
        phiSignal = 0
        def findPeriod(x):
            self.CrystalPeriod = x
            return np.linalg.norm(self.__deltak0vec(speedCom, lamPump, thetaPump, phiPump, lamSignal, 1e-12, phiSignal, T0))
        sol = minimize_scalar(findPeriod, bounds=(self.__minPeriod, self.__maxPeriod), method='bounded')
        if np.isnan(sol.fun):
            raise Exception('Optimization did not terminate successfully.')
        elif sol.fun > self.__maxWaveMismatch:
            raise Exception('Wavevector mismatch is bigger than ' + str(self.__maxWaveMismatch))
        period = sol.x
        deltak = sol.fun
        return np.array([T0, period, deltak, sol.success])

    def __findTemp(self, speedCom, lamPump, thetaPump, phiPump, lamSignal,  PP0):
        phiSignal = 0
        if PP0 > 0:
            self.CrystalPeriod = PP0
        def findTemp(x):
            T = x
            return np.linalg.norm(self.__deltak0vec(speedCom, lamPump, thetaPump, phiPump, lamSignal, 1e-12, phiSignal, T))
        sol = minimize_scalar(findTemp, bounds=(self.__minTemp, self.__maxTemp), method='bounded')
        if np.isnan(sol.fun):
            raise Exception('Optimization did not terminate successfully.')
        elif sol.fun > self.__maxWaveMismatch:
            raise Exception('Wavevector mismatch is bigger than ' + str(self.__maxWaveMismatch))
        T = sol.x
        deltak = sol.fun
        PP0 = self.CrystalPeriod
        return np.array([T, PP0, deltak, sol.success])

    def __findOptimum(self, speedCom, lamPump, thetaPump, phiPump, lamSignal, T0, PP0):
        phiSignal = 0
        if PP0 < 0:
            PP0 = self.CrystalPeriod
        def findTemp(x):
            T = x[0]
            self.CrystalPeriod = x[1]
            return np.linalg.norm(self.__deltak0vec(speedCom, lamPump, thetaPump, phiPump, lamSignal, 1e-12, phiSignal, T))
        sol = minimize(findTemp, np.array([T0, PP0]), bounds=((self.__minTemp, self.__maxTemp), (self.__minPeriod, self.__maxPeriod)))
        if np.isnan(sol.fun):
            raise Exception('Optimization did not terminate successfully.')
        elif sol.fun > self.__maxWaveMismatch:
            raise Exception('Wavevector mismatch is bigger than ' + str(self.__maxWaveMismatch))
        T = sol.x[0]
        deltak = sol.fun
        PP0 = self.CrystalPeriod
        return np.array([T, PP0, deltak, sol.success])

    def findCollinearOPO(self, mode, lamPump, lamSignal, X0 = np.nan, quiet = True):
        temp = self.CrystalPeriod
        print('findCollinearOPO: starting search...')
        list = []
        pumpDirs = np.array([[0, 0], [np.pi/2, 0], [np.pi/2, np.pi/2]])
        Crystalsign = self.PPsign
        for dir in pumpDirs:
            [thetaPump, phiPump] = dir
            for sign in [1, -1]:
                self.PPsign = sign
                for pumpSpeed in Speeds:
                    for signalSpeed in Speeds:
                        for idlerSpeed in Speeds:
                            speedCom = np.array([pumpSpeed, signalSpeed, idlerSpeed])
                            self.CrystalPeriod = temp
                            try:
                                if mode == Mode.Temperature:
                                    if np.isnan(X0):
                                        PP0 = -1
                                    else:
                                        PP0 = X0
                                    [T, period, deltak, issuccess] = self.__findTemp(speedCom, lamPump, thetaPump, phiPump, lamSignal, PP0)
                                elif mode == Mode.CrystalPeriod:
                                    if np.isnan(X0):
                                        T0 = RoomTemp
                                    else:
                                        T0 = X0
                                    [T, period, deltak, issuccess] = self.__findPeriod(speedCom, lamPump, thetaPump, phiPump, lamSignal, T0)
                                elif mode == Mode.TempPeriod:
                                    if np.isnan(X0):
                                        T0 = RoomTemp
                                        PP0 = -1
                                    else:
                                        [T0, PP0] = X0
                                    [T, period, deltak, issuccess] = self.__findOptimum(speedCom, lamPump, thetaPump, phiPump, lamSignal, T0, PP0)
                                else:
                                    raise Exception('Incorrect mode')
                                lamIdler = 1 / (1 / lamPump - 1 / lamSignal)
                                instance = OpoSetup(issuccess, deltak, lamPump, lamSignal, lamIdler, thetaPump, phiPump, T, period, sign, pumpSpeed, idlerSpeed, signalSpeed, 0, 0)
                                instance.disp()
                                list.append(instance)
                            except Exception as inst:
                                if not quiet:
                                    print(inst.args)
                            finally:
                                self.CrystalPeriod = temp
        self.PPsign = Crystalsign
        print('findCollinearOPO: search finished. ' + str(len(list)) + ' setups found. \n')
        return list