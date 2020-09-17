import numpy as np

class OpoSetup:

    def __init__(self, issuccess, deltak, lamPump,lamSignal,lamIdler, thetaPump, phiPump, temperature, crystalPeriod, signPseudoVector, polPump, polIdler, polSignal, thetaSignal, thetaIdler):
        if issuccess == 1:
            self.__issuccess = True
        elif issuccess == 0:
            self.__issuccess = False
        else:
            self.__issuccess = np.nan
        self.__deltak = deltak
        self.__lamPump = lamPump
        self.__lamSignal = lamSignal
        self.__lamIdler = lamIdler
        self.__thetaPump = thetaPump
        self.__phiPump = phiPump
        self.__temperature = temperature
        self.__crystalPeriod = crystalPeriod
        self.__signPseudoVector = signPseudoVector
        self.__polPump = polPump
        self.__polIdler = polIdler
        self.__polSignal = polSignal
        self.__thetaSignal = thetaSignal
        self.__thetaIdler = thetaIdler
        self.__dirPump = np.array([np.sin(thetaPump) * np.cos(phiPump), np.sin(thetaPump) * np.sin(phiPump), np.cos(thetaPump)])

    def deltak(self):
        return self.__deltak

    def lamPump(self):
        return self.__lamPump

    def lamSignal(self):
        return self.__lamSignal

    def lamIdler(self):
        return self.__lamIdler

    def thetaPump(self):
        return self.__thetaPump

    def phiPump(self):
        return self.__phiPump

    def temperature(self):
        return self.__temperature

    def crystalPeriod(self):
        return self.__crystalPeriod

    def signPseudoVector(self):
        return self.__signPseudoVector

    def polPump(self):
        return self.__polPump

    def polSignal(self):
        return self.__polSignal

    def polIdler(self):
        return self.__polIdler

    def thetaPump(self):
        return self.__thetaPump

    def thetaSignal(self):
        return self.__thetaSignal

    def thetaIdler(self):
        return self.__thetaIdler

    def dirPump(self):
        return self.__dirPump

    def disp(self):
        print ('\033[1m' + 'Is optmial?: ' + str(self.__issuccess) + '\033[0m')
        print('Phase mismatch: ' + str(self.deltak()) + ' [1/um]')
        print('Pump and crystal setup: \t wavelength: ' + str(self.lamPump()) + ' [um]' + '\t Theta: ' + str(self.thetaPump()) + ' [rad]' + '\t Phi: ' + str(self.phiPump()) + ' [rad]' + '\t Temperature: ' + str(self.temperature()) + ' [C]')
        print('Crystal period: \t' + str(self.crystalPeriod()) + ' [um]')
        if self.signPseudoVector() == 1:
            print('Sign of pseudo vector: +')
        else:
            print('Sign of pseudo vector: -')
        print('Polarization Setup: \t Pump: ' + str(self.polPump()) + '\t Signal: ' + str(self.polSignal()) + '\t Idler: ' + str(self.polIdler()))
        print('\t Signal wavelength: ' + str(self.lamSignal()) + ' [um]')
        print('\t Idler wavelength: ' + str(self.lamIdler()) + ' [um]')
        print('\t Signal opening angle: ' + str(np.abs(self.thetaSignal())) + ' [rad]')
        print('\t Idler opening angle: ' + str(np.abs(self.thetaIdler())) + ' [rad]')
