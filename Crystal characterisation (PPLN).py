import numpy as np
from Dispersion import PPLNgayer2008
from Dispersion import Speeds
from Dispersion import OpoPol, Mode

##Question: can a PPLN crystal (Sellmeier coefficents from Gayer2008) support given OPO/SPDC process?
##following process can be easily (hopefully!) repeated for any crystal, uniaxial or biaxial. All you need to do is to create your own crystal class.

##first decide periodic polling direction and period. If you don't know the period yet, place 0 as first argument and decide only on direction.
Period = 10
PeriodDir = [1, 0, 0]
##First create an instance of PPLNgayer2008
PPcrystal = PPLNgayer2008(Period, PeriodDir)
##Let's say we are looking for PPLN crystal which will allow for 775 -> 1550 + 1550 conversion. We want both signal and idler to have the same polarization
##First, with our guess for PP and room temperature let's what is possible to do with 0.755 um pump wavelength
#list = PPcrystal.findOPO(0.775, 25)
##Right away we can find possible polarization candidates for our conversion: (FAST, FAST, FAST), (SLOW, SLOW, SLOW). (FAST, SLOW, SLOW) is possible but not along periodic polling
## so it corresponds to down conversion in regular LN crystal.
## We will take a look at first polarization setup. We will see how conversion will change with period given room temperature.
speeds = (Speeds.FAST, Speeds.FAST, Speeds.FAST)
PPmax = 50
PPmin = 2
lamPump = 0.775
lamSignal = 1.550
thetaPump = np.pi / 2
phiPump = 0
T = 25
##We will try to plot OPO
#PPcrystal.plotWaveOPO(speeds, 0.8, 0.6, thetaPump, phiPump, T) # function of pumping wavelength
#PPcrystal.plotTempOPO(speeds, 150, 1, lamPump, thetaPump, phiPump) # function of temperature
#PPcrystal.plotPPolOPO(speeds, PPmax, PPmin, lamPump, thetaPump, phiPump, T) # function of period
## From plots it is easy to observe that OPO is weakly dependent on temperature and strongly on period.
##Altough minimalization function in PPcrystal.plotPPolOPO  coudnt find minimum around 18 - 20 um it seems like somewhere there two stright lines would cross. So lets guess 18 um and look if we can find temerature which would give us our wavelength and polarizations.
Period = 18
PPcrystal = PPLNgayer2008(Period, PeriodDir)
#list = PPcrystal.findCrystalTemperature(0.775, 1.550, OpoPol.HOMO)
##The result printed in consol states that for 18 um crystal we can get our OPO (note difference in polarisations) in -7 C.
##Let's look for a crystal which could work in room temperature
#list = PPcrystal.findCrystalPeriod(lamPump, lamSignal, OpoPol.HOMO, T0=T)
## There are few candidates, each with different polarisation and period. Let's take a look at crystal with approx 19.4um period
Period = 19.4
PPcrystal = PPLNgayer2008(Period, PeriodDir)
speeds = (Speeds.FAST, Speeds.FAST, Speeds.FAST)
#PPcrystal.plotTempOPO(speeds, 140, 1, lamPump, thetaPump, phiPump)
## It is easy to observe that desired OPO still is possible around room temperature but it's has a stronger dependence on temperature then previously plotted OPO

## Alternative approche - if we assome that our OPO is collinear, then it is easier to look for setups. We can run at first function calculating optimal period and polarization
#list = PPcrystal.findCollinearOPO(Mode.TempPeriod, 0.775, 1.550)
## Let's focus on setup for ~20.3 um crystal
Period = 20.3
PPSign = - 1
PPcrystal = PPLNgayer2008(Period, PeriodDir, PPSign)
speeds = (Speeds.FAST, Speeds.SLOW, Speeds.SLOW)
PPcrystal.plotTempOPO(speeds, 40, 20, lamPump, thetaPump, phiPump)
## It is easy to see that there is very strong dependence on temperature. Even few degrees of deviation from ideal (around 25) temperature can potentially lead to spectrally distinguishable beams.
## We can calculate perfect temerature with following function:
list = PPcrystal.find_collinear_opo(Mode.Temperature, 0.775, 1.550, )

##Not needed in above examples:
list = PPcrystal.find_collinear_opo(Mode.CrystalPeriod, 0.775, 1.550, 30)


quit()

list = PPcrystal.find_crystal_temperature(0.775, 1.550, OpoPol.HETERO)
