# nonlinear_crystal_characterisation

This package is supposed to provide a relatively easy way to characterise nonlinear crystals. These crystals might be 'regular' nonlinear crystals or periodically polled ones. 

By characterisation I understand the following:
1) plotting refractive index dependence on wavelength and temperature
2) plotting given SHG-process dependence on periodic polling and temperature. 
3) plotting given OPO(SPDC) dependence on periodic polling, pumping wavelength or temperature
4) finding possible OPO (SPDC) process that given crystal can support
5) finding setup (polarization, direction of propagation, temperature,  polarization) for given OPO (SPDC) (defined as a set of pumping, idler and signal wavelength)to be possible in a given crystal

This package has been developed so that adding additional crystals should be straightforward. Although right now there are only two crystals defined in this package (PPKTP from DOI: 10.1088/2399-6528/aaccac" and PPLN from DOI: 10.1007/s00340-008-2998-2) adding some specific crystal should not pose any challenges: all one needs to do is to define Sellmeyer coefficients and temperature coefficients and then check if such crystal behaves as outlined in the literature.

Additionally, I hope that adding new features and methods for additional calculations will be easy. I intend to use it for SPDC calculations and this package from the start was built with the idea of simplifying these types of calculations.
