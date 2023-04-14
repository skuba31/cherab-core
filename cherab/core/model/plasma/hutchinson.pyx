# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# cython: language_level=3

import numpy as np
from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.atomic cimport FreeFreeGauntFactor
from cherab.core.species cimport Species
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, SPEED_OF_LIGHT, PLANCK_CONSTANT, ELECTRON_REST_MASS, VACUUM_PERMITTIVITY
from libc.math cimport sqrt, log, exp, M_PI
cimport cython


cdef double PH_TO_J_FACTOR = PLANCK_CONSTANT * SPEED_OF_LIGHT * 1e9

cdef double EXP_FACTOR = PH_TO_J_FACTOR / ELEMENTARY_CHARGE

cdef double FRAC_1 = (ELEMENTARY_CHARGE**2 * RECIP_4_PI / VACUUM_PERMITTIVITY)**3
cdef double FRAC_2 = 32 * M_PI**2 / (3 * sqrt(3) * ELECTRON_REST_MASS**2 * SPEED_OF_LIGHT**3) 
cdef double FRAC_3 = sqrt(2 * ELECTRON_REST_MASS / M_PI )


# todo: doppler shift?
cdef class HutchinsonBremsstrahlung(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object.

    The bremsstrahlung formula implemented is equation 5.3.40 from I. H. Hutchinson,
    'Principles of Plasma Diagnostics', second edition, ISBN: 9780511613630

    Temperature averaged gaunt factor was replaced by Maxwellian Free-Free Gaunt Factor.
    The data is taken from M.A. de Avillez and D. Breitschwerdt, 'Temperature-averaged and total 
    free-free Gaunt factors for κ and Maxwellian distributions of electrons', 2015, 
    Astron. & Astrophys. 580, A124 (Table A.1).

    .. math::
        \\varepsilon_{\\mathrm{ff}(f) = n_\\mathrm{e} \\sum_i \\left( n_\\mathrm{i} g_\\mathrm{ff} (Z_\\mathrm{i}, T_\\mathrm{e}, f) Z_\\mathrm{i}^2 \\right)
        \\left( \\frac{e^2}{4 \\pi \\varepsilon_0 c} \\right)^3 
        \\frac{32 \\pi^2}{3 \\sqrt{3} m_\\mathrm{e}^2 c^3} 
        \\sqrt{\\frac{2 m_\\mathrm{e}^3}{\\pi T_\\mathrm{e}}} \\mathrm{e}^{-\\frac{hf}{T_\\mathrm{e}}} g_\\mathrm{ff} \\,,
        
    where the emission :math:`\\varepsilon{\\mathrm{ff} (f)` is in units of radiance (W/m^3/Hz) to obtain value in (w/m^3/nm/sr)
    
    .. math::
        \\varepsilon{\\mathrm{ff} (\\lambda) = \\frac{c \cdot 10^{-9}}{4 \\pi \\lambda^2} \\varepsilon{\\mathrm{ff} (f)

    :ivar Plasma plasma: The plasma to which this emission model is attached. Default is None.
    :ivar AtomicData atomic_data: The atomic data provider for this model. Default is None.
    :ivar FreeFreeGauntFactor gaunt_factor: Free-free Gaunt factor as a function of Z, Te and
                                            wavelength. If not provided,
                                            the `atomic_data` is used.
    """

    def __init__(self, Plasma plasma=None, AtomicData atomic_data=None, FreeFreeGauntFactor gaunt_factor=None):

        super().__init__(plasma, atomic_data)

        self.gaunt_factor = gaunt_factor

        # ensure that cache is initialised
        self._change()

    @property
    def gaunt_factor(self):

        return self._gaunt_factor

    @gaunt_factor.setter
    def gaunt_factor(self, value):

        self._gaunt_factor = value
        self._user_provided_gaunt_factor = True if value else False

    def __repr__(self):
        return '<PlasmaModel - Bremsstrahlung>'

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te
            double lower_wavelength, upper_wavelength
            double lower_sample, upper_sample
            Species species
            int i

        # cache data on first run
        if self._species_charge is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0:
            return spectrum
        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0:
            return spectrum

        # collect densities of charged species
        i = 0
        for species in self._plasma.get_composition():
            if species.charge > 0:
                self._species_density_mv[i] = species.distribution.density(point.x, point.y, point.z)
                i += 1

        # numerically integrate using trapezium rule
        # todo: add sub-sampling to increase numerical accuracy
        lower_wavelength = spectrum.min_wavelength
        lower_sample = self._bremsstrahlung(lower_wavelength, te, ne)
        for i in range(spectrum.bins):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)
            upper_sample = self._bremsstrahlung(upper_wavelength, te, ne)

            spectrum.samples_mv[i] += 0.5 * (lower_sample + upper_sample)

            lower_wavelength = upper_wavelength
            lower_sample = upper_sample

        return spectrum

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _bremsstrahlung(self, double wvl, double te, double ne):
        """
        :param double wvl: Wavelength in nm.
        :param double te: Electron temperature in eV
        :param double ne: Electron density in m^-3
        :return:
        """

        cdef double ni_gff_z2, radiance, pre_factor, ni, z, fre
        cdef int i


        ni_gff_z2 = 0
        for i in range(self._species_charge_mv.shape[0]):
            z = self._species_charge_mv[i]
            ni = self._species_density_mv[i]
            if ni > 0:
                ni_gff_z2 += ni * self._gaunt_factor.evaluate(z, te, wvl) * z * z

        wvl = wvl * 1e-9  # nm -> m
        fre = SPEED_OF_LIGHT / wvl # Hz
        te = te * ELEMENTARY_CHARGE  # eV -> J

        # radiance = (ELEMENTARY_CHARGE**2 * RECIP_4_PI / VACUUM_PERMITTIVITY)**3
        # radiance *= 32 * M_PI**2 / (3 * sqrt(3) * ELECTRON_REST_MASS**2 * SPEED_OF_LIGHT**3) 
        # radiance *= sqrt(2 * ELECTRON_REST_MASS / M_PI / te)
        # emissivity per Hz [W m^-3 Hz^-1]
        radiance = ne * ni_gff_z2 * FRAC_1 * FRAC_2 * FRAC_3 / sqrt(te) * exp(- PLANCK_CONSTANT * fre / te)  # [W m^-3 Hz^-1]
        # transform to wavelength and convert to per steradian per nanometer [W m^-3 nm^-1 sr^-1]
        radiance *= SPEED_OF_LIGHT / wvl / wvl * RECIP_4_PI * 1e-9

        
        # # bremsstrahlung equation W/m^3/str/nm
        # pre_factor = 0.95e-19 * RECIP_4_PI * ni_gff_z2 * ne / (sqrt(te) * wvl)
        # radiance = pre_factor * exp(- EXP_FACTOR / (te * wvl)) * PH_TO_J_FACTOR

        # convert to W/m^3/str/nm
        # return radiance / wvl
        return radiance

    cdef int _populate_cache(self) except -1:

        cdef list species_charge
        cdef Species species

        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")

        if self._gaunt_factor is None:
            if self._atomic_data is None:
                raise RuntimeError("The emission model is not connected to an atomic data source.")

            # initialise Gaunt factor on first run using the atomic data
            self._gaunt_factor = self._atomic_data.free_free_gaunt_factor()

        species_charge = []
        for species in self._plasma.get_composition():
            if species.charge > 0:
                species_charge.append(species.charge)

        # Gaunt factor takes Z as double to support Zeff, so caching Z as float64
        self._species_charge = np.array(species_charge, dtype=np.float64)
        self._species_charge_mv = self._species_charge

        self._species_density = np.zeros_like(self._species_charge)
        self._species_density_mv = self._species_density

    def _change(self):

        # clear cache to force regeneration on first use
        if not self._user_provided_gaunt_factor:
            self._gaunt_factor = None
        self._species_charge = None
        self._species_charge_mv = None
        self._species_density = None
        self._species_density_mv = None
