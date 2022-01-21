
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from numpy import ndarray
from raysect.optical.observer import FibreOptic

from cherab.tools.observers.group.base import Observer0DGroup


class FibreOpticGroup(Observer0DGroup):
    """
    A group of fibre optics under a single scene-graph node.

    A scene-graph object regrouping a series of 'FibreOptic'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list observers: A list of fibre optics (FibreOptic instances) in this
                            group.
    :ivar list/float acceptance_angle: The angle in degrees between the z axis and the cone
                                       surface which defines the fibres solid angle sampling
                                       area. The same value can be shared between all observers,
                                       or each observer can be assigned with individual value.
    :ivar list/float radius: The radius of the fibre tip in metres. This radius defines a circular
                             area at the fibre tip which will be sampled over. The same value
                             can be shared between all observers, or each observer can be
                             assigned with individual value.

    .. code-block:: pycon

       >>> from math import cos, sin, pi
       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.optical.observer import SpectralPowerPipeline0D, PowerPipeline0D, FibreOptic
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import FibreOpticGroup
       >>> from cherab.tools.observers.plotting import plot_group_total, plot_group_spectra
       >>>
       >>> world = World()
       ...
       >>> group = FibreOpticGroup(parent=world)
       >>> group.add_observer(FibreOptic(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, sin(pi/10)), name="Fibre 1"))
       >>> group.add_observer(FibreOptic(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="Fibre 2"))
       >>> group.add_observer(FibreOptic(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, -sin(pi/10)), name="Fibre 3"))
       >>> group.connect_pipelines([(SpectralPowerPipeline0D, 'MySpectralPipeline', None),
                                    (PowerPipeline0D, 'MyMonoPipeline', None)])  # add pipelines to all fibres in the group
       >>> group.acceptance_angle = 2  # same value for all fibres in the group
       >>> group.radius = 2.e-3
       >>> group.spectral_bins = 512
       >>> group.pixel_samples = [2000, 1000, 2000]  # individual value for each fibre in the group
       >>> group.display_progress = False  # control pipeline parameters through the group observer
       >>> group.observe()  # combined observation
       >>> 
       >>> plot_group_spectra(group, item='MySpectralPipeline', in_photons=True)  # plot the spectra
       >>> plot_group_total(group, item='MyMonoPipeline')  # plot the total signals
       >>> plt.show()
    """

    @Observer0DGroup.observers.setter
    def observers(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("The observers attribute of FibreOpticGroup must be a list or tuple of FibreOptics.")

        for observer in value:
            if not isinstance(observer, FibreOptic):
                raise TypeError("The observers attribute of FibreOpticGroup must be a list or tuple of "
                                "FibreOptics. Value {} is not a FibreOptic.".format(observer))

        # Prevent external changes being made to this list
        for observer in value:
            observer.parent = self

        self._observers = tuple(value)
    
    def add_observer(self, fibre):
        """
        Adds new fibre optic to the group.

        :param FibreOptic fibre: Fibre optic to add.
        """
        if not isinstance(fibre, FibreOptic):
            raise TypeError("The fiber argument must be of type FibreOptic.")
        fibre.parent = self
        self._observers = self._observers + (fibre, )

    @property
    def acceptance_angle(self):
        # The angle in degrees between the z axis and the cone surface which defines the fibres
        # solid angle sampling area.
        return [observer.acceptance_angle for observer in self._observers]

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.acceptance_angle = v
            else:
                raise ValueError("The length of 'acceptance_angle' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.acceptance_angle = value

    @property
    def radius(self):
        # The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
        # which will be sampled over.
        return [observer.radius for observer in self._observers]

    @radius.setter
    def radius(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.radius = v
            else:
                raise ValueError("The length of 'radius' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.radius = value
