
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
from raysect.core import Node
from raysect.core.workflow import RenderEngine
from raysect.optical.observer import Observer0D


class Observer0DGroup(Node):
    """
    A base class for handling groups of nonimaging observers as one Node.

    A scene-graph object regrouping a series of observers as a scene-graph parent.
    Allows combined observation and display control simultaneously.
    Note that for any property except `names` and `pipelines`, the same value can be shared between
    all observers, or each observer can be assigned with individual value.

    :ivar list names: A list of observer names.
    :ivar list pipelines: A list of all pipelines connected to each observer in the group.
    :ivar list/RenderEngine render_engine: Rendering engine used by the observers.
                                           Note that if the engine is shared, changing its
                                           parameters for one observer in a group will affect
                                           all observers.
    :ivar list/float min_wavelength: Lower wavelength bound for sampled spectral range.
    :ivar list/float max_wavelength: Upper wavelength bound for sampled spectral range.
    :ivar list/int spectral_bins: The number of spectral samples over the wavelength range.
    :ivar list/float ray_extinction_prob: Probability of ray extinction after every material
                                          intersection.
    :ivar list/float ray_extinction_min_depth: Minimum number of paths before russian roulette
                                               style ray extinction.
    :ivar list/int ray_max_depth: Maximum number of Ray paths before terminating Ray.
    :ivar list/float ray_important_path_weight: Relative weight of important path sampling.
    :ivar list/int pixel_samples: The number of samples to take per pixel.
    :ivar list/int samples_per_task: Minimum number of samples to request per task.
    """
    def __init__(self, parent=None, transform=None, name=None, observers=None):
        super().__init__(parent=parent, transform=transform, name=name)
        if observers is None:
            self._observers = tuple()
        else:
            for observer in observers:
                self.add_observer(observer)

    def __getitem__(self, item):

        if isinstance(item, int):
            try:
                return self._observers[item]
            except IndexError:
                raise IndexError("observer number {} not available in this {} "
                                 "with only {} observers.".format(item, self.__class__.__name__, len(self._observers)))
        elif isinstance(item, str):
            observers = [observer for observer in self._observers if observer.name == item]
            if len(observers) == 1:
                return observers[0]

            if len(observers) == 0:
                raise ValueError("observer '{}' was not found in this {}.".format(item, self.__class__.__name__))

            raise ValueError("Found {} observers with name {} in this {}.".format(len(observers), item, self.__class__.__name__))
        else:
            raise TypeError("{} key must be of type int or str.".format(self.__class__.__name__))

    @property
    def observers(self):
        """
        A list of all Observer0D object assigned to the group.
        The group is set as a parent to any added observer."""
        return self._observers

    @observers.setter
    def observers(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("The observers attribute of Observer0DGroup must be a list or tuple of Observer0D.")

        if not all([isinstance(val, Observer0D) for val in value]):
            raise ValueError('All observers assigned to the group must be of type Observer0D')
        
        for observer in value:
            observer.parent = self

        self._observers = tuple(value)
    
    def add_observer(self, observer):
        """
        Adds new observer to the group

        :param Observer0D observer: observer to add
        """
        if not isinstance(observer, Observer0D):
            raise ValueError("Can only add Observer0D objects")
        observer.parent = self
        self._observers = self._observers + (observer, )

    @property
    def names(self):
        # A list of observer names.
        return [observer.name for observer in self._observers]

    @names.setter
    def names(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.name = v
            else:
                raise ValueError("The length of 'names' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            raise TypeError("The names attribute must be a list or tuple.")
    
    @property
    def pipelines(self):
        """A list of all pipelines connected to each observer in the group"""
        return [observer.pipelines for observer in self._observers]
    
    @pipelines.setter
    def pipelines(self, pipelist):
        if len(pipelist) == len(self._observers):
            for observer, pipelines in zip(self._observers, pipelist):
                observer.pipelines = pipelines
        else:
            raise ValueError('Length of pipelines list do not match number of observers in the group.')
    
    @property
    def render_engine(self):
        # Rendering engine used by the observers.
        return [observer.render_engine for observer in self._observers]

    @render_engine.setter
    def render_engine(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    if isinstance(v, RenderEngine):
                        observer.render_engine = v
                    else:
                        raise TypeError("The list 'render_engine' must contain only RenderEngine instances.")
            else:
                raise ValueError("The length of 'render_engine' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            if not isinstance(value, RenderEngine):
                raise TypeError("The list 'render_engine' must contain only RenderEngine instances.")
            for observer in self._observers:
                observer.render_engine = value

    @property
    def min_wavelength(self):
        # Lower wavelength bound for sampled spectral range.
        return [observer.min_wavelength for observer in self._observers]

    @min_wavelength.setter
    def min_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.min_wavelength = v
            else:
                raise ValueError("The length of 'min_wavelength' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.min_wavelength = value

    @property
    def max_wavelength(self):
        # Upper wavelength bound for sampled spectral range.
        return [observer.max_wavelength for observer in self._observers]

    @max_wavelength.setter
    def max_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.max_wavelength = v
            else:
                raise ValueError("The length of 'max_wavelength' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.max_wavelength = value

    @property
    def spectral_bins(self):
        # The number of spectral samples over the wavelength range.
        return [observer.spectral_bins for observer in self._observers]

    @spectral_bins.setter
    def spectral_bins(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.spectral_bins = v
            else:
                raise ValueError("The length of 'spectral_bins' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.spectral_bins = value

    @property
    def ray_extinction_prob(self):
        # Probability of ray extinction after every material intersection.
        return [observer.ray_extinction_prob for observer in self._observers]

    @ray_extinction_prob.setter
    def ray_extinction_prob(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.ray_extinction_prob = v
            else:
                raise ValueError("The length of 'ray_extinction_prob' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.ray_extinction_prob = value

    @property
    def ray_extinction_min_depth(self):
        # Minimum number of paths before russian roulette style ray extinction.
        return [observer.ray_extinction_min_depth for observer in self._observers]

    @ray_extinction_min_depth.setter
    def ray_extinction_min_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.ray_extinction_min_depth = v
            else:
                raise ValueError("The length of 'ray_extinction_min_depth' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.ray_extinction_min_depth = value

    @property
    def ray_max_depth(self):
        # Maximum number of Ray paths before terminating Ray.
        return [observer.ray_max_depth for observer in self._observers]

    @ray_max_depth.setter
    def ray_max_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.ray_max_depth = v
            else:
                raise ValueError("The length of 'ray_max_depth' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.ray_max_depth = value
   
    @property
    def ray_important_path_weight(self):
        # Relative weight of important path sampling.
        return [observer.ray_important_path_weight for observer in self._observers]

    @ray_important_path_weight.setter
    def ray_important_path_weight(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.ray_important_path_weight = v
            else:
                raise ValueError("The length of 'ray_important_path_weight' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.ray_important_path_weight = value

    @property
    def pixel_samples(self):
        # The number of samples to take per pixel.
        return [observer.pixel_samples for observer in self._observers]

    @pixel_samples.setter
    def pixel_samples(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.pixel_samples = v
            else:
                raise ValueError("The length of 'pixel_samples' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.pixel_samples = value

    @property
    def samples_per_task(self):
        # Minimum number of samples to request per task.
        return [observer.samples_per_task for observer in self._observers]

    @samples_per_task.setter
    def samples_per_task(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.samples_per_task = v
            else:
                raise ValueError("The length of 'samples_per_task' ({}) "
                                 "mismatches the number of observers ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.samples_per_task = value
    
    def observe(self):
        """
        Starts the observation.
        """
        for observer in self._observers:
            observer.observe()
    
    @property
    def pipelines(self):
        # A list of all pipelines connected to each sight-line in the group.
        return [observer.pipelines for observer in self._observers]

    def connect_pipelines(self, pipeline_classes, keywords_list=None, suppress_display_progress=True):
        """
        Creates and connects a new set of given pipelines to each observer in the group.

        Pipeline classes are instantiated using parameters specified in approrpiate dict from keywords list.
        If keywords list is provided, it length must match the number of provided pipeline classes.

        :param list pipeline_classes: list of pipeline classes to be connected with observers
        :param list keywords_list: list of dicts with keywords passed to init methods of pipeline classes
                                   its length must match the number of pipeline classes
                                   for default parameters place an empty dict to approriate place in the list
        :param bool suppress_display_progress: Toggles setting display_progress to False for each compatible pipeline (default=True)
        """
        if keywords_list is None:
            keywords_list = [dict() for ppln in pipeline_classes]
        if len(pipeline_classes) !=len (keywords_list):
            raise ValueError('The number of given pipeline classes does not match the number of dicts in keyword list.\
                              For each pipeline class there must be a parameter dict.')
        for observer in self._observers:
            pipelines = []
            for PipelineClass, kwargs in zip(pipeline_classes, keywords_list):
                pipeline = PipelineClass(**kwargs)
                if suppress_display_progress:
                    try:
                        pipeline.display_progress = False
                    except AttributeError:
                        pass
                pipelines.append(pipeline)
            observer.pipelines = pipelines
        return
