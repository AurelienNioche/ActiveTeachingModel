from abc import ABC

import numpy as np
from learner.act_r import ActR
# np.seterr(all='raise')


class ActRMeaning(ActR, ABC):

    version = 3.1
    bounds = {**ActR.bounds, **{'m': (0.0, 5)}}

    def __init__(self, semantic_connections, param=None, metaclass=False,
                 **kwargs):

        super().__init__(metaclass=True, **kwargs)

        if not metaclass:

            self.semantic_connections = semantic_connections

            # Decay parameter
            self.d = None
            # Retrieval threshold
            self.tau = None
            # Noise in the activation levels
            self.s = None

            # Helping from items with close meaning
            self.m = None

            # For easiness
            self.temp = None
            self.x = None
            self.c_x = None

            self.set_cognitive_parameters(param)

        self.n_item = len(self.semantic_connections)
        self.items = np.arange(self.n_item)

    def _get_presentation_effect_i_and_j(self, item, time, time_index):

        pr_effect_i = self._presentation_effect(item,
                                                time=time,
                                                time_index=time_index)
        if not pr_effect_i:
            return 0, None, None

        # For connected items
        list_j = self.items[self.items != item]
        pr_effect_j = np.array(
            [self._presentation_effect(j,
                                       time=time,
                                       time_index=time_index
                                       ) for j in list_j])

        return pr_effect_i, pr_effect_j, list_j

    def p_recall(self, item, time=None, time_index=None):

        pr_effect_i, pr_effect_j, list_j = \
             self._get_presentation_effect_i_and_j(item=item, time=time,
                                                   time_index=time_index)
        if pr_effect_i == 0:
            return 0

        contrib = (self.c_x[item, list_j] * pr_effect_j).sum() * self.x

        _sum = pr_effect_i + contrib
        if _sum <= 0:
            return 0

        try:
            base_activation = np.log(_sum)

        except FloatingPointError as e:
            print(pr_effect_j, contrib, pr_effect_i + contrib)
            raise e

        return self._sigmoid_function(base_activation)

    def init(self):

        # Short cut
        self.temp = self.s * np.square(2)

        # Aliases for easiness of computation
        self.x = self.m
        self.c_x = self.semantic_connections

# ========================================================================== #


class ActRGraphic(ActRMeaning, ABC):

    bounds = {**ActR.bounds, **{'g': (0.0, 5)}}

    def __init__(self, graphic_connections, param, **kwargs):

        super().__init__(metaclass=True, **kwargs)

        # Decay parameter
        self.d = None

        # Retrieval threshold
        self.tau = None

        # Noise in the activation levels
        self.s = None

        # Graphic 'help'
        self.g = None

        # For easiness
        self.temp = None
        self.c_x = None
        self.x = None

        self.graphic_connection = graphic_connections
        self.set_cognitive_parameters(param)

    def init(self):

        # Short cut
        self.temp = self.s * np.square(2)

        # Aliases for the easiness of computation
        self.c_x = self.graphic_connection
        self.x = self.g


# ========================================================================== #


class ActRPlus(ActRMeaning):

    bounds = {**ActR.bounds, **{'m': (0.0, 5)}, **{'g': (0.0, 5)}}

    def __init__(self, graphic_connections,
                 semantic_connections, param, **kwargs):

        super().__init__(metaclass=True, **kwargs)

        # Decay parameter
        self.d = None
        # Retrieval threshold
        self.tau = None
        # Noise in the activation levels
        self.s = None

        self.g = None
        self.m = None

        self.graphic_connections = graphic_connections
        self.semantic_connections = semantic_connections

        self.set_cognitive_parameters(param)

    def p_recall(self, item, time=None, time_index=None):

        pr_effect_i, pr_effect_j, list_j = \
            self._get_presentation_effect_i_and_j(item=item, time=time,
                                                  time_index=time_index)

        if pr_effect_i == 0:
            return 0

        semantic_contrib = \
            (self.semantic_connections[item, list_j] * pr_effect_j).sum() \
            * self.m
        graphic_contrib = \
            (self.graphic_connections[item, list_j] * pr_effect_j).sum() \
            * self.g

        _sum = pr_effect_i + semantic_contrib + graphic_contrib
        if _sum <= 0:
            return 0

        base_activation = np.log(_sum)

        return self._sigmoid_function(base_activation)
