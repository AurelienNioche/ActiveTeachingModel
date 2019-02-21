import numpy as np
import json
import itertools as it


class Task:

    def __init__(self, config_file='model/parameters/parameters.json'):

        self.parameters = self.import_parameters(config_file)

        self.t_max = int(self.parameters['t_max'])
        self.n = self.parameters['n']

        self.questions = np.arange(self.n)
        self.replies = np.arange(self.n)

    @staticmethod
    def import_parameters(config_file):

        try:
            parameters = json.load(open(config_file))

        except FileNotFoundError:
            config_file = 'model/parameters/default.json'
            parameters = json.load(open(config_file))

        finally:
            print(f"Using the config file '{config_file}'")

        return parameters

    def get_reply(self, question):

        return self.replies[question]


class TaskConnect(Task):

    def __init__(self, config_file='model/parameters/parameters.json'):

        super().__init__(config_file)

        # Let assume that semantic is 2d and graphic is 2d
        self.n_graphic_dimension = 2
        self.n_semantic_dimension = 2

        self.c_graphic = np.zeros((self.n, self.n))
        self.c_semantic = np.zeros((self.n, self.n))

    def compute_distances(self):

        pass

        # p_graphic = np.zeros((self.n, ))
        #
        # for i, j in it.combinations(range(self.n), r=2):
        #
        #     g, s = np.random.random(), np.random.random()
        #
        #     for x, y in [(i, j), (j, i)]:
        #
        #         c_graphic[x, y] = g
        #         c_semantic[x, y] = s

# dist = numpy.linalg.norm(a-b)



