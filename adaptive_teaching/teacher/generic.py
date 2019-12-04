class GenericTeacher:

    version = 0.0

    def __init__(self, task_param, learner_model=None,
                 confidence_threshold=9999):

        self.n_item = task_param['n_item']
        self.learner_model = learner_model
        self.confidence_threshold = confidence_threshold

    def ask(self, best_param):
        raise NotImplementedError

    def update(self, item, response):
        pass
