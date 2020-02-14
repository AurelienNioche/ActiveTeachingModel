from . teacher import Teacher


class TeacherPerfectInfo(Teacher):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def get_item(self, param, *args, **kwargs):

        super().get_item(param=self.param, *args, **kwargs)
