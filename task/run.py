from model.learner import QLearner
from model.teacher import Teacher

from task import exercise


def run():

    t_max = exercise.t_max

    learner = QLearner()
    teacher = Teacher()

    for t in range(t_max):

        question = teacher.choose_question(t=t)
        reply = learner.decide(question=question)
        correct_answer, success = teacher.evaluate(t=t, reply=reply)
        learner.learn(question=question, reply=reply, correct_answer=correct_answer)

    return teacher.summarize()
