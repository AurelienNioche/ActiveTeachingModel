from main_avya import run

from learner.act_r_custom import ActRMeaning
from teacher.leitner import LeitnerTeacher


def main():

    run(student_model=ActRMeaning, teacher_model=LeitnerTeacher,
        student_param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.1},
        n_item=25, t_max=150)


if __name__ == '__main__':
    main()