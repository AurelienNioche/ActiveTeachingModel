from learner.act_r_tracking import ActRTracking
from teacher.tracking_teacher import TrackingTeacher
import matplotlib.pyplot as plt


def main(t_max=300, n_items=6):

    teacher = TrackingTeacher(t_max=t_max, n_items=n_items, handle_similarities=False)
    agent = ActRTracking(parameters={"d": 0.5, "tau": 0.01, "s": 0.06}, task_features=teacher.task_features)
    questions, replies, successes = teacher.teach(agent=agent)

    plt.plot(agent.p)
    plt.show()


if __name__ == "__main__":
    main()
