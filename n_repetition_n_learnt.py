import numpy as np
import matplotlib.pyplot as plt

N_ITER = 1000
THR = 0.9
PARAM = (0.02, 0.2)


def exp_forget(delta, fr):
    return np.exp(-fr*delta)


def forgetting_rate(n_pres):
    return PARAM[0] * (1-PARAM[1])**(n_pres-1)


def delta_threshold(fr):
    return - np.log(THR) / fr


def number_learnt(n_pres):
    delta_thr = delta_threshold(forgetting_rate(n_pres))
    print(delta_thr)
    n = np.zeros(len(n_pres), dtype=int)
    n[:] = (N_ITER - delta_thr) / n_pres
    n[n <= 0] = 0
    time_training = n * n_pres
    added_time = time_training + delta_thr
    print(added_time)
    assert np.all(added_time <= N_ITER)
    # r = np.zeros(len(n_pres), dtype=int) - 1
    # enough_time = (time_training + delta_thr) < N_ITER
    # r[enough_time] = delta_thr
    # not_enough_time = np.logical_not(enough_time)
    #
    #
    # n[long_training] = np.maximum(np.ones(np.sum(long_training)), N_ITER - time_training[long_training])
    # short_training =
    # for i in range(len(n_pres)):
    #     n = int()
    #     while True:
    #         time_training = n * n_pres[i]
    #         if time_training > delta_thr[i]:
    #             r[i] = n-1
    #             break

    # # v = (N_ITER - delta_thr) / n_pres
    # # v2 = np.minimum(delta_thr, v)
    # n_iter = np.full(len(n_pres), N_ITER)
    # time_available_for_training = N_ITER-delta_thr
    # v_final = np.minimum(n_iter, delta_thr)
    # # inf_to_zero = v < 0
    # # n_inf_to_zero = np.sum(inf_to_zero)
    # # n_iter = np.full(n_inf_to_zero, N_ITER)
    # # v[inf_to_zero] = np.minimum(n_iter, delta_thr[inf_to_zero])
    return n


def main():

    # fr = 0.02
    #
    # delta_thr = delta_threshold(fr)
    #
    # x = np.arange(100)
    # y = exp_forget(delta=x, fr=fr)
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.axvline(delta_thr, color='black', alpha=0.1, linestyle='--')
    # ax.axhline(THR, color='black', alpha=0.1, linestyle=':')
    # plt.show()

    fig, axes = plt.subplots(nrows=4)

    ax = axes[0]

    n_rep = np.arange(1, 5)

    for i, n in enumerate(n_rep):
        fr = forgetting_rate(n)
        t = np.arange(1000)
        y = exp_forget(delta=t, fr=fr)
        ax.plot(t, y)
        ax.axvline(delta_threshold(fr), color=f'C{i}', alpha=0.1, linestyle='--')
        ax.axhline(THR, color='black', alpha=0.1, linestyle=':')

    fr = forgetting_rate(n_pres=n_rep)
    ax = axes[1]
    ax.plot(n_rep, fr)

    ax = axes[2]
    delta_thr = delta_threshold(fr=fr)
    ax.plot(fr, delta_thr)
    # ax.axvline(delta_thr, color='black', alpha=0.1, linestyle='--')
    # ax.axhline(THR, color='black', alpha=0.1, linestyle=':')

    ax = axes[3]
    ax.plot(n_rep, delta_thr)
    ax.set_xlabel("N repetition")
    ax.set_ylabel("Delta THR")

    # ax = axes[3]
    # n_learnt = number_learnt(n_rep)
    # ax.plot(n_rep, n_learnt)
    # ax.set_xlabel("N repetition")
    # ax.set_ylabel("N learnt")

    #print(np.max(n_learnt))

    plt.show()






if __name__ == '__main__':
    main()
