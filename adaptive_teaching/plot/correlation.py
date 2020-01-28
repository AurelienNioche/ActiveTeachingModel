import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

from scipy.optimize import curve_fit

from utils.plot import save_fig

from scipy.stats.distributions import t

def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def stats(y, p_opt, p_cov, alpha=0.01):  # 95% confidence interval = 100*(1-alpha)

    n = len(y)  # number of data points
    p = len(p_opt)  # number of parameters

    # print(n, p)

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - alpha / 2., dof)

    print(f"student-t value: {tval:.2f}")

    p_err = np.sqrt(np.diag(p_cov))

    for i, p, std in zip(range(n), p_opt, p_err):

        me = std * tval
        print(f'p{i}: {p:.2f} [{p - me:.2f}  {p + me:.2f}]')


def fig_correlation(x_data, y_data, f=sigmoid, fig_folder=None,
        fig_name=None):

    fig, ax = plt.subplots()

    ax.scatter(x_data, y_data)
    # slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    # ax.plot(x, intercept + slope * x)

    # try:
    #
    #     # Fit sigmoid
    #     p_opt, p_cov = curve_fit(f, x_data, y_data)
    #
    #     # Do stats about fit
    #     stats(y=y_data, p_cov=p_cov, p_opt=p_opt)
    #
    #     # Plot
    #     n_points = 50  # Arbitrary neither too small, or too large
    #     x = np.linspace(min(x_data), max(x_data), n_points)
    #     y = f(x, *p_opt)
    #
    #     ax.plot(x, y)
    #
    # except RuntimeError as e:
    #     print(e)

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    else:
        plt.show()