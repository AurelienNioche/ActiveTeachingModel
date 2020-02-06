import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import itertools as it


def antti_example():
    def f_u(x):
        return x**2 + 0.1 + 0.1 * np.random.randn()
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]
    myBopt = GPyOpt.methods.BayesianOptimization(
        f=f_u, domain=bounds, acquisition_type='EI',
        exact_feval = False, initial_design_numdata=10, normalize_Y=False)
    max_iter = 0; max_time = 60; eps = 10e-6

    myBopt.run_optimization(max_iter, eps)
    myBopt._update_model()
    #myBopt.plot_acquisition()
    #myBopt.model.model.plot([0.0,1.0])
    model = myBopt.model.model

    x_grid = np.arange(0, 1, 0.001)
    x_grid = x_grid.reshape(len(x_grid),1)
    m, v = model.predict(x_grid)

    model.plot_density([0,1], alpha=.5)

    plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
    plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
    plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

    Xdata, Ydata = myBopt.get_evaluations()

    plt.plot(Xdata, Ydata, 'r.', markersize=10)

    plt.show()


def example2D():
    def f_u(param):
        x, y = param[0]
        z = x ** 2 - 3*y
        return z

    bounds = [
        {'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)},
    ]
    myBopt = GPyOpt.methods.BayesianOptimization(
        f=f_u, domain=bounds, acquisition_type='EI',
        exact_feval = False, initial_design_numdata=10, normalize_Y=False)
    max_iter = 10
    # max_time = 60
    eps = 10e-6

    myBopt.run_optimization(max_iter, eps)

    print("x_opt", myBopt.x_opt)
    print("fx_opt", myBopt.fx_opt)

    # myBopt._update_model()
    myBopt.plot_acquisition()

    model = myBopt.model.model

    eval_points = np.array(list(it.product(np.linspace(0, 1, 10), repeat=2)))
    m, v = model.predict(eval_points)
    for i, m_i, v_i in zip(eval_points, m, v):
        print(i, m_i[0], v_i[0])


def example3D():
    def f_u(param):
        x, y, z = param[0]
        print(x)
        print(y)
        z = x ** 2 - 3 * y + z
        print(z)
        print()
        return z

    bounds = [
        {'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'var_3', 'type': 'continuous', 'domain': (0, 1)},
    ]
    myBopt = GPyOpt.methods.BayesianOptimization(
        f=f_u, domain=bounds, acquisition_type='EI',
        exact_feval=False, initial_design_numdata=10, normalize_Y=False)
    max_iter = 2000
    max_time = 10
    eps = 10e-6

    myBopt.run_optimization(max_time=max_time, max_iter=max_iter, eps=eps)

    print("x_opt", myBopt.x_opt)
    print("fx_opt", myBopt.fx_opt)

    # myBopt._update_model()
    myBopt.plot_convergence()
    # #myBopt.model.model.plot([0.0,1.0])
    # model = myBopt.model.model
    #
    # x_grid = np.arange(0, 1, 0.001)
    # x_grid = x_grid.reshape(len(x_grid),1)

    import itertools as it

    # m, v = model.predict(np.array(list(it.product(np.linspace(0, 1, 10), repeat=2))))

    # model.plot_density([0,1], alpha=.5)
    #
    # plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
    # plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
    # plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)
    #
    # Xdata, Ydata = myBopt.get_evaluations()
    #
    # plt.plot(Xdata, Ydata, 'r.', markersize=10)
    #
    # plt.show()

example2D()