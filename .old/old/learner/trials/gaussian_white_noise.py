import matplotlib.pyplot as plt
import numpy as np


class GaussianNoise:
    def __init__(self, mu=0, sigma=65**0.5, n_iteration=2000, xi_0=65,
                 n_neurons=10**5, n_population=3871):

        self.xi_0 = xi_0
        self.mu = mu
        self.sigma = sigma

        self.n_iteration = n_iteration
        self.data = np.zeros(n_iteration)

        self.n_population = n_population
        self.n_neurons = n_neurons
        self.s_v = 1 / self.n_neurons * n_population
        self.amplitude = self.xi_0 * self.s_v * self.n_neurons
        print(self.amplitude)

    def normal_distribution(self, x):

        return 1 / self.sigma * (2 * np.pi)**0.5 * np.exp(
            (self.mu - x)**2 / 2 * self.sigma**2
        )

    def probability_density(self, x):
        return self.amplitude * np.exp(
            (self.mu - x)**2 / 2 * self.sigma**2
        )

    def compute_noise_std_amplitude(self):
        """
        Modified from http://greenteapress.com/thinkdsp/html/thinkdsp005.html:

        Uncorrelated uniform (UU) spectrum has equal power at all frequencies,
        on average, therefore UU noise is white.

        When people talk about “white noise”, they don’t always mean UU
        noise. In fact, more often they mean uncorrelated Gaussian (UG) noise.

        Uncorrelated Gaussian (UG) noise is similar in many ways to UU noise.
        The spectrum has equal power at all frequencies, on average, so UG is
        also white. And it has one other interesting property: the spectrum of
        UG noise is also UG noise. More precisely, the real and imaginary parts
        of the spectrum are uncorrelated Gaussian values.
        """

        for i in range(self.n_iteration):
            self.data[i] = np.random.normal(loc=0, scale=self.amplitude)

    def compute_noise_xi_0_var(self):
        for i in range(self.n_iteration):
            self.data[i] = np.random.normal(loc=0, scale=self.xi_0**0.5)

    def compute_noise_xi_0_std(self):
        for i in range(self.n_iteration):
            self.data[i] = np.random.normal(loc=0, scale=self.xi_0)

    def compute_noise_xi_0_var_n(self):
        for i in range(self.n_iteration):
            self.data[i] = np.random.normal(loc=0, scale=self.xi_0
                                            * self.n_neurons)

    def compute_noise_value_amplitude(self):
        for i in range(self.n_iteration):
            self.data[i] = np.random.normal(loc=0, scale=1) * self.amplitude

    def compute_noise_sum(self):
        self.data = np.sum(
            np.random.normal(loc=0, scale=self.xi_0**0.5,
                             size=(self.n_population, self.n_iteration)), axis=0)

    def compute_noise_sqrt_xi_0_n(self):

        self.data = np.random.normal(
            loc=0, scale=(self.xi_0 ** 0.5) * self.n_population,
            size=self.n_iteration)


def plot_noise(noise):
    x = np.arange(0, noise.data.size, 1)
    y = noise.data
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("noise")
    plt.plot(x, y)
    plt.show()


def plot_hist(noise):
    y = noise.data
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.title("noise")
    plt.hist(y)
    plt.show()


def plot_all(noise):
    x = np.arange(0, noise.data.size, 1)

    noise.compute_noise_std_amplitude()
    y1 = np.copy(noise.data)

    noise.compute_noise_xi_0_var()
    y2 = np.copy(noise.data)

    noise.compute_noise_xi_0_std()
    y3 = np.copy(noise.data)

    noise.compute_noise_xi_0_var_n()
    y4 = np.copy(noise.data)

    noise.compute_noise_value_amplitude()
    y5 = np.copy(noise.data)

    fig_all = plt.figure(0)

    plt.subplot(5, 1, 1)
    plt.plot(x, y1)
    plt.title("Amplitude is the std")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.subplot(5, 1, 2)
    plt.plot(x, y2)
    plt.title("$\\xi_{0}$ is the variance")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.subplot(5, 1, 3)
    plt.plot(x, y3)
    plt.ylabel("Noise value")
    plt.title("$\\xi_{0}$ is the std")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.subplot(5, 1, 4)
    plt.plot(x, y4)
    plt.title("$\\xi_{0} * N$ is std")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.subplot(5, 1, 5)
    plt.plot(x, y5)
    plt.title("std $= 1$ and noise value $* N$")
    plt.xlabel("step")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    fig_all.show()


def plot_single_vs_population(noise):
    x = np.arange(0, noise.data.size, 1)

    noise.compute_noise_xi_0_var()
    y1 = np.copy(noise.data) * noise.n_population

    noise.compute_noise_std_amplitude()
    y2 = np.copy(noise.data)

    fig_vs = plt.figure(1)

    plt.subplot(1, 2, 1)
    plt.plot(x, y1)
    plt.title(f"Noise according to {noise.n_population} single neurons")
    plt.xlabel("step")

    plt.subplot(1, 2, 2)
    plt.plot(x, y2)
    plt.title(f"Noise according to population (w/ amplitude)")
    plt.xlabel("step")

    fig_vs.show()


def plot_sum_vs_population(noise):
    x = np.arange(0, noise.data.size, 1)

    noise.compute_noise_sum()
    y1 = np.copy(noise.data) * noise.n_population

    noise.compute_noise_sqrt_xi_0_n()
    y2 = np.copy(noise.data)

    fig_vs = plt.figure(1)

    plt.subplot(1, 2, 1)
    plt.plot(x, y1)
    plt.title(f"Noise according to {noise.n_population} single neurons")
    plt.xlabel("step")

    plt.subplot(1, 2, 2)
    plt.plot(x, y2)
    plt.title(f"Noise according to population (w/ amplitude)")
    plt.xlabel("step")

    fig_vs.show()


def main():
    np.random.seed(np.random.randint(0, 2**10))

    xi_0 = 65
    n_neurons = 1000
    n_iteration = 2000

    data1 = np.sum(
            np.random.normal(loc=0, scale=xi_0**0.5,
                             size=(n_neurons, n_iteration)), axis=0)

    data2 = np.random.normal(
        loc=0, scale=(xi_0 * n_neurons) ** 0.5, size=n_iteration)

    fig, axes = plt.subplots(ncols=2)

    ax1 = axes[0]
    ax1.plot(data1)
    ax1.set_title("Single unit")

    ax2 = axes[1]
    ax2.plot(data2)
    ax2.set_title("Population")

    plt.show()
    # gaussian_noise = GaussianNoise()
    # # gaussian_noise.compute_noise_std_amplitude()
    # # plot_noise(gaussian_noise)
    # # plot_hist(gaussian_noise)
    # plot_all(gaussian_noise)
    # plot_sum_vs_population(gaussian_noise)


if __name__ == '__main__':
    main()


# def f(x, a, s, mean=0):
#     return a*np.exp(
#         -((x-mean)**2) / (2*s**2)
#     )

# Gaussian white noise
# amplitude = 251615
# sigma = 65**0.5
#
# X = np.linspace(start=-100, stop=100, num=1000)
#
# plt.plot(X, f(X, a=amplitude, s=sigma))
# plt.show()
