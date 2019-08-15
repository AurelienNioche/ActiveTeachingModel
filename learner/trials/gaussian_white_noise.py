import matplotlib.pyplot as plt
import numpy as np


class GaussianNoise:
    def __init__(self, mu=0, sigma=65**0.5, data_size=2000, xi_0=65,
                 n_neurons=10**5, n_population=3871):

        self.xi_0 = xi_0
        self.mu = mu
        self.sigma = sigma

        self.data_size = data_size
        self.data = np.zeros(data_size)

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

    def compute_noise(self):
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

        for i in range(self.data_size):
            self.data[i] = np.random.normal(loc=0, scale=self.amplitude)


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


def main():
    np.random.seed(123)

    gaussian_noise = GaussianNoise()
    gaussian_noise.compute_noise()
    # plot_noise(gaussian_noise)
    plot_hist(gaussian_noise)


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
