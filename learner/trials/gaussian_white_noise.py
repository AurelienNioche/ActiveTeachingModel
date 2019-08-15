import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
        :return:
        """
        # print(self.data)
        # sigma = 1 #/ ((2 * np.pi)**0.5 * self.amplitude)
        # for i in range(self.data_size):
            # self.data[i] = np.random.normal(loc=0, scale=self.xi_0**0.5)
            # self.data[i] = np.random.normal(loc=0, scale=sigma)

        # print(self.data)

        # self.data *= self.amplitude
        # print(self.data)

        for i in range(self.data_size):
            self.data[i] = np.random.normal(loc=0, scale=self.amplitude)


class DeterministicGen(stats.rv_continuous):

    def _cdf(self, x):

        # return np.where(x < 0, 0., 1.)

        return self.amplitude * np.exp(
            (self.mu - x)**2 / 2 * self.sigma**2
        )

    def _stats(self):
        return 0., 0., 0., 0.


def plot_noise(noise):
    x = np.arange(0, noise.data.size, 1)
    y = noise.data
    plt.title("aa")
    plt.plot(x, y)


def plot_hist(noise):
    y = noise.data
    plt.hist(y)
    plt.show()


def plot_hist2(noise):
    mu = 0
    sigma = noise.xi_0**0.5
    s = noise.data
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()


def main():
    gaussian_noise = GaussianNoise()#n_population=10)
    gaussian_noise.compute_noise()
    # plot_noise(gaussian_noise)
    plot_hist(gaussian_noise)
    # plot_hist2(gaussian_noise)
    # deterministic = DeterministicGen(name="deterministic")
    # print(deterministic.cdf(np.arange(-20, 20, 0.5)))


if __name__ == '__main__':
    main()


def f(x, a, s, mean=0):
    return a*np.exp(
        -((x-mean)**2) / (2*s**2)
    )

# Gaussian white noise
# amplitude = 251615
# sigma = 65**0.5
#
# X = np.linspace(start=-100, stop=100, num=1000)
#
# plt.plot(X, f(X, a=amplitude, s=sigma))
# plt.show()
