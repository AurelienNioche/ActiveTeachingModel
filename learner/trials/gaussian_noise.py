import numpy as np
import matplotlib.pyplot as plt


class GaussianNoise:
    def __init__(self, data_size=2000, xi_0=65, n_neurons=10**5,
                 n_population=3871):

        self.xi_0 = xi_0

        self.data_size = data_size
        self.data = np.zeros(data_size)

        self.n_population = n_population
        self.n_neurons = n_neurons
        self.s_v = 1 / self.n_neurons * n_population
        self.amplitude = self.xi_0 * self.s_v * self.n_neurons
        print(self.amplitude)

        self.compute_noise()

    def compute_noise(self):
        print(self.data)
        sigma = 1 / ((2 * np.pi)**0.5 * self.amplitude)
        for i in range(self.data_size):
            # self.data[i] = np.random.normal(loc=0, scale=self.xi_0**0.5)
            self.data[i] = np.random.normal(loc=0, scale=sigma)

        print(self.data)

        # self.data *= self.amplitude
        print(self.data)


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
    # plot_noise(gaussian_noise)
    plot_hist(gaussian_noise)
    # plot_hist2(gaussian_noise)


if __name__ == '__main__':
    main()
