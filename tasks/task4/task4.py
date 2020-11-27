import numpy as np
import matplotlib.pyplot as plt


class RandomProcess:

    def __init__(self, size: int, dispersion: int, alpha: int, step: int):
        self.size = size
        self.dispersion = dispersion
        self.alpha = alpha
        self.step = step
        self.tau = step / size
        self.noise = self.discrete_white_noise()

    def discrete_white_noise(self) -> list:
        return np.random.standard_normal(self.size)

    def temporary_generation(self) -> list:
        return np.arange(0, self.step, self.tau)

    def sliding_func(self, k: int) -> list:
        f = self.autocorrelation_func
        return [f(self.tau * n) * self.noise[n - k] for n in range(1, k + 1)]

    def sliding_summation(self) -> list:
        return [sum(self.sliding_func(k)) for k in range(self.size)]

    def autocorrelation_func(self, tau: int) -> float:
        return self.dispersion * np.exp(-self.alpha * np.abs(tau)) * \
            (
                1 + self.alpha * np.abs(self.tau) +
                (self.alpha ** 2 * tau ** 2) / 3
            )

    def expected_value(self, sequence: list) -> float:
        return np.mean(sequence)

    def dispersion_value(self, sequence: list) -> float:
        return np.var(sequence)


def calculations(process: object) -> None:
    sequence = process.sliding_summation()
    x, y = process.temporary_generation(), process.sliding_summation()

    print(f'\nExpected value: {process.expected_value(sequence)}')
    print(f'Dispersion: {process.dispersion_value(sequence)}')

    print(f'\nExpected value X: {process.expected_value(x)}')
    print(f'Dispersion X: {process.dispersion_value(x)}')

    print(f'\nExpected value Y: {process.expected_value(y)}')
    print(f'Dispersion Y: {process.dispersion_value(y)}')


def plotting(process: object) -> None:
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    x, y = range(len(process.noise)), process.noise
    axs[0].scatter(x, y, c=np.random.rand(process.size))

    x, y = process.temporary_generation(), process.sliding_summation()
    axs[1].plot(x, y, c='green')

    axs[0].set_title('White noise')
    axs[1].set_title('Random process')
    plt.show()


def main():
    size, dispersion, alpha, step = 100, 1, 1, 10
    process = RandomProcess(size, dispersion, alpha, step)

    calculations(process)
    plotting(process)


if __name__ == '__main__':
    main()
