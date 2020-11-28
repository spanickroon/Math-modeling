import time
import numpy as np
import matplotlib.pyplot as plt


class RandomProcess:

    def __init__(self, size: int, dispersion: float, alpha: float, step: int):
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

    def expected_value(self, sequence: list) -> float:
        return np.mean(sequence)

    def dispersion_value(self, sequence: list) -> float:
        return np.var(sequence)

    def normalized_correlation_func(self, tau: float) -> float:
        return self.autocorrelation_func(tau) / self.autocorrelation_func(0)

    def autocorrelation_func(self, tau: float) -> float:
        return self.dispersion * np.exp(-self.alpha * np.abs(tau)) * \
            (
                1 + self.alpha * np.abs(self.tau) +
                (self.alpha ** 2 * tau ** 2) / 3
            )

    def power_spectral_densities(self, sigma: float) -> float:
        return (
            (8 * self.dispersion ** 2 * self.alpha ** 5) /
            3 * np.pi * (self.alpha ** 2 + sigma ** 2) ** 3
            )


def timer(func: object):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f'\nFunction: {func.__name__}.')
        print(f'Lead time: {time.time() - start_time}s')
        return result
    return wrapper


def correlation_coeff(process: object, step_1: int, step_2: int) -> float:
    size, dispersion, alpha = list(process.__dict__.values())[:3]

    process_1 = RandomProcess(size, dispersion, alpha, step_1)
    process_2 = RandomProcess(size, dispersion, alpha, step_1)

    sequence_1 = process_1.sliding_summation()
    sequence_2 = process_2.sliding_summation()

    af_1 = process_1.autocorrelation_func
    af_2 = process_2.autocorrelation_func

    ds_1 = process_1.dispersion_value
    ds_2 = process_2.dispersion_value

    tau_1 = step_1 / size
    tau_2 = step_2 / size

    return (
        (af_1(tau_1 * tau_2) - af_1(tau_1) * af_2(tau_2)) /
        (ds_1(sequence_1) * ds_2(sequence_2)) ** (1/2)
    )


@timer
def calculations(process: object) -> None:
    sequence = process.sliding_summation()
    x, y = process.temporary_generation(), process.sliding_summation()

    print(f'\nExpected value: {process.expected_value(sequence)}')
    print(f'Dispersion: {process.dispersion_value(sequence)}')

    print(f'\nExpected value X: {process.expected_value(x)}')
    print(f'Dispersion X: {process.dispersion_value(x)}')

    print(f'\nExpected value Y: {process.expected_value(y)}')
    print(f'Dispersion Y: {process.dispersion_value(y)}')

    ncf = process.normalized_correlation_func
    print('\nNormalized correlation function:')
    print(f'NCF(-t) = NCF(t) | t = 0.5 | {ncf(-0.5):.9f} = {ncf(0.5):.9f}')
    print(f'NCF(t) <= 1 | t = 0.8 | {ncf(0.8):.9f} <= 1')
    print(f'NCF(0) = 1 | t = 0 | {ncf(0):.9f}')

    print(f'\nCorrelation coefficient: {correlation_coeff(process, 10, 20)}')

    sigma = 0.05
    print('Power spectral densities: ', end='')
    print(f'{process.power_spectral_densities(sigma)}')


@timer
def plotting(process: object) -> None:
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

    x, y = range(len(process.noise)), process.noise
    axs[0][0].scatter(x, y, c=np.random.rand(process.size))
    axs[1][0].scatter(x, y, c='green')

    x, y = process.temporary_generation(), process.sliding_summation()
    axs[0][1].plot(x, y, c='green')
    axs[1][1].plot(x, y, c='green')

    for color in ['red', 'blue']:
        process.size += 2
        process.tau = process.step / process.size
        process.noise = process.discrete_white_noise()

        x, y = range(len(process.noise)), process.noise
        axs[1][0].scatter(x, y, c=color)

        x, y = process.temporary_generation(), process.sliding_summation()
        axs[1][1].plot(x, y, c=color)

    axs[0][0].set_title('White noise')
    axs[1][0].set_title('White noises')
    axs[0][1].set_title('Random process')
    axs[1][1].set_title('Random processes')

    plt.show()


def main():
    size, dispersion, alpha, step = 200, 1, 1, 10
    process = RandomProcess(size, dispersion, alpha, step)

    calculations(process)
    plotting(process)


if __name__ == '__main__':
    main()
