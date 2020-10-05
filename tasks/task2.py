import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from pprint import pprint

from task1 import RandomVariableSensor, RandomVariableTester


def timer(func: object):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func()
        print(f'\n\nLead time: {time.time() - start_time}s')
    return wrapper


class GenerationRandomVariables:

    def __init__(self, number_amount: int) -> None:
        self.number_amount = number_amount

    def density(self, x: float, y: float) -> float:
        return math.exp(-x - y)

    def neumann_method(self, a: float, b: float) -> list:
        return [self.generate_xy() for i in range(self.number_amount)]

    def generate_xy(self) -> tuple:
        m, k = 2**31, 16807
        start = random.randint(10**8, 10**9)
        value = random.randint(10**8, 10**9)

        xy = RandomVariableSensor(2).mult_congruent_method(start, m, k)
        w = random.random()

        return xy if w <= self.density(*xy) else self.generate_xy()

    def correlation(
            self, x: list, y: list,
            m_x: float, m_y: float,
            d_x: float, d_y: float):
        x, y = np.array(x), np.array(y)
        return (sum(x * y) / len(x) - m_x * m_y) / (d_x * d_y) ** (1/2)

    def generate_matrix(self, m: int, n: int) -> list:
        numbers = np.array([i for i in range(1, m * n + 1)])
        return np.array_split(numbers / sum(numbers), m)


@timer
def main():
    start, quantity = 19941995, 10000
    m, k = 2**31, 16807
    a, b = 0, 1
    intervals = 10
    s = 2
    m, n = 5, 5

    generator = GenerationRandomVariables(quantity)
    sensor = RandomVariableSensor(quantity)

    new_xy = generator.neumann_method(a, b)
    x, y = zip(*new_xy)

    hits_rate_x = RandomVariableTester(intervals, x).hits_rate()
    hits_rate_y = RandomVariableTester(intervals, y).hits_rate()

    m_x, d_x, h_x = RandomVariableTester(intervals, x).uniformity_testing()
    m_y, d_y, h_y = RandomVariableTester(intervals, y).uniformity_testing()

    print('\nRelative hit rates X:')
    pprint(hits_rate_x)
    print(f'Sum of probabilities: {round(sum(hits_rate_x.values()), 10)}')

    print('\nRelative hit rates Y:')
    pprint(hits_rate_y)
    print(f'Sum of probabilities: {round(sum(hits_rate_y.values()), 10)}')

    print('\nExpected value, dispersion, histogram border for X:')
    print(m_x, d_x, h_x)

    print('\nExpected value, dispersion, histogram border for Y:')
    print(m_y, d_y, h_y)

    m_t = -2 * math.exp(-1) + 1
    d_t = -5 * math.exp(-1) + 2 - m_t ** 2

    print('\nTheoretical expected value, dispersion for X:')
    print(f'M[x]={m_t}')
    print(f'D[x]={d_t}')

    print('\nTheoretical expected value, dispersion for Y:')
    print(f'M[y]={m_t}')
    print(f'D[y]={d_t}')

    print('\nCorrelation coefficient X:')
    print(RandomVariableTester(intervals, x).independence_testing(s))

    print('\nCorrelation coefficient Y:')
    print(RandomVariableTester(intervals, y).independence_testing(s))

    print('\nCorrelation coefficient XY:')
    print(generator.correlation(x, y, m_x, m_y, d_x, d_y))

    print('\nTheoretical correlation coefficient XY:')
    print(generator.correlation(x, y, m_t, m_t, d_t, d_t))

    matrix_xy = generator.generate_matrix(m, n)
    x_probabilities = [sum(i) for i in matrix_xy]
    y_probabilities = sum(matrix_xy)

    print('\nProbability matrix XY:')
    pprint(matrix_xy)
    print(f'Sum of probabilities XY: {sum(sum(matrix_xy))}')

    print('\nProbability X:')
    print(x_probabilities)

    print('\nProbability Y:')
    print(y_probabilities)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].bar(
        sorted([i[0] for i in hits_rate_x.keys()]),
        hits_rate_x.values(),
        width=1/intervals, align='edge')
    axs[0].set_title('X')

    axs[1].bar(
        sorted([i[0] for i in hits_rate_y.keys()]),
        hits_rate_y.values(),
        width=1/intervals, align='edge')

    axs[1].set_title('Y')

    plt.show()


if __name__ == '__main__':
    main()
