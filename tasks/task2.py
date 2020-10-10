import math
import numpy as np
import random
import time
import pylab
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import accumulate
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
        random.shuffle(numbers)
        return np.array_split(numbers / sum(numbers), m)

    def generate_vector(self, size: int) -> list:
        return [i for i in range(size)]

    def distribution_func(self, probabilities: list) -> list:
        return [0] + list(accumulate(probabilities))

    def generate_matrix_y(self, matrix: list, x_probabilities: list) -> list:
        return [row / x_probabilities[i] for i, row in enumerate(matrix)]

    def discrete_random_variables(
            self, v_x: list, v_y: list,
            f_x: list, f_y: list):
        x = [random.random() for i in range(self.number_amount)]
        y = [random.random() for i in range(self.number_amount)]

        new_x, new_y = [], []

        for i, j in zip(x, y):
            new_x.append(
                v_x[f_x.index(next(filter(lambda x: i < x, f_x))) - 1])

            z = new_x[0]

            new_y.append(
                v_y[f_y[z].index(next(filter(lambda y: j < y, f_y[z]))) - 1])

        drv = list(zip(new_x, new_y))

        matrix = np.zeros((len(v_x), len(v_y)))
        empir_matrix = np.copy(matrix)

        for pair in drv:
            matrix[pair[0]][pair[1]] += 1
            empir_matrix[pair[0]][pair[1]] += (1 / self.number_amount)

        return matrix, empir_matrix, drv

    def construction_distrib_func(self, vector: list, dist: list) -> tuple:
        x = sum([[i, j, np.nan] for i, j in zip(vector[:-1], vector[1:])], [])
        y = sum([[i, i, np.nan] for i in dist[1:]], [])
        return x + [x[-2], x[-2] + (x[-2] - x[-3]), np.nan], y


@timer
def main():
    start, quantity = 19941995, 100_000
    m, k = 2**31, 16807
    a, b = 0, 1
    intervals = 10
    s = 2
    m, n = 4, 4

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

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].set_yticks(np.arange(0, 1, 0.02), minor=False)

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

    x, y = np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)
    x, y = np.meshgrid(x, y)
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x, y, np.exp(-x - y), rstride=1, cstride=1, cmap=cm.jet)

    matrix_xy = generator.generate_matrix(m, n)

    print('\nProbability matrix XY:')
    pprint(matrix_xy)
    print(f'Sum of probabilities XY: {sum(sum(matrix_xy))}')

    vector_x = generator.generate_vector(m)
    vector_y = generator.generate_vector(n)

    print(f'\nVector X:\n{vector_x}')
    print(f'\nVector Y:\n{vector_y}')

    x_probabilities = np.array([sum(i) for i in matrix_xy])
    print(f'\nProbability X:\n{x_probabilities}')

    distribution_x = generator.distribution_func(x_probabilities)
    print(f'\nDistribution function F(X):\n{distribution_x}')

    y_propabities = generator.generate_matrix_y(matrix_xy, x_probabilities)
    print('\nProbability matrix Y:')
    pprint(y_propabities)

    distribution_y = [generator.distribution_func(i) for i in y_propabities]
    print('\nDistribution function F(Y):')
    pprint(distribution_y)

    drv = generator.discrete_random_variables(
        vector_x, vector_y,
        distribution_x, distribution_y)

    print(f'\nDRV matrix:\n{drv[0]}')
    print(f'\nEmpirical matrix:\n{drv[1]}')

    xy = np.array(drv[2])
    x, y = xy[:, 0], xy[:, 1]
    m_x, d_x = np.mean(x), sum(x ** 2 - np.mean(x) ** 2) / len(x)
    m_y, d_y = np.mean(y), sum(y ** 2 - np.mean(y) ** 2) / len(y)

    print(f'\nExpected value, dispersion X:\n{m_x} {d_x}')
    print(f'\nExpected value, dispersion Y:\n{m_y} {d_y}')

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    x, y = generator.construction_distrib_func(vector_x, distribution_x)
    axs.set_yticks(np.arange(0, 1.05, 0.05), minor=False)
    axs.plot(x, y, 'o-')
    axs.set_title('Distribution function F(X):')

    fig, axs = plt.subplots(1, len(distribution_y), sharey=True, tight_layout=True)

    for i, distribution in enumerate(distribution_y):
        x, y = generator.construction_distrib_func(vector_y, distribution)
        axs[i].set_yticks(np.arange(0, 1.05, 0.05), minor=False)
        axs[i].plot(x, y, 'o-')
        axs[i].set_title(f'Distribution function F(Y)[{i}]:')

    plt.show()
    pylab.show()


if __name__ == '__main__':
    main()
