import time
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from collections import Counter
from pprint import pprint


class RandomVariableSensor:

    def __init__(self, number_amount: int) -> None:
        self.number_amount = number_amount

    def midsquare_method(self, start: int, bit: int) -> list:
        result = []
        temp_number = start
        start = (bit - bit // 2) // 2
        stop = -start

        for number in range(self.number_amount):
            new_number = int(str(f'{temp_number ** 2:0{bit}}')[start: stop])
            result.append(new_number / (10 ** len(str(new_number))))
            temp_number = new_number

        return result

    def mult_congruent_method(self, start: int, m: int, k: int) -> list:
        result = []
        temp_number = start

        for number in range(self.number_amount):
            new_number = self._residue_generator(temp_number, m, k) / m
            result.append(new_number)
            temp_number = new_number * m

        return result

    def _residue_generator(self, a: int, m: int, k: int) -> float:
        return (k * a) % m


class RandomVariableTester:

    def __init__(self, intervals: int, list_random: list) -> None:
        self.intervals = intervals
        self.list_random = list_random
        self.len_random = len(list_random)

    def uniformity_testing(self) -> tuple:
        hits_rate = self.hits_rate()

        histogram_border = 1 / self.intervals
        factor = (1 / self.len_random)

        expected_value = self._expected_value(factor, self.list_random)
        dispersion = self._dispersion(factor, self.list_random, expected_value)

        return (expected_value, dispersion, histogram_border)

    def independence_testing(self, s: int) -> float:
        x = np.array(self.list_random[:-s])
        y = np.array(self.list_random[s:])

        M = self._expected_value(1 / len(x), x)
        D = self._dispersion(1 / len(y), x, M)
        return self._correl_coeff(M, D, x, y)

    def rounding(self, number: float) -> float:
        return float(str(number)[:2 + math.ceil(self.intervals / 10)])

    def hits_rate(self) -> dict:
        hits_segments = []
        step = 1 / self.intervals
        segments = np.arange(0, 1 + step, step)

        for number in self.list_random:
            for i in segments:
                if number < i:
                    end = i
                    break
            hits_segments.append((
                self.rounding(end-step), self.rounding(end)))

        hits = Counter(hits_segments)
        return {k: (v / self.len_random) for k, v in hits.items()}

    def _expected_value(self, factor: float, members_row: list) -> float:
        return factor * sum(members_row)

    def _dispersion(
            self, factor: float, members_row: list,
            expected_value: float) -> float:
        return factor * sum(
            np.array(members_row) ** 2 - expected_value ** 2)

    def _correl_coeff(self, M: float, D: float, x: float, y: float) -> float:
        return (sum(x * y) / len(x) - M * M) / (D * D) ** (1/2)


def main():
    start, bit, quantity = 19941995, 16, 100000
    m, k = 2**31, 16807
    intervals = 10
    s = 10
    start_time = time.time()

    m1 = RandomVariableSensor(quantity).midsquare_method(start, bit)
    m2 = RandomVariableSensor(quantity).mult_congruent_method(start, m, k)

    hits_rate_1 = RandomVariableTester(intervals, m1).hits_rate()
    hits_rate_2 = RandomVariableTester(intervals, m2).hits_rate()

    print('Relative hit rates Mid-square method:')
    pprint(hits_rate_1)
    print(f'Sum of probabilities: {round(sum(hits_rate_1.values()), 10)}')

    print('\nRelative hit rates Multiplicative congruent method:')
    pprint(hits_rate_2)
    print(f'Sum of probabilities: {round(sum(hits_rate_2.values()), 10)}')

    print('\nExpected value, dispersion, histogram border Mid-Sq method:')
    print(RandomVariableTester(intervals, m1).uniformity_testing())

    print('\nExpected value, dispersion, histogram border Mult-cong method:')
    print(RandomVariableTester(intervals, m2).uniformity_testing())

    print('\nCorrelation coefficient Mid-Sq method:')
    print(RandomVariableTester(intervals, m1).independence_testing(s))

    print('\nCorrelation Mult-cong method:')
    print(RandomVariableTester(intervals, m2).independence_testing(s))

    print(f'\nLead time: {time.time() - start_time}s')

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].bar(
        sorted([i[0] for i in hits_rate_1.keys()]),
        hits_rate_1.values(),
        width=1/intervals, align='edge')
    axs[0].set_title('Mid-square method')

    axs[1].bar(
        sorted([i[0] for i in hits_rate_2.keys()]),
        hits_rate_2.values(),
        width=1/intervals, align='edge')

    axs[1].set_title('Multiplicative congruent method')

    plt.show()


if __name__ == '__main__':
    main()
