from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2))
    b = mean(ys) - m * mean(xs)

    return m, b


def squared_errors(ys_real, ys_calculated):
    return sum((ys_real - ys_calculated) ** 2)


def coefficient_of_determination(ys_real, ys_calculated):
    y_mean_line = [mean(ys_real) for _ in ys_real]

    squared_error_calculated = squared_errors(ys_real, ys_calculated)
    squared_error_y_mean = squared_errors(ys_real, y_mean_line)

    return 1 - (squared_error_calculated / squared_error_y_mean)


m, b = best_fit_slope(xs, ys)
regression_line = [m * x + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
