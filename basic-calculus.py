import numpy as np
from numpy.typing import NDArray
from numpy import floating
import plotext as plt


def f(x: float | NDArray[floating]) -> float | NDArray[floating]:
    return 3 * x**2 - 4 * x + 5


if __name__ == "__main__":
    print(f(3.0))

    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)

    print(xs)
    print(ys)

    # Console plot
    plt.plot(xs, ys)
    # plt.show()

    h = 0.001
    x = 3.0
    print(f(x))
    print(f(x + h))
