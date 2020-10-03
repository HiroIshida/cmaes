import numpy as np
from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    cov = np.eye(2)*0.01
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, cov=cov)
    while True:
        solutions = []
        X = optimizer.ask_all(inball=True)
        values = quadratic(X[:, 0], X[:, 1]).tolist()
        optimizer.tell(list(zip(X, values)))
        print(optimizer._mean)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
