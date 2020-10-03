import numpy as np
from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

def compute_variance(X):
    mean = np.mean(X, axis=0)
    return sum([np.outer(x - mean, x-mean) for x in X]) / (len(X) - 1)

if __name__ == "__main__":
    # main routin
    cov = np.eye(2)*0.01
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, cov=cov, cm=0.5, population_size=10000)
    while True:
        solutions = []
        X = optimizer.ask_all(inball=True)
        C_emp1 = compute_variance(X)
        X2 = optimizer.ask_all(inball=False)
        C_emp2 = compute_variance(X2)
        print("=== testing covariance =====")
        print(C_emp1)
        print(optimizer._C * optimizer._sigma**2)
        values = quadratic(X[:, 0], X[:, 1]).tolist()
        optimizer.tell(list(zip(X, values)))
        print(optimizer._mean)

        if optimizer.should_stop():
            break
