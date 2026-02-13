import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    
    data = np.random.normal(0, 1, n)

    plt.figure()
    plt.hist(data, bins=10)
    plt.title("Normal Distribution (0,1)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.close()

    return data


def uniform_histogram(n):
    
    data = np.random.uniform(0, 10, n)

    plt.figure()
    plt.hist(data, bins=10)
    plt.title("Uniform Distribution (0,10)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.close()

    return data


def bernoulli_histogram(n):
    
    data = np.random.binomial(1, 0.5, n)

    plt.figure()
    plt.hist(data, bins=10)
    plt.title("Bernoulli Distribution (p=0.5)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.close()

    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    
    total = 0
    count = 0

    for value in data:
        total += value
        count += 1

    return total / count


def sample_variance(data):
    
    mean = sample_mean(data)
    total = 0
    count = 0

    for value in data:
        total += (value - mean) ** 2
        count += 1

    return total / (count - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    
    data = list(data)
    data.sort()
    n = len(data)

    minimum = data[0]
    maximum = data[-1]

    
    if n % 2 == 1:
        median = data[n // 2]
    else:
        median = (data[n // 2 - 1] + data[n // 2]) / 2

    

    q1_index = int(0.25 * (n - 1))
    q3_index = int(0.75 * (n - 1))

    q1 = data[q1_index]
    q3 = data[q3_index]

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    total = 0
    count = 0

    for xi, yi in zip(x, y):
        total += (xi - mean_x) * (yi - mean_y)
        count += 1

    return total / (count - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
