## Grant Gasser
## Discrete Probability Distributions
## https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/
## 9/19/19

from numpy.random import binomial

# Examples of bernoulli: medical test (+/-), coin flip (H/T), cat or dog, etc.
# Binomial is just a sequence of independent bernoulli trials

def simulate_binomial(p, k, n):
    """
    Function simulates n Bernoulli processes (binomial distribution)

    Args:
        p (float): probability a given trial is a success
        k (int): number of bernoulli trials (failure/success) to perform
            in the process

    Returns:
        p_estimate (float): an "estimate" of p

    NOTE: if X is the random variable representing the number of successes,
    E[X] = k*p. So we would expect p_estimate to be close to (k*p). Also, as n -> inf,
    p_estimate -> p. Try it out!
    """
    p_estimate = 0

    for x in range(n):
        num_success = binomial(k, p)
        p_estimate += num_success

    p_estimate = p_estimate / n

    return p_estimate


def main():
    """ Test simulate_binomial """
    print(simulate_binomial(.43, 100, 100)) # expect close to 43

    print(simulate_binomial(.7, 10, 20)) # somewhat close to 7

    print(simulate_binomial(.67, 100, 10000)) # really close to 67


    """ Test simulate multinomial """



main()
