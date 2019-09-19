## Grant Gasser
## Discrete Probability Distributions
## Guide: https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/
## 9/19/19

import numpy as np
from numpy.random import binomial, multinomial, randint

# Examples of bernoulli: medical test (+/-), coin flip (H/T), cat or dog, etc.
# Binomial is just a sequence of independent bernoulli trials

def simulate_binomial(p, k, n):
    """
    Function simulates n Bernoulli processes (binomial distribution)

    Args:
        p (float): probability a given trial is a success
        k (int): number of bernoulli trials (failure/success) to perform
            in the process
        n (int): number of bernoulli processes to run

    Returns:
        p_estimate (float): an "estimate" of p

    NOTE: if X is the random variable representing the number of successes,
    E[X] = k*p. So we would expect p_estimate to be close to (k*p). Also, as n -> inf,
    p_estimate -> p. Try it out!
    """
    p_estimate = 0

    for _ in range(n):
        num_success = binomial(k, p)
        p_estimate += num_success

    p_estimate = p_estimate / n

    return p_estimate


# Examples of multinoulli (1 trial, c classes): rolling a die (1,2,3,4,5,6)
# Multinomial: multiple independent multinoulli trials
# Examples of multinomial: sequence of independent dice rolls

def simulate_multinomial(p, k, c, n):
    """
    Function simulates n Multinoulli processes (multinomial distribution)

    Args:
        p (List[float]): probability of each class/category occuring
        k (int): number of multinoulli trials
        c (int): number of classes/categories [2,10]
        n (int): number of multionoulli processes to run

    Returns:
        hmmmmm
    """
    total_cat = np.zeros(c) # aggregate then average the outcome in each cat for each process

    for _ in range(n):
        cat = np.array(multinomial(k, p))
        total_cat += cat

    total_cat = total_cat / n

    return total_cat




def main():
    """ Test simulate_binomial """
    print('\n\nSimulating binomial...')
    print(simulate_binomial(.43, 100, 100)) # expect close to 43

    print(simulate_binomial(.7, 10, 20)) # somewhat close to 7

    print(simulate_binomial(.67, 100, 10000)) # really close to 67


    """ Test simulate multinomial """
    print('\n\nSimulating multinomial...')
    c = randint(2,10) # c for number of classes

    p = [1.0/float(c) for _ in range(c)]
    k = 100

    print('Expecting each entry in list to be close to', (1.0/float(c))*k)
    print(simulate_multinomial(p, k, c, 100)) # expect each cat in list to be close to (1/c) * k



main()
