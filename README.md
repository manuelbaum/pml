# Probabilistic Machine Learning in PyTorch
Pytorch Implementations for Probabilistic Machine Learning book by Kevin Murphy.

The implementations in here are in some way a trade-off between readability, execution speed and me trying to get stuff
done quickly. This means that I only had limited time to optimize the code and that e.g. stuff that I could potentially
parallelize is still being iterated over using a for loop.

It's also not a collection of solutions to the exercises in the book. Instead it's a collection of algorithms from the
book that I found interesting or useful to implement.

# Installation

pip install .

# Implementations

[11.4.2 EM for GMMs](src/pml/gmm_em.py)

[17.4.2 Forwards Algorithm for HMMs with Gaussian likelihood](src/pml/hmm_gaussian.py)

[17.4.3 Forwards-Backwards Algorithm for HMMs with Gaussian likelihood](src/pml/hmm_gaussian.py)

[17.5.2 EM for HMMs with Gaussian likelihood](src/pml/hmm_gaussian.py)
