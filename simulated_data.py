import numpy as np
import scipy.linalg as la
from scipy.stats import multinomial

def random_multivar_normal(n, d, k, sigma=.1):
    '''
    Generate random samples from a random multivariate normal distribution
    with covariance A A^T + sigma^2 I.
    Input:
        n: int, number of samples
        d: int, dimension of samples
        k: int, number of samples approximated
        sigma: optional float > 0, default .1, the standard deviation
            of the sample noise.

    Output:
        cov: d x d array, the covariance matrix for the distibution
        A0: d x d array, the eigenvectors we want to estimate (note the
            eigenvectors are columns of the array, in descending order)
        X: n x d array of n d-dimensional samples.
    '''
    A0 = la.qr(np.random.rand(d, k), mode='economic')[0]
    cov = A0 @ A0.T + sigma**2 * np.eye(d)
    X = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    return cov, A0, X

def spiked_covariance(n, d, k, sigma=.1):
    '''
    Generate random samples from a random multivariate normal distribution
    with covariance A D A^T + sigma^2 I.
    Here A is a set of k orthogonal vectors and D is a diagonal matrix with
    random, uniform entries, sorted and scaled so that the first entry = 1.
    Input:
        n: int, number of samples
        d: int, dimension of samples
        k: int, number of samples approximated
        sigma: optional float > 0, default .1, the standard deviation
            of the sample noise.

    Output:
        cov: d x d array, the covariance matrix for the distibution
        w: d vector of the diagonal values from matrix D.
        A0: d x k array, the eigenvectors we want to estimate (note the
            eigenvectors are columns of the array, in descending order)
        X: n x d array of n d-dimensional samples.
    '''
    A0 = la.qr(np.random.rand(d, k), mode='economic')[0]
    w = np.sort(np.random.rand(k, 1), axis=0)[::-1]
    w /= w.max()

    cov = A0 @ (w**2 * A0.T) + sigma**2 * np.eye(d)
    X = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    return cov, w, A0, X

def random_multinomial(n, d, trials=100, mean0 = True, scale=1):
    '''
    Generate random samples from a random multinomial distribution with p_i ~ U(0,1).
    Input:
        n: int, number of samples
        d: int, dimension of samples
        trials: optional int, the number of trials for each sample from the multinomial distribution
           default is 100.
        mean0: optional boolean, default True. Indicates whether to normalize the samples
            so they are mean 0.

    Output:
       cov: d x d array, the covariance matrix for the distribution
       e: d-dimensional array, the eigenvalues of the covariance matrix
       v: d x d array, the eigenvectors of the covariance matrix
       X: n x d array of n d-dimensional samples from the random_dirichlet distribution
           with covariance cov.
    '''
    # Initialize p values
    p = np.random.rand(d)
    p /= p.sum()

    # Calculate the covariance matrix for the multinomial distribution
    # For large d > 10000, use multinomial.cov(d,p)
    if d >= 10000:
        cov = multinomial.cov(trials, p)
    else:
        cov = -np.outer(p, p) * trials
        cov[np.diag_indices(d)] = trials * p * (1-p)

    cov *= scale**2

    # Obtain the eigenvectors of the covariance matrix.
    e, v = la.eigh(cov)
    e = e[::-1]
    v = v[:,::-1]

    # Obtain a sample from the multinomial distribution of size n
    X = np.random.multinomial(trials, p, n).astype(float)

    if mean0:
        # Let X have mean 0
        X -= trials * p

    X *= scale
    return cov, e, v, X

def random_dirichlet(n, d, mean0=True, scale=1):
    '''
    Generate random samples from a random dirichlet distribution with a_i ~ U(0,1).
    Input:
        n: int, number of samples
        d: int, dimension of samples
        mean0: optional boolean, default True. Indicates whether to normalize the samples
            so they are mean 0.

    Output:
       cov: d x d array, the covariance matrix for the distribution
       e: d-dimensional array, the eigenvalues of the covariance matrix
       v: d x d array, the eigenvectors of the covariance matrix
           (note the eigenvectors are columns of the array, in descending order)
       X: n x d array of n d-dimensional samples from the random_dirichlet distribution
           with covariance cov.
    '''
    # Initialize a random set of parameters a drawn from the
    # uniform distribution.
    a = np.random.rand(d)
    a0 = a.sum()
    a_denom = a0**2 * (a0 + 1)

    # Obtain the covariance matrix for the dirichlet distribution.
    # Note that scipy doesn't currently have a builtin method for this
    # (I may add one myself)
    cov = -np.outer(a, a) / a_denom # i neq j case
    cov[np.diag_indices(d)] = a * (a0 - a) / a_denom # i = j case
    cov *= scale**2

    # Obtain the eigenvectors of the covariance matrix.
    e, v = la.eigh(cov)
    e = e[::-1]
    v = v[:,::-1]

    X = np.random.dirichlet(a, n)
    if mean0:
        X -= a / a0

    X *= scale

    return cov, e, v, X
