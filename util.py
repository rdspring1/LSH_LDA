import numpy as np
from scipy.special import gammaln
from scipy.special import gamma

def log_multi_beta(alpha):
	"""
	Logarithm of the multivariate beta function.
	"""
	return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

def multi_beta(alpha):
	"""
	Multivariate beta function.
	"""
	return np.prod(gamma(alpha)) / gamma(np.sum(alpha))
