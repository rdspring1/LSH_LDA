import numpy as np
from scipy.special import gammaln

def log_multi_beta(alpha):
	"""
	Logarithm of the multinomial beta function.
	"""
	return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
