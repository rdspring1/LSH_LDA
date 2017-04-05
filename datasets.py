import numpy as np
from numpy import random
import string

def generate(K, N, L, alpha):
	"""
	Generate data:
		1) Sample the number of entities K using log-normal prior
		2) For each entity, draw a distribution of tokens from a Dirichlet prior.
		3) For each mention, assign an entity using uniform distribution
		4) Then, using the mention's entity, draw a token from the entity's dictionary.
	"""
	data = []
	entities = []
	theta = np.random.dirichlet([alpha] * L, K)
	token_ids = [idx for idx in range(L)]
	for idx in range(N):
		entity =  np.random.randint(K)
		token = np.random.choice(token_ids, p=theta[entity, :])
		data.append(token)
		entities.append(entity)
	return data, entities
