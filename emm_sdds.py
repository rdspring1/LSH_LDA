from scipy import stats
from scipy.special import gammaln

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import random
import string
import util

class EMM_SDDS():
	def __init__(self, num_entities, num_tokens, alpha):
		"""
        num_topics: desired number of topics
        alpha: a scalar
		"""
		self.num_entities = num_entities
		self.num_tokens = num_tokens
		self.alpha = alpha

	def _initialize(self, data):
		# (token) -> entity
		self.token_entity = list()

		# number of times token l is assigned to entity k
		self.nkl = dict()

		# (entity) -> (mention, tokens)
		self.entities = dict()

		for idx in range(self.num_entities):
			self.entities[idx] = list()
			self.nkl[idx] = np.zeros((self.num_tokens))
		self.key_list = [key for key in self.entities.keys()]	

		self.available_keys = list()

		for key, value in enumerate(data):
			z =  np.random.randint(self.num_entities)
			self.token_entity.append(z)
			self.entities[z].append((key, value))
			self.nkl[z, value] += 1

	def split_merge(self):
		# TODO
		# 1) smart_split, dumb_merge, dumb_split, smart_merge
		# 2) create proposal [add], [remove], forward_prob, reverse_prob
		# 3) calculate acceptance probability
		# 4) execute update if new state is accepted 

	def gibbs(self, key, value):
		# select entity for current mention
		z = self.entities[key]
		self.nkl[z, value] -= 1

		# remove current entity from current state
		p_z = self._conditional_distribution(value)
		z = self._sample(p_z)

		# add new entity, mention pair to distribution
		self.nkl[z, value] += 1

		# update entity
		self.entities[key] = z

	def smart_split(self):
		# TODO
		return 0

	def smart_merge(self):
		# TODO
		return 0

	def dumb_split(self):
		e12 = np.random.choice(self.key_list)
		p = 1.0 / (len(self.entities) * pow(2, len(self.entities(e12))))

		e1 = list()
		e2 = list()
		for item in self.entities[e12]:
			if random.random() <= 0.5:
				e1.append(item)
			else:
				e2.append(item)
		return [e1, e2], [e12], p

	def dumb_merge(self):
		e1, e2 = np.random.choice(self.key_list, 2)
		e12 = list()
		for item in self.entities[e1]:
			e12.append(item)
		for item in self.entities[e2]:
			e12.append(item)
		p = 1.0 / (len(self.entities) * (len(self.entities-1)))
		return [e12], [e1, e2], p

	def update(self, add, remove):
		# remove old entities
		for item in remove:
			self.entities.pop(item)
			self.nkl.pop(item)
			self.available_key.append(item)

		# add new entities
		for item in add:
			new_key = len(self.entities) if not self.available_key else self.available_key.pop()
			self.entities[new_key] = item
			self.nkl[new_key] = np.zeros((self.num_tokens))
			for mention, token in item:
				self.nkl[new_key][token] += 1
				self.token_entity[mention] = new_key

		self.key_list = [key for key in self.entities.keys()]	

	def run(self, data, maxiter=100):
		"""
		Run SDDS MCMC
		"""
		self._initialize(data)

		for it in range(maxiter):
			print(self.loglikelihood())
			for key, value in enumerate(data):
				self.split_merge()
				self.gibbs(key, value)

"""
Entity Mixture Model - EMM
K entities
N mentions 
L tokens (A, B, C, etc)
S - latent entities
M - observed mentions
"""
