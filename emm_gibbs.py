from scipy import stats

import numpy as np
from numpy import random

import matplotlib.pyplot as plt
import string
import util

class EMM_Gibbs():
	def __init__(self, num_entities, num_tokens, alpha):
		"""
        num_topics: desired number of topics
        alpha: a scalar
		"""
		self.num_entities = num_entities
		self.num_tokens = num_tokens
		self.alpha = alpha
		self.entity_ids = [idx for idx in range(self.num_entities)]

	def _initialize(self, data):
		# number of times token l is assigned to entity k
		self.nkl = np.zeros((self.num_entities, self.num_tokens))

		# (token) -> entity
		self.entities = list()

		for key, value in enumerate(data):
			z =  np.random.randint(self.num_entities)
			self.entities.append(z)
			self.nkl[z, value] += 1

	def _conditional_distribution(self, l):
		num = self.alpha + self.nkl[:, l]
		den = np.sum([self.alpha] * self.num_tokens + self.nkl, axis=1)
		p_z = num / den
		p_z /= np.sum(p_z) # normalize
		return p_z

	def _sample(self, p_z):
		return np.random.choice(self.entity_ids, p=p_z)

	def loglikelihood(self):
		"""
		P(m|K,S) = probability of data given state
		"""
		ak = [self.alpha] * self.num_tokens
		score = 0.0
		for idx in range(self.num_entities):
			num = util.log_multi_beta(ak + self.nkl[idx, :])
			den = util.log_multi_beta(ak)
			score += (num - den)
		return score

	def run(self, data, maxiter=100):
		"""
		Run Gibbs sampler
		"""
		self._initialize(data)

		for it in range(maxiter):
			print(self.loglikelihood())
			for key, value in enumerate(data):
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
