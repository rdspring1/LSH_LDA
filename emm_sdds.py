from scipy import stats
from scipy.special import gammaln

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import random
import string
import util

"""
Entity Mixture Model - EMM
K entities
N mentions 
L tokens (A, B, C, etc)
S - latent entities
M - observed mentions
"""

class EMM_SDDS():
	def __init__(self, num_entities, num_tokens, alpha):
		"""
        num_topics: desired number of topics
        alpha: a scalar
		"""
		self.num_entities = num_entities
		self.num_tokens = num_tokens
		self.alpha = alpha
		self.ak = [self.alpha] * self.num_tokens

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

		self.key_list = list(self.entities.keys())
		self.available_keys = list()

		for key, value in enumerate(data):
			z =  np.random.randint(self.num_entities)
			self.token_entity.append(z)
			self.entities[z].append((key, value))
			self.nkl[z, value] += 1

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

	def new_loglikelihood(self, add, remove):
		"""
		P(m|K,S) = probability of data given potential new state
		"""
		score = 0.0
		den = util.log_multi_beta(self.ak)

		new_nkl = [self.count(e) for e in add]
		for idx in range(len(new_nkl)):
			num = util.log_multi_beta(self.ak + new_nkl[idx])
			score += (num-den)
		
		for key in self.nkl.keys():
			if key not in remove:
				num = util.log_multi_beta(self.ak + self.nkl[key])
				score += (num-den)
		return score

	def loglikelihood(self):
		"""
		P(m|K,S) = probability of data given state
		"""
		score = 0.0
		den = util.log_multi_beta(self.ak)
		for key in self.nkl.keys():
			num = util.log_multi_beta(self.ak + self.nkl[key])
			score += (num-den)
		return score

	def split_merge(self):
		# 1) smart_split, dumb_merge, dumb_split, smart_merge
		choice = np.random.choice([idx for idx in range(4)])

		# 2) create proposal [add], [remove], forward_prob, reverse_prob
		if choice == 0:
			add, remove, fp, bp = self.smart_split()
		elif choice == 1:
			add, remove, fp, bp = self.dumb_merge()
		elif choice == 2:
			add, remove, fp, bp = self.smart_merge()
		else:
			add, remove, fp, bp = self.dumb_split()
		
		num = bp * np.exp(self.new_loglikelihood(add, remove))
		den = fp * np.exp(self.loglikelihood())
		accept = num / den

		# 3) calculate acceptance probability
		if random.random() < accept:
			# 4) execute update if new state is accepted 
			self.update(add, remove)

	def split_prob(self):
		"""
		1 / P(m_ek | E_k) - inverse probability of data given entity
		"""
		scores = dict()
		z = 0.0
		den = util.log_multi_beta(self.ak)

		for key in self.nkl.keys():
			num = util.log_multi_beta(self.ak + self.nkl[key])
			score = np.exp(num-den)
			scores[key] = score
			z += score

		for key, value in scores.items()
			scores[key] = value / z
		return scores

	def merge_prob(self, e1):
		"""
		P(m_ei_ej, E_j | E_i) - probability of data i+j, entity j given entity i
		"""
		den = util.log_multi_beta(self.ak)
		scores = dict()
		z = 0.0
		for e2 in self.nkl.keys():
			if e1 != e2:
				num = util.log_multi_beta(self.ak + self.nkl[e1] + self.nkl[e2])
				score = np.exp(num-den)
				z += score
				scores[e2] = score

		for key, value in scores.items()
			scores[key] /= z
		return scores

	def smart_split_p(self, item, e, a=1):
		nkl = self.count(e)
		num = a + nkl[item]
		den = 0.0
		for idx in len(self.num_tokens):
			den += (a + nkl[idx])
		return num/den

	def smart_split(self):
		"""
			forward - smart split
			backward - dumb merge
		"""
		# draw entity based on lowest likelihood
		# calculate likelihood of merge with other entities
		scores = self.split_prob(e1)
		keys, prob = zip(*scores.items())
		e12 = np.random.choice(keys, p=prob)

		e1 = list()
		e2 = list()
		split_p = 1.0
		for item in self.entities[e12]:
			e1_p = self.smart_split_p(item, e1)
			e2_p = self.smart_split_p(item, e2)
			p = e1 / (e1_p + e2_p)
			if random.random() <= p:
				e1.append(item)
				split_p *= p
			else:
				e2.append(item)
				split_p *= (1.0 - p)

		K = len(self.entities) 
		fp = scores[e12] * split_p
		bp = 2.0 / (K * (K-1))
		return [e1, e2], [e12], fp, bp

	def smart_merge(self):
		"""
			forward - smart merge
			backward - dumb split
		"""
		# Select random entity e1
		e1 = np.random.choice(self.key_list)

		# calculate likelihood of merge with other entities
		scores = self.merge_prob(e1)

		# draw other entity e2
		keys, prob = zip(*scores.items())
		e2 = np.random.choice(keys, p=prob)
		assert(e1 != e2)

		# merge entities
		e12 = list()
		for item in self.entities[e1]:
			e12.append(item)
		for item in self.entities[e2]:
			e12.append(item)

		fp = scores[e2]
		bp = 1.0 / (len(self.entities) * pow(2, len(e12)))
		return [e12], [e1, e2], fp, bp

	def dumb_split(self):
		"""
			forward - dumb split
			backward - smart merge
		"""
		e12 = np.random.choice(self.key_list)

		e1 = list()
		e2 = list()
		for item in self.entities[e12]:
			if random.random() <= 0.5:
				e1.append(item)
			else:
				e2.append(item)

		fp = 1.0 / (len(self.entities) * pow(2, len(self.entities(e12))))
		bp = self.reverse_merge(e1, e2, e12)
		return [e1, e2], [e12], fp, bp

	def dumb_merge(self):
		"""
			forward - dumb merge
			backward - smart split
		"""
		e1, e2 = np.random.choice(self.key_list, 2)
		assert(e1 != e2)

		e12 = list()
		for item in self.entities[e1]:
			e12.append(item)
		for item in self.entities[e2]:
			e12.append(item)

		K = len(self.entities) 
		fp = 1.0 / (K * (K-1))
		bp = self.reverse_split(e12, e1, e2)
		return [e12], [e1, e2], fp, bp

	def count(self, e):
		nkl = np.zeros((self.num_tokens))
		for key, value in e:
			nkl[value] += 1
		return nkl

	def reverse_split(self, e12, e1, e2):
		"""
			Probability smart-split reverses dumb-merge
			Selecting merged entity in new state	
		"""
		den = util.log_multi_beta(self.ak)
		scores = list()

		e12_nkl = self.count(e12)
		num = util.log_multi_beta(self.ak + e12_nkl)
		e12_score = np.exp(num-den)
		scores.append(e12_score)

		for key in self.nkl.keys():
			if key != e1 and key != e2:
				num = util.log_multi_beta(self.ak + self.nkl[key])
				scores.append(np.exp(num-den))

		e12_z = np.sum(scores)
		p = e12_z / e12_score
		return p
		
	def reverse_merge(self, e1, e2, e12):
		"""
			Probability smart-merge reverses dumb-split
			Selecting other entity given first entity in new state
		"""
		den = util.log_multi_beta(self.ak)
		e1_nkl = self.count(e1)
		e2_nkl = self.count(e2)

		e1_scores = list()
		e2_scores = list()
		num = util.log_multi_beta(self.ak + e1_nkl + e2_nkl)
		new_score = np.exp(num-den)
		e1_scores.append(new_score)
		e2_scores.append(new_score)

		for key in self.nkl.keys():
			if key != e12:
				e1_num = util.log_multi_beta(self.ak + e1_nkl + self.nkl[key])
				e2_num = util.log_multi_beta(self.ak + e2_nkl + self.nkl[key])
				e1_scores.append(np.exp(e1_num-den))
				e2_scores.append(np.exp(e2_num-den))

		e1_z = np.sum(e1_scores)
		e2_z = np.sum(e2_scores)
		e1_p = new_score / e1_z
		e2_p = new_score / e2_z
		p = (1.0 / (len(self.entities)+1)) * np.mean([e1_p, e2_p])
		return p

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
