import datasets
from emm_gibbs import EMM_Gibbs

"""
Entity Mixture Model - EMM
K entities
N mentions 
L tokens (A, B, C, etc)
S - latent entities
M - observed mentions
"""
# log-normal prior
mu = 0.5
sigma = 1.0

# dirichlet prior
alpha = 0.001

K = 10 # Number of Entities
N = 100 # Number of Mentions
L = 10 # Number of Tokens

data, entities = datasets.generate(K, N, L, alpha)
emm = EMM_Gibbs(K, L, alpha)
emm.run(data)
