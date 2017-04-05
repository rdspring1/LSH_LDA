import os
from zipfile import ZipFile
import numpy as np

zipname = "glove.6B.zip"
filename = "glove.6B.100d.txt"
embeddings_index = {}
with ZipFile(os.path.join(os.getcwd(), zipname)) as myzip:
	with myzip.open(filename) as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs

for key in embeddings_index.keys():
	print(key)
