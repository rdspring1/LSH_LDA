from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from lda_gibbs import LdaSampler

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

docs = [doc1, doc2, doc3, doc4, doc5]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in docs]

def create_dictionary(docs):
    dictionary = dict()
    inv_dict = dict()
    index = 0
    for doc in docs:
        for word in doc:
            if word not in dictionary:
                dictionary[word] = index
                inv_dict[index] = word
                index += 1
    return dictionary, inv_dict

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# Create the LDA model using gensim library
# Running and Training LDA model on the document term matrix.

'''
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
lda = gensim.models.ldamodel.LdaModel
lda_model = lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=10000)
print(lda_model.print_topics(num_topics=3, num_words=3))
'''

dictionary, inv_dict = create_dictionary(doc_clean)
doc_term_matrix = [[dictionary.get(word) for word in doc] for doc in doc_clean]
sampler = LdaSampler(num_topics=3, id2word=inv_dict)
sampler.run(doc_term_matrix, 10000)
sampler.topk()
