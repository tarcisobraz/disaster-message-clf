# import libraries

#Basic DS libs
import numpy as np
import pandas as pd

#Helper Libs
import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

#NLTK
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#Gensim
from gensim.test.utils import datapath
from gensim import utils
import gensim.models

#Vectorizers/Transformers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

#Glove
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#Doc2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
import multiprocessing

# Tokenizer Functions
def tokenize_to_list(text, lemmatizer = WordNetLemmatizer()):
    '''
    INPUT
    text - text string to be tokenized
    lemmatizer - lemmatizer object to be used to process text tokens (defaults to WordNetLemmatizer)
    
    OUTPUT
    A list of tokens extracted from the input text
    
    This function receives raw text as input a pre-processes it for NLP analysis, removing punctuation and
    special characters, normalizing case and removing extra spaces, as well as removing stop words and 
    applying lemmatization
    '''
    tokens = nltk.tokenize.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip()))
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopwords.words("english")]

    return clean_tokens

def tokenize_to_str(text, lemmatizer = WordNetLemmatizer()):
    '''
    INPUT
    text - text string to be tokenized
    lemmatizer - lemmatizer object to be used to process text tokens (defaults to WordNetLemmatizer)
    
    OUTPUT
    A string with the tokens extracted from the input text concatenated by spaces
    
    This function receives raw text as input a pre-processes it for NLP analysis, removing punctuation and
    special characters, normalizing case and removing extra spaces, as well as removing stop words and 
    applying lemmatization
    '''
    tokens = nltk.tokenize.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip()))
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopwords.words("english")]
    #Return tokens list as a string joined by whitespaces
    clean_tokens_str = ' '.join(clean_tokens)

    return clean_tokens_str

# Feature Generators/Aggregators

class MeanEmbeddingTrainVectorizer(BaseEstimator):

    def __init__(self, word2vec_model=None, num_dims=100):
        if word2vec_model is None:
            self.word2vec_model = None
            self.num_dims = num_dims
            self.workers = multiprocessing.cpu_count() - 1
            
        else:
            self.word2vec_model = word2vec_model
            self.num_dims = word2vec_model.vector_size
            
        print(self.num_dims)
            
    def fit(self, X, y):
        if self.word2vec_model is None:
            self.word2vec_model = gensim.models.Word2Vec(X, size=self.num_dims, 
                                                         workers=self.workers)
        
        return self 

    def transform(self, X):
        mean_embeddings = np.empty([X.shape[0],self.num_dims])
        
        for i in range(X.shape[0]):
            doc_tokens = X[i]
            
            words_vectors_concat = [self.word2vec_model[w] for w in doc_tokens if w in self.word2vec_model]

            if (len(words_vectors_concat) == 0):
                words_vectors_concat = [np.zeros(self.num_dims)]
                
            #print(np.mean(words_vectors_concat, axis=0))
                
            mean_embeddings[i] = np.mean(words_vectors_concat, axis=0)
            
        return mean_embeddings
    
class TfidfEmbeddingTrainVectorizer(BaseEstimator):
    
    def __init__(self, word2vec_model=None, num_dims=100):
        self.word2vec_model = word2vec_model
        self.num_dims = num_dims
        
    def fit(self, X, y):
        if self.word2vec_model is None:
            self.workers_ = multiprocessing.cpu_count() - 1
            self.word2vec_model = gensim.models.Word2Vec(X, size=self.num_dims, 
                                                         workers=self.workers_)
        self.num_dims = self.word2vec_model.vector_size
            
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        
        tfidf_weights = [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        self.word_weights_ = defaultdict(lambda: max_idf, tfidf_weights)
    
        return self
    
    def transform(self, X):
        mean_embeddings = np.empty([X.shape[0],self.num_dims])
        
        for i in range(X.shape[0]):
            doc_tokens = X[i]
            
            words_vectors_concat = [self.word2vec_model[w]*self.word_weights_[w] for w in doc_tokens if w in self.word2vec_model]

            if (len(words_vectors_concat) == 0):
                words_vectors_concat = [np.zeros(self.num_dims)]
                
            #print(np.mean(words_vectors_concat, axis=0))
                
            mean_embeddings[i] = np.mean(words_vectors_concat, axis=0)
            
        return mean_embeddings


class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, epochs=20):
        self.epochs = epochs
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1
        self.d2v_model = None

    def fit(self, X, y):
        tagged_x = [TaggedDocument(tokens_str.split(), [index]) for index, tokens_str in np.ndenumerate(X)]
        self.d2v_model = Doc2Vec(vector_size=self.vector_size, workers=self.workers, epochs=self.epochs)
        self.d2v_model.build_vocab(tagged_x)
        self.d2v_model.train(tagged_x, total_examples=self.d2v_model.corpus_count, 
                             epochs=self.epochs)

        return self

    def transform(self, X):
        return np.asmatrix(np.array([self.d2v_model.infer_vector(tokens_str.split())
                                     for tokens_str in X]))
    
class CategoriesSimilarity(BaseEstimator):
    
    def __init__(self, categories_tokens, word2vec_model=None, num_dims=100):
        self.categories_tokens = categories_tokens
        self.word2vec_model = word2vec_model    
        self.num_dims = num_dims
        
    def compute_mean_embeddings(self, tokens_array):    
        mean_embeddings = np.empty([tokens_array.shape[0],self.num_dims])
        
        for i in range(tokens_array.shape[0]):
            doc_tokens = tokens_array[i]
            
            words_vectors_concat = [self.word2vec_model[w] for w in doc_tokens if w in self.word2vec_model]

            if (len(words_vectors_concat) == 0):
                words_vectors_concat = [np.zeros(self.num_dims)]
                
            #print(np.mean(words_vectors_concat, axis=0))
                
            mean_embeddings[i] = np.mean(words_vectors_concat, axis=0)
            
        return mean_embeddings
                    
    def fit(self, X, y):
        if self.word2vec_model is None:
            self.workers_ = multiprocessing.cpu_count() - 1
            self.word2vec_model = gensim.models.Word2Vec(X, size=self.num_dims, 
                                                         workers=self.workers_)
        self.num_dims = self.word2vec_model.vector_size        
        self.categories_vectors_ = self.compute_mean_embeddings(self.categories_tokens)
        return self 

    def transform(self, X):
        mean_embeddings = self.compute_mean_embeddings(X)
        cats_similarities = cosine_similarity(mean_embeddings, self.categories_vectors_)
            
        return cats_similarities