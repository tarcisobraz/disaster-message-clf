# import libraries

#Basic DS libs
import numpy as np
import pandas as pd

#Helper Libs
import os
import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

#NLTK
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

for corpora in ['punkt','wordnet','stopwords']:
    _ = nltk.download(corpora, quiet=True)

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

def get_ngrams_freqs(messages_array, n=1):
    '''
    INPUT
    messages_array - numpy array, string messages from where n-grams will be extracted
    n - int, n-gram size, defaults to 1
    
    OUTPUT
    words_freq_df, pandas DataFrame, a dataframe with all the n-grams found within the data
    along with their respective counts.
    
    This function receives a numpy array with messages from which all n-grams of size n will be 
    recognized and counted, returning a dataframe with the distribution of such n-grams.
    '''
    vec = CountVectorizer(ngram_range=(n, n)).fit(messages_array)
    bag_of_words = vec.transform(messages_array)
    word_count = bag_of_words.sum(axis=0)
    words_freq = [(word, n, word_count[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[2], reverse=True)
    words_freq_df = pd.DataFrame(data = words_freq, columns = ['ngram','n','count'])
    return words_freq_df

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
    """ 
    A feature vector aggregator which uses the mean as the aggregation function.

    Parameters
    ----------
    word2vec_model : gensim.models.Word2Vec, default=None
        A pre-built word2vec model which will be used to generate the feature vectors
    num_dims : int, default=100
        The number of dimensions of the feature vector to be generated

    Attributes
    ----------
    workers : int
        The number of workers to be used if training the model from local data
        when calling :meth:`fit`.
    """

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
        """
        If a pre-built model is not passed in the initialization, this method
        fits a word2vec model on the input data given the number of dimensions passed.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.word2vec_model is None:
            self.word2vec_model = gensim.models.Word2Vec(X, size=self.num_dims, 
                                                         workers=self.workers)
        
        return self 

    def transform(self, X):
        """ 
        Extracts the feature vector from the data using the word2vec model
        and aggregates the vector for each element in ``X`` using the mean.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        mean_embeddings : array, shape (n_samples, n_features)
            The array containing the element-wise mean of the feature vectors obtained from ``X``.
        """
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
    """ 
    A feature vector aggregator which uses the TF-IDF as the aggregation function.

    Parameters
    ----------
    word2vec_model : gensim.models.Word2Vec, default=None
        A pre-built word2vec model which will be used to generate the feature vectors
    num_dims : int, default=100
        The number of dimensions of the feature vector to be generated

    Attributes
    ----------
    workers : int
        The number of workers to be used if training the model from local data
        when calling :meth:`fit`.
    word_weights : dict
        The TF-IDF weights dictionary for every word of the corpus
    """
    def __init__(self, word2vec_model=None, num_dims=100):
        self.word2vec_model = word2vec_model
        self.num_dims = num_dims
        
    def fit(self, X, y):
        """
        If a pre-built model is not passed in the initialization, this method
        fits a word2vec model on the input data given the number of dimensions passed,
        and computes the word_weights dict using TF-IDF. If the word cannot be found
        within the dict, it returns the max_idf value.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
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
        """ 
        Extracts the feature vector from the data using the word2vec model
        and aggregates the vector for each element in ``X`` using the TF-IDF
        word weights.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        mean_embeddings : array, shape (n_samples, n_features)
            The array containing the element-wise TF-IDF of the feature vectors obtained from ``X``.
        """
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
    """ 
    A Doc2Vec feature generator.
    For more info on Doc2Vec, see: https://radimrehurek.com/gensim/models/doc2vec.html

    Parameters
    ----------
    vector_size : int, default=100
        The number of dimensions of the feature vector to be generated
    epochs : int, default=20
        The number of epochs to be used when training the Doc2Vec model

    Attributes
    ----------
    workers : int
        The number of workers to be used if training the model from local data
        when calling :meth:`fit`.
    d2v_model : gensim.models.doc2vec.Doc2Vec, default=None
        The doc2vec model to be fit and used to generate the feature vectors
    """
    def __init__(self, vector_size=100, epochs=20):
        self.epochs = epochs
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1
        self.d2v_model = None

    def fit(self, X, y):
        """
        Fits a doc2vec model on the input data given the number of dimensions 
        and epochs passed.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        tagged_x = [TaggedDocument(tokens_str.split(), [index]) for index, tokens_str in np.ndenumerate(X)]
        self.d2v_model = Doc2Vec(vector_size=self.vector_size, workers=self.workers, epochs=self.epochs)
        self.d2v_model.build_vocab(tagged_x)
        self.d2v_model.train(tagged_x, total_examples=self.d2v_model.corpus_count, 
                             epochs=self.epochs)

        return self

    def transform(self, X):
        """ 
        Extracts the feature vector from the data using the doc2vec model 
        for each element in ``X``.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        _ : array, shape (n_samples, n_features)
            The array containing the element-wise doc2vec feature vectors obtained from ``X``.
        """
        return np.asmatrix(np.array([self.d2v_model.infer_vector(tokens_str.split())
                                     for tokens_str in X]))
    
class CategoriesSimilarity(BaseEstimator):
    """ 
    A Category Similarity feature generator.
    This is a custom feature thought for this specific project, 
    whose idea is to take advantage of the supervised characteristic of the problem, 
    by comparing the messages feature vectors to the categories names feature vectors, 
    computing the cosine distance between them. I suspect that messages whose words are
    close to their categories words should have a short distance to them.
    The format of this feature is a vector of size num_categories with the 
    cosine distance between the message and each category.

    Parameters
    ----------
    categories_tokens : {array-like, sparse matrix}, shape (n_categories, 1)
        The category names tokens.
    word2vec_model : gensim.models.Word2Vec, default=None
        A pre-built word2vec model which will be used to generate the feature vectors
    num_dims : int, default=100
        The number of dimensions of the feature vector to be generated

    Attributes
    ----------
    workers : int
        The number of workers to be used if training the model from local data
        when calling :meth:`fit`.
    categories_vectors : {array-like, sparse matrix}, shape (n_categories, n_features)
        The array containing the mean feature vectors generated for the category tokens
        using the word2vec model
    """
    def __init__(self, categories_tokens, word2vec_model=None, num_dims=100):
        self.categories_tokens = categories_tokens
        self.word2vec_model = word2vec_model    
        self.num_dims = num_dims
        
    def compute_mean_embeddings(self, tokens_array):    
        """ 
        Extracts the feature vector from the data using the word2vec model
        and aggregates the vector for each element in ``X`` using the mean.

        Parameters
        ----------
        tokens_array : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        mean_embeddings : array, shape (n_samples, n_features)
            The array containing the element-wise mean of the feature vectors 
            obtained from ``tokens_array``.
        """
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
        """
        Fits a word2vec model on the input data given the number of dimensions passed.
        Then, generates the mean feature vectors for the category tokens array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.word2vec_model is None:
            self.workers_ = multiprocessing.cpu_count() - 1
            self.word2vec_model = gensim.models.Word2Vec(X, size=self.num_dims, 
                                                         workers=self.workers_)
        self.num_dims = self.word2vec_model.vector_size        
        self.categories_vectors_ = self.compute_mean_embeddings(self.categories_tokens)
        return self 

    def transform(self, X):
        """ 
        Extracts the feature vector from the data using the word2vec model 
        for each element in ``X`` and then computes the distance between each
        element's mean feature vector and the mean feature vector of each category.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        cats_similarities : array, shape (n_samples, n_features)
            The array containing the element-wise distance array between mean feature vectors
            obtained from ``X`` and mean feature vectors obtained from category tokens.
        """
        mean_embeddings = self.compute_mean_embeddings(X)
        cats_similarities = cosine_similarity(mean_embeddings, self.categories_vectors_)
            
        return cats_similarities