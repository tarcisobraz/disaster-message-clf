#General libs
import sys
import os

#Data wrangling libs
import pandas as pd
import numpy as np

#DB related libs
from sqlalchemy import create_engine

#ML models related libs
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#Gensim
from gensim.models import KeyedVectors

#Custom Transformers and Estimators
import nlp_estimators

#Model Saver
import dill

glove_models_by_size = {50: None,
                        100: None,
                        300: None}

def get_or_load_glove_model(num_dims, glove_models_folderpath):
    if glove_models_by_size[num_dims] == None:
        print('Pre-trained Glove Model with {} dims not found. '\
                '\nLoading it from file...'.format(num_dims))
        glove_models_by_size[num_dims] = KeyedVectors.load_word2vec_format(
            os.path.join(glove_models_folderpath,
            'glove.6B.{}d_word2vec.txt'.format(num_dims)),
            binary=False)
    return glove_models_by_size[num_dims]

def load_glove_models(glove_models_folderpath):
    print('Loading Glove Models...')
    glove_models_by_size[50] = get_or_load_glove_model(50, glove_models_folderpath)
    glove_models_by_size[100] = get_or_load_glove_model(100, glove_models_folderpath)
    glove_models_by_size[300] = get_or_load_glove_model(300, glove_models_folderpath)
    print('\tDone Loading Glove Models!')

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    messages_df = pd.read_sql_table(con=engine, table_name='Message')
    categories_df = pd.read_sql_table(con=engine, table_name='CorpusWide')
    messages_tokens = pd.read_sql_table(con=engine, table_name='MessageTokens')
    
    X = messages_df.message.values
    X_tokenized = messages_tokens.tokens_str.values
    Y_df = categories_df.drop(['message_id', 'message', 'original', 'genre'], axis=1)
    Y = Y_df.values
    category_columns = Y_df.columns
    categories_tokens = np.array([np.array(cat.split('_')) for cat in category_columns])
    
    return X, X_tokenized, Y, category_columns, categories_tokens


def build_model(model_config,glove_models_folderpath,categories_tokens):
    feature_set = model_config['feature_set']
    pipeline = grid_search_params = grid_search_cv = None
    jobs = -1
    score = 'f1_micro'
    def_cv = 2
    verbosity_level=10
    
    if feature_set == 'local_w2v':
        pipeline = Pipeline([
                            ('local_w2v', nlp_estimators.TfidfEmbeddingTrainVectorizer()),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])
        
        grid_search_params = {'local_w2v__num_dims' : [50]}

    elif feature_set == 'glove':
        pipeline = Pipeline([
                            ('glove', nlp_estimators.TfidfEmbeddingTrainVectorizer(
                                get_or_load_glove_model(50,glove_models_folderpath))),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = {'glove__word2vec_model' : 
                            [get_or_load_glove_model(50,glove_models_folderpath)]}

    elif feature_set == 'doc2vec':
        pipeline = Pipeline([
                            ('doc2vec', nlp_estimators.Doc2VecTransformer()),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = {'doc2vec__vector_size' : [50]}

    elif feature_set == 'cats_sim':
        pipeline = Pipeline([
                            ('cats_sim', nlp_estimators.CategoriesSimilarity(
                                categories_tokens=categories_tokens)),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = {'cats_sim__word2vec_model' : 
                            [get_or_load_glove_model(50,glove_models_folderpath)]}

    elif feature_set == 'all_feats':
        pipeline = Pipeline([
                            ('features', FeatureUnion([
                                ('local_w2v', nlp_estimators.TfidfEmbeddingTrainVectorizer(num_dims=50)),
                                ('glove', nlp_estimators.TfidfEmbeddingTrainVectorizer(
                                    get_or_load_glove_model(50,glove_models_folderpath)
                                )),
                                ('doc2vec', nlp_estimators.Doc2VecTransformer(vector_size=50)),
                                ('cats_sim', nlp_estimators.CategoriesSimilarity(categories_tokens=categories_tokens,
                                word2vec_model=get_or_load_glove_model(50,glove_models_folderpath)))
                            ])),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])
        
        grid_search_params = {}
        
#         params_options_models_simple = [MultiOutputClassifier(RandomForestClassifier(random_state=199,
#                                                         n_estimators=50,
#                                                         max_depth=100,
#                                                         min_samples_split=5)),
#                                             MultiOutputClassifier(LogisticRegression(random_state=199,
#                                                                                     solver='liblinear',
#                                                                                     C=1,
#                                                                                     penalty='l2'))]
        
#         grid_search_params = {'local_w2v__num_dims' : [50,100,300],
#                               'clf' : params_options_models_simple}
        
    else:
        print("Error: Invalid Feature Set: " + feature_set)
        sys.exit(1)
        
    grid_search_cv = GridSearchCV(estimator=pipeline,
            param_grid=grid_search_params,
            scoring=score,
            cv=def_cv,
            n_jobs=jobs,
            verbose=verbosity_level)
    
    return grid_search_cv


def evaluate_model(model, X_test, Y_test, category_names):
    print('Best params: %s' % model.best_params_)
    # Best training data accuracy
    print('Best training score: %.3f' % model.best_score_)
    # Predict on test data with best params
    test_score = model.score(X_test, Y_test)
    # Test data accuracy of model with best params
    print('Test set score for best params: %.3f ' % test_score)


def save_model(model, model_filepath):
    # Output a pickle file for the model
    #joblib.dump(model, model_filepath) 
    with open(model_filepath,'wb') as f:
        dill.dump(model, f)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, glove_models_folderpath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, X_tokenized, Y, category_names, categories_tokens = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_tokenized, Y, test_size=0.25)
        
        print('Building model...')
        model = build_model({'feature_set':'all_feats'},
                            glove_models_folderpath,
                            categories_tokens)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the filepath of the pickle file to '\
              'save the model to as the second argument, and the folder where '\
              'pre-built Glove models are stored as the third argument. \n\n'\
              'Example: python train_classifier.py ../data/DisasterResponse.db '\
              'best-classifier.pkl ./glove-pretrained')


if __name__ == '__main__':
    main()
