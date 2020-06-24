#General libs
import sys
import os
import json
from datetime import datetime
import time

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
from sklearn.linear_model import LogisticRegression

#Gensim
from gensim.models import KeyedVectors

#Custom Transformers and Estimators
import nlp_estimators

#Model Saver
import dill

#Workspace Utils
from workspace_utils import active_session

#Glove Models dictionary (to be filled in when needed)
glove_models_by_size = {50: None,
                        100: None,
                        300: None}

#Train Configurations to be filled in when script is called
train_configs = {}

def get_or_load_glove_model(num_dims):
    '''
    INPUT
    num_dims - int, number of dimensions of the Glove model to be loaded
    

    OUTPUT
    glove_model - object, the pre-trained glove model with the specified number of dimensions
    
    This function either retrieves the already-stored glove model or loads and 
    stores it from file using the train configuration `glove_models_folderpath`
    '''
    if glove_models_by_size[num_dims] == None:
        print('Pre-trained Glove Model with {} dims not found. '\
                '\nLoading it from file...'.format(num_dims))
        glove_models_by_size[num_dims] = KeyedVectors.load_word2vec_format(
            os.path.join(train_configs['glove_models_folderpath'],
            'glove.6B.{}d_word2vec.txt'.format(num_dims)),
            binary=False)
    return glove_models_by_size[num_dims]

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - string, filepath of database from which data will be loaded
    

    OUTPUT
    X - numpy array, The raw messages ready to be used to train the pipelines
    X_tokenized - numpy array, The tokenized messages ready to be used to train the pipelines
    Y - numpy array, The list of categories to which each message belongs
    category_columns - pandas series, The names of the categories
    categories_tokens - numpy array, The tokenized categories names (to be used by cats_sim feature set)
    
    
    This function loads and prepares data for the models training
    '''
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

def build_estimator_obj(estimator_code):
    '''
    INPUT
    estimator_code - string, the code of the classifier object to be built
    

    OUTPUT
    classifier_obj - sklearn estimator, the built classifier object
    
    
    This function builds a classifier object based on the estimator code received as input.
    For unexpected codes, it prints an error and exits the script execution
    '''
    classifier_obj = None
    if estimator_code == 'rf':
        classifier_obj = RandomForestClassifier()
    elif estimator_code == 'lr':
        classifier_obj = LogisticRegression()
    else:
        print("Invalid Classifier Estimator Code " +  estimator_code)
        exit(1)

    return classifier_obj

def build_classifiers_build_params(classifiers_configs):
    '''
    INPUT
    classifiers_configs - dict, a dictionary containing the configuration for each classifier
    

    OUTPUT
    classifiers_params_dict - dict, a dictionary containing the grid params to be used for
    each classifier in the training process
    
    
    This function builds a dictionary with grid params to be used in training process for each
    classifier whose configurations were given as input.
    It can handle a single classifier or a list of classifiers.
    '''
    if len(classifiers_configs) > 1:
        classifiers_params_list = []
        classifiers_params_dict = {}
        for classifier in classifiers_configs:
            classifier_estimator = classifier['estimator']
            classifier_obj = build_estimator_obj(classifier_estimator)
            classifier_obj = MultiOutputClassifier(classifier_obj.set_params(**classifier['params']))
            classifiers_params_list.append(classifier_obj)
        
        classifiers_params_dict['clf'] = classifiers_params_list
    elif len(classifiers_configs) == 1:
        classifier = classifiers_configs[0]
        classifier_estimator = classifier['estimator']
        classifier_obj = build_estimator_obj(classifier_estimator)
        classifier_obj = MultiOutputClassifier(classifier_obj)
        classifiers_params_dict = {'clf' : [classifier_obj]}
        classifiers_params_dict.update(classifier['params'])

    print(classifiers_params_dict)
    return classifiers_params_dict
        


def build_model(model_config,classifiers_params,categories_tokens):
    '''
    INPUT
    model_config - dict, a dictionary containing the configuration for a model pipeline
    classifiers_configs - dict, a dictionary containing the configuration for each classifier
    categories_tokens - numpy array, array containing the tokenized categories names
    

    OUTPUT
    grid_search_cv - sklearn GridSearchCV, a grid search CV object containing specifications
    on how to train the model based on the input configs
    
    
    This function builds a Grid Search CV object with specifications for training process for a
    given model and its classifiers whose configurations were given as input.
    It can handle different feature_sets:
      - Local Word2Vec
      - Pre-Trained Glove
      - Doc2Vec
      - Category Similarity
      - All Features Sets together
    '''
    feature_set = model_config['feature_set']
    print("Building Model for feature set: {}".format(feature_set))
    print("Grid Params: {}".format(model_config['grid_params']))
    pipeline = grid_search_params = grid_search_cv = None
    jobs = -1
    score = 'f1_micro'
    def_cv = 3
    verbosity_level=10
    
    if feature_set == 'local_w2v':
        pipeline = Pipeline([
                            ('local_w2v', nlp_estimators.TfidfEmbeddingTrainVectorizer()),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = model_config['grid_params']

    elif feature_set == 'glove':
        pipeline = Pipeline([
                            ('glove', nlp_estimators.TfidfEmbeddingTrainVectorizer(
                                get_or_load_glove_model(50))),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = {'glove__word2vec_model' : 
            [get_or_load_glove_model(num_dims) for num_dims in 
                model_config['grid_params']['glove__num_dims']]}

    elif feature_set == 'doc2vec':
        pipeline = Pipeline([
                            ('doc2vec', nlp_estimators.Doc2VecTransformer()),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = model_config['grid_params']

    elif feature_set == 'cats_sim':
        pipeline = Pipeline([
                            ('cats_sim', nlp_estimators.CategoriesSimilarity(
                                categories_tokens=categories_tokens)),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])

        grid_search_params = {'cats_sim__word2vec_model' : 
            [get_or_load_glove_model(num_dims) for num_dims in 
                model_config['grid_params']['cats_sim__num_dims']]}

    elif feature_set == 'all_feats':
        pipeline = Pipeline([
                            ('features', FeatureUnion([
                                ('local_w2v', nlp_estimators.TfidfEmbeddingTrainVectorizer(num_dims=50)),
                                ('glove', nlp_estimators.TfidfEmbeddingTrainVectorizer(
                                    get_or_load_glove_model(50)
                                )),
                                ('doc2vec', nlp_estimators.Doc2VecTransformer(vector_size=50)),
                                ('cats_sim', nlp_estimators.CategoriesSimilarity(categories_tokens=categories_tokens,
                                word2vec_model=get_or_load_glove_model(50)))
                            ])),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])
        
        grid_search_params = model_config['grid_params']
               
    else:
        print("Error: Invalid Feature Set: " + feature_set)
        sys.exit(1)

    # Adds classifiers params to grid params
    grid_search_params.update(classifiers_params)
        
    grid_search_cv = GridSearchCV(estimator=pipeline,
            param_grid=grid_search_params,
            scoring=score,
            cv=def_cv,
            n_jobs=jobs,
            verbose=verbosity_level)
    
    return grid_search_cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - sklearn GridSearchCV, the GridSearch containing the model with best performance on the training set
    X_test - numpy array, tokenized messages ready to be used to test the fit pipelines
    Y_test - numpy array, array containing the tokenized categories names for the test set
    category_names - pandas series, the categories names
    

    OUTPUT
    test_score - float, the score of the input model on the test data
    
    
    This function runs the model with best performance on the training set on the test dataset,
    printing the precision, recall and f-1 per category and returning the overall prediction score.
    '''
    print('Best params: %s' % model.best_params_)
    # Best training data accuracy
    print('Best training score: %.3f' % model.best_score_)
    # Predict on test data with best params
    Y_pred = model.predict(X_test)
    test_score = model.score(X_test, Y_test)
    # Test data accuracy of model with best params
    print('Test set score for best params: %.3f ' % test_score)

    for category_idx in range(len(category_names)):
        print(classification_report(y_pred=Y_pred[:,category_idx],
                                    y_true=Y_test[:,category_idx], 
                                    labels=[0,1], 
                                    target_names=[category_names[category_idx] + '-0',
                                                  category_names[category_idx] + '-1']))

    return test_score


def save_model(model, model_filepath):
    '''
    INPUT
    model - sklearn Estimator, the model with best performance on the training set
    model_filepath - string, path where model picke will be saved
    

    This function saves the model with best performance on the training set to a given filepath.
    '''
    # Output a pickle file for the model
    with open(model_filepath,'wb') as f:
        dill.dump(model, f)

def build_grid_search_results_df(gs_results, gs_name, test_score):
    '''
    INPUT
    gs_results - dict, dictionary containing the results of GridSearchCV training
    gs_name - string, the name of the GridSearchCV feature set
    test_score - float, the score of the best performing model of the GridSearchCV on the test set
    

    OUTPUT
    gs_results_df - pandas DataFrame, a dataframe holding information of the GridSearchCV results 
    (train and test) for record
    
    
    This function builds a dataframe with information of the GridSearchCV results 
    (train and test) for record.
    '''
    gs_results_df = pd.DataFrame(gs_results)
    gs_results_df['grid_id'] = gs_name
    gs_results_df['best_model_test_score'] = test_score
    gs_results_df['param_set_order'] = np.arange(len(gs_results_df))

    return gs_results_df

def run_grid_search():
    '''
    This function runs the whole model selection phase:
        - Load Data from DB
        - Build Model
        - Run GridSearch
        - Save results to file
        - Save best model pickle file
    '''
    start = time.time()
    print("Train configuration:")
    print(json.dumps(train_configs, indent=4))
    print('Loading data...\n    DATABASE: {}'.format(train_configs['database_filepath']))
    X, X_tokenized, Y, category_names, categories_tokens = load_data(train_configs['database_filepath'])
    X_train, X_test, Y_train, Y_test = train_test_split(X_tokenized, Y, test_size=0.25)
    classifiers_params = build_classifiers_build_params(train_configs['classifiers'])

    print('Running GridSearch on models parameters...')
    best_score = 0.0
    best_gs = ''
    overall_results_df = pd.DataFrame()

    for model_config in train_configs['models']:
        print('Building model...')
        model = build_model(model_config,
                            classifiers_params,
                            categories_tokens)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        test_score = evaluate_model(model, X_test, Y_test, category_names)

        gs_results_df = build_grid_search_results_df(model.cv_results_, 
                                    model_config['feature_set'], 
                                    test_score)

        overall_results_df = pd.concat([overall_results_df, gs_results_df])

        print('Saving model...\n    MODEL: {}'.format(
            model_config['model_ouput_filepath']))
        save_model(model.best_estimator_, model_config['model_ouput_filepath'])

        print('Trained model saved!')

        # Track best (highest test accuracy) model
        if test_score > best_score:
            best_score = test_score
            best_gs = model_config['feature_set']

    output_filepath =  train_configs['results_folderpath'] + \
                        'res-' + train_configs['name'] + '-' + \
                        datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + \
                        '.csv'

    print('Saving Results...\n    FILEPATH: {}'.format(output_filepath))
    overall_results_df.to_csv(output_filepath, index=False)

    print('\nClassifier with best test set accuracy: %s' % best_gs)

    end = time.time()
    print("Training Time: " + str(int(end - start)) + "s")

def main():
    if len(sys.argv) >= 3:
        train_config_filepath, using_udacity_workspace = sys.argv[1:]

        # Read train config from file
        with open(train_config_filepath, 'r') as f:
            global train_configs
            train_configs = json.load(f)

        if using_udacity_workspace == 1:
            with active_session():
                run_grid_search()
        else:
            run_grid_search()

    else:
        print('Please provide the filepath of train configuration file and '\
            ' whether or not you are using udacity workspace (0,1) \n\n'\
            'Example running local: python train_classifier.py configs/train_config_simple.json 0'\
            '\nExample running at Udacity: python train_classifier.py configs/train_config_simple.json 1')


if __name__ == '__main__':
    main()
