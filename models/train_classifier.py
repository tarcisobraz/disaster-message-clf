#General libs
import sys

#Data wrangling libs
import pandas as pd

#NLP libs
import nltk
from nltk import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

#DB related libs
from sqlalchemy import create_engine

#ML models related libs
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

#Custom Transformers and Estimators
from nlp_estimators import *

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
    
    return X, X_tokenized, Y, category_columns


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text.lower().strip())
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return clean_tokens


def build_model(model_config):
    feature_set = model_config['feature_set']
    pipeline = grid_search_params = grid_search_cv = None
    jobs = -1
    score = 'f1_micro'
    def_cv = 3
    verbosity_level=10
    
    if feature_set == 'local_w2v':
        pipeline = Pipeline([
                            ('local_w2v', TfidfEmbeddingTrainVectorizer()),
                            ('clf', MultiOutputClassifier(GaussianNB()))
                        ])
        
        grid_search_params = {'local_w2v__num_dims' : [50]}
        
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
    
#     for category_idx in range(len(category_names)):
#         print(classification_report(y_pred=Y_pred[:,category_idx],
#                                     y_true=Y_test[:,category_idx], 
#                                     labels=[0,1], 
#                                     target_names=[category_names[category_idx] + '-0',
#                                                   category_names[category_idx] + '-1']))


def save_model(model, model_filepath):
    # Output a pickle file for the model
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, X_tokenized, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_tokenized, Y, test_size=0.25)
        
        print('Building model...')
        model = build_model({'feature_set':'local_w2v'})
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
