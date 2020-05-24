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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(con=engine, table_name='Message')
    X = df.message.values
    Y_df = df.drop(['id','message','original','genre'], axis=1)
    Y = Y_df.values
    category_names = Y_df.columns
    
    return X, Y, category_names


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text.lower().strip())
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier(random_state=199)))
    ])
    
    parameters = {
        #'multi_clf__estimator__n_estimators': [20,50],
        #'multi_clf__estimator__max_depth': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)
    
    for category_idx in range(len(category_names)):
        print(classification_report(y_pred=Y_pred[:,category_idx],
                                    y_true=Y_test[:,category_idx], 
                                    labels=[0,1], 
                                    target_names=[category_names[category_idx] + '-0',
                                                  category_names[category_idx] + '-1']))


def save_model(model, model_filepath):
    # Output a pickle file for the model
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()