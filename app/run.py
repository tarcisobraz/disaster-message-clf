import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as plt_gos
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def count_words(messages):
    num_words_per_message = messages.apply(lambda x: len(x.split(' ')))
    return num_words_per_message.median()

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
messages = pd.read_sql_table('Message', engine)
messages_tokens = pd.read_sql_table('MessageTokens', engine)
messages_cats_wide = pd.read_sql_table('MessageCategoryWide', engine)
messages_cats_long = pd.read_sql_table('MessageCategoryLong', engine)

messages_wide = messages.merge(messages_cats_wide, on='message_id')
messages_long = messages.merge(messages_cats_long, on='message_id')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    num_words_dist = messages.message.apply(lambda x: len(x.split())).to_frame(name='num_words')

    word_counts_by_category = messages_long.groupby('category').message \
        .agg(lambda x: count_words(x)).reset_index(name='num_words')
    word_counts_by_category

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                plt_gos.Histogram(
                    x=num_words_dist.num_words
                )
            ],

            'layout': {
                'title': 'Distribution of Number of Words (Overall)',
                'yaxis': {
                    'title': "Count",
                    'type': "log"
                },
                'xaxis': {
                    'title': "Number of Words"
                }
            }
        },
        {
            'data': [
                plt_gos.Bar(
                    x=word_counts_by_category.num_words,
                    y=word_counts_by_category.category,
                    orientation='h',
                )
            ],

            'layout': {
                'title': 'Distribution of Number of Words (Per Category)',
                'height': 800, 
                'width': 600,
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Number of Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(messages_wide.columns[4:], classification_labels))

    # print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
