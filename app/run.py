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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
messages = pd.read_sql_table('Message', engine)
categories = pd.read_sql_table('Category', engine)
corpus_wide = pd.read_sql_table('CorpusWide', engine)
ngram_freqs = pd.read_sql_table('NGramsFreqs', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    top_unigrams = ngram_freqs[ngram_freqs['n'] == 1].head(20)
    top_bigrams = ngram_freqs[ngram_freqs['n'] == 2].head(20)
    top_trigrams = ngram_freqs[ngram_freqs['n'] == 3].head(20)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                plt_gos.Histogram(
                    x=messages.num_words
                )
            ],

            'layout': {
                'title': 'Distribution of Number of Words (Overall)',
                'yaxis': {
                    'title': "Count (log scale)",
                    'type': "log",
                    'showgrid': False,
                },
                'xaxis': {
                    'title': "Number of Words",
                    'showgrid': False,
                }
            },
        },
        {
            'data': [
                plt_gos.Bar(
                    x=categories.num_msgs,
                    y=categories.category,
                    orientation='h',
                ),
                plt_gos.Bar(
                    x=categories.num_words,
                    y=categories.category,
                    orientation='h',
                    xaxis="x2",
                    yaxis="y2"
                )
            ],

            'layout': {
                'showlegend': False,
                'height': 800,
                'title': 'Number of Messages/Words Per Category',
                'yaxis': {
                    'title': "Category"
                },
                'yaxis2': {
                    'anchor': "x2"
                },
                'xaxis': {
                    'title': "Number of Messages (log scale)",
                    'domain': [0.1, 0.5],
                    'showgrid': False,
                    'type': "log"
                },
                'xaxis2': {
                    'title': "Number of Words",
                    'domain': [0.6, 1],
                    'showgrid': False,
                }
            }
        },
        {
            'data': [
                plt_gos.Bar(
                    x=top_unigrams.ngram,
                    y=top_unigrams['count']
                )
            ],

            'layout': {
                'title': 'Top Unigrams',
                'yaxis': {
                    'title': "Num. Occurrences",
                    'showgrid': False,
                },
                'xaxis': {
                    'title': "Word (Unigram)",
                    'showgrid': False,
                }
            }
        },
        {
            'data': [
                plt_gos.Bar(
                    x=top_bigrams.ngram,
                    y=top_bigrams['count']
                )
            ],

            'layout': {
                'title': 'Top Bigrams',
                'yaxis': {
                    'title': "Num. Occurrences",
                    'showgrid': False,
                },
                'xaxis': {
                    'title': "Words (Bigram)",
                    'showgrid': False,
                }
            }
        },
        {
            'data': [
                plt_gos.Bar(
                    x=top_trigrams.ngram,
                    y=top_trigrams['count']
                )
            ],

            'layout': {
                'title': 'Top Trigrams',
                'yaxis': {
                    'title': "Num. Occurrences",
                    'showgrid': False,
                },
                'xaxis': {
                    'title': "Words (Trigram)",
                    'showgrid': False,
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
    classification_results = dict(zip(corpus_wide.columns[4:], classification_labels))

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
