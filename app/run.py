#General libs
import sys
import json

#Visualization libs
import plotly
import plotly.graph_objs as plt_gos

#Data Wrangling libs
import pandas as pd
import numpy as np

#NLP libs
from nltk import WordNetLemmatizer

#Server libs
from flask import Flask
from flask import render_template, request, jsonify, url_for

#Data Persistence libs
import dill
from sqlalchemy import create_engine

#Import tokenization function from another directory
sys.path.insert(1, '../models/')
from nlp_estimators import tokenize_to_str

#Initialize Flask App
app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
messages = pd.read_sql_table('Message', engine)
categories = pd.read_sql_table('Category', engine)
corpus_wide = pd.read_sql_table('CorpusWide', engine)
ngram_freqs = pd.read_sql_table('NGramsFreqs', engine)

# load model
with open('../models/best-models/best-classifier.pkl', 'rb') as f:
    model = dill.load(f)


# index webpage displays project overview info and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page with plotly graphs
    # This will render the master.html Please see that file. 
    return render_template('master.html')
    

# web page that handles user query and displays model results
@app.route('/go', endpoint='go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    tokenized_query = tokenize_to_str(query, lemmatizer=WordNetLemmatizer())
    classification_labels = model.predict(np.array([tokenized_query]))[0]
    classification_results = dict(zip(corpus_wide.columns[4:], classification_labels))

    # print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# web page that displays Exploratory Data Analysis on messages/categories data
@app.route('/words_msgs_dist', endpoint='words_msgs_dist')
def words_msgs_dist():
    # create visuals using Plotly low level API
    # graphs below show:
    #  1 - The Overall Distribution of Number of Words per Message
    #  2 - The Number of Messages/Words Per Category
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
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    # This will render the words_msgs_dist.html Please see that file.
    return render_template('words_msgs_dist.html', ids=ids, graphJSON=graphJSON)

# web page that displays Exploratory Data Analysis on N-Grams data extracted from messages
@app.route('/ngrams_dist', endpoint='ngrams_dist')
def ngrams_dist():
    # extract top n-grams data needed for visuals
    top_unigrams = ngram_freqs[ngram_freqs['n'] == 1].head(20)
    top_bigrams = ngram_freqs[ngram_freqs['n'] == 2].head(20)
    top_trigrams = ngram_freqs[ngram_freqs['n'] == 3].head(20)
    
    # create visuals using Plotly low level API
    # graphs below show:
    #  1 - The Top-20 Unigrams in a bar chart
    #  2 - The Top-20 Bigrams in a bar chart
    #  3 - The Top-20 Trigrams in a bar chart

    graphs = [
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
                    'title': "Unigrams",
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
                    'title': "Bigrams",
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
                    'title': "Trigrams",
                    'showgrid': False,
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    # This will render the ngrams_dist.html Please see that file.
    return render_template('ngrams_dist.html', ids=ids, graphJSON=graphJSON)
    
# web page that displays WordClouds of Top N-Grams extracted from messages data
@app.route('/ngrams_wordcloud', endpoint='ngrams_wordcloud')
def ngrams_wordcloud():
    static_imgs_folder = '/static/imgs/'
    
    imgs = {static_imgs_folder + 'uni_wordcloud.png':'Top Unigrams WordCloud',
            static_imgs_folder + 'bi_wordcloud.png' :'Top Bigrams WordCloud', 
            static_imgs_folder + 'tri_wordcloud.png':'Top Trigrams WordCloud'}
    
    # This will render the ngrams_wordcloud.html Please see that file. 
    return render_template('ngrams_wordcloud.html', imgs=imgs)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
