#General Libs
import sys

#Data Science Libs
import pandas as pd

#Database Handling Libs
from sqlalchemy import create_engine

#WordCloud generation Libs
import wordcloud

#NGrams Names (labels dictionary)
ngrams_names = {
    1 : 'uni',
    2 : 'bi',
    3 : 'tri'
}

def save_wordcloud(ngrams_freqs, n,output_filepath):
    '''
    INPUT
    ngrams_freqs - pandas DataFrame, dataframe containing the counts of n-grams in the dataset
    n - the size of the ngram
    output_filepath - string, filepath where wordcloud image will be saved to
    
    
    This function generates a wordcloud image for the top n-grams given the input parameters 
    and saves the results to file.
    '''
    filtered_ngrams = ngrams_freqs[ngrams_freqs['n'] == n]
    ngrams_freqs_dict = dict(zip(filtered_ngrams['ngram'], filtered_ngrams['count']))
    n_wordcloud = wordcloud.WordCloud(background_color ='white', 
                min_font_size = 10,
                max_words= 30).generate_from_frequencies(ngrams_freqs_dict)  
    n_wordcloud.to_file(output_filepath)

def main():
    if len(sys.argv) >= 2:
        db_filepath, output_folderpath = sys.argv[1:]

        engine = create_engine('sqlite:///{}'.format(db_filepath))
        ngrams_freqs = pd.read_sql_table('NGramsFreqs', engine)

        for n in range(1,4):
            wordcloud_filepath = '{}/{}_wordcloud.png'.format(output_folderpath,ngrams_names[n])
            save_wordcloud(ngrams_freqs, n, wordcloud_filepath)

    else:
        print('Please provide the filepath of database file and the output folderpath of the wordcloud images.\n\n'\
                'Example: python generate-ngrams-wordclouds.py ../data/DisasterResponse.db static/imgs/')


if __name__ == '__main__':
    main()