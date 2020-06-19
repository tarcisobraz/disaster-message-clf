#General libs
import sys
import pandas as pd

#DB related libs
from sqlalchemy import create_engine

#NLP libs
import nltk
from nltk import WordNetLemmatizer

#Import tokenization and ngrams extraction functions from another directory
sys.path.insert(1, '../models/')
from nlp_estimators import tokenize_to_str, get_ngrams_freqs

def count_words(messages):
    '''
    INPUT
    messages - pandas series, disaster messages
    

    OUTPUT
    median_num_words_per_message - int, median number of words per message in input messages set
    
    This function computes the median number of words per message in input messages set.
    '''
    num_words_per_message = messages.apply(lambda x: len(x.split(' ')))
    return num_words_per_message.median()

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - string, file path where messages data is stored
    categories_filepath - string, file path where categories data is stored
    

    OUTPUT
    messages_raw - pandas DataFrame, dataframe containing raw messages data
    categories_raw - pandas DataFrame, dataframe containing raw categories data
    
    This function reads and returns raw messages and categories data.
    '''
    # load raw messages and categories data
    messages_raw = pd.read_csv(messages_filepath)
    categories_raw = pd.read_csv(categories_filepath)
    
    return (messages_raw, categories_raw)


def clean_data(messages_raw, categories_raw):
    '''
    INPUT
    messages_raw - pandas DataFrame, dataframe containing raw messages data
    categories_raw - pandas DataFrame, dataframe containing raw categories data
    

    OUTPUT
    messages - pandas DataFrame, dataframe containing cleansed (non-duplicated) messages data
    categories - pandas DataFrame, dataframe containing cleansed (non-duplicated) categories data
    categories_wide - pandas DataFrame, dataframe containing categories per message in wide format 
        (one category per column)
    categories_long - pandas DataFrame, dataframe containing categories per message in long format 
        (one category per line)
    messages_wide - pandas DataFrame, dataframe containing messages data with their respective 
        categories in wide format (one category per column)
    messages_long - pandas DataFrame, dataframe containing messages data with their respective 
        categories in long format (one category per line)

    
    This function reads raw data and cleans/processes/organizes it, returning different views of the
    same data to faccilitate analysis.
    '''
    # rename id column and drop message duplicates
    messages = messages_raw.rename(index=str, columns={'id':'message_id'})
    messages = messages.drop_duplicates()

    # rename id column and drop categories duplicates
    categories = categories_raw.rename(index=str, columns={'id':'message_id'})
    categories = categories.drop_duplicates()

    # check for messages which have more than one unique set of categories
    duplicated_msg_cats = categories.groupby('message_id').agg(lambda x: len(x)).reset_index().query('categories > 1')
    
    # remove messages with duplicated category assignments from messages data
    messages = messages[~messages['message_id'].isin(duplicated_msg_cats['message_id'])]

    #Remove messages with duplicated category assignments from categories data
    categories = categories[~categories['message_id'].isin(duplicated_msg_cats['message_id'])]

    # create a dataframe of the 36 individual category columns
    categories_wide = categories.categories.str.split(';', expand=True)
    # extract column names from the values
    category_colnames = categories_wide.head(1).apply(
        lambda x: x.str.slice(stop=-2)).values[0].tolist()
    categories_wide.columns = category_colnames
    
    # simplify category encoding by using booleans (0/1) instead of strings
    for column in category_colnames:
        # set each value to be the last character of the string
        categories_wide[column] = categories_wide[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories_wide[column] = categories_wide[column].astype(int)

    # remove child_alone category, as there is no example of message from this category
    categories_wide = categories_wide.drop('child_alone', axis=1)

    # fix problem with related category, replacing every value 2 to 1
    categories_wide['related'] = categories_wide['related'].apply(lambda x: x if x < 2 else 1)

    # create `categories_wide` and `categories_long` DataFrame for better data processing
    categories_wide = pd.concat([categories.message_id,categories_wide], axis=1)
    categories_long = pd.melt(categories_wide, id_vars=['message_id'], var_name='category')
    categories_long = categories_long[categories_long['value'] == 1].drop('value', axis=1)
        
    # merge datasets, creating messages_wide and messages_long dataframes
    messages_wide = pd.merge(messages, categories_wide, on='message_id')
    messages_long = pd.merge(messages, categories_long, on='message_id')
    
    return (messages, categories, categories_wide, categories_long, messages_wide, messages_long)

def prep_data_for_analysis(messages, categories, categories_wide, categories_long, 
                            messages_wide, messages_long):
    '''
    INPUT
    messages - pandas DataFrame, dataframe containing cleansed (non-duplicated) messages data
    categories - pandas DataFrame, dataframe containing cleansed (non-duplicated) categories data
    categories_wide - pandas DataFrame, dataframe containing categories per message in wide format 
        (one category per column)
    categories_long - pandas DataFrame, dataframe containing categories per message in long format 
        (one category per line)
    messages_wide - pandas DataFrame, dataframe containing messages data with their respective 
        categories in wide format (one category per column)
    messages_long - pandas DataFrame, dataframe containing messages data with their respective 
        categories in long format (one category per line)
    

    OUTPUT
    datasets_tables - dict, dictionary mapping to-be DB table names to their respective data-holding
        pandas dataframe. Datasets comprised are:
            - messages_final - pandas DataFrame, dataframe containing the final set of messages 
                for processing (with computed properties)
            - categories_final - pandas DataFrame, dataframe containing the final set of categories 
                for processing (with computed properties)
            - genres_final - pandas DataFrame, dataframe containing the number of messages per genre
            - corpus_wide - pandas DataFrame, dataframe containing the final set of messages 
                for processing in wide format (messages in lines and categories in columns)
            - message_categories - pandas DataFrame, dataframe containing the final set of messages 
                for processing
            - messages_tokens - pandas DataFrame, dataframe containing categories per message in long format 
                (one category per line)
            - ngrams_freqs - pandas DataFrame, dataframe containing the count of n-grams for the 
                whole dataset with values of n ranging from 1 to 3
    

    
    This function reads clean data and generates the final set of datasets/views to be saved in DB and
    used in the next steps of the analysis.
    '''
    # compute number of messages per genre
    messages_per_genre = messages_wide.genre.value_counts().rename_axis('genre').reset_index(name='num_msgs')

    # compute number of messages per category
    num_msgs_per_cat = messages_long.category.value_counts().rename_axis('category').reset_index(name='num_msgs')

    # compute number of words per message
    num_words_dist = messages.message.apply(lambda x: len(x.split())).to_frame(name='num_words')

    # compute median number of words per category-message
    word_counts_by_category = messages_long.groupby('category').message.agg(lambda x: count_words(x)).reset_index(name='num_words')

    # tokenize messages for further analysis
    print('Tokenizing Messages...')
    lemmatizer = WordNetLemmatizer()
    messages_tokens = messages[['message']]
    messages_tokens.loc[:,'tokens_str'] = messages_tokens.message.apply(lambda x: tokenize_to_str(x, lemmatizer))

    # count n-grams in messages
    print('Counting N-Grams...')
    ngrams_freqs = pd.DataFrame()
    for i in range(1,4):
        ngrams_freqs = pd.concat([ngrams_freqs,get_ngrams_freqs(messages_tokens.tokens_str,n=i)])

    # build final datasets to save to db

    # build final messages df by adding the number of words per message to messages df
    messages_final = pd.concat([messages, num_words_dist], axis=1)

    # build final categories df by joining the categories info with number of messages/words-per-message per category
    categories_final = categories_long.category.drop_duplicates().reset_index().drop('index', axis=1) \
                    .merge(num_msgs_per_cat, on='category') \
                    .merge(word_counts_by_category, on='category')

    # build final genres df by saving the number of messages per genre
    genres_final = messages.genre.drop_duplicates().reset_index().drop('index', axis=1) \
                    .merge(messages_per_genre, on='genre')

    # build_corpus wide df merging the messages and categories_wide dfs
    corpus_wide = messages.merge(categories_wide, on='message_id')

    # build message_categories df by renaming categories_long df
    message_categories = categories_long.rename(index=str, columns={'id':'message_id'})

    datasets_tables = {
        'Message' : messages_final,
        'Category' : categories_final,
        'Genre' : genres_final,
        'CorpusWide' : corpus_wide,
        'MessageCategoryLong' : message_categories,
        'MessageTokens' : messages_tokens,
        'NGramsFreqs' : ngrams_freqs
    }

    return datasets_tables

def save_data(datasets_tables, database_filepath):
    '''
    INPUT
    datasets_tables - dict, dictionary mapping to-be DB table names to their respective data-holding
        pandas dataframe
    database_filepath - string, filepath where database file will be saved to
    
    
    This function saves the dataset tables from the input dictionary to the database.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    for table_name, dataset in datasets_tables.items():
        dataset.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_raw, categories_raw = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        messages, categories, categories_wide, categories_long, messages_wide, messages_long = \
                clean_data(messages_raw, categories_raw)

        print('Preparing data for Analysis...')
        datasets_tables = prep_data_for_analysis(messages, categories, 
                            categories_wide, categories_long, messages_wide, 
                            messages_long)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(datasets_tables, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()