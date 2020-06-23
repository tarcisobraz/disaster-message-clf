## Disaster Response Pipeline Project

### Table of Contents

1. [Motivation](#motivation)
2. [Repository Structure / Files](#files)
3. [Model Training](#model_training)
4. [Application](#application)
5. [Run it yourself!](#run_yourself)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Motivation<a name="motivation"></a>

As part of the activities of Udacity's Data Science Nanodegree I'm enrolled, I created this project, which aims at developing a Natural Language Processing (NLP) Classifier for text messages obtained in a context of disaster. 

The idea is that, at the occasion of a disaster (natural / human / machine caused), many messages are collected by the government authorities and they have to select which messages are important (related to the disaster), and have to group them into buckets by subject so it can be passed on to the right entities for providing help.

In this project, I use a dataset comprised of messages sent in a context of disaster with their respective categories to train a classifier which will be able to determine the category of a fresh message, helping disaster monitoring organizations. The data was gently provided by [Figure8](https://www.linkedin.com/company/figure8app/), now part of [Appen](https://appen.com/).

## Repository Structure / Files <a name="files"></a>

- The `data` folder comprises:
  * The messages and categories datasets
  * A draft jupyter notebook used for data preparation and exploratory analysis
  * The process_data.py script, used to prepare data for analysis and application presentation
  * The database file where data is stored for both training step and application data presentation
  
- The `models` folder comprises:
  * A draft jupyter notebook used for feature set/model building and analysis
  * The train_classifier.py script, used to run the train pipelines to select the best model
  * The nlp_estimators.py file, which defines custom estimators used in the different feature sets developed for this project
  * The workspace_utils.py file, which defines functions to keep workspace active while running scripts
  * The config folder, containing different train configuration files used to direct the execution of train_classifier script
  * The results folder, where both CSV files with GridSearchCV results and log files are stored
  * The best-models folder, where the best model for each GridSearchCV execution is stored
  
- The `app` folder comprises:
  * The run.py script which launches the web application using a Flask server
  * The templates folder, containing the html files for each page of the web application
  * The static folder, containing the n-grams wordcloud images to be displayed in the web application
  * The generate-ngrams-wordclouds.py script, used to generate the n-grams wordcloud images

## Model Training<a name="model_training"></a>

### Feature Sets

In order to develop the classification model, I invested some time developing feature extractors for the data:
 
 - Local Word2Vec model
   * [Widely known NLP model](https://en.wikipedia.org/wiki/Word2vec) used to generate word embeddings whose idea is to position the text words in a vector space such that words with similar meaning are close to each other, and words with opposite meaning are away from each other.
   * This model was trained locally, using the [Gensim library](https://radimrehurek.com/gensim/).
   * In this project, I use the trained model to extract the vectors for each token in the message text and then aggregate the vectors using a custom TF-IDF aggregator.
 - Pre-trained Glove model
   * [Glove](https://nlp.stanford.edu/projects/glove/) is another strategy to generate word embeddings developed by Stanford researchers. They provide a set of pre-trained models (trained on corpus with billions of words) with different vector sizes.
   * In this project, just like with the Word2Vec model, I use the pre-trained Glove model to extract the vectors for each token in the message text and then aggregate the vectors using a custom TF-IDF aggregator.
 - Doc2Vec
   * [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) is an NLP model which follows the same idea of Word2Vec, but instead of mapping words to feature vectors, maps whole documents to feature vectors.
   * This model was trained locally, using the [Gensim library](https://radimrehurek.com/gensim/).
   * As the model generates one feature vector per message, there was no need to aggregate the vectors as I did for the previous features.
 - Category Similarity
   * This is a custom feature thought for this specific project, whose idea is to take advantage of the supervised characteristic of the problem, by comparing the messages feature vectors to the categories names feature vectors, computing the cosine distance between them. I suspect that messages whose words are close to their categories words should have a short distance to them.
   * The format of this feature is a vector of size num_categories with the cosine distance between the message and each category.
 - All Features
   * All the above feature sets together (concatenated)
 
### Classifiers

 - Naive Bayes - Baseline classifier used to test the pipeline execution
 - Logistic Regression - Linear Classifier with good results on NLP tasks
 - Random Forest - Widely used Ensemble Classifier with good results on NLP tasks
 
### Feature/Model Evaluation and Selection

In order to select the best feature set and model to use in our classifier, I've used the following approach:

1. Run a Grid Search for each individual Feature Set varying the size of the feature vector using Random Forest and Logistic Regression models with fixed (average) parameters.

2. Run a second Grid Search for the All Features set using the best params from each feature set obtained from the previous grid search using Random Forest and Logistic Regression models with fixed (average) parameters.

3. Run a third Grid Search using the feature set and model with best performance from the above two grid searches, having fixed the feature set best parameters, but now using a grid to search the model hyperparameters.

4. Save the best model obtained from this third Grid Search to be used by the web application.

### Results

The table below shows the Top-5 FeatureSet-Model combinations according to their score on the test set.

| Feature Set                    | Model                                                          | Test Score (F1-Score) |
| ------------------------------ | -------------------------------------------------------------- | --------------------- |
| TF-IDF Local W2V (num_dims=300)| RandomForest(n_estimators=50,max_depth=100,min_num_samples=5)  | 0.474161              |
| TF-IDF Local W2V (num_dims=300)| RandomForest(n_estimators=100,max_depth=100,min_num_samples=5) | 0.471549              |
| TF-IDF Local W2V (num_dims=100)| RandomForest(n_estimators=50,max_depth=100,min_num_samples=5)  | 0.470284              |
| All Features with Best Params  | RandomForest(n_estimators=50,max_depth=100,min_num_samples=5)  | 0.470169              |
| TF-IDF Local W2V (num_dims=300)| RandomForest(n_estimators=50,max_depth=100,min_num_samples=5)  | 0.469709              |

As we can see, the TF-IDF aggregated Local Word2Vec outweights all others, making 4 of the top-5. In addition, all Top-5 pipelines use RandomForest as the classifier model. The best pipeline uses a feature vector with 300 dimensions and 50 trees (estimators) in the Random Forest.

The final test score is not high (below 50%), but we have to take into consideration the high complexity of the problem (multilabel classification), the large number of classes and the small number of samples in the dataset, which contributed greatly towards this result. Besides, I did not have a more robust infrastructure to test a wider range of grid params for the feature sets and models.

Notice: the F1-Score was computed by applying micro-averaging accross all the classes, accounting for class imbalance, as literature suggests.

## Application<a name="application"></a>

The results of the project are presented in the form of a Web Application. Next, I show some screenshots of it to depict its features:

<p align="center">
  <img width="800" height="200" src="app/screenshots/classifier-header.png">
</p>

At the header of the website, there is a text box where the user can type in a message and get back the classification results:

<p align="center">
  <img width="800" height="200" src="app/screenshots/classification-results.png">
</p>

Besides, there are some analysis on the dataset, showing:

- the number of messages/words per category
<p align="center">
  <img width="800" height="800" src="app/screenshots/messages-eda.png">
</p>

- the most common n-grams (in the form of histograms)
<p align="center">
  <img width="800" height="400" src="app/screenshots/bigrams-hist.png">
</p>

- the most common n-grams (in the form of wordclouds)
<p align="center">
  <img width="600" height="200" src="app/static/imgs/bi_wordcloud.png">
</p>

## Run it yourself!<a name="run_yourself"></a>

### Dependencies

The project needs a few extra libraries which don't come along with Anaconda 3's default package:
- Gensim
- dill
- plotly

In order to faccilitate the reproduction of the results, I've added to the repository a requirements.txt file with all the packages (and their respective versions) I used in the conda environment I created locally for this project.

1. The code assumes you use Anaconda (Python 3). Use the requirements.txt file at the repo root folder to recreate the conda environment with the needed libraries: `conda create --name <env_name> --file requirements.txt`.

2. Download the [pre-trained Glove models](https://drive.google.com/file/d/1XGzkIEgx6Y2IjzVYGDvn_shd77d_ZKki/view?usp=sharing) if you want to train models with Glove feature vectors. Unzip it into a local folder and set the `glove_models_folderpath` config in the train config file.

3. Run the following commands to prepare the data and model for application:

    - To activate the Anaconda environment created above, run the following command in the root folder:
    
        `conda activate <env_name>`
        
    - To run ETL pipeline that cleans data and stores in database, run the following command in the `data` folder:
    
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves, run the following command in the `models` folder:
    
        `python train_classifier.py configs/train_config_best_model.json 0`
        
    - To generate the wordclouds for the application, run the following command in the `apps` folder:
    
        `python generate-ngrams-wordclouds.py ../data/DisasterResponse.db static/imgs/`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/ to access the application.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

In order to achieve the results presented in this project, I've read many articles of specialists/enthusiasts in the field of NLP to get insights and learn how to deal with the problems I've faced along the way. Below, I cite some of them:

- [A Comprehensive Guide to Understand and Implement Text Classification in Python](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/)
- [Managing Machine Learning Workflows with Scikit-learn Pipelines Part 3: Multiple Models, Pipelines, and Grid Searches](https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.html)
- [Deep dive into multi-label classification..! (With detailed Case Study)](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff)
- [How to Build a Reusable Custom NLP Pipeline with Scikit-Learn](https://towardsdatascience.com/how-to-build-a-reusable-nlp-code-pipeline-with-scikit-learn-with-an-emphasis-on-feature-504f8aa14699)
- [[NLP] Performance of Different Word Embeddings on Text Classification](https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b)
- [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)

Feel free to use the code provided that you give credits / cite this repo, as well as to contribute.
