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

## Application<a name="application"></a>

### Run it yourself!<a name="run_yourself"></a>

1. The code assumes you use Anaconda (Python 3). Use the requirements.txt file at the repo root folder to recreate the conda environment with the needed libraries: `conda create --name <env_name> --file requirements.txt`.

2. Download the [pre-trained Glove models](https://drive.google.com/file/d/1XGzkIEgx6Y2IjzVYGDvn_shd77d_ZKki/view?usp=sharing) if you want to train models with Glove feature vectors. Unzip it into a local folder and set the `glove_models_folderpath` config in the train config file.

3. Run the following commands to prepare the data and model for application:

    - To activate the Anaconda environment created above, run the following command in the root folder:
        `conda activate <env_name>`
    - To run ETL pipeline that cleans data and stores in database, run the following command in the `data` folder:
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves, run the following command in the `models` folder:
        `python train_classifier.py configs/train_config_simple.json 0`
    - To generate the wordclouds for the application, run the following command in the `apps` folder:
        `python generate-ngrams-wordclouds.py ../data/DisasterResponse.db static/imgs/`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/ to access the application.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

