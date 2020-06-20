## Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Motivation](#motivation)
3. [Repository Structure / Files](#files)
4. [Model Training](#model_training)
5. [Application](#application)
6. [Run it yourself!](#run_yourself)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code assumes you use Anaconda (Python 3). Use the requirements.txt file at the repo root folder to recreate the conda environment with the needed libraries: `conda create --name <env> --file requirements.txt`

## Motivation<a name="motivation"></a>

As part of the activities of Udacity's Data Science Nanodegree I'm enrolled, I created this project, which aims at developing a Natural Language Processing (NLP) Classifier for text messages obtained in a context of disaster. 

The idea is that, at the occasion of a disaster (natural / human / machine caused), many messages are collected by the government authorities and they have to select which messages are important (related to the disaster), and have to group them into buckets by subject so it can be passed on to the right entities for providing help.

In this project, I use a dataset comprised of messages sent in a context of disaster with their respective categories to train a classifier which will be able to determine the category of a fresh message, helping disaster monitoring organizations.

## Repository Structure / Files <a name="files"></a>

## Model Training<a name="model_training"></a>

## Application<a name="application"></a>

### Run it yourself!<a name="run_yourself"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

