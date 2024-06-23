# IMDB Movie Reviews Sentiment Analysis App

## Overview

This Streamlit app demonstrates various Natural Language Processing (NLP) techniques applied to the IMDB movie reviews dataset. The app showcases:
- A word cloud generated from the reviews.
- The typical NLP process as described in the textbook.
- An NLP process using NLTK for text preprocessing.
- Topic modeling using Latent Dirichlet Allocation (LDA).

## Installation

To run this app, you need to have Python installed on your machine. Then, install the required libraries using pip:

```sh
pip install streamlit pandas matplotlib scikit-learn nltk wordcloud

Running the App
To run the app, use the following command in your terminal:
streamlit run imdb_app.py

Project Structure
imdb_app.py: The main Streamlit app script.
Pictures/imdb_wordcloud.png: The word cloud image used in the home page (ensure this path is correct).
App Pages
Home
This page provides an overview of the IMDB dataset and displays a word cloud generated from the movie reviews.

Textbook Process
This page explains the typical NLP process as described in the textbook, including steps such as:

Loading the dataset.
Text preprocessing.
Feature extraction.
Model training.
Model evaluation.
NLTK Process
This page explains the NLP process using NLTK for text preprocessing, including steps such as:

Tokenization.
Part-of-Speech (POS) tagging.
Lemmatization and stemming.
Feature extraction.
Model training.
Model evaluation.
An interactive demonstration is included, where users can input a movie review and see how it is processed at each step.

Topic Modeling (LDA)
This page explains the topic modeling process using Latent Dirichlet Allocation (LDA), including steps such as:

Feature extraction using CountVectorizer.
Applying LDA to identify topics.
Displaying the top words for each topic.
Preprocessing Functions
get_wordnet_pos(treebank_tag)
This function converts Treebank POS tags to WordNet POS tags.

preprocess_text(text)
This function tokenizes the text, performs POS tagging, and applies lemmatization and stemming.

text_preprocessing_steps(text)
This function performs all preprocessing steps and returns a dictionary with the original text, tokenized text, POS tagged text, lemmatized text, and stemmed text.

Dependencies
Streamlit: For creating the web app.
Pandas: For data manipulation.
Matplotlib: For plotting.
Scikit-learn: For machine learning and feature extraction.
NLTK: For natural language processing tasks.
Wordcloud: For generating word clouds.