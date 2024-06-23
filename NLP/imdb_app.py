# imdb_app.py
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(stemmed_word)
    return ' '.join(processed_tokens)

def text_preprocessing_steps(text):
    steps = {}
    
    # Original text
    steps['Original'] = text
    
    # Tokenization
    tokens = word_tokenize(text)
    steps['Tokenized'] = ' '.join(tokens)
    
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    steps['POS Tagged'] = ' '.join([f"{word}/{tag}" for word, tag in pos_tags])
    
    # Lemmatization and Stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    steps['Lemmatized'] = ' '.join(lemmatized_tokens)
    steps['Stemmed'] = ' '.join(stemmed_tokens)
    
    return steps

# Set up the page configuration
st.set_page_config(page_title='IMDB NLP App', layout='wide')

st.title('IMDB Movie Reviews Sentiment Analysis')

# Sidebar for navigation
st.sidebar.header('Navigation')
pages = ['Home', 'Textbook Process', 'NLTK Process', 'Topic Modeling (LDA)', 'Model Comparison']
selection = st.sidebar.radio('Go to', pages)

if selection == 'Home':
    st.subheader('Welcome to the IMDB NLP Project')

    st.image('Pictures/imdb_wordcloud.png', caption='Word Cloud for IMDB Reviews')

    st.write("## IMDB Dataset")
    st.write("""
    The IMDB dataset consists of 50,000 movie reviews from IMDB, labeled by sentiment (positive/negative). 
    This dataset is often used for binary sentiment classification. This project aims to analyze and 
    classify these reviews using various NLP techniques and machine learning models.
    """)

if selection == 'Textbook Process':
    st.subheader('IMDB Process from the Textbook')
    
    st.write("### Chapter 8: Applying Machine Learning to Sentiment Analysis")

    st.write("""
    **Textbook Process Overview:**
    1. **Load the dataset**: Load the IMDB dataset which consists of movie reviews and their sentiments.
    2. **Text preprocessing**: Convert the reviews to lowercase, remove punctuation, and tokenize the text.
    3. **Feature extraction**: Use TfidfVectorizer to convert the text data into TF-IDF features.
    4. **Model training**: Train a Linear Support Vector Classifier (LinearSVC) on the TF-IDF features.
    5. **Model evaluation**: Evaluate the model's performance using metrics like accuracy, confusion matrix, and classification report.
    """)

if selection == 'NLTK Process':
    st.subheader('IMDB Process using NLTK')

    st.write("""
    **NLTK Process Overview:**
    1. **Text preprocessing**: Use NLTK to tokenize the text, perform POS tagging, and apply lemmatization and stemming.
    2. **Feature extraction**: Convert the preprocessed text data into TF-IDF features.
    3. **Model training**: Train a Linear Support Vector Classifier (LinearSVC) on the TF-IDF features.
    4. **Model evaluation**: Evaluate the model's performance using metrics like accuracy, confusion matrix, and classification report.
    """)

    st.write("### Interactive Text Preprocessing Demonstration")

    # User input for text preprocessing demonstration
    user_input = st.text_area("Enter a movie review text:", "I loved the movie! It was fantastic.")
    if st.button("Show Preprocessing Steps"):
        steps = text_preprocessing_steps(user_input)
        for step, result in steps.items():
            st.write(f"**{step}:**")
            st.write(result)

if selection == 'Topic Modeling (LDA)':
    st.subheader('Topic Modeling using LDA')

    st.write("""
    **LDA Process Overview:**
    1. **Feature extraction**: Use CountVectorizer to convert the text data into a bag-of-words representation.
    2. **Topic modeling**: Apply Latent Dirichlet Allocation (LDA) to identify topics in the text data.
    3. **Display topics**: Show the top words for each topic identified by the LDA model.
    """)

review_1 =  """The monster will look very familiar to you. So will the rest of the film, if you've seen a half-dozen of these teenagers-trapped-in-the-woods movies. Okay, so they're not teenagers, this time, but they may as well be. Three couples decide it might be a good idea to check out a nearly-abandoned ghost town, in hopes of finding the gold that people were killed over a scant century-and-a-half before. You'd think that with a title like "Miner's Massacre" some interesting things might happen. They don't. In fact, only about 1/10 of the film actually takes place in the mine. I had envisioned teams of terrified miners scampering for their lives in the cavernous confines of their workplace, praying that Black Lung Disease would get them before The Grim Reaper exacted his grisly revenge, but instead I got terrestrial twenty-somethings fornicating--and, in one case, defecating--in the woods, a gang of morons with a collective I.Q. that would have difficulty pulling a plastic ring out of a box of Cracker Jacks, much less a buried treasure from an abandoned mine. No suspense, no scares, and plenty of embarrassing performances give this turkey a 3 for nudity."""


if selection == 'Model Comparision':

    st.write(f"""
    ###Model Caparison Table


    | Review | SVC Model Prediction | LLM Prediction | Actual Value |
    |---|---|---|---|
    | {review_1}| Negative | NEGATIVE. | Negative/0|
    | {review_2}| Row 2, Cell 2 | Row 2, Cell 3 | Actual |
    | {review_3}| Row 2, Cell 2 | Row 2, Cell 3 | Actual |
    | {review_4}| Row 2, Cell 2 | Row 2, Cell 3 | Actual |
    | {review_5}| Row 2, Cell 2 | Row 2, Cell 3 | Actual |
    """)
    st.divider()
