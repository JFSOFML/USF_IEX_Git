"""Handles logic for NLP streamlit app"""

import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


def get_wordnet_pos(treebank_tag):
    """Part fo speech tagging w/o NLTK"""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    """Preprocess text with NLTK"""
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(stemmed_word)
    return " ".join(processed_tokens)


def text_preprocessing_steps(text):
    """Preprocess text step by step"""
    nlp_steps = {}

    # Original text
    nlp_steps["Original"] = text

    # Tokenization
    tokens = word_tokenize(text)
    nlp_steps["Tokenized"] = " ".join(tokens)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    nlp_steps["POS Tagged"] = " ".join([f"{word}/{tag}" for word, tag in pos_tags])

    # Lemmatization and Stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    nlp_steps["Lemmatized"] = " ".join(lemmatized_tokens)
    nlp_steps["Stemmed"] = " ".join(stemmed_tokens)

    return nlp_steps


# Set up the page configuration
st.set_page_config(page_title="IMDB NLP App", layout="wide")

st.title("IMDB Movie Reviews Sentiment Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
pages = [
    "Home",
    "Textbook Process",
    "NLTK Process",
    "Topic Modeling (LDA)",
    "Model Comparison",
]
selection = st.sidebar.radio("Go to", pages)

if selection == "Home":
    st.subheader("Welcome to the IMDB NLP Project")

    st.image(r"Pictures/imdb_wordcloud.png", caption="Word Cloud for IMDB Reviews")

    st.write("## IMDB Dataset")
    st.write(
        """
    The IMDB dataset consists of 50,000 movie reviews from \
        IMDB, labeled by sentiment (positive/negative). 
    This dataset is often used for binary sentiment \
        classification. This project aims to analyze and 
    classify these reviews using various NLP techniques \
        and machine learning models.
    """
    )

if selection == "Textbook Process":
    st.subheader("IMDB Process from the Textbook")

    st.write(
        "### Chapter 8: Applying Machine Learning \
             to Sentiment Analysis"
    )

    st.write(
        """
    **Textbook Process Overview:**
    1. **Load the dataset**: Load the IMDB dataset which \
        consists of movie reviews and their sentiments.
    2. **Text preprocessing**: Convert the reviews to \
        lowercase, remove punctuation, and tokenize the text.
    3. **Feature extraction**: Use TfidfVectorizer to \
        convert the text data into TF-IDF features.
    4. **Model training**: Train a Linear Support Vector \
        Classifier (LinearSVC) on the TF-IDF features.
    5. **Model evaluation**: Evaluate the model's \
        performance using metrics like accuracy, \
            confusion matrix, and classification report.
    """
    )

if selection == "NLTK Process":
    st.subheader("IMDB Process using NLTK")

    st.write(
        """
    **NLTK Process Overview:**
    1. **Text preprocessing**: Use NLTK to tokenize the text, \
        perform POS tagging, and apply lemmatization and stemming.
    2. **Feature extraction**: Convert the preprocessed text \
        data into TF-IDF features.
    3. **Model training**: Train a Linear Support Vector \
        Classifier (LinearSVC) on the TF-IDF features.
    4. **Model evaluation**: Evaluate the model's performance \
        using metrics like accuracy, confusion matrix, and \
            classification report.
    """
    )

    st.write("### Interactive Text Preprocessing Demonstration")

    # User input for text preprocessing demonstration
    user_input = st.text_area(
        "Enter a movie review text:", "I loved the movie! It was fantastic."
    )
    if st.button("Show Preprocessing Steps"):
        steps = text_preprocessing_steps(user_input)
        for step, result in steps.items():
            st.write(f"**{step}:**")
            st.write(result)

if selection == "Topic Modeling (LDA)":
    st.subheader("Topic Modeling using LDA")

    st.write(
        """
    **LDA Process Overview:**
    1. **Feature extraction**: Use CountVectorizer to \
        convert the text data into a bag-of-words representation.
    2. **Topic modeling**: Apply Latent Dirichlet Allocation \
        (LDA) to identify topics in the text data.
    3. **Display topics**: Show the top words for each topic \
        identified by the LDA model.
    """
    )

REVIEW_1 = """The monster will look very familiar to you. \
    So will the rest of the film, if you've seen a half-dozen \
        of these teenagers-trapped-in-the-woods movies. Okay, \
        so they're not teenagers, this time, but they may \
        as well be. Three couples decide it might be a \
        good idea to check out a nearly-abandoned ghost town, \
        in hopes of finding the gold that people were \
        killed over a scant century-and-a-half before. \
        You'd think that with a title like "Miner's Massacre" \
        some interesting things might happen. They don't. In fact, \
        only about 1/10 of the film actually takes place in the mine. \
        I had envisioned teams of terrified miners scampering for \
        their lives in the cavernous confines of their workplace, \
        praying that Black Lung Disease would get them before The \
        Grim Reaper exacted his grisly revenge, but instead I got \
        terrestrial twenty-somethings fornicating--and, in one case, \
        defecating--in the woods, a gang of morons with a collective \
        I.Q. that would have difficulty pulling a plastic ring out of \
        a box of Cracker Jacks, much less a buried treasure from an \
        abandoned mine. No suspense, no scares, and plenty of \
        embarrassing performances give this turkey a 3 for nudity."""
REVIEW_2 = """It's another variation on the oft-told tale of two \
    people getting married and having to share their brood of kids. \
        WITH SIX YOU GET EGG ROLL is directed by Howard Morris \
        (from television) and it shows, because it's the kind \
        of tale that plays like a half-hour situation comedy \
        padded out to feature film length--but with a scarcity of \
        laughs, or to put it differently, only the number of laughs \
        that would have been possible within the half-hour limits of \
        a TV show.<br /><br />DORIS DAY decided to call it quits after \
        this film--and it's rather easy to see why. Even the presence of \
        some fairly reliable actors in the cast doesn't help. BRIAN KEITH, \
        BARBARA HERSHEY, PAT CARROLL and ALICE GHOSTLEY do their best, \
        but the script is the real problem and should have been left \
        untouched for the big screen.<br /><br />Nothing much can be said \
        in favor of it. Skip it and see Miss Day in any number of her more \
        worthwhile films.
"""
REVIEW_3 = """I watched this film with a sort of dangerous fascination, \
    like a hedgehog trapped in the headlights. There is no doubt that \
    (even if you enjoyed it) it's a bad movie, but the important question is \
    why? It has a good cast; it's lively; it's prepared to tackle sex head on,\
     with some of the characters actually getting some of it here and there, \
    which is unusual for a British comedy. It also has Johnny Vegas and \
    Mackenzie Crook, Marmite performers agreed but they've have had their \
    moments in the past.<br /><br />What it's principally lacking is charm. \
    The characters are impossibly idiotic, unbelievable and alienating, so that \
    instead of a film of Men Behaving Badly the producers have made Game On. \
    Any mediocre writer wanting to make a film about the sexual attitudes of \
    dozy, sexist British men would have got hold of a few copies of Loaded, \
    Zoo or even Viz to read Sid the Sexist and the thing would have written \
    itself. Instead, the producers clearly tried to make up some moronic, \
    difficult to care about, characters. Character comedy - as opposed to \
    slapstick etc - only works if the audience can recognise some human \
    truth to the situation. But watching this film is like being told an \
    annoying joke that you know is not going to end up funny but you can't \
    stop it.<br /><br />Sadly, the film is also poorly made. The plot \
    structure is weak, there's little character delineation or development, \
    and many of the scenes aren't funny. Time after time the same lame \
    reggae chips in to divide scenes, pointlessly and gratingly. There's a \
    lot of needless repetition - when you've done one joke about parking \
    outside a sex party you don't need to do it again. One wonders what \
    the UK Film Council saw in the script.<br /><br />This is a world \
    where most men are rakes, and most women are continually up for it. \
    The Apartment and Alfie satirised much the same world view, but the \
    producers of this film accept it without criticism. Thus they've ended \
    up with a kind of inferior update of Confessions of a Window Cleaner. \
    Somebody British needs to have another go at this kind of thing, and \
    do it properly  a good next project for Simon Pegg and Edgar Wright \
    I think...
Prediction:Negative"""
REVIEW_4 = """Well not actually. This movie is very entertaining though. \
Went and saw it with the girlfriend last night and had to use the "I think \
there might be something in my eye" routine. The movie is a great \
    combination of comedy and typical romance. Jim Carey is superb \
    as a down on his luck reporter who is given the power to change \
        himself and the city in which he resides. In fact all the \
            characters are great. The movie is not overly funny or \
                sappy, good flick to go see with the wife.<br /><br />\
                    All in All 8/10....note * I am not an easy grader. \
                        Thats all from BigV over and out!
"""
REVIEW_5 = """If you like Deep Purple, you will enjoy in this excellent \
    movie with Stephen Rea in main role. The story is about the most \
        famous rock group back there in 70s, Strange Fruits, and they \
            decided to play together again. But, of course, there is going\
                  to be lots of problem during theirs concerts. \
                    Jimmy Nail and Bill Nighy are great, and song "\
                        The Flame Still Burns" is perfect. You have \
                            to watch it.
"""

if selection == "Model Comparison":

    data = {
        "Review": [REVIEW_1, REVIEW_2, REVIEW_3, REVIEW_4, REVIEW_5],
        "SVC Model Prediction": [
            "Negative",
            "Positive",
            "Negative",
            "Positive",
            "Positive",
        ],
        "LLM Prediction": ["NEGATIVE", "NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE"],
        "Actual Value": [
            "Negative/0",
            "Negative/0",
            "Negative/0",
            "Positive/1",
            "Positive/1",
        ],
    }

    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)

    st.write("### Model Comparison Table")
    st.table(df)
    st.divider()

    st.header("Model Comparison Summary")
    st.write(
        """
LinerSVC Model Test-Accuracy: 89% | 80% from Sample size
                
LLM Model Accuracy: 100% (from sample size) 

### Summary
The LLM model outperforms the LinerSVC model in this sample size.
             
However, I could not run the entire dataset through the LLM because of \
    the limitations of my device. It would take approximately 625 Hours\
        to run sentitment analysis with LLM Studios on the entire dataset.
            
             """
    )
