import re
import numpy as np
import streamlit as st
import nltk
from nltk import word_tokenize, pos_tag
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from nltk.corpus import stopwords

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# pd.set_option("max_columns", None) # show all cols
# pd.set_option('max_colwidth', None) # show full width of showing cols
# pd.set_option("expand_frame_repr", False) # print cols side by side as it's supposed to be
# pd.set_option('display.max_colwidth', -1)

text = "Assignment essays are developed from set questions that give students a period of time to research a topic and produce their answer with references to their sources of information. While there are some disadvantages with using assignment essays as an assessment tool, there are sound educational purposes underpinning this practice. This essay examines the reasons why assignment essays are beneficial for student learning and considers some of the problems with this method of assessment."

spell = SpellChecker()

# Function to generate a word cloud
def generate_word_cloud(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Count word frequencies
    word_count = Counter(tokens)

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_count)

    return wordcloud

# Function to analyze text
def analyze_text(text):

    # Filter out non-alphanumeric characters
    clean_text = re.sub(r'[^A-Za-z\s]', '', text)

    # Tokenize the text and perform part-of-speech tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    clean_tokens = word_tokenize(clean_text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in clean_tokens if word.lower() not in stop_words]

    # Calculate word count
    total_word_count = len(clean_tokens)
    unique_word_count = len(set(clean_tokens))

    # Extract nouns and verbs
    nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    verbs = [word for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    adjectives = [word for word, pos in pos_tags if pos in ['JJ', 'JJR', 'JJS']]
    adverbs = [word for word, pos in pos_tags if pos in ['RB', 'RBR', 'RBS']]
    preposition = [word for word, pos in pos_tags if pos in ['IN']]

    # Count frequencies
    word_count = Counter(tokens)
    clean_word_count = Counter(clean_tokens)
    noun_count = Counter(nouns)
    verb_count = Counter(verbs)
    adjective_count = Counter(adjectives)
    adverb_count = Counter(adverbs)
    preposition_count = Counter(preposition)
    
    # Find the longest and shortest words
    # words = [token for token, _ in pos_tags if token.isalpha()]
    longest_word = max(filtered_words, key=len)
    shortest_word = min(filtered_words, key=len)

    # Find the longest and shortest sentences
    sentences = text.split('.')
    longest_sentence = max(sentences, key=len)
    shortest_sentence = min(sentences, key=len)

    # Find the most common words and their frequencies
    most_common_words = word_count.most_common(10)

    # Find misspelled words
    misspelled_words = spell.unknown(clean_tokens)
    

    return total_word_count, unique_word_count, clean_word_count, nouns, noun_count, verbs, verb_count, longest_word, shortest_word, longest_sentence, shortest_sentence, most_common_words, word_count, adjectives, adjective_count, adverbs, adverb_count, preposition, preposition_count, misspelled_words

# Streamlit UI
st.title("Text Analysis")
st.write("Enter text to analyze:")
user_input = st.text_area("Text input", value=text, height=200)

if st.button("Analyze"):
    if user_input:
        total_word_count, unique_word_count, clean_word_count, nouns, noun_count, verbs, verb_count, longest_word, shortest_word, longest_sentence, shortest_sentence, most_common_words, word_count, adjectives, adjective_count, adverbs, adverb_count, preposition, preposition_count, misspelled_words = analyze_text(user_input)

        wordcloud = generate_word_cloud(user_input)

        st.write("---") 

        # st.subheader("Analysis Results")

        # # st.image(wordcloud.to_array())

        # st.write("---") 

        st.markdown(f'<p style="font-size:25px"><b>Total Word Count:</b> {total_word_count}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:25px"><b>Unique Word Count:</b> {unique_word_count}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:25px"><b>Longest Word:</b> {longest_word}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:25px"><b>Shortest Word:</b> {shortest_word}</p>', unsafe_allow_html=True)
        
        st.write("---") 

        st.subheader("Word Frequency")

        # Display the word count as a bar chart
        df = pd.DataFrame.from_dict(clean_word_count, orient='index', columns=['Count'])
        st.bar_chart(df)

        st.write("---") 

        st.subheader("Part-of-speech Tagging")

        st.subheader("Nouns:")
        noun_data = {'Identified Nouns': ', '.join(noun_count.keys()), 'Count': [sum(noun_count.values())], 'Unique Count': len(set(nouns))}
        df_nouns = pd.DataFrame(noun_data)
        st.table(df_nouns)

        st.subheader("Verbs:")
        verb_data = {'Identified Verbs': ', '.join(verb_count.keys()), 'Count': [sum(verb_count.values())], 'Unique Count': len(set(verbs))}
        df_verbs = pd.DataFrame(verb_data)
        st.table(df_verbs)

        st.subheader("Adverbs:")
        adverb_data = {'Identified Adverbs': ', '.join(adverb_count.keys()), 'Count': [sum(adverb_count.values())], 'Unique Count': len(set(adverbs))}
        df_adverbs = pd.DataFrame(adverb_data)
        st.table(df_adverbs)

        st.subheader("Adjectives:")
        adjective_data = {'Identified Adjectives': ', '.join(adjective_count.keys()), 'Count': [sum(adjective_count.values())], 'Unique Count': len(set(adjectives))}
        df_adjectives = pd.DataFrame(adjective_data)
        st.table(df_adjectives)

        st.subheader("Prepositions:")
        preposition_data = {'Identified Prepositions': ', '.join(preposition_count.keys()), 'Count': [sum(preposition_count.values())], 'Unique Count': len(set(preposition))}
        df_prepositions = pd.DataFrame(preposition_data)
        st.table(df_prepositions)

        st.write("---") 
        st.subheader("Misspelled Words")
        if misspelled_words:
            for word in misspelled_words:
                st.write(word)
        else:
            st.write("No misspelled words found in the text.")