import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(phrase):
    return nltk.word_tokenize(phrase)

def stem_and_lower(token):
    return stemmer.stem(token.lower())

def bag_of_words(tokenized_phrase, all_words):
    tokenized_phrase = [stem_and_lower(word) for word in tokenized_phrase]
    bag = np.zeros(len(all_words), dt=np.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_phrase:
            bag[index] = 1.0
            
    return bag