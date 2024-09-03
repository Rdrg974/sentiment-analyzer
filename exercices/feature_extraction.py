import nltk

from nltk import FreqDist
from prep_data import positive_words, negative_words

# Calculer les distributions de fréquence
positive_fd = FreqDist(positive_words)
negative_fd = FreqDist(negative_words)

# Retirer les mots communs
common_set = set(positive_fd).intersection(negative_fd)
for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

# Top 100 mots les plus fréquents
top_100_positive = {word for word, _ in positive_fd.most_common(100)}
top_100_negative = {word for word, _ in negative_fd.most_common(100)}

# Extraire les caractéristiques
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = []
    positive_scores = []

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features
