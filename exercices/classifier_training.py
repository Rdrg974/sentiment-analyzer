from feature_extraction import extract_features
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from random import shuffle

# Créer la liste des caractéristiques
features = [
    (extract_features(movie_reviews.raw(review)), "pos")
    for review in movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(movie_reviews.raw(review)), "neg")
    for review in movie_reviews.fileids(categories=["neg"])
])

# Mélanger et diviser les données
shuffle(features)
train_count = len(features) // 4

# Entraîner le classificateur
classifier = NaiveBayesClassifier.train(features[:train_count])

# Évaluer la précision
print(f"Accuracy: {accuracy(classifier, features[train_count:]):.2%}")
