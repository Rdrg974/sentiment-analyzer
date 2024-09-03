import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords, names
from random import shuffle
from statistics import mean

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Initialise l'analyseur de sentiments VADER
sia = SentimentIntensityAnalyzer()

# Définir les stopwords et noms indésirables
unwanted = set(stopwords.words("english"))
unwanted.update([w.lower() for w in names.words()])

# Définir les 100 mots les plus courants dans les critiques positives
def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):  # Exclure les noms
        return False
    return True

# Récupérer les critiques positives et négatives
positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(movie_reviews.words(categories=["pos"]))
)]

negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(movie_reviews.words(categories=["neg"]))
)]

positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)
for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

# Fonction pour extraire les caractéristiques
def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

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

# Créer le jeu de caractéristiques pour les critiques positives et négatives
features = [
    (extract_features(movie_reviews.raw(review)), "pos")
    for review in movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(movie_reviews.raw(review)), "neg")
    for review in movie_reviews.fileids(categories=["neg"])
])

# Entraîner le modèle
shuffle(features)
train_count = len(features) // 4
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])

# Évaluer le modèle
accuracy = nltk.classify.accuracy(classifier, features[train_count:])
print(f"Accuracy: {accuracy:.2%}")

# Montrer les caractéristiques les plus informatives
classifier.show_most_informative_features(10)

# Tester sur une nouvelle critique
new_review = "I absolutely loved this movie. The plot was fantastic and the acting was great."
print("New review classification:", classifier.classify(extract_features(new_review)))

# Création des classificateurs scikit-learn
classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME'),
}

# Entraînement et évaluation
from nltk.classify import SklearnClassifier
from nltk.classify.util import accuracy
from random import shuffle

# Mélanger les données et définir la taille d'entraînement
shuffle(features)
train_count = len(features) // 4

# Entraîner et évaluer chaque classificateur
for name, sklearn_classifier in classifiers.items():
    classifier = SklearnClassifier(sklearn_classifier)
    classifier.train(features[:train_count])
    acc = accuracy(classifier, features[train_count:])
    print(f"{acc:.2%} - {name}")

