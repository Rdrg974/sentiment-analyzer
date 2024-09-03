from nltk.classify import accuracy
from feature_extraction import extract_features
from classifier_training import features, train_count

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.classify import SklearnClassifier

# Définir les classificateurs scikit-learn
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

# Entraîner et évaluer chaque classificateur
for name, sklearn_classifier in classifiers.items():
    classifier = SklearnClassifier(sklearn_classifier)
    classifier.train(features[:train_count])
    accuracy_score = accuracy(classifier, features[train_count:])
    print(f"{accuracy_score:.2%} - {name}")
