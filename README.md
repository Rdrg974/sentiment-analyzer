# SentimentAnalyzer

## Description

SentimentAnalyzer est un projet Python qui utilise le machine learning pour analyser les sentiments des textes. Ce projet permet de classer des textes en fonction de leur polarité (positif, négatif ou neutre) à l'aide de techniques de traitement du langage naturel (NLP). Cela peut se faire en analysant des ensembles de textes, tels que des commentaires, tweets, et avis de produits, afin d'obtenir des insights sur votre audience.


## Fonctionnalité with NLTK

La bibliothèque NLTK offre divers outils pour manipuler et analyser efficacement les données linguistiques. Parmi ses fonctionnalités avancées se trouvent des classificateurs de texte, utilisables pour différents types de classification, y compris l'analyse de sentiment.

L'analyse de sentiment consiste à utiliser des algorithmes pour classer des échantillons de texte en catégories globales de positif et négatif. Avec NLTK, vous pouvez utiliser ces algorithmes via des opérations d'apprentissage automatique intégrées pour obtenir des insights à partir des données linguistiques.

## Installation et Configuration de NLTK 

* Pour commencer à utiliser NLTK (Natural Language Toolkit), installe le module avec pip :

python3 -m pip install nltk

* Téléchargement des Ressources NLTK

Après l'installation de NLTK, tu dois télécharger des ressources supplémentaires nécessaires pour le traitement du langage naturel. Voici comment procéder :

import nltk
nltk.download()

Cela ouvre une interface graphique te permettant de choisir et télécharger les ressources nécessaires.
Téléchargement direct : Tu peux également télécharger directement les ressources spécifiques en utilisant les identifiants suivants :

import nltk
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

* Gestion des Ressources Manquantes

Si tu essaies d'utiliser une ressource qui n'a pas encore été téléchargée, NLTK affichera une LookupError avec des instructions pour télécharger la ressource manquante. Par exemple :

import nltk
w = nltk.corpus.shakespeare.words()

* Si la ressource "shakespeare" n'est pas trouvée, tu verras une erreur comme :

LookupError:
**********************************************************************
  Resource shakespeare not found.
  Please use the NLTK Downloader to obtain the resource:
  >>> import nltk
  >>> nltk.download('shakespeare')
**********************************************************************

Suis les instructions fournies pour télécharger la ressource en utilisant nltk.download('shakespeare').

## Compilation des Données avec NLTK

### Charger et Nettoyer les Données

* Charger les mots : Utilise nltk.corpus.state_union.words() et filtre pour ne garder que les mots alphabétiques :

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

* Retirer les stop words : Exclue les mots courants (stop words) avec :

stopwords = nltk.corpus.stopwords.words("english")
words = [w for w in words if w.lower() not in stopwords]

### Créer un Corpus et Tokeniser

* Créer un corpus : Charge du texte brut ou utilise des annotations pour créer ton propre corpus.
* Tokeniser le texte : Divise le texte en mots avec nltk.word_tokenize() et filtre la ponctuation :

from pprint import pprint
text = """For some quick analysis, creating a corpus could be overkill. If all you need is a word list,there are simpler ways to achieve that goal."""
words = [w for w in nltk.word_tokenize(text) if w.isalpha()]
pprint(words)

Ces étapes permettent de préparer des données textuelles pour une analyse efficace.

## Création de Distributions de Fréquence avec NLTK

Une distribution de fréquence est essentiellement un tableau ou un résumé qui montre combien de fois chaque mot apparaît dans un texte donné. Dans NLTK, cela se fait à l'aide de la classe FreqDist, qui fonctionne comme un dictionnaire Python mais avec des fonctionnalités supplémentaires pour l'analyse des fréquences des mots.

### Créer une Distribution de Fréquence

Tokeniser le texte : Divise le texte en mots.

words = nltk.word_tokenize(text)

Construire la distribution : Utilise nltk.FreqDist() pour créer une distribution des fréquences des mots.

fd = nltk.FreqDist(words)

* Utiliser les Méthodes

.most_common(n) : Affiche les n mots les plus fréquents.

fd.most_common(3)

.tabulate(n) : Affiche les n mots les plus fréquents dans un format tabulaire.

fd.tabulate(3)

* Analyse Personnalisée

Requête de mots : Compte les occurrences exactes des mots, en tenant compte de la casse.

fd["America"]

Normalisation : Crée une distribution basée sur des mots en minuscules pour une analyse plus précise.

lower_fd = nltk.FreqDist([w.lower() for w in words])

## Extraction de Concordances et de Collocations avec NLTK

### Concordances

Une concordance est une collection d'occurrences d'un mot avec le contexte environnant. Elle permet de voir :

Combien de fois un mot apparaît.
Où chaque occurrence se trouve dans le texte.
Quels mots entourent chaque occurrence.

Créer une instance nltk.Text avec une liste de mots :

text = nltk.Text(nltk.corpus.state_union.words())

Extraire des concordances avec .concordance() pour voir les occurrences d'un mot avec contexte :

text.concordance("america", lines=5)

Obtenir une liste détaillée des concordances avec .concordance_list() :

concordance_list = text.concordance_list("america", lines=2)

### Collocations

Les collocations sont des combinaisons de mots qui apparaissent fréquemment ensemble dans un texte. Par exemple, dans le corpus des discours de l'état de l'Union, "United States" est une collocation.

Trouver des collocations (combinaisons fréquentes de mots) avec CollocationFinder :

finder = nltk.collocations.TrigramCollocationFinder.from_words(words)

Afficher les collocations les plus fréquentes :

finder.ngram_fd.most_common(2)
finder.ngram_fd.tabulate(2)

Ces outils permettent d'explorer les occurrences et combinaisons de mots dans un texte de manière approfondie.

## Difference entre nltk.FreqDist et nltk.Text:

nltk.FreqDist et nltk.Text sont des outils complémentaires dans NLTK. nltk.FreqDist est utilisé pour analyser la fréquence des mots dans un texte, offrant des méthodes pour afficher les mots les plus fréquents, créer des graphiques de distribution, et calculer les fréquences relatives. C'est idéal pour des analyses statistiques simples.

nltk.Text, en revanche, est conçu pour des analyses contextuelles. Il permet de rechercher des occurrences de mots avec leur contexte, de découvrir des collocations (mots souvent associés), et de créer une distribution de fréquence simplifiée. Utilise nltk.FreqDist pour des statistiques de fréquence et nltk.Text pour explorer le contexte et les associations de mots.

## Analyse de Sentiments avec VADER

NLTK inclut VADER (Valence Aware Dictionary and sEntiment Reasoner), un analyseur de sentiments pré-entraîné, idéal pour les textes courts et informels comme les tweets.

* Utilisation de VADER

Initialisation :

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

Analyse de Texte :

scores = sia.polarity_scores("Texte à analyser")

Scores :
neg : Sentiment négatif
neu : Sentiment neutre
pos : Sentiment positif
compound : Score composite global, de -1 à 1

Analyse de Tweets :

tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

Classification :

def is_positive(tweet: str) -> bool:
    return sia.polarity_scores(tweet)["compound"] > 0

* Analyse de Critiques de Films :

Charger les critiques :

positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])

Classifier les critiques :

def is_positive(review_id: str) -> bool:
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [sia.polarity_scores(sentence)["compound"] for sentence in nltk.sent_tokenize(text)]
    return mean(scores) > 0

Précision : VADER a montré une précision de 64% sur les critiques de films.

VADER est un excellent point de départ pour l'analyse de sentiments, particulièrement adapté aux médias sociaux. Pour des résultats plus précis, des ajustements et des évaluations supplémentaires peuvent être nécessaires.

## Personnalisation de l'Analyse de Sentiments avec NLTK

NLTK propose des classificateurs intégrés pour l'analyse des sentiments, capables de catégoriser des textes en sentiments positifs, négatifs ou neutres. Pour améliorer l'efficacité de ces classificateurs, il est crucial de bien sélectionner les features (caractéristiques) des données. Les features sont des aspects spécifiques du texte, comme la fréquence de certains mots, qui aident le modèle à faire des prédictions précises.

La sélection des features consiste à choisir les caractéristiques les plus pertinentes pour la tâche de classification. Bien que ce guide ne couvre pas en détail le processus de sélection et d’ingénierie des features, il est important de noter que la qualité des features choisies peut significativement influencer la précision des résultats d'analyse de sentiments. Ajuster et optimiser ces caractéristiques peut aider à améliorer les performances des outils d'analyse fournis par NLTK.

### Sélection des Features Utiles pour l'Analyse des Sentiments

Pour améliorer l'analyse des sentiments avec NLTK, il est crucial de sélectionner des caractéristiques (features) pertinentes de vos données. Voici comment procéder :

* Préparation des Données :

Exclusion des Mots Non Souhaités : Créez une liste de mots à exclure, incluant les mots très courants (stopwords) et les noms propres (ex. noms d'acteurs). Utilisez la fonction skip_unwanted() pour filtrer ces mots en excluant également les noms (balises grammaticales commençant par "NN").

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

* Création des Listes de Mots Positifs et Négatifs :

Extraction des Mots : Utilisez nltk.pos_tag() pour taguer les mots par partie du discours et filtrez-les selon les critères définis pour obtenir des listes de mots significatifs pour les critiques positives et négatives.

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]

negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]

* Analyse des Fréquences :

Distribution des Fréquences : Créez des objets FreqDist pour les mots positifs et négatifs. Identifiez et supprimez les mots communs aux deux listes pour éviter les biais, puis sélectionnez les mots les plus fréquents pour chaque catégorie.

positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)
for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

* Amélioration avec les Bigrammes :

Utilisation des Bigrammes : Identifiez les combinaisons fréquentes de mots (bigrams) dans les critiques positives et négatives pour capturer des phrases significatives, telles que les expressions courantes dans chaque catégorie.

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])

negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
])

En expérimentant avec ces méthodes et en ajustant les caractéristiques sélectionnées, vous pouvez affiner les résultats de votre analyse des sentiments et obtenir des insights plus précis.

### Entraînement et Utilisation d'un Classificateur de Sentiments

Pour créer un classificateur de sentiments, suivez ces étapes :

1. Définir une Fonction d'Extraction des Caractéristiques

La première étape consiste à créer une fonction qui extrait des caractéristiques pertinentes d'un texte. Voici un exemple de fonction extract_features(text) qui calcule la moyenne des scores composés et positifs de VADER pour chaque phrase et compte le nombre de mots présents dans une liste de mots positifs courants :

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

Cette fonction renvoie un dictionnaire avec trois caractéristiques :

* mean_compound: La moyenne des scores composés des phrases, augmentée de 1 pour éviter les valeurs négatives.
* mean_positive: La moyenne des scores positifs des phrases.
* wordcount: Le nombre de mots du texte qui apparaissent dans la liste des 100 mots les plus fréquents des critiques positives.

2. Préparer les Données d'Entraînement

Construisez une liste de tuples où chaque tuple contient les caractéristiques extraites d'une critique et sa catégorie ("pos" pour positif, "neg" pour négatif). Voici comment préparer cette liste à partir du corpus movie_reviews de NLTK :

features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
    for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
])

Cette liste est ensuite divisée en deux ensembles : un pour l'entraînement et un pour l'évaluation. Utilisez la méthode .train() pour entraîner un classificateur Naive Bayes :

train_count = len(features) // 4
shuffle(features)
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])

3. Évaluer et Utiliser le Classificateur

Après l'entraînement, évaluez la performance du classificateur sur l'ensemble de test et affichez les caractéristiques les plus informatives :

accuracy = nltk.classify.accuracy(classifier, features[train_count:])
print(f"Accuracy: {accuracy:.2%}")
classifier.show_most_informative_features(10)

Vous pouvez utiliser le classificateur pour prédire la catégorie de nouvelles critiques en passant les caractéristiques extraites de ces critiques à classifier.classify(). Ajustez les caractéristiques pour améliorer la précision du modèle, en explorant différentes combinaisons et en affinant votre extraction de caractéristiques.

En utilisant cette approche, vous pouvez créer un classificateur de sentiments qui apprend à partir de données préexistantes et peut être utilisé pour analyser de nouveaux textes avec une précision améliorée.

### Comparaison des Classificateurs Supplémentaires avec NLTK et scikit-learn

NLTK permet d'intégrer des classificateurs du framework scikit-learn, offrant une variété de modèles de machine learning pour l'analyse de sentiments. Après avoir installé scikit-learn via pip ($ python3 -m pip install scikit-learn), vous pouvez importer divers classificateurs, tels que :

Naive Bayes : BernoulliNB, ComplementNB, MultinomialNB
K-Nearest Neighbors : KNeighborsClassifier
Arbres de Décision : DecisionTreeClassifier
Forêts Aléatoires : RandomForestClassifier
Régression Logistique : LogisticRegression
Réseaux de Neurones : MLPClassifier
Boosting : AdaBoostClassifier
Analyse Discriminante : QuadraticDiscriminantAnalysis

Une fois installé, vous pouvez importer divers classificateurs de scikit-learn dans votre code Python. Voici quelques-uns des classificateurs disponibles :

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Créez des instances de ces classificateurs avec leurs paramètres par défaut, ce qui simplifie l'évaluation comparative. Utilisez un dictionnaire pour gérer ces instances :

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}

Ces classificateurs peuvent être entraînés et évalués sur vos données pour déterminer lequel offre les meilleures performances pour votre tâche d'analyse de sentiments. Vous pouvez ainsi comparer leur précision et choisir le modèle le plus adapté à vos besoins.

## Conclusion

Ce tutoriel vous a permis de découvrir les fonctionnalités de NLTK pour traiter et analyser des données textuelles. Vous avez appris à préparer et filtrer les données textuelles, analyser la fréquence des mots, et identifier les concordances et collocations. Vous avez également utilisé l'analyse de sentiment rapide avec VADER intégré à NLTK, défini des caractéristiques pour la classification personnalisée, et comparé différents classificateurs de scikit-learn au sein de NLTK.

Avec ces compétences, vous êtes prêt à appliquer NLTK dans vos propres projets, que ce soit pour créer des visualiseurs d'analyse de sentiment ou enrichir le traitement du texte dans des applications web en Python. N'hésitez pas à explorer des paquets populaires supplémentaires pour continuer à approfondir vos connaissances.