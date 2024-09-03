import nltk
from nltk.corpus import stopwords, movie_reviews

# Récupération des mots non voulus (stopwords + noms propres)
unwanted = set(stopwords.words('english'))
unwanted.update([w.lower() for w in nltk.corpus.names.words()])

# Fonction pour filtrer les mots non voulus
def skip_unwanted(word, tag):
    if not word.isalpha() or word.lower() in unwanted or tag.startswith('NN'):
        return False
    return True

# Préparer les critiques positives et négatives
def extract_words_from_reviews(category):
    words = []
    for word, tag in nltk.pos_tag(movie_reviews.words(categories=[category])):
        if skip_unwanted(word, tag):
            words.append(word.lower())
    return words

positive_words = extract_words_from_reviews("pos")
negative_words = extract_words_from_reviews("neg")
