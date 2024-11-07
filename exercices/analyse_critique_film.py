import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
from time import sleep

# Télécharger les données nécessaires de NLTK (décommente la ligne suivante si c'est la première fois que tu exécutes ce script)
# nltk.download('vader_lexicon')

def get_reviews(url, start):
    """Récupère les critiques depuis IMDb en utilisant le paramètre 'start'"""
    full_url = f"{url}&start={start}"
    try:
        response = requests.get(full_url)
        response.raise_for_status()  # Vérifie si la requête a échoué
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = soup.find_all('div', class_='text show-more__control')
        return [review.get_text() for review in reviews]
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des critiques : {e}")
        return []

# URL du film sur IMDb (remplace avec l'URL du film que tu veux analyser, ici c'est Inception)
base_url = 'https://www.imdb.com/title/tt6718170/reviews/?ref_=tt_ov_ql_2'

# Nombre de pages à analyser
num_pages = 2  # Ajuste ce nombre selon tes besoins

all_reviews = []
for page in range(num_pages):
    start = page * 10
    print(f"Récupération des critiques de la page {page + 1} (start={start})")
    reviews = get_reviews(base_url, start)
    if reviews:
        all_reviews.extend(reviews)
    else:
        print("Aucune critique trouvée sur cette page.")
    sleep(2)  # Attendre entre les requêtes pour éviter les restrictions IP

print(f"Nombre total de critiques récupérées : {len(all_reviews)}")

# Initialiser l'analyseur de sentiments VADER
sia = SentimentIntensityAnalyzer()

# Calculer les scores compound pour chaque critique
compound_scores = []
for review in all_reviews:
    sentiment_score = sia.polarity_scores(review)
    compound_scores.append(sentiment_score['compound'])
print("Compound pour le premier critique : ", compound_scores[0])

# Calculer la moyenne des scores compound
if compound_scores:
    average_compound_score = mean(compound_scores)
    print(f"Moyenne des scores de sentiment : {average_compound_score:.2f}")
else:
    print("Aucune critique trouvée ou aucune analyse effectuée.")
