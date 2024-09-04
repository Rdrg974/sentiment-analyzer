import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Télécharger les données nécessaires de NLTK (décommente la ligne suivante si c'est la première fois que tu exécutes ce script)
# nltk.download('vader_lexicon')

# Analyser le sentiment d'un texte
def analyze_sentiment(text):
    # Initialiser l'analyseur de sentiment VADER
    sia = SentimentIntensityAnalyzer()
    
    # Obtenir les scores de polarité pour le texte donné
    scores = sia.polarity_scores(text)
    
    # Classer le texte en fonction du score "compound"
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = "Positif"
    elif compound_score <= -0.05:
        sentiment = "Négatif"
    else:
        sentiment = "Neutre"
    
    return scores, sentiment

if __name__ == "__main__":
    # Demander à l'utilisateur d'entrer une phrase
    user_input = input("Entrez une phrase à analyser : ")
    
    # Analyser le sentiment de la phrase
    result, sentiment = analyze_sentiment(user_input)
    
    # Afficher les résultats de l'analyse
    print(f"Résultats de l'analyse : {result}")
    print(f"Le sentiment global est : {sentiment}")
