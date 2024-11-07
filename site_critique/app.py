from flask import Flask, request, render_template, jsonify
import requests
from dotenv import load_dotenv
import os
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")  # Clé API de The Movie Database (TMDb)

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_movie():
    movie_title = request.form['title']
    url = f'https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}'
    response = requests.get(url)
    movies = response.json().get('results', [])
    return render_template('results.html', movies=movies, title=movie_title)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    movie_id = request.form['movie_id']
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}'
    response = requests.get(url)
    reviews = response.json().get('results', [])

    # Analyser les sentiments des critiques
    sentiment_results = []
    for review in reviews:
        score = sia.polarity_scores(review['content'])
        sentiment_results.append({
            'review': review['content'],
            'neg': score['neg'],
            'neu': score['neu'],
            'pos': score['pos'],
            'compound': score['compound']
        })

    # Calculer le pourcentage de critiques positives, neutres et négatives
    positive_reviews = [res for res in sentiment_results if res['compound'] > 0.05]
    neutral_reviews = [res for res in sentiment_results if -0.05 <= res['compound'] <= 0.05]
    negative_reviews = [res for res in sentiment_results if res['compound'] < -0.05]

    return render_template(
        'sentiment_results.html',
        positive_count=len(positive_reviews),
        neutral_count=len(neutral_reviews),
        negative_count=len(negative_reviews),
        reviews=sentiment_results
    )

if __name__ == '__main__':
    app.run(debug=True)
