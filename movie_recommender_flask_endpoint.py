from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare data once at startup
data = pd.read_csv('TMDB_movie_dataset_v11.csv')

filtered = data[
    (data['status'] == 'Released') &
    (data['vote_count'] >= 50) &
    (data['overview'].notna()) &
    (data['genres'].notna())
].copy().reset_index(drop=True)

movies = filtered[['id', 'title', 'genres', 'overview', 'tagline', 'vote_average', 'vote_count']].copy()
movies['tagline'] = movies['tagline'].fillna('')
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['genres'] + ' ' + movies['overview'] + ' ' + movies['tagline']

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(title, n=10):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'vote_average']].iloc[movie_indices].to_dict('records')

@app.route('/recommend', methods=['GET'])
def recommend_endpoint():
    title = request.args.get('title')
    n = int(request.args.get('n', 10))
    
    if not title:
        return jsonify({'error': 'Please provide a title parameter'}), 400
    
    results = recommend(title, n)
    if results is None:
        return jsonify({'error': f"'{title}' not found in the dataset"}), 404
    
    return jsonify({'query': title, 'recommendations': results})

if __name__ == '__main__':
    app.run(debug=True)
