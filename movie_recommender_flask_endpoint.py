from flask import Flask, request, jsonify, render_template_string
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

indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()

def recommend(title, n=10):
    title = title.lower()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'vote_average']].iloc[movie_indices].to_dict('records')

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Movie Recommender</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        input[type=text] { padding: 8px; width: 300px; }
        button { padding: 8px 16px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f4f4f4; }
        .error { color: red; }
        .note { color: #666; font-size: 0.9em; margin-top: 30px; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h2>Movie Recommender</h2>
    <form method="get" action="/search">
        <input type="text" name="title" placeholder="Enter a movie title..." value="{{ query or '' }}" required>
        <button type="submit">Search</button>
    </form>

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if results %}
        <h3>Movies similar to "{{ query }}"</h3>
        <table>
            <tr><th>Title</th><th>Genres</th><th>Rating</th></tr>
            {% for movie in results %}
            <tr>
                <td>{{ movie.title }}</td>
                <td>{{ movie.genres }}</td>
                <td>{{ movie.vote_average }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}

    <p class="note">
        This page is for humans. Developers can query the JSON API directly at
        <code>/recommend?title=&lt;movie title&gt;&amp;n=&lt;number of results&gt;</code>
    </p>
</body>
</html>
"""

# --- HTML interface, for humans ---
@app.route('/')
def home():
    return render_template_string(PAGE_TEMPLATE)

@app.route('/search')
def search():
    title = request.args.get('title', '')
    results = recommend(title, n=10)
    error = None
    if results is None:
        error = f"'{title}' not found in the dataset."
    return render_template_string(PAGE_TEMPLATE, query=title, results=results, error=error)

# --- JSON API, for developers ---
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
    app.run(debug=True, use_reloader=False)
