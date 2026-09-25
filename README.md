# Movie Recommendation System

A content-based movie recommender that suggests similar films based on genre, overview, and tagline text, built using TF-IDF vectorization and cosine similarity. Deployed as a local Flask API.

## Project Overview

This project walks through building a content-based recommendation system end to end:

- Loading and filtering a large-scale movie metadata dataset
- Cleaning and combining text-based content features
- Vectorizing movie content with TF-IDF
- Computing similarity between movies using cosine similarity
- Building a recommendation function and evaluating it qualitatively with sample queries
- Packaging the recommender as a Flask API endpoint

## Dataset

The dataset (`TMDB_movie_dataset_v11.csv`) originally contains **994,914 movies** with 23 columns, including title, genres, overview, tagline, vote average, vote count, and production metadata.

Given the dataset's size, it was filtered down to **27,844 movies** meeting the following criteria:
- `status == 'Released'`
- `vote_count >= 50` (removes obscure/low-data entries)
- Non-null `overview` and `genres`

This filtering step was necessary both to remove low-quality/incomplete records and to keep the pairwise similarity computation computationally feasible (a full similarity matrix across ~1 million movies would be infeasible to compute or store).

## Methodology

1. **Data cleaning** — filtered to released movies with sufficient vote counts and complete metadata, then filled remaining missing values in `tagline`.
2. **Feature engineering** — combined `genres`, `overview`, and `tagline` into a single text field (`content`) per movie.
3. **Vectorization** — converted `content` into numeric features using `TfidfVectorizer` (English stopwords removed, vocabulary capped at 10,000 features), producing a `(27844, 10000)` TF-IDF matrix.
4. **Similarity computation** — calculated pairwise cosine similarity across all movies, producing a `(27844, 27844)` similarity matrix.
5. **Recommendation function** — given a movie title, returns the top N most similar movies based on cosine similarity score.
6. **Qualitative evaluation** — tested the recommender against multiple sample queries to assess recommendation quality.
7. **Deployment** — packaged the recommender as a Flask API with a `/recommend` endpoint.

## Evaluation

Testing across several sample movies revealed a clear pattern:

- **Franchise films performed excellently**: queries like "The Dark Knight" and "Ice Age" returned highly relevant sequels and spin-offs, since shared vocabulary (character names, franchise terms) reinforced genre similarity.
- **Standalone films with distinctive narrative concepts were harder**: "Inception" returned genre-appropriate sci-fi/action films, but didn't fully capture its specific "dream-heist" theme, since that concept isn't always explicit in the overview text.
- Occasional unrelated matches appeared (e.g. a horror film recommended alongside "Iron Man"), reflecting a known limitation: content-based filtering can only detect similarity present in the literal text provided, not deeper semantic meaning.

## Flask API

The recommender is deployed as a local Flask API:

```
GET /recommend?title=<movie title>&n=<number of results>
```

**Example:**
```
http://127.0.0.1:5000/recommend?title=The%20Dark%20Knight&n=5
```

**Example response:**
```json
{
  "query": "The Dark Knight",
  "recommendations": [
    {"title": "Batman: The Long Halloween, Part Two", "genres": "Animation, Mystery, Action, Crime", "vote_average": 7.459},
    {"title": "The Dark Knight Rises", "genres": "Action, Crime, Drama, Thriller", "vote_average": 7.777}
  ]
}
```

> Note: this is a local development server only (Flask's built-in server), not intended for production use.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (TF-IDF vectorization, cosine similarity)
- Flask (API deployment)

## Project Structure

```
├── TMDB_movie_dataset_v11.csv       # Dataset (not included in repo due to size — see Data Source)
├── Movie_Recommender.ipynb          # Full analysis and model-building notebook
├── movie_recommender_flask_endpoint.py                           # Flask API for serving recommendations
└── README.md
```

## Data Source

Full TMDB Movies Dataset 2024 (1M Movies), available on Kaggle.

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn flask
   ```
2. Download the dataset from Kaggle and place it in the project directory.
3. To explore the analysis: open `Movie_Recommender.ipynb` in Jupyter or VS Code and run all cells in order.
4. To run the API:
   ```bash
   python movie_recommender_flask_endpoint.py
   ```
   Then query it at `http://127.0.0.1:5000/recommend?title=<movie title>`.

## Future Improvements

- Incorporate cast, crew, or keyword data to improve thematic matching for standalone films
- Use n-grams or weight `overview` more heavily to capture multi-word concepts
- Explore a hybrid approach combining content-based filtering with collaborative filtering (using user rating data)
- Pre-compute and save the similarity matrix rather than recalculating it on every API startup, for faster load times and production readiness
