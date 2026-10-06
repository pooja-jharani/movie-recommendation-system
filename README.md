# AI-Powered Movie Recommendation Engine 🎬🤖

An end-to-end recommendation system implementing and comparing two classic
approaches: **Content-Based Filtering** and **Collaborative Filtering**,
deployed as an interactive Streamlit app.

## Problem Statement
Build a movie recommender that can suggest relevant movies either from a
single seed movie (no user history needed) or personalized to a specific
user's rating history, and quantitatively compare which approach works
better and why.

## Dataset
MovieLens `ml-latest-small`: 9,742 movies, 100,836 ratings from 610 users
(GroupLens Research, University of Minnesota).

## Approach

### 1. Content-Based Filtering (`src/content_based.py`)
- TF-IDF vectorization over movie genres
- Cosine similarity between movies
- Recommends: "movies similar to X" — works even for movies with zero ratings (no cold-start)

### 2. Collaborative Filtering (`src/collaborative.py`)
- User-item ratings matrix → mean-centered → **Truncated SVD** (matrix factorization,
  the same family of technique used in the Netflix Prize)
- Learns latent "taste" factors purely from rating patterns
- Recommends: personalized picks for a specific user

### 3. Evaluation & Comparison (`src/evaluate.py`)
| Approach | Metric | Result |
|---|---|---|
| Collaborative Filtering | RMSE / MAE (80/20 train-test split) | **RMSE 0.93**, MAE 0.72 |
| Content-Based Filtering | Precision@10 (genre overlap proxy) | **0.987** |

**Takeaway:** Collaborative filtering predicts *how much* a specific user
will like a movie (personalized), but fails for new users/movies with no
rating history (cold-start problem). Content-based filtering has no
cold-start problem and gives explainable recommendations, but doesn't
adapt to individual taste beyond genre — it can't tell a user prefers dark
comedies specifically. A production system would combine both (hybrid).

### 4. Deployment (`app.py`)
Streamlit app with two tabs — recommend by movie, or by user ID — showing
both models side by side.

## How to run
```bash
pip install -r requirements.txt

# Run individual modules
python src/content_based.py
python src/collaborative.py
python src/evaluate.py

# Launch the app
streamlit run app.py
```

## Project structure
```
movie-recommender-system/
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── src/
│   ├── data_preprocessing.py
│   ├── content_based.py
│   ├── collaborative.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

## Tech Stack
Python · Pandas · NumPy · SciPy · scikit-learn · Streamlit

## Author
Pooja Jharani — [GitHub](https://github.com/pooja-jharani)
