# Movie Recommendation System

A recommender that implements and compares two classic approaches, **content-based filtering** and **collaborative filtering**, with an interactive Streamlit app.

**Live demo:** [add link after deploying]

## Problem Statement
Build a movie recommender that can suggest relevant movies either from a single seed movie (no user history needed) or personalized to a specific user's rating history, and compare the trade-offs of the two approaches.

## Dataset
[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/): 9,742 movies and 100,836 ratings from 610 users (GroupLens Research, University of Minnesota).

## Approach

### 1. Content-Based Filtering (`src/content_based.py`)
- TF-IDF vectorization over movie genres
- Cosine similarity between movies
- Recommends "movies similar to X" and works even for movies with zero ratings (no cold-start)

### 2. Collaborative Filtering (`src/collaborative.py`)
- User-item ratings matrix, mean-centered, then **Truncated SVD** (matrix factorization, the same family of technique used in the Netflix Prize)
- Learns latent "taste" factors purely from rating patterns
- Recommends personalized picks for a specific user

### 3. Evaluation (`src/evaluate.py`)
- SVD: 80/20 train-test split, evaluated on 19,355 held-out ratings
- Content-based: Precision@10 using genre overlap as a proxy for relevance

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Collaborative (SVD) | RMSE | 0.9304 |
| Collaborative (SVD) | MAE | 0.7192 |
| Content-based (TF-IDF + cosine) | Precision@10 (genre overlap) | 0.9869 |

**Note:** The content-based score is high partly by design. The model recommends by genre similarity and the metric checks genre overlap, so it measures consistency, not user satisfaction. The two metrics are not directly comparable.

| | Collaborative | Content-based |
|---|---|---|
| Strength | Personalized from rating patterns | Works for new movies (no cold-start), explainable |
| Weakness | Cold-start for new users and movies | Less personalized, only uses genres |

**Takeaway:** Collaborative filtering predicts how much a specific user will like a movie, but fails without rating history. Content-based filtering has no cold-start problem, but it can't tell that a user prefers dark comedies specifically. A production system would combine both (hybrid).

## App (`app.py`)
Streamlit app with two tabs, recommend by movie or by user ID, showing both models side by side.

[Add 1-2 screenshots here]

## How to run
```bash
pip install -r requirements.txt

# Run individual modules
python src/content_based.py
python src/collaborative.py
python src/evaluate.py

# Launch the app
python -m streamlit run app.py
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
Pooja Jharani · [GitHub](https://github.com/pooja-jharani)