"""
Data loading & preprocessing for the Movie Recommender System.
Dataset: MovieLens ml-latest-small (9,742 movies, 100,836 ratings, 610 users).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    # Extract release year from title, e.g. "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    # genres as space-separated string (needed for TF-IDF) and as list
    movies["genres_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    movies["genres_str"] = movies["genres_list"].apply(lambda lst: " ".join(lst))
    return movies


def load_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return ratings


def load_all():
    movies = load_movies()
    ratings = load_ratings()
    print(f"Loaded {len(movies):,} movies and {len(ratings):,} ratings "
          f"from {ratings['userId'].nunique():,} users")
    return movies, ratings


if __name__ == "__main__":
    movies, ratings = load_all()
    print(movies.head())
    print(ratings.head())
