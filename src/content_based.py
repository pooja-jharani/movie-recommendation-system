"""
Content-Based Filtering
========================
Recommends movies similar to a given movie using TF-IDF over genres
and cosine similarity. Works well for new/cold-start movies with no
ratings history, and gives explainable "because you liked X" results.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import load_movies


class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.title_to_idx = pd.Series(
            self.movies.index, index=self.movies["clean_title"].str.lower()
        )
        self._fit()

    def _fit(self):
        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")  # genres already tokenized on spaces
        tfidf_matrix = tfidf.fit_transform(self.movies["genres_str"])
        self.similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend(self, title: str, n: int = 10) -> pd.DataFrame:
        key = title.lower()
        if key not in self.title_to_idx:
            matches = self.movies[self.movies["clean_title"].str.lower().str.contains(key)]
            if matches.empty:
                raise ValueError(f"Movie '{title}' not found in dataset.")
            idx = matches.index[0]
        else:
            idx = self.title_to_idx[key]
            if isinstance(idx, pd.Series):  # duplicate titles
                idx = idx.iloc[0]

        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:n]

        result_idx = [i for i, _ in scores]
        result = self.movies.iloc[result_idx][["clean_title", "genres", "year"]].copy()
        result["similarity_score"] = [round(s, 3) for _, s in scores]
        return result.reset_index(drop=True)


if __name__ == "__main__":
    movies = load_movies()
    recommender = ContentBasedRecommender(movies)
    print(recommender.recommend("Toy Story", n=10))
