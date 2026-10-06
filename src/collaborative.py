"""
Collaborative Filtering
========================
Latent-factor model using truncated SVD (matrix factorization) over the
user-item ratings matrix -- the same family of technique that won the
Netflix Prize. Learns hidden "taste" factors from rating patterns, so it
can recommend movies a user has never interacted with based on similar
users' behaviour, without needing any genre/metadata info.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from data_preprocessing import load_movies, load_ratings


class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, n_factors: int = 50):
        self.movies = movies
        self.ratings = ratings
        self.n_factors = n_factors
        self._fit()

    def _fit(self):
        # Build user-item matrix
        self.user_item = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        self.user_ids = self.user_item.index.to_numpy()
        self.movie_ids = self.user_item.columns.to_numpy()

        matrix = self.user_item.to_numpy()
        # Mean only over RATED items per user (zeros are "unrated", not "rated 0")
        rated_mask = matrix > 0
        self.user_means = np.divide(
            matrix.sum(axis=1), rated_mask.sum(axis=1),
            out=np.zeros(matrix.shape[0]), where=rated_mask.sum(axis=1) > 0
        )
        matrix_demeaned = np.where(rated_mask, matrix - self.user_means.reshape(-1, 1), 0)

        k = min(self.n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma = np.diag(sigma)

        self.predicted_ratings = (
            U @ sigma @ Vt + self.user_means.reshape(-1, 1)
        )

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in dataset.")

        user_row_idx = np.where(self.user_ids == user_id)[0][0]
        pred_scores = self.predicted_ratings[user_row_idx]

        already_rated = set(
            self.ratings.loc[self.ratings["userId"] == user_id, "movieId"]
        )

        preds = pd.Series(pred_scores, index=self.movie_ids)
        preds = preds[~preds.index.isin(already_rated)]
        top_n = preds.sort_values(ascending=False).head(n)

        result = self.movies[self.movies["movieId"].isin(top_n.index)][
            ["movieId", "clean_title", "genres", "year"]
        ].copy()
        result["predicted_rating"] = result["movieId"].map(top_n).round(2)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()
    recommender = CollaborativeRecommender(ratings, movies)
    print(recommender.recommend(user_id=1, n=10))
