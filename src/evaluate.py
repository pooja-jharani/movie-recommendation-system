"""
Evaluation & Comparison
========================
Quantitatively evaluates both recommenders:
  - Collaborative filtering: RMSE/MAE on a held-out test split of ratings
    (standard rating-prediction accuracy metric).
  - Content-based filtering: genre-overlap Precision@K -- of the top-K
    recommended movies, what fraction share at least one genre with the
    seed movie (a proxy for relevance since there's no rating to predict).
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_movies, load_ratings
from content_based import ContentBasedRecommender


def evaluate_collaborative(ratings: pd.DataFrame, n_factors: int = 50, test_size: float = 0.2):
    train, test = train_test_split(ratings, test_size=test_size, random_state=42)

    user_item = train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    user_ids = user_item.index.to_numpy()
    movie_ids = user_item.columns.to_numpy()

    matrix = user_item.to_numpy()
    rated_mask = matrix > 0
    user_means = np.divide(
        matrix.sum(axis=1), rated_mask.sum(axis=1),
        out=np.zeros(matrix.shape[0]), where=rated_mask.sum(axis=1) > 0
    )
    matrix_demeaned = np.where(rated_mask, matrix - user_means.reshape(-1, 1), 0)

    k = min(n_factors, min(matrix.shape) - 1)
    U, sigma, Vt = svds(matrix_demeaned, k=k)
    sigma = np.diag(sigma)
    predicted = U @ sigma @ Vt + user_means.reshape(-1, 1)

    pred_df = pd.DataFrame(predicted, index=user_ids, columns=movie_ids)

    y_true, y_pred = [], []
    for row in test.itertuples():
        if row.userId in pred_df.index and row.movieId in pred_df.columns:
            y_true.append(row.rating)
            y_pred.append(pred_df.loc[row.userId, row.movieId])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "n_test_ratings_evaluated": len(y_true)}


def evaluate_content_based(movies: pd.DataFrame, recommender: ContentBasedRecommender,
                            sample_size: int = 200, k: int = 10):
    sample_titles = movies["clean_title"].sample(sample_size, random_state=42)
    precisions = []

    genre_lookup = movies.set_index("clean_title")["genres_list"].to_dict()

    for title in sample_titles:
        seed_genres = set(genre_lookup.get(title, []))
        if not seed_genres:
            continue
        try:
            recs = recommender.recommend(title, n=k)
        except ValueError:
            continue
        hits = 0
        for rec_title in recs["clean_title"]:
            rec_genres = set(genre_lookup.get(rec_title, []))
            if seed_genres & rec_genres:
                hits += 1
        precisions.append(hits / k)

    return {
        "Precision@K (genre overlap)": round(np.mean(precisions), 4),
        "K": k,
        "movies_evaluated": len(precisions),
    }


def main():
    movies = load_movies()
    ratings = load_ratings()

    print("Evaluating Collaborative Filtering (SVD)...")
    collab_metrics = evaluate_collaborative(ratings)
    print(collab_metrics)

    print("\nEvaluating Content-Based Filtering (TF-IDF + Cosine Similarity)...")
    recommender = ContentBasedRecommender(movies)
    content_metrics = evaluate_content_based(movies, recommender)
    print(content_metrics)

    print("\n=== Comparison Summary ===")
    print("Collaborative Filtering:")
    print("  - Learns from rating patterns across users, no metadata needed")
    print("  - Better for users with rating history; suffers from cold-start for new users/movies")
    print(f"  - RMSE: {collab_metrics['RMSE']}, MAE: {collab_metrics['MAE']}")
    print("Content-Based Filtering:")
    print("  - Uses movie genres, works even for brand-new movies (no cold-start)")
    print("  - Recommendations are explainable (same genres) but less personalized")
    print(f"  - Precision@{content_metrics['K']} (genre overlap): "
          f"{content_metrics['Precision@K (genre overlap)']}")


if __name__ == "__main__":
    main()
