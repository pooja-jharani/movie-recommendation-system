"""
Movie Recommendation System -- Streamlit App
Run: streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import load_movies, load_ratings
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_models():
    movies = load_movies()
    ratings = load_ratings()
    content_model = ContentBasedRecommender(movies)
    collab_model = CollaborativeRecommender(ratings, movies)
    return movies, ratings, content_model, collab_model


movies, ratings, content_model, collab_model = load_models()

st.title("🎬 AI-Powered Movie Recommendation Engine")
st.caption("Content-Based Filtering (TF-IDF + Cosine Similarity) vs. "
           "Collaborative Filtering (SVD Matrix Factorization) — side by side")

tab1, tab2 = st.tabs(["🎭 By Movie (Content-Based)", "👤 By User (Collaborative)"])

with tab1:
    st.subheader("Find movies similar to one you like")
    movie_titles = sorted(movies["clean_title"].unique())
    selected_movie = st.selectbox("Pick a movie", movie_titles, index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0)
    n_recs = st.slider("Number of recommendations", 5, 20, 10, key="content_n")

    if st.button("Recommend similar movies", type="primary"):
        with st.spinner("Finding similar movies..."):
            recs = content_model.recommend(selected_movie, n=n_recs)
        st.success(f"Movies similar to **{selected_movie}**:")
        st.dataframe(
            recs.rename(columns={
                "clean_title": "Title", "genres": "Genres",
                "year": "Year", "similarity_score": "Similarity"
            }),
            use_container_width=True, hide_index=True
        )

with tab2:
    st.subheader("Personalized recommendations for a user")
    user_ids = sorted(ratings["userId"].unique())
    selected_user = st.selectbox("Pick a user ID", user_ids)
    n_recs_collab = st.slider("Number of recommendations", 5, 20, 10, key="collab_n")

    if st.button("Recommend for this user", type="primary"):
        with st.spinner("Computing personalized recommendations..."):
            recs = collab_model.recommend(selected_user, n=n_recs_collab)
        st.success(f"Top picks for **User {selected_user}**:")
        st.dataframe(
            recs.rename(columns={
                "clean_title": "Title", "genres": "Genres",
                "year": "Year", "predicted_rating": "Predicted Rating"
            })[["Title", "Genres", "Year", "Predicted Rating"]],
            use_container_width=True, hide_index=True
        )

        with st.expander(f"See what User {selected_user} has already rated highly"):
            user_history = ratings[ratings["userId"] == selected_user].merge(
                movies[["movieId", "clean_title", "genres"]], on="movieId"
            ).sort_values("rating", ascending=False).head(10)
            st.dataframe(
                user_history[["clean_title", "genres", "rating"]].rename(
                    columns={"clean_title": "Title", "genres": "Genres", "rating": "Rating"}
                ),
                use_container_width=True, hide_index=True
            )

st.divider()
st.caption("Dataset: MovieLens ml-latest-small (9,742 movies, 100,836 ratings, 610 users)")
