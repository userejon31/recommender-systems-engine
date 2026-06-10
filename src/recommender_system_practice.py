"""MovieLens recommender system practice.

This module contains the exercises from the assignment in executable form:
data loading, user-based collaborative filtering, and content-based filtering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Resolve the dataset location relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

movies_dataset = pd.read_csv(DATA_DIR / "movies.csv")
links_dataset = pd.read_csv(DATA_DIR / "links.csv")
ratings_dataset = pd.read_csv(DATA_DIR / "ratings.csv")
tags_dataset = pd.read_csv(DATA_DIR / "tags.csv")


# ---------------------------------------------------------------------------
# Activity 1 - Basic exploration counts
# ---------------------------------------------------------------------------

# Distinct users in the ratings file.
num_users = ratings_dataset["userId"].nunique()

# Distinct movies in the catalog.
num_movies = movies_dataset["movieId"].nunique()

# Number of rating events.
num_ratings = ratings_dataset.shape[0]

# Number of tag events.
num_tags = tags_dataset.shape[0]


# ---------------------------------------------------------------------------
# Activity 2 - User-item matrix
# ---------------------------------------------------------------------------

# Build the user-item matrix.
user_dataset = ratings_dataset.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean",
    fill_value=0,
)


# ---------------------------------------------------------------------------
# Activity 4 - Distance and similarity metrics
# ---------------------------------------------------------------------------

def mink_distance(data: pd.DataFrame, user1: int, user2: int, p: int = 2) -> float:
    """Compute the Minkowski distance between two users.

    Parameters
    ----------
    data:
        User-item matrix where rows are users and columns are movies.
    user1, user2:
        Row labels of the two users to compare.
    p:
        Minkowski order. When p=2 the distance becomes the Euclidean distance.

    Returns
    -------
    float
        The Minkowski distance between the two user rating vectors.

    Notes
    -----
    The mathematical formula is:

        d(x, y) = (sum_i |x_i - y_i|^p)^(1/p)

    where x and y are the vectors of ratings for the two users.
    """

    # Convert the two user profiles to NumPy arrays so vectorized operations can
    # be applied efficiently.
    x1 = np.array(data.loc[user1, :], dtype=float)
    x2 = np.array(data.loc[user2, :], dtype=float)

    # Compute the absolute difference component by component, raise it to p,
    # sum all the contributions, and finally apply the 1/p power.
    distance = np.sum(np.abs(x1 - x2) ** p) ** (1.0 / p)
    return float(distance)


def cos_similarity(data: pd.DataFrame, user1: int, user2: int) -> float:
    """Compute cosine similarity between two users using all columns.

    Parameters
    ----------
    data:
        User-item matrix where rows are users and columns are movies.
    user1, user2:
        Row labels of the two users to compare.

    Returns
    -------
    float
        Cosine similarity between the two vectors.

    Notes
    -----
    Cosine similarity is defined as:

        sim(x, y) = (x dot y) / (||x|| * ||y||)

    The dot product measures alignment, while the norms normalize the vector
    lengths so the result depends on direction rather than scale.
    """

    # Extract the two rating vectors as floating-point NumPy arrays.
    x1 = np.array(data.loc[user1, :], dtype=float)
    x2 = np.array(data.loc[user2, :], dtype=float)

    # Dot product measures how much the two vectors point in the same direction.
    dot_product = np.dot(x1, x2)

    # Norms represent the magnitude of each vector.
    normx1 = norm(x1)
    normx2 = norm(x2)

    # Avoid division by zero if one of the vectors is all zeros.
    if normx1 == 0 or normx2 == 0:
        return 0.0

    similarity = dot_product / (normx1 * normx2)
    return float(similarity)


def similarity(data: pd.DataFrame, user1: int, user2: int) -> float:
    """Compute cosine similarity using only the movies rated by both users.

    Parameters
    ----------
    data:
        User-item matrix where rows are users and columns are movies.
    user1, user2:
        Row labels of the two users to compare.

    Returns
    -------
    float
        Cosine similarity restricted to the intersection of commonly rated
        movies. If there are no common ratings, the function returns 0.

    Notes
    -----
    The practice stores unrated movies as 0. To compare users fairly, we keep
    only the coordinates where both users have a rating greater than 0. This
    avoids penalizing a user for not having seen a movie the other user rated.
    """

    # Retrieve the complete rating profiles for both users.
    x1 = data.loc[user1, :]
    x2 = data.loc[user2, :]

    # common_movies is a Boolean mask: True only where both users rated the
    # same movie. Because unrated values were replaced by 0, the logical test is
    # simply "greater than zero" for both vectors.
    common_movies = (x1 > 0) & (x2 > 0)

    # Keep only the overlapping ratings.
    x1 = np.array(x1[common_movies], dtype=float)
    x2 = np.array(x2[common_movies], dtype=float)

    # If there are no common movies, the similarity is undefined. Returning 0
    # is a practical default for recommender ranking.
    if x1.size == 0 or x2.size == 0:
        return 0.0

    # Compute cosine similarity on the reduced vectors.
    dot_product = np.dot(x1, x2)
    normx1 = norm(x1)
    normx2 = norm(x2)

    if normx1 == 0 or normx2 == 0:
        return 0.0

    similarity_value = dot_product / (normx1 * normx2)
    return float(similarity_value)


# ---------------------------------------------------------------------------
# Activities 6, 7, and 8 - User ranking and collaborative recommendations
# ---------------------------------------------------------------------------

def top_users(
    data: pd.DataFrame,
    user1: int,
    n: int = 5,
    similarity_function=similarity,
) -> list[tuple[float, int]]:
    """Return the top-N users most similar to a target user.

    Parameters
    ----------
    data:
        User-item matrix.
    user1:
        Target user ID.
    n:
        Number of neighbors to return.
    similarity_function:
        Function used to compare users. By default it uses the corrected
        cosine similarity restricted to common movies.

    Returns
    -------
    list[tuple[float, int]]
        A list of (similarity_score, user_id) tuples ordered from highest to
        lowest similarity.
    """

    # Compute the similarity between the target user and every other user.
    score: list[tuple[float, int]] = []
    for user2 in data.index:
        if user2 != user1:
            value = similarity_function(data, user1, user2)
            score.append((value, user2))

    # Sort by similarity in descending order, so the best match appears first.
    score = sorted(score, key=lambda item: item[0], reverse=True)
    return score[:n]


def recommend_from_best_user(
    data: pd.DataFrame,
    movies_data: pd.DataFrame,
    target_user: int,
    n_neighbors: int = 5,
    top_n_movies: int = 3,
) -> pd.DataFrame:
    """Recommend movies using only the single most similar user.

    This helper implements the logic requested in Activity 7.

    Parameters
    ----------
    data:
        User-item matrix.
    movies_data:
        Movie metadata table with movieId and title columns.
    target_user:
        User ID for whom recommendations are generated.
    n_neighbors:
        Number of nearest users to inspect. The recommendation itself uses only
        the best-ranked neighbor, as in the assignment.
    top_n_movies:
        Number of recommendations to keep after sorting by rating.

    Returns
    -------
    pd.DataFrame
        Table with the recommended movie IDs, titles, and the similar user's
        rating.
    """

    # Obtain the ordered list of similar users and keep only the best one.
    similar_users = top_users(data, target_user, n=n_neighbors)
    best_similarity, best_user = similar_users[0]

    # The target and neighbor profiles are Series indexed by movieId.
    target_profile = data.loc[target_user, :]
    similar_profile = data.loc[best_user, :]

    movies_to_suggest: list[tuple[int, float]] = []

    # Iterate over every movie in the matrix.
    for movie_id in data.columns:
        target_rating = target_profile[movie_id]
        similar_rating = similar_profile[movie_id]

        # We only propose movies the target user has not rated but the similar
        # user has rated positively.
        if target_rating == 0 and similar_rating > 0:
            movies_to_suggest.append((movie_id, float(similar_rating)))

    # Sort by the similar user's rating in descending order and keep the best
    # candidates.
    movies_to_suggest = sorted(movies_to_suggest, key=lambda item: item[1], reverse=True)
    movies_to_suggest = movies_to_suggest[:top_n_movies]

    recommended_ids = [movie_id for movie_id, _ in movies_to_suggest]

    # Join the recommendation IDs with the movie catalog so titles are visible.
    recommended_movies = movies_data[movies_data["movieId"].isin(recommended_ids)][
        ["movieId", "title"]
    ].copy()

    # Preserve the ranking order from the recommendation list.
    recommended_movies["similar_user_rating"] = recommended_movies["movieId"].map(
        dict(movies_to_suggest)
    )
    recommended_movies = recommended_movies.sort_values(
        by="similar_user_rating", ascending=False
    ).reset_index(drop=True)

    # The similarity value is useful for debugging or reporting, so we include
    # it as an attribute in case the caller wants to inspect it later.
    recommended_movies.attrs["best_user"] = best_user
    recommended_movies.attrs["best_similarity"] = best_similarity

    return recommended_movies


def recommend_from_top_users(
    data: pd.DataFrame,
    movies_data: pd.DataFrame,
    target_user: int,
    n_neighbors: int = 5,
    top_n_movies: int = 5,
) -> pd.DataFrame:
    """Recommend movies by averaging ratings from the top similar users.

    This helper implements the logic requested in Activity 8.

    Parameters
    ----------
    data:
        User-item matrix.
    movies_data:
        Movie metadata table with movieId and title columns.
    target_user:
        User ID for whom recommendations are generated.
    n_neighbors:
        Number of similar users to consider.
    top_n_movies:
        Number of final recommendations to return.

    Returns
    -------
    pd.DataFrame
        Table with movie IDs, titles, and the average rating among the similar
        users who rated each candidate movie.
    """

    similar_users = top_users(data, target_user, n=n_neighbors)
    target_profile = data.loc[target_user, :]

    # candidate_movies collects, for each movie, the ratings given by similar
    # users. The structure is:
    #
    #     {movie_id: [rating_from_user_A, rating_from_user_B, ...]}
    candidate_movies: dict[int, list[float]] = {}

    for sim_value, user_id in similar_users:
        similar_profile = data.loc[user_id, :]

        for movie_id in data.columns:
            target_rating = target_profile[movie_id]
            similar_rating = similar_profile[movie_id]

            # Keep only movies unseen by the target user and positively rated by
            # the similar user.
            if target_rating == 0 and similar_rating > 0:
                if movie_id not in candidate_movies:
                    candidate_movies[movie_id] = []
                candidate_movies[movie_id].append(float(similar_rating))

    movies_to_suggest: list[tuple[int, float]] = []

    # Convert the list of ratings into a single average score for each movie.
    for movie_id, ratings in candidate_movies.items():
        avg_rating = float(np.mean(ratings))
        movies_to_suggest.append((movie_id, avg_rating))

    movies_to_suggest = sorted(movies_to_suggest, key=lambda item: item[1], reverse=True)
    movies_to_suggest = movies_to_suggest[:top_n_movies]

    recommended_ids = [movie_id for movie_id, _ in movies_to_suggest]
    recommended_movies = movies_data[movies_data["movieId"].isin(recommended_ids)][
        ["movieId", "title"]
    ].copy()
    recommended_movies["average_similar_user_rating"] = recommended_movies["movieId"].map(
        dict(movies_to_suggest)
    )
    recommended_movies = recommended_movies.sort_values(
        by="average_similar_user_rating", ascending=False
    ).reset_index(drop=True)

    return recommended_movies


# Example objects matching the activity wording for the target user 1.
similar_users_user_1 = top_users(user_dataset, 1, n=5)
best_user_recommendations = recommend_from_best_user(
    user_dataset, movies_dataset, target_user=1, n_neighbors=5, top_n_movies=3
)
top_users_recommendations = recommend_from_top_users(
    user_dataset, movies_dataset, target_user=1, n_neighbors=5, top_n_movies=5
)


# ---------------------------------------------------------------------------
# Activities 10, 11, and 12 - Content-based filtering
# ---------------------------------------------------------------------------

# Split every genre string on the pipe character and collect the unique genre
# labels that appear in the catalog.
#
# The list is sorted only to keep the output deterministic and easy to inspect.
genre_list = sorted(
    {
        genre
        for genres in movies_dataset["genres"].str.split("|")
        for genre in genres
        if genre != "(no genres listed)"
    }
)

# Build a feature table with the movie metadata and one binary column per genre.
# Each 1 means the movie belongs to that genre; each 0 means it does not.
movie_features = movies_dataset[["movieId", "title", "genres"]].copy().reset_index(drop=True)

for genre in genre_list:
    # str.contains performs a string search inside the genre field. Because the
    # genres are pipe-separated labels, using regex=False prevents special
    # characters such as hyphens from being treated as regular-expression syntax.
    movie_features[genre] = movie_features["genres"].str.contains(genre, regex=False).astype(int)

# Keep only the genre columns for the vector-space comparison.
genre_matrix = movie_features[genre_list].copy()


def movie_similarity(
    features: pd.DataFrame,
    movie1_index: int,
    movie2_index: int,
) -> float:
    """Compute cosine similarity between two movies using genre features.

    Parameters
    ----------
    features:
        Matrix where each row is a movie and each column is a genre feature.
    movie1_index, movie2_index:
        Positional indexes of the two movies to compare.

    Returns
    -------
    float
        Cosine similarity between the two movie vectors.
    """

    # Convert the genre rows into numeric vectors.
    x1 = np.array(features.iloc[movie1_index, :], dtype=float)
    x2 = np.array(features.iloc[movie2_index, :], dtype=float)

    # Compute cosine similarity using the same mathematical structure as the
    # user-based version.
    dot_product = np.dot(x1, x2)
    normx1 = norm(x1)
    normx2 = norm(x2)

    if normx1 == 0 or normx2 == 0:
        return 0.0

    similarity_value = dot_product / (normx1 * normx2)
    return float(similarity_value)


def similar_movies(
    movie_features: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    movie_title: str,
    n: int = 5,
) -> pd.DataFrame:
    """Find the n movies most similar to a given movie title.

    Parameters
    ----------
    movie_features:
        Table that contains movie IDs, titles, and genre metadata.
    genre_matrix:
        Matrix of genre features used for similarity calculations.
    movie_title:
        Title of the reference movie.
    n:
        Number of similar movies to return.

    Returns
    -------
    pd.DataFrame
        The top-n most similar movies with their similarity score.
    """

    # Locate the movie row by title. The assignment usually expects a single
    # exact match.
    matches = movie_features.index[movie_features["title"] == movie_title].tolist()
    if not matches:
        raise ValueError(f"Movie title not found: {movie_title}")

    movie_index = matches[0]

    similarities: list[tuple[int, str, float]] = []

    for other_index in range(len(movie_features)):
        if other_index == movie_index:
            continue

        score = movie_similarity(genre_matrix, movie_index, other_index)
        similarities.append(
            (
                int(movie_features.loc[other_index, "movieId"]),
                str(movie_features.loc[other_index, "title"]),
                float(score),
            )
        )

    similarities = sorted(similarities, key=lambda item: item[2], reverse=True)
    similarities = similarities[:n]

    return pd.DataFrame(similarities, columns=["movieId", "title", "similarity"])


# Example object matching the activity wording.
example_movie_similarity = movie_similarity(genre_matrix, 0, 1)


if __name__ == "__main__":
    # Print the core outputs so the script can be run directly if desired.
    print("num_users:", num_users)
    print("num_movies:", num_movies)
    print("num_ratings:", num_ratings)
    print("num_tags:", num_tags)
    print("user_dataset shape:", user_dataset.shape)
    print("top users for user 1:", similar_users_user_1)
    print("best-user recommendations for user 1:\n", best_user_recommendations)
    print("top-users recommendations for user 1:\n", top_users_recommendations)
    print("example movie similarity between rows 0 and 1:", example_movie_similarity)