import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from surprise import Dataset, Reader, SVD

from .models import UserRating
from library.models import Cartoon
from django.contrib.auth import get_user_model

from fcmeans import FCM
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

User = get_user_model()


# -------------------------
# Utility: fetch cartoon objects in the order of ids list
# -------------------------
def fetch_cartoons_by_id_ordered(id_list):
    """
    Given a list of Cartoon.id (primary keys), return a list of Cartoon model
    objects in the same order (skip missing).
    """
    if not id_list:
        return []

    cartoons_qs = Cartoon.objects.filter(id__in=id_list)
    cartoons_map = {c.id: c for c in cartoons_qs}
    ordered = [cartoons_map.get(i) for i in id_list if cartoons_map.get(i) is not None]
    return ordered


# LOAD & PREPROCESS DATA
def load_data():
    """
    Returns:
        ratings: pandas.DataFrame with columns ['user_id', 'cartoon_id', 'rating']
        cartoons: pandas.DataFrame with columns matching Cartoon fields
    Notes:
        - ratings.rating coerced to numeric (non-numeric -> NaN -> filled to 0.0)
        - cartoons.rating coerced to float; missing cartoon ratings replaced with global mean
    """
    # ratings
    ratings_qs = UserRating.objects.all().values("user_id", "cartoon_id", "rating")
    ratings = pd.DataFrame.from_records(ratings_qs)

    # cartoons
    cartoons = pd.DataFrame.from_records(Cartoon.objects.all().values(
        "id", "tvmaze_id", "name", "summary", "genres", "rating", "image_medium", "image_original", "popularity"
    ))

    # Ensure cartoons columns exist
    if cartoons.empty:
        cartoons = pd.DataFrame(columns=[
            "id", "tvmaze_id", "name", "summary", "genres", "rating", "image_medium", "image_original", "popularity"
        ])

    # Clean text fields
    cartoons["summary"] = cartoons["summary"].fillna("").astype(str)
    cartoons["genres"] = cartoons["genres"].fillna("").astype(str)

    # Coerce ratings to numeric and replace missing with global mean (safer than 0)
    cartoons["rating"] = pd.to_numeric(cartoons.get("rating"), errors="coerce")
    if cartoons["rating"].notna().any():
        mean_rating = float(cartoons["rating"].mean(skipna=True))
    else:
        mean_rating = 0.0
    cartoons["rating"] = cartoons["rating"].fillna(mean_rating).astype(float)

    # Ratings table: coerce rating numeric, fill NaN -> 0.0 (user ratings may be optional)
    if not ratings.empty:
        ratings["rating"] = pd.to_numeric(ratings.get("rating"), errors="coerce").fillna(0.0).astype(float)

    return ratings, cartoons


# COLLABORATIVE FILTERING (SVD)
def collaborative_filtering(user_id, top_n=10):
    ratings, _ = load_data()

    # Fallback to top-rated if no rating data
    if ratings.empty:
        return list(Cartoon.objects.order_by("-rating")[:top_n])

    # If user hasn't rated anything, fallback
    if ratings[ratings["user_id"] == user_id].empty:
        return list(Cartoon.objects.order_by("-rating")[:top_n])

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(ratings[["user_id", "cartoon_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    all_item_ids = ratings["cartoon_id"].unique().tolist()
    seen_ids = ratings[ratings["user_id"] == user_id]["cartoon_id"].tolist()
    unseen = [cid for cid in all_item_ids if cid not in seen_ids]

    preds = []
    for cid in unseen:
        try:
            est = algo.predict(user_id, cid).est
            preds.append((cid, float(est)))
        except Exception:
            continue

    preds.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in preds[:top_n]]

    return fetch_cartoons_by_id_ordered(top_ids)


# CONTENT-BASED FILTERING (TF-IDF)
def content_based_filtering(user_id, top_n=10, max_features=5000):
    ratings, cartoons = load_data()

    # If no cartoons, return safe empty shapes
    if cartoons.empty:
        return [], np.empty((0, 0)), np.array([])

    user_rated = pd.DataFrame()
    if not ratings.empty:
        user_rated = ratings[ratings["user_id"] == user_id].sort_values(by="rating", ascending=False)

    # If user hasn't rated anything, fallback to top-rated cartoons (model instances)
    if user_rated.empty:
        top_ids = cartoons.sort_values("rating", ascending=False).head(top_n)["id"].tolist()
        return fetch_cartoons_by_id_ordered(top_ids), np.empty((0, 0)), np.array([])

    # Build text: summary + genres
    cartoons = cartoons.reset_index(drop=True)
    cartoons["text"] = cartoons["summary"].astype(str) + " " + cartoons["genres"].astype(str)

    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(cartoons["text"])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # mapping id -> index in cartoons df
    id_to_idx = {int(r["id"]): idx for idx, r in cartoons.reset_index().iterrows()}

    seen_ids = user_rated["cartoon_id"].tolist()
    scores = {}

    for cid in seen_ids:
        idx = id_to_idx.get(int(cid))
        if idx is None:
            continue
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i, sim in sim_scores:
            target_id = int(cartoons.iloc[i]["id"])
            if target_id in seen_ids:
                continue
            scores[target_id] = scores.get(target_id, 0.0) + float(sim)

    if not scores:
        # fallback to top-rated if TF-IDF produced nothing
        top_ids = cartoons.sort_values("rating", ascending=False).head(top_n)["id"].tolist()
        return fetch_cartoons_by_id_ordered(top_ids), np.empty((0, 0)), tfidf.get_feature_names_out()

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in sorted_scores[:top_n]]

    # Build features matrix aligned with top_ids (preserve order)
    indices = [id_to_idx[cid] for cid in top_ids if cid in id_to_idx]
    if indices:
        features = tfidf_matrix[indices].toarray().astype(np.float32)
    else:
        features = np.empty((0, 0), dtype=np.float32)

    top_cartoons = fetch_cartoons_by_id_ordered(top_ids)

    return top_cartoons, features, tfidf.get_feature_names_out()


# HYBRID FILTERING (weighted combination)
def hybrid_filtering(user_id, top_n=10, w_collab=0.5, w_content=0.5):
    ratings, _ = load_data()

    # Fallback to top-rated if insufficient data
    if ratings.empty or ratings[ratings["user_id"] == user_id].empty:
        return list(Cartoon.objects.order_by("-rating")[:top_n])

    collab_list = collaborative_filtering(user_id, 50)  # list of Cartoon objects
    content_list, _, _ = content_based_filtering(user_id, 50)  # list of Cartoon objects

    scores = {}

    for idx, c in enumerate(collab_list):
        scores[c.id] = scores.get(c.id, 0.0) + w_collab * (50 - idx)

    for idx, c in enumerate(content_list):
        scores[c.id] = scores.get(c.id, 0.0) + w_content * (50 - idx)

    if not scores:
        return list(Cartoon.objects.order_by("-rating")[:top_n])

    sorted_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    return fetch_cartoons_by_id_ordered(sorted_ids)


# FUZZY C-MEANS CLUSTERING (FCM)
def fuzzy_c_means_clustering(features, n_clusters=4):
    features = np.array(features, dtype=np.float32)
    if features.size == 0:
        return np.array([], dtype=int), np.array([])

    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(features)

    labels = fcm.predict(features)
    centers = fcm.centers
    return np.array(labels, dtype=int), np.array(centers)


# SELF-ORGANIZING MAP (SOM)
def som_clustering(features, x=3, y=3):
    features = np.array(features)
    if features.size == 0 or features.shape[1] == 0:
        return np.zeros(len(features), dtype=int)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    som = MiniSom(x=x, y=y, input_len=scaled.shape[1], sigma=0.5, learning_rate=0.5)
    som.random_weights_init(scaled)
    som.train_random(scaled, 300)

    labels = [som.winner(v)[0] * y + som.winner(v)[1] for v in scaled]
    return np.array(labels, dtype=int)


#DOMINANT FEATURES FROM CLUSTER CENTERS
def extract_dominant_features(cluster_centers, feature_names, top_n=8):
    if cluster_centers is None or len(cluster_centers) == 0:
        return []
    dom = []
    for center in cluster_centers:
        idxs = np.argsort(center)[-top_n:][::-1]
        dom.append([feature_names[i] for i in idxs if i < len(feature_names)])
    return dom


# CLUSTER VALIDATION
def validate_clusters(features, labels):
    features = np.array(features)
    labels = np.array(labels)

    if features.size == 0 or len(set(labels)) < 2:
        return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": 0.0}

    try:
        return {
            "silhouette": float(silhouette_score(features, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
            "davies_bouldin": float(davies_bouldin_score(features, labels))
        }
    except Exception:
        return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": 0.0}


# HYBRID + CLUSTERING (group content-based recs into clusters)
def hybrid_with_clustering(user_id, top_n=12, n_clusters=4):
    ratings, cartoons = load_data()

    # fallback when no user ratings
    if ratings.empty or ratings[ratings["user_id"] == user_id].empty:
        return [{"cluster_name": "Trending Now", "cartoons": list(Cartoon.objects.order_by("-rating")[:top_n])}]

    content_list, features, feature_names = content_based_filtering(user_id, 50)

    if not content_list or features.size == 0:
        # fallback: a couple of simple clusters
        return [
            {"cluster_name": "Top Rated", "cartoons": list(Cartoon.objects.order_by("-rating")[:top_n])},
            {"cluster_name": "Popular Among Users", "cartoons": collaborative_filtering(user_id, top_n)}
        ]

    labels, centers = fuzzy_c_means_clustering(features, n_clusters=n_clusters)
    dominant = extract_dominant_features(centers, feature_names)

    # Validate clusters (optional print)
    _val = validate_clusters(features, labels)

    # Group cartoons by cluster id (clusters may not be contiguous)
    grouped = {}
    for idx, cartoon in enumerate(content_list):
        cluster_id = int(labels[idx]) if idx < len(labels) else 0
        grouped.setdefault(cluster_id, []).append(cartoon)

    # Sort cluster ids to have deterministic order
    cluster_ids_sorted = sorted(grouped.keys())

    cluster_names = ['Top Picks', 'Hidden Gems', 'Action Favourites', 'Heartwarming Picks', 'Trending Mix']

    final = []
    for i, cid in enumerate(cluster_ids_sorted):
        items = grouped[cid]
        keywords = dominant[i] if i < len(dominant) else []
        final.append({
            "cluster_name": cluster_names[i % len(cluster_names)],
            "cartoons": items,
            "keywords": keywords
        })

    # Add a collaborative section at the end
    final.append({"cluster_name": "Popular Among Viewers", "cartoons": collaborative_filtering(user_id, top_n)})

    return final
