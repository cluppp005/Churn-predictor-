import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_raw() -> tuple[list[dict], list[dict]]:
    with open(RAW_DIR / "movies.json") as f:
        movies = json.load(f)
    with open(RAW_DIR / "users.json") as f:
        users = json.load(f)
    return movies, users


def build_movie_index(movies: list[dict]) -> dict[str, dict]:
    return {m["imdbID"]: m for m in movies}


def parse_runtime(runtime_str: str) -> float:
    try:
        return float(runtime_str.replace(" min", "").strip())
    except Exception:
        return np.nan


def parse_imdb_rating(rating_str: str) -> float:
    try:
        return float(rating_str)
    except Exception:
        return np.nan


def build_features(users: list[dict], movie_index: dict[str, dict]) -> pd.DataFrame:
    records = []
    now = pd.Timestamp("2024-06-01")

    for u in users:
        join_date = pd.Timestamp(u["join_date"])
        last_active = pd.Timestamp(u["last_active"])
        days_since_join = (now - join_date).days
        days_since_last_watch = (now - last_active).days

        ratings = u["ratings"]
        watched_ids = u["watched_movies"]
        watched_movies = [movie_index[mid] for mid in watched_ids if mid in movie_index]

        # Feature 1: avg_rating_given — mean rating the user assigns
        avg_rating_given = float(np.mean(ratings)) if ratings else 0.0

        # Feature 2: rating_std — how varied the user's ratings are
        rating_std = float(np.std(ratings)) if len(ratings) > 1 else 0.0

        # Feature 3: watch_rate_per_week — movies watched per week since joining
        weeks_active = max(days_since_join / 7, 1)
        watch_rate_per_week = len(watched_ids) / weeks_active

        # Feature 4: days_since_last_watch — recency of last activity
        days_since_last_watch_val = float(days_since_last_watch)

        # Feature 5: genre_diversity — number of unique genres watched
        all_genres = []
        for m in watched_movies:
            all_genres.extend(m.get("Genre", "").split(", "))
        genre_diversity = float(len(set(g for g in all_genres if g)))

        # Feature 6: preferred_genre_ratio — fraction of watched movies matching preferred genres
        preferred = set(u["preferred_genres"])
        matches = sum(
            1 for m in watched_movies
            if any(g in m.get("Genre", "") for g in preferred)
        )
        preferred_genre_ratio = matches / len(watched_movies) if watched_movies else 0.0

        # Feature 7: avg_imdb_rating_watched — avg IMDb rating of movies the user chose
        imdb_ratings = [
            parse_imdb_rating(m.get("imdbRating", "N/A"))
            for m in watched_movies
        ]
        imdb_ratings = [r for r in imdb_ratings if not np.isnan(r)]
        avg_imdb_rating_watched = float(np.mean(imdb_ratings)) if imdb_ratings else 0.0

        # Feature 8: avg_runtime_watched — avg runtime in minutes of watched movies
        runtimes = [
            parse_runtime(m.get("Runtime", "N/A"))
            for m in watched_movies
        ]
        runtimes = [r for r in runtimes if not np.isnan(r)]
        avg_runtime_watched = float(np.mean(runtimes)) if runtimes else 0.0

        # Feature 9: recent_watch_count — movies watched in last 30 days
        recent_watch_count = float(u["recent_watch_count"])

        # Feature 10: account_age_days — how long the user has been registered
        account_age_days = float(days_since_join)

        records.append({
            "user_id": u["user_id"],
            "avg_rating_given": avg_rating_given,
            "rating_std": rating_std,
            "watch_rate_per_week": watch_rate_per_week,
            "days_since_last_watch": days_since_last_watch_val,
            "genre_diversity": genre_diversity,
            "preferred_genre_ratio": preferred_genre_ratio,
            "avg_imdb_rating_watched": avg_imdb_rating_watched,
            "avg_runtime_watched": avg_runtime_watched,
            "recent_watch_count": recent_watch_count,
            "account_age_days": account_age_days,
            "churned": u["churned"],
        })

    return pd.DataFrame(records)


FEATURE_COLS = [
    "avg_rating_given",
    "rating_std",
    "watch_rate_per_week",
    "days_since_last_watch",
    "genre_diversity",
    "preferred_genre_ratio",
    "avg_imdb_rating_watched",
    "avg_runtime_watched",
    "recent_watch_count",
    "account_age_days",
]


if __name__ == "__main__":
    movies, users = load_raw()
    movie_index = build_movie_index(movies)
    df = build_features(users, movie_index)
    print(df[FEATURE_COLS + ["churned"]].describe())
    print(f"\nChurn rate: {df['churned'].mean():.1%}")
    df.to_csv(RAW_DIR / "features.csv", index=False)
    print(f"Saved features.csv with {len(df)} rows and {len(FEATURE_COLS)} features")