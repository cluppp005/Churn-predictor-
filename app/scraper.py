import os
import json
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
BASE_URL = "http://www.omdbapi.com/"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_TERMS = [
    "action", "comedy", "drama", "thriller", "horror",
    "romance", "sci-fi", "animation", "crime", "adventure"
]

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Horror",
          "Romance", "Sci-Fi", "Animation", "Crime", "Adventure"]


def fetch_movies(max_per_term: int = 10) -> list[dict]:
    movies = {}
    for term in SEARCH_TERMS:
        resp = requests.get(BASE_URL, params={
            "apikey": OMDB_API_KEY,
            "s": term,
            "type": "movie",
        })
        data = resp.json()
        if data.get("Response") != "True":
            continue
        for item in data.get("Search", [])[:max_per_term]:
            detail = requests.get(BASE_URL, params={
                "apikey": OMDB_API_KEY,
                "i": item["imdbID"],
                "plot": "short",
            }).json()
            if detail.get("Response") == "True":
                movies[item["imdbID"]] = detail
    return list(movies.values())


def simulate_users(movies: list[dict], n_users: int = 300, seed: int = 42) -> list[dict]:
    random.seed(seed)
    users = []
    now = datetime(2024, 6, 1)

    for uid in range(n_users):
        join_date = now - timedelta(days=random.randint(90, 730))
        days_active = (now - join_date).days

        # Simulate last activity — churned users have longer inactivity
        is_churned = random.random() < 0.35
        if is_churned:
            last_active = now - timedelta(days=random.randint(61, 300))
        else:
            last_active = now - timedelta(days=random.randint(0, 30))

        # Genre preference (1-2 preferred genres per user)
        preferred_genres = random.sample(GENRES, k=random.randint(1, 3))

        # Sample movies watched — churned users watched fewer recently
        pool_size = max(5, int(days_active / 10))
        upper = max(3, min(pool_size, len(movies)))
        watched_count = random.randint(3, upper)
        watched = random.sample(movies, k=watched_count)

        # Assign ratings biased by genre match
        ratings = []
        for m in watched:
            movie_genres = m.get("Genre", "")
            genre_match = any(g in movie_genres for g in preferred_genres)
            base = random.gauss(7.0 if genre_match else 5.5, 1.2)
            ratings.append(round(max(1.0, min(10.0, base)), 1))

        # Recent activity: movies watched in last 30 days
        recent_watch_count = 0 if is_churned else random.randint(1, 8)

        users.append({
            "user_id": f"user_{uid:04d}",
            "join_date": join_date.isoformat(),
            "last_active": last_active.isoformat(),
            "preferred_genres": preferred_genres,
            "watched_movies": [m["imdbID"] for m in watched],
            "ratings": ratings,
            "recent_watch_count": recent_watch_count,
            "churned": int(is_churned),
        })

    return users


def save_raw(movies: list[dict], users: list[dict]):
    with open(RAW_DIR / "movies.json", "w") as f:
        json.dump(movies, f, indent=2)
    with open(RAW_DIR / "users.json", "w") as f:
        json.dump(users, f, indent=2)
    print(f"Saved {len(movies)} movies and {len(users)} users to {RAW_DIR}")


if __name__ == "__main__":
    if not OMDB_API_KEY:
        raise SystemExit("ERROR: OMDB_API_KEY not set. Check your .env file.")

    print("Fetching movies from OMDb...")
    movies = fetch_movies(max_per_term=10)
    print(f"Fetched {len(movies)} movies")

    if not movies:
        raise SystemExit("ERROR: No movies fetched. Check your API key is valid.")

    print("Simulating users...")
    users = simulate_users(movies, n_users=300)
    churned = sum(u["churned"] for u in users)
    print(f"Simulated {len(users)} users ({churned} churned, {len(users)-churned} active)")

    save_raw(movies, users)