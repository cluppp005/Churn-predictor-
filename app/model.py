import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import build_features, load_raw, build_movie_index, FEATURE_COLS

MODEL_PATH = Path(__file__).parent.parent / "data" / "model.pkl"


def train(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[FEATURE_COLS]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["active", "churned"]))
    return model


def save(model: RandomForestClassifier):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load() -> RandomForestClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run model.py first.")
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    movies, users = load_raw()
    movie_index = build_movie_index(movies)
    df = build_features(users, movie_index)
    model = train(df)
    save(model)