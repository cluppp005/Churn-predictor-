# churn-predictor

Customer churn prediction system for a simulated media streaming platform.
Built with OMDb API data, scikit-learn, FastAPI, and Docker.

Final project — Introduction to Data Science
Universidad Politécnica Taiwan-Paraguay (UPTP)

---

## Stack

- Python 3.11
- FastAPI + Uvicorn
- scikit-learn (Random Forest)
- OMDb API
- Docker + docker-compose

---

## Project Structure

```
churn-predictor/
├── app/
│   ├── scraper.py       # fetches OMDb data + simulates 300 users
│   ├── features.py      # engineers 10 behavioral features
│   ├── model.py         # trains and saves the classifier
│   └── main.py          # FastAPI /predict endpoint
├── notebooks/
│   └── analysis.ipynb   # EDA + 4 feature selection methods
├── data/
│   ├── raw/             # movies.json, users.json, features.csv (generated)
│   └── model.pkl        # trained model (generated)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/your-username/churn-predictor.git
cd churn-predictor
cp .env.example .env
# edit .env and add your OMDb API key
```

Get a free OMDb API key at https://www.omdbapi.com/apikey.aspx
Make sure to activate it via the confirmation email.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Fetch data and train the model

Run these in order:

```bash
python app/scraper.py    # fetches movies, simulates users → data/raw/
python app/features.py   # engineers features → data/raw/features.csv
python app/model.py      # trains classifier → data/model.pkl
```

### 4. Run the API locally

```bash
python -m uvicorn app.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 5. Run with Docker

```bash
docker-compose up --build
```

---

## API

### GET /health
```json
{ "status": "ok" }
```

### POST /predict
Request body:
```json
{
  "avg_rating_given": 5.2,
  "rating_std": 1.4,
  "watch_rate_per_week": 0.8,
  "days_since_last_watch": 75,
  "genre_diversity": 3.0,
  "preferred_genre_ratio": 0.45,
  "avg_imdb_rating_watched": 6.8,
  "avg_runtime_watched": 102.0,
  "recent_watch_count": 0.0,
  "account_age_days": 320.0
}
```

Response:
```json
{
  "churned": true,
  "churn_probability": 0.7831
}
```

---

## Notebook

The analysis notebook covers:
- EDA and feature distributions
- Feature engineering justification
- 4 feature selection methods: Filter (ANOVA), RFE, Decision Tree, Random Forest
- Comparison table and rank heatmap
- Retention analysis and intervention recommendations

```bash
python -m pip install jupyter matplotlib seaborn
jupyter notebook notebooks/analysis.ipynb
```
