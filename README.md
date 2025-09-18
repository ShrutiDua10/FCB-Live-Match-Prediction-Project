# FCB-Live-Match-Prediction-Project


## 1. Introduction

This project predicts **live football match outcomes** every 3 minutes for FC Barcelona using both structured match statistics and computer vision–extracted features. I have seen the 
increase in use of Computer Vision (CV) in sports analytics to track players, the ball, and their trajectories to analyse various soccer games. I would like to see if this analysis can be converted to data 
to help predict game wins in real time. I am focusing on my favorite soccer club for the project FCB specific to the local La Liga league. 
The models will be trained on matches of FCB with other teams in the league of seasons from the past 10 years from the season 2014-2015 to now. 
Once complete this model could be expanded to incorportate more teams and leagues and make a real time prediction system for any match provided computing power allows to run models ran on that much data!

I structured the project in 2 big steps. 
- Building the FBref-only ML model first to get an understanding of the match specific statistics we have available.
- Building CV models that extract features from the match video and including them in the ML model explored in the above step

---

## 2. ML Production (FBref Stats Only)

### 2.1 Data

**Features** from **FBref match statistics** via the [`soccerdata`](https://pypi.org/project/soccerdata/) Python Package:
- We will extract the match statistics using the read_team_match_stats() filtered to FCB. This includes **schedule statistics** like date, opponent, venue (Home/Away), result (W/D/L), goals_for, goals_against,
**shooting statistics** like xG (expected goals), shots, shots_on_target, **possesion and passing statistics** like passes_completed, pass accuracy,  **defense statistics** like tackles, blocks and finally **opponent statistics** which gives similar statistics but for the opponent team FCB is playing against.

**Engineered Features**:
- Club rating is a feature that can be calculated through the process of ELO rating (https://krinya.medium.com/understanding-the-elo-rating-system-a-practical-example-on-the-english-premier-league-using-python-59d56ea19d9d). This rating can be caluclated manually as the article shows or we can use soccerdata that extracts similar rating from Club Elo. Currently leaning toward manually calculating to learn more about rating algorithms.
- Home advantage
- Season phase - Early/Mid/Late to incorporate pressure the team might be under.
- Differential statistics comparing FCB's stats with opponents.

### 2.2 Model Training, Validation, and Testing Strategy

We will be predicting if FCB will win or not or if FCB will have a W/D/L making this a classification problem. We can further advance this to regression problems like how many goals FCB or the opponent will score, how many goals saved etc.

- Models to review: Logistic Regression, Random Forest, XGBoost/LightGBM. Additionally, we will be exploring Baeysian approaches to statistical learning as well through the PyMC package since our tabular data is small of around 400 matches only
- Targets: Binary (Win/Not Win), Multiclass (Home Win / Draw / Away Win)
- Validation: I will be slightly adjusting the basic Stratified k-fold Cross Validation and random search hyperparamter tuning. We use random search over a small hyperparameter space (30–50 configurations), combined with walk-forward time-series cross-validation for each fold.
We’ll train on past matches and test on future ones (by kickoff date) using time‑series splits with k‑fold‑style cross‑validation for hyperparameter tuning, while adding a small embargo gap so no future information leaks into training since we might incorporate rolling averages from the past 3 days. This avoids overfitting while efficiently exploring the best model settings.
- Metrics: log loss, accuracy, AUC

### 2.3 Deployment (FBref-only)
I will be doing cloud deployment for just this model along with the overall model with video features as well to get more practice on good deployment and monitoring
- FastAPI + Uvicorn - to define a predict end point and uvicorn for server
- Docker to containerize and package my model
- Will deploy on AWS EC2
- Optional: As i have very less experience in monitoring and automation of my deplpyed models, I would like to incorporate it here. I will possibly use AWS CloudWatch for server monitoing and MLflow to log model training runs, track hyperparameters and metrics, and version deployed models.
- For automation, I would like to explore a GitHub Actions CI/CD pipeline.

**API**:
- `POST /predict_fbref_only`: FBref stats → prediction
- `GET /health`: health check

**Input Example:**
```json
{
  "xg_diff": 0.75,
  "possession_diff": 8.5,
  "shots_diff": 4,
  "home_adv": 1
}
```

**Output:**
```json
{
  "predicted_class": "Win",
  "probabilities": {
    "Win": 0.62,
    "Draw": 0.23,
    "Loss": 0.15
  }
}
```
Since I am focusing on this project in 2 parts, I have created a complete pipeline just for prediction with fbref statistics as well.


```
fcb_fbref_model/
├── main.py                    # FastAPI app entrypoint
├── config.py                  # Configs
│
├── fbref_pipeline/
│   ├── data_loader.py         # Load FBref stats using soccerdata
│   ├── feature_engineering.py # Clean, create differentials, rolling features
│   ├── model_training.py      # time series CV, tuning, training
│   ├── predictor.py           # Load model, run inference
│
├── api/
│   ├── routes.py              # API routes: /predict_fbref_only, /health
│   ├── schemas.py             # Pydantic input/output models
│
├── models/                    # Trained model.pkl, scaler.pkl
├── data/                      # Raw + processed FBref data
├── scripts/                   # CLI tools (e.g., run training, fetch data)
├── notebooks/                 # EDA, model evaluation, charts
├── tests/                     # Unit + integration tests
├── requirements.txt           # Dependencies
└── Makefile                   # Setting up virtual env and docker build

```
---

## 3. Computer Vision Production

### 3.1 Roboflow Dataset

- Annotated objects: `player_home`, `player_away`, `ball`, `goalpost`, `referee`
- Roboflow export: YOLOv8 or COCO format
- Tracking: ByteTrack/StrongSORT
- Homography: field calibration via line detection or manual config

### 3.2 CV Models & Features

- Detector: YOLOv8 / RF-DETR (Roboflow-trained)
- Tracker: ByteTrack + team assignment via color clustering
- Frame features:
  - Ball → goal distance
  - Ball → nearest player (team)
  - Possession proxy
  - Final third entries
  - Defensive line height, width, compactness
  - Ball speed, shot heuristics

### 3.3 Rolling Aggregation + Prediction

- 3-min rolling windows, updated every 1 min
- Features aggregated and merged with FBref context
- Output probability: Win/Draw/Loss

**CSV Output (per window):**
```
match_id, team, window_start, ball_goal_dist_mean, box_entries, sprints, ...
```

---

## 4. Full Deployment Stack



- `FastAPI` app serves predictions every 1 min
- Fallback to FBref-only model if CV quality low
- Deployed via Docker

---

## 5. Project Structure

```
fcb_match_prediction/
├── main.py                # FastAPI entrypoint
├── config.py              # Global configs
│
├── fbref_pipeline/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predictor.py
│
├── cv_pipeline/
│   ├── video_reader.py
│   ├── detector.py
│   ├── tracker.py
│   ├── field_mapper.py
│   ├── feature_extractor.py
│   ├── window_aggregator.py
│
├── services/
│   ├── prediction_service.py
│   ├── fallback_handler.py
│
├── api/
│   ├── routes.py
│   ├── schemas.py
│
├── models/                # Trained model.pkl, scaler.pkl
├── data/                  # Raw FBref + CV feature data
├── notebooks/             # EDA + training
├── tests/                 # Unit tests
├── requirements.txt
└── README.md
```
