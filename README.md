# Heart Disease Prediction

A complete data science project that predicts heart disease risk from the UCI Cleveland Heart Disease dataset.

## What is included

- `src/app.py` — Flask web application with a polished frontend for risk prediction.
- `src/data_loader.py` — downloads, loads, and cleans the UCI heart disease dataset.
- `src/model_training.py` — trains a Random Forest classifier, evaluates it, and saves a model.
- `src/templates/index.html` — responsive user interface for predictions and dataset preview.
- `src/static/css/style.css` — high-contrast UI styling with accessible colors.
- `notebooks/heart_disease_prediction.ipynb` — interactive analysis notebook.
- `data/heart.csv` — local data cache downloaded from UCI.
- `models/heart_model.pkl` — saved model artifact after training.
- `requirements.txt` — dependencies for backend, model training, and analysis.

## Project features

- Dataset loading and cleaning from the UCI repository
- Binary heart disease prediction (`0` = no disease, `1` = disease)
- Model training with a Random Forest classifier
- Evaluation metrics: accuracy, classification report, confusion matrix
- Frontend form to enter patient values and receive predictions
- Dataset preview table in the web UI
- Modern dark theme with contrast-first styling

## Setup and usage

1. Create and activate a Python virtual environment:

```bash
cd /workspaces/Heart-Disease-Prediction
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and generate the saved artifact:

```bash
python src/model_training.py
```

4. Start the Flask web app:

```bash
python src/app.py
```

5. Open the app in your browser:

```bash
http://localhost:5000
```

6. Optionally run the notebook for deeper analysis:

```bash
jupyter notebook notebooks/heart_disease_prediction.ipynb
```

## Backend and frontend workflow

- The backend loads the dataset and retrains the model at startup.
- The frontend accepts feature input from users and submits the form to Flask.
- Flask predicts the target and returns a result message and confidence score.
- The page also displays a preview of the dataset and the model accuracy.

## Data schema

The model uses the following features:

- `age` — patient age in years
- `sex` — 1 = male, 0 = female
- `cp` — chest pain type
- `trestbps` — resting blood pressure
- `chol` — serum cholesterol
- `fbs` — fasting blood sugar > 120 mg/dL
- `restecg` — resting ECG results
- `thalach` — max heart rate achieved
- `exang` — exercise-induced angina
- `oldpeak` — ST depression induced by exercise
- `slope` — slope of the peak exercise ST segment
- `ca` — number of major vessels colored by fluoroscopy
- `thal` — thalassemia status

## Notes

- The current model is a baseline proof-of-concept and can be improved with feature engineering or hyperparameter tuning.
- The app uses a locally cached dataset stored in `data/heart.csv`.
- Retraining is fast for this dataset and happens on app startup.

## Next improvements

- Add API endpoints for external predictions
- Implement persistent model storage and incremental retraining
- Add visualization charts inside the frontend
- Add form validation and better UX messages
