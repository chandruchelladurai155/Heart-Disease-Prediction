# Heart Disease Prediction

A local-ready data science project that predicts heart disease risk using the UCI Cleveland Heart Disease dataset.

## What is included

- `run.py` — local app launcher for the Flask frontend.
- `train.py` — local training launcher to build and save the model.
- `src/app.py` — Flask backend and frontend integration.
- `src/data_loader.py` — dataset downloader, loader, and cleaner.
- `src/model_training.py` — model training, evaluation, and model persistence.
- `src/templates/index.html` — responsive user interface for entering patient data.
- `src/static/css/style.css` — modern theme with high-contrast styling.
- `notebooks/heart_disease_prediction.ipynb` — interactive analysis notebook.
- `requirements.txt` — Python dependency list.

## Project features

- Automatic dataset download from the UCI repository if the local file is missing.
- Binary heart disease risk prediction (`0` = no disease, `1` = disease).
- Random Forest model training and evaluation.
- Flask frontend with a prediction form, dataset preview, and score display.
- Accessible, contrast-focused UI with dropdown and numeric inputs.
- Local-ready startup scripts for simplified execution.

## Setup on a local PC

1. Clone the repository.

```bash
git clone https://github.com/chandruchelladurai155/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. Create and activate a Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Train the model locally (this downloads the dataset and saves the model under `models/`).

```bash
python train.py
```

5. Start the web app locally.

```bash
python run.py
```

6. Open your browser and visit:

```bash
http://localhost:5000
```

## Alternate local commands

If you prefer, these direct commands also work:

- Train directly: `python src/model_training.py`
- Run the app directly: `python src/app.py`
- Open analysis notebook: `jupyter notebook notebooks/heart_disease_prediction.ipynb`

## Local workflow

- `train.py` downloads the dataset and trains the model if needed.
- `run.py` launches the Flask app with correct local paths.
- The frontend sends feature data to Flask, which predicts heart disease risk.
- The app displays prediction results and dataset preview.

## Data schema

The model uses these features:

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

- The app is configured for local execution and should work from the project root.
- The dataset is downloaded automatically if missing.
- The model is saved to `models/heart_model.pkl` after training.

## Recommended next steps

- Improve model accuracy with feature engineering or hyperparameter tuning.
- Add charts or data visualizations to the frontend.
- Add an API endpoint for external prediction integration.
