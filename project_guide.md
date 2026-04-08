# Heart Disease Prediction Project: Complete Guide

## Project Overview

The Heart Disease Prediction project is a full-stack data science application that predicts the risk of heart disease in patients using machine learning. It combines a Flask web backend, a responsive frontend, and a trained Random Forest model based on the UCI Cleveland Heart Disease dataset.

### Key Features
- **Web Interface**: User-friendly form for entering patient data and viewing predictions.
- **Machine Learning Model**: Random Forest classifier trained on clinical data.
- **Data Visualization**: Preview of the dataset and model performance metrics.
- **Local Execution**: Designed to run on a local PC with simple setup.

### Target Audience
- Data scientists learning end-to-end ML projects.
- Developers interested in Flask and web-based ML apps.
- Medical professionals or students exploring predictive analytics in healthcare.

## Technology Stack

### Backend
- **Python 3.12**: Core programming language.
- **Flask**: Lightweight web framework for the API and frontend rendering.
- **Scikit-learn**: Machine learning library for model training and prediction.
- **Pandas & NumPy**: Data manipulation and numerical computing.
- **Joblib**: Model serialization for saving and loading trained models.

### Frontend
- **HTML5**: Structure of the web pages.
- **CSS3**: Styling with a modern, accessible dark theme.
- **Jinja2**: Templating engine for dynamic content in Flask.

### Data & Tools
- **UCI Cleveland Heart Disease Dataset**: Source of clinical data.
- **Jupyter Notebook**: For exploratory data analysis.
- **Git**: Version control.

### Dependencies
All listed in `requirements.txt`:
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- requests
- joblib

## Project Structure

```
Heart-Disease-Prediction/
├── run.py                 # Main app launcher
├── train.py               # Model training launcher
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── src/
│   ├── app.py             # Flask application
│   ├── data_loader.py     # Dataset handling
│   ├── model_training.py  # ML model training
│   ├── __init__.py        # Package marker
│   ├── templates/
│   │   └── index.html     # Main web page
│   └── static/
│       └── css/
│           └── style.css  # Stylesheet
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Analysis notebook
├── data/
│   └── heart.csv          # Cached dataset
└── models/
    └── heart_model.pkl    # Trained model
```

## Dataset Explanation

### Source
The project uses the **UCI Cleveland Heart Disease dataset** from the UCI Machine Learning Repository. This is a classic dataset for binary classification tasks in healthcare.

### Data Description
- **Size**: 303 instances (patients), 14 attributes.
- **Target**: Binary classification (0 = no heart disease, 1 = heart disease present).
- **Features**: 13 clinical measurements.

### Feature Details

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Patient age in years | Numeric | 29-77 |
| sex | Gender (1 = male, 0 = female) | Categorical | 0-1 |
| cp | Chest pain type | Categorical | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | Numeric | 94-200 |
| chol | Serum cholesterol (mg/dL) | Numeric | 126-564 |
| fbs | Fasting blood sugar > 120 mg/dL | Categorical | 0-1 |
| restecg | Resting electrocardiographic results | Categorical | 0-2 |
| thalach | Maximum heart rate achieved | Numeric | 71-202 |
| exang | Exercise-induced angina | Categorical | 0-1 |
| oldpeak | ST depression induced by exercise | Numeric | 0-6.2 |
| slope | Slope of the peak exercise ST segment | Categorical | 0-2 |
| ca | Number of major vessels colored by fluoroscopy | Categorical | 0-3 |
| thal | Thalassemia status | Categorical | 3,6,7 |

### Data Cleaning
- Missing values (marked as '?') are dropped.
- Numeric columns are converted to appropriate types.
- Target is binarized (values > 0 become 1).

## Machine Learning Model

### Algorithm
**Random Forest Classifier** from scikit-learn.

### Why Random Forest?
- Handles mixed data types (numeric and categorical).
- Robust to overfitting.
- Provides feature importance.
- Good baseline performance for tabular data.

### Training Process
1. Load and clean the dataset.
2. Split into train/test sets (75/25 split, stratified).
3. Standardize features using `StandardScaler`.
4. Train the model with 100 trees.
5. Evaluate on test set.

### Model Performance
- **Accuracy**: ~85% on test set.
- **Metrics**: Precision, recall, F1-score for both classes.
- **Confusion Matrix**: True positives, false positives, etc.

### Model Persistence
- Saved as `models/heart_model.pkl` using joblib.
- Loaded at app startup for predictions.

## Backend Architecture

### Flask Application (`src/app.py`)
- **Initialization**: Loads dataset and model on startup.
- **Routes**: Single route `/` for GET/POST requests.
- **Prediction Logic**: Parses form data, makes predictions, returns results.
- **Template Rendering**: Passes data to Jinja2 templates.

### Data Loader (`src/data_loader.py`)
- Downloads dataset from UCI if not present.
- Cleans and preprocesses data.
- Returns pandas DataFrame.

### Model Training (`src/model_training.py`)
- Defines the ML pipeline (scaler + classifier).
- Trains and evaluates the model.
- Saves the trained model.

## Frontend Design

### User Interface (`src/templates/index.html`)
- **Hero Section**: Project title, description, model accuracy.
- **Prediction Form**: Input fields for all 13 features.
- **Results Display**: Shows prediction and confidence.
- **Dataset Preview**: Table with first 10 rows.

### Styling (`src/static/css/style.css`)
- **Theme**: Dark blue background with orange accents.
- **Accessibility**: High contrast, readable fonts.
- **Responsive**: Works on desktop and mobile.
- **Components**: Cards, forms, tables with modern design.

### Form Features
- Dropdowns for categorical features with tooltips.
- Number inputs with min/max validation.
- Required fields to ensure complete data.
- Form persistence (retains values after submission).

## How to Run the Project

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/chandruchelladurai155/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**
   ```bash
   python train.py
   ```

5. **Run the Web App**
   ```bash
   python run.py
   ```

6. **Access the Application**
   - Open browser to `http://localhost:5000`
   - Enter patient data and click "Predict Risk"

### Alternative Commands
- Direct training: `python src/model_training.py`
- Direct app run: `python src/app.py`
- Notebook analysis: `jupyter notebook notebooks/heart_disease_prediction.ipynb`

## Workflow Explanation

### Training Phase
1. `train.py` calls `run_training()` from `model_training.py`.
2. Dataset is loaded and cleaned.
3. Model is trained and evaluated.
4. Model is saved to disk.

### Prediction Phase
1. User visits `http://localhost:5000`.
2. Flask renders the form with dataset preview.
3. User fills form and submits.
4. Flask parses inputs, scales features, makes prediction.
5. Result is displayed with confidence score.

### Data Flow
- Dataset → Cleaning → Training → Model Save
- User Input → Parsing → Prediction → Display

## Code Walkthrough

### Key Functions

#### `load_heart_disease_data()`
```python
def load_heart_disease_data(csv_path=None, download_if_missing=True):
    # Downloads from UCI if needed
    # Cleans missing values
    # Returns DataFrame
```

#### `train_model(df)`
```python
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model, X_test, y_test
```

#### `predict_risk(form_data)`
```python
def predict_risk(form_data):
    values = [parse_feature_value(form_data[name], name) for name in FEATURE_ORDER]
    prediction = MODEL.predict([values])[0]
    probability = MODEL.predict_proba([values])[0][1]
    return prediction, probability
```

### Flask Route
```python
@app.route("/", methods=["GET", "POST"])
def index():
    # Handle form submission
    # Render template with data
```

## Potential Improvements

### Model Enhancements
- Try other algorithms (XGBoost, SVM).
- Hyperparameter tuning with GridSearchCV.
- Feature engineering (polynomial features, interactions).

### Frontend Improvements
- Add data visualization charts (matplotlib plots).
- Implement form validation with JavaScript.
- Add prediction history or batch processing.

### Backend Improvements
- Add API endpoints for external integrations.
- Implement user authentication.
- Add database for storing predictions.

### Deployment
- Containerize with Docker.
- Deploy to cloud platforms (Heroku, AWS).
- Add CI/CD pipeline.

## Common Questions & Answers

### Q: What if the dataset download fails?
A: Check internet connection. The dataset is cached locally after first download.

### Q: Can I use my own dataset?
A: Modify `data_loader.py` to load your CSV, ensuring same column names.

### Q: How accurate is the model?
A: ~85% accuracy on test set. This is a baseline; real medical models need more validation.

### Q: Is this ready for medical use?
A: No, this is educational. Consult professionals for real diagnoses.

### Q: Can I run this on Windows/Mac?
A: Yes, adjust virtual environment activation commands.

## Conclusion

This project demonstrates a complete ML workflow: data acquisition, preprocessing, modeling, evaluation, and web deployment. It serves as an excellent starting point for learning full-stack data science applications.

The combination of Flask's simplicity, scikit-learn's power, and a clean UI makes it accessible for beginners while being extensible for advanced users.

For further learning, explore the Jupyter notebook for data analysis details, or experiment with model improvements.
