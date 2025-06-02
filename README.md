# Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A machine learning-based web application that predicts the risk of heart disease based on clinical parameters.

## Overview

This project implements a machine learning pipeline to analyze various health metrics and identify hidden trends between different features in the dataset to predict the likelihood of heart disease. The system provides an interactive web interface for users to input their health parameters and receive risk assessments.

## Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| Age | Age of the patient | Numeric (years) |
| Sex | Gender of the patient | Categorical (M: Male, F: Female) |
| ChestPainType | Type of chest pain | Categorical (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic) |
| RestingBP | Resting blood pressure | Numeric (mm Hg) |
| Cholesterol | Serum cholesterol | Numeric (mm/dl) |
| FastingBS | Fasting blood sugar | Binary (1: if > 120 mg/dl, 0: otherwise) |
| RestingECG | Resting electrocardiogram results | Categorical (Normal, ST: ST-T wave abnormality, LVH: left ventricular hypertrophy) |
| MaxHR | Maximum heart rate achieved | Numeric (60-202) |
| ExerciseAngina | Exercise-induced angina | Binary (Y: Yes, N: No) |
| Oldpeak | ST depression induced by exercise | Numeric |
| ST_Slope | Slope of the peak exercise ST segment | Categorical (Up: upsloping, Flat: flat, Down: downsloping) |
| HeartDisease | Target variable | Binary (1: heart disease, 0: Normal) |

## Project Structure

```
├── data/
│   └── heart.csv                # Dataset file
├── notebook/
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── MODEL TRAINING.ipynb     # Model training and evaluation
├── src/
│   ├── components/              # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/                # Prediction pipeline
│   │   └── predict_pipeline.py
│   └── exception.py             # Custom exception handling
├── templates/                   # HTML templates
│   ├── home.html
│   └── index.html
├── app.py                       # Flask application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda (recommended) or pip package manager

### Environment Setup

#### Using Anaconda (Recommended)

1. Open Anaconda Prompt and navigate to your project directory:
   ```bash
   cd path/to/project
   ```

2. Create a new conda environment:
   ```bash
   conda create -p venv python=3.8 -y
   ```

3. Activate the environment:
   ```bash
   conda activate ./venv
   ```

#### Using pip and venv

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

### Installing Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:1910/
   ```

3. Enter the required health parameters in the form and click "Calculate Risk" to get a prediction.

## Model Development

The project evaluates multiple classification algorithms including:
- Random Forest
- Logistic Regression
- Gradient Boosting
- XGBoost
- CatBoost
- SVM
- Decision Tree
- K-Nearest Neighbors
- AdaBoost

Models are compared using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to select the best performer for heart disease prediction.

## Web Application

The Flask web application provides:
- A user-friendly interface for data input
- Real-time prediction of heart disease risk
- Visual presentation of results
- Responsive design for various devices

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Dataset source: Heart Disease Dataset
- Special thanks to all contributors and mentors who provided guidance