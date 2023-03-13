## Cardiovascular Input based Prediction (ML)
The given code consists of an HTML form that collects user input, a Flask application that loads a dataset, trains a random forest classifier on it, and provides a function to predict cardiovascular disease based on user input, and JavaScript code that sends user input to the Flask API and displays the result on the webpage.

The HTML form collects the user's input as described in the features below. The Flask application loads a dataset called cardio_train.csv, which is a dataset of individuals with and without cardiovascular disease, and trains a random forest classifier on the dataset. The JavaScript code sends the user input to the Flask API and displays the prediction result on the webpage.


## Natural Language Processing(Interpreting direct human input)
- Apart from prediction, accuracy scores and graphical metrics
- Natural language input is also acceptable but within the given parameters
- The model, then accurately returns in natural language an accurate prediction of cvd likelihood

## Obtaininement of Dataset from `Kaggle`

- Objective: factual information;
- Examination: results of medical examination;
- Subjective: information given by the patient.

Features:

- Age | Objective Feature | age | int (days) ✓
- Height | Objective Feature | height | int (cm) | ✓
- Weight | Objective Feature | weight | float (kg) | ✓
- Gender | Objective Feature | gender | categorical code | ✓
- Systolic blood pressure | Examination Feature | ap_hi | int | ✓
- Diastolic blood pressure | Examination Feature | ap_lo | int | ✓
- Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal | ✓
- Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal | ✓
- Smoking | Subjective Feature | smoke | binary | ✓
- Alcohol intake | Subjective Feature | alco | binary | ✓
- Physical activity | Subjective Feature | active | binary | ✓
- Presence or absence of cardiovascular disease | Target Variable | cardio | binary | ✓

**All of the dataset values were collected at the moment of medical examination.**

_The dataset consists of 70 000 records of patients data, 11 features + target._