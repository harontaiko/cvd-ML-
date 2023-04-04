## Cardiovascular Input based Prediction (ML)
This code defines a Flask app that receives user input about cardiovascular disease risk factors, predicts whether the user has a high or low likelihood of having cardiovascular disease, and returns the prediction along with a plot showing the feature importance for the prediction. The app loads a dataset of cardiovascular disease risk factors, trains a random forest classifier on the data, and calculates the model's accuracy. When a user submits their data, the app preprocesses the data, makes a prediction using the trained model, calculates the accuracy of the prediction, and creates a plot of the feature importance. Finally, the app returns the prediction, plot, and accuracy to the user.

## Output Graph
- feature importances from the trained clf classifier.

- Sorting feature importances in descending order using `np.argsort()` and reverse the order using `[::-1]`

- Reorder the user data according to the sorted feature importances.

- plot graph


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

> **All of the dataset values were collected at the moment of medical examination.**
> 
> _The dataset consists of 70 000 records of patients data, 11 features + target._

## Installing and running (locally & Production)
- Pre-requisites (Python3, pip, vscode(or other IDE))
- run `pip install` to install necessary packages
- run `python -m venv env` to create a virtual envirnment
- activate the environmnet through `projectfolder\env\Scripts\Activate`
- run `python app.py`, the app will open on port `5000 or any free localhost port`
- The production version is found at [Heroku](https://cvd-ml-haron.herokuapp.com)