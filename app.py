import unittest
import base64
import spacy
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io


app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

# Load data into pandas
df = pd.read_csv("templates/cardio_train.csv", sep=";")
df.columns = ['id', 'age', 'gender', 'height', 'weight', 'systolic_bp',
              'diastolic_bp', 'cholesterol', 'glucose', 'smoke', 'alco', 'active', 'cardio']

# Pre process data for consistency


def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    df["systolic_bp"] = pd.to_numeric(
        df["systolic_bp"], errors="coerce").fillna(0)
    df["diastolic_bp"] = pd.to_numeric(
        df["diastolic_bp"], errors="coerce").fillna(0)
    return df


df = preprocess_data(df)

# Split
X = df.drop("cardio", axis=1)
y = df["cardio"]

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Preprocess data b4 training the model
X = preprocess_data(X)

clf.fit(X, y)


def calculate_accuracy(user_data):
    user_df = pd.DataFrame(user_data, index=[0])

    user_df_preprocessed = preprocess_data(user_df)

    prediction = clf.predict(user_df_preprocessed)

    accuracy = accuracy_score(y, clf.predict(X))

    accuracy_percentage = round(accuracy * 100, 2)

    return accuracy_percentage


def plot_graph(clf, X):
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(feature_importances.index, feature_importances.values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url



nlp = spacy.load('en_core_web_sm')


def process_text(text):
    doc = nlp(text)
    age = None
    gender = None
    height = None
    weight = None
    systolic_bp = None
    diastolic_bp = None
    cholesterol = None
    glucose = None
    smoke = None
    alco = None
    active = None

    # Extract information from the text
    for ent in doc.ents:
        if ent.label_ == 'AGE':
            age = int(ent.text)
        elif ent.label_ == 'GENDER':
            gender = 1 if ent.text.lower() == 'male' else 0
        elif ent.label_ == 'HEIGHT':
            height = float(ent.text.split()[0])
        elif ent.label_ == 'WEIGHT':
            weight = float(ent.text.split()[0])
        elif ent.label_ == 'SYS':
            systolic_bp = int(ent.text)
        elif ent.label_ == 'DIA':
            diastolic_bp = int(ent.text)
        elif ent.label_ == 'CHOL':
            cholesterol = int(ent.text)
        elif ent.label_ == 'GLUCOSE':
            glucose = int(ent.text)
        elif ent.label_ == 'SMOKE':
            smoke = 1 if ent.text.lower() == 'yes' else 0
        elif ent.label_ == 'ALCO':
            alco = 1 if ent.text.lower() == 'yes' else 0
        elif ent.label_ == 'ACTIVE':
            active = 1 if ent.text.lower() == 'yes' else 0

    # Create a new DataFrame with the extracted information
    user_data_n = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "cholesterol": [cholesterol],
        "glucose": [glucose],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    # Preprocess the data
    inputs_n = preprocess_data(user_data_n)

    # Make the prediction using the trained model
    prediction_n = clf.predict(inputs_n)

    # Return the prediction in natural language
    if prediction_n[0] == 1:
        return "High likelihood of cardiovascular disease."
    else:
        return "Low likelihood of cardiovascular disease."

# Predict function/runs natural language fx


@app.route("/ask", methods=["POST"])
def predictn():
    try:
        data = request.get_json()
        text = data['text']

        # Process the text to extract relevant information and make a prediction
        predictiontext = process_text(text)

        # Return the prediction in natural language
        return {'predictiontext': predictiontext}

    except Exception as e:
        return {'error': str(e)}


# Function to make a prediction based on user inputs and plot the graph
@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        age = int(data["age"])
        gender = int(data["gender"])
        height = float(data["height"])
        weight = float(data["weight"])
        systolic_bp = int(data["systolic_bp"]) if "systolic_bp" in data else 0
        diastolic_bp = int(data["diastolic_bp"]
                           ) if "diastolic_bp" in data else 0
        cholesterol = int(data["cholesterol"])
        glucose = int(data["glucose"])
        smoke = int(data["smoke"])
        alco = int(data["alco"])
        active = int(data["active"])

        # Create a new DataFrame with the user inputs
        user_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "height": [height],
            "weight": [weight],
            "systolic_bp": [systolic_bp],
            "diastolic_bp": [diastolic_bp],
            "cholesterol": [cholesterol],
            "glucose": [glucose],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        })

        inputs = preprocess_data(user_data)

        prediction = clf.predict(inputs)

        plot_url = plot_graph(clf, X)

        # Calculate accuracy in % of the prediction
        accuracy_percentage = calculate_accuracy(user_data)

        # Return the prediction and the graph
        return {'prediction': "High Likelihood of CVD" if prediction[0] == 1 else "Low Likelihood of CVD", 'plot': plot_url, 'accuracy': accuracy_percentage}

    except Exception as e:
        return {'error': str(e)}


@app.route("/")
def index():

    X_preprocessed = preprocess_data(X)
    prediction = clf.predict(X_preprocessed)
    plot_url = plot_graph(clf, X)
    return render_template("index.html", plot=plot_url)


@app.route("/docs")
def docs():
    return render_template("docs.html")


if __name__ == "__main__":
    app.run(debug=True)
