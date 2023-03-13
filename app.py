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

# Load the data into a pandas DataFrame
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

# Split the data into features and target
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


def plot_graph(df, prediction):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red" if p == 1 else "green" for p in prediction]
    ax.scatter(df["age"], df["systolic_bp"], color=colors, alpha=0.5, s=50)
    ax.set_xlabel("Age")
    ax.set_ylabel("Systolic Blood Pressure")
    ax.set_title(
        "Correlation between Age, Systolic Blood Pressure and Cardiovascular Disease")
    ax.grid(True)
    # Convert the plot to a base64 encoded string
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


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

        plot_url = plot_graph(inputs, prediction)

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
    plot_url = plot_graph(df, prediction)
    return render_template("index.html", plot=plot_url)


if __name__ == "__main__":
    app.run(debug=True)
