import spacy

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

    # Preprocess the data
    inputs = preprocess_data(user_data)

    # Make the prediction using the trained model
    prediction = clf.predict(inputs)

    # Return the prediction in natural language
    if prediction[0] == 1:
        return "High likelihood of cardiovascular disease."
    else:
        return "Low likelihood of cardiovascular disease."


@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data['text']

        # Process the text to extract relevant information and make a prediction
        prediction = process_text(text)

        # Return the prediction in natural language
        return {'prediction': prediction}

    except Exception as e:
        return {'error': str(e)}
