from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

# Load all your trained model 
model = joblib.load('C:\\Users\\KIIT\\Projects\\Parkinson-Desease-Prediction-Model-Using-Machine-Learning-main\\model.pkl')
app = Flask(__name__)

# Define the expected features based on the model’s requirements in the notebook
EXPECTED_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", 
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", 
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", 
    "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

def get_default_data():
    # Initialize default feature values. Adjust values as necessary based on typical ranges
    return np.zeros(len(EXPECTED_FEATURES))

def map_symptoms_to_input(symptoms):
    """
    Convert selected symptoms to model-compatible input array based on EXPECTED_FEATURES.
    """
    # Initialize all features with default (neutral) values
    input_data = get_default_data()

    # Map symptoms to corresponding feature indices and assign values for features linked to each symptom
    if "tremor" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:Fo(Hz)")] = 200  # Example value
        input_data[EXPECTED_FEATURES.index("MDVP:Jitter(%)")] = 0.02

    if "bradykinesia" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:Fhi(Hz)")] = 180
        input_data[EXPECTED_FEATURES.index("MDVP:Flo(Hz)")] = 85

    if "rigid_muscles" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:RAP")] = 0.03
        input_data[EXPECTED_FEATURES.index("Shimmer:APQ5")] = 0.04

    if "impaired_posture" in symptoms:
        input_data[EXPECTED_FEATURES.index("spread1")] = -6
        input_data[EXPECTED_FEATURES.index("spread2")] = 0.5
        input_data[EXPECTED_FEATURES.index("D2")] = 0.4

    if "loss_of_automatic_movements" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:PPQ")] = 0.025
        input_data[EXPECTED_FEATURES.index("Jitter:DDP")] = 0.035

    if "speech_changes" in symptoms:
        input_data[EXPECTED_FEATURES.index("HNR")] = 15
        input_data[EXPECTED_FEATURES.index("NHR")] = 0.5
        input_data[EXPECTED_FEATURES.index("MDVP:Shimmer")] = 0.04
        input_data[EXPECTED_FEATURES.index("MDVP:Shimmer(dB)")] = 0.3

    if "overall_neurological_impact" in symptoms:
        input_data[EXPECTED_FEATURES.index("RPDE")] = 0.6
        input_data[EXPECTED_FEATURES.index("DFA")] = 0.7
        input_data[EXPECTED_FEATURES.index("PPE")] = 0.5

    if "fatigue" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:Jitter(Abs)")] = 0.001
        input_data[EXPECTED_FEATURES.index("Shimmer:APQ3")] = 0.02

    if "swallowing_difficulties" in symptoms:
        input_data[EXPECTED_FEATURES.index("MDVP:Shimmer(dB)")] = 0.5
        input_data[EXPECTED_FEATURES.index("MDVP:APQ")] = 0.03

    if "handwriting_changes" in symptoms:
        input_data[EXPECTED_FEATURES.index("Shimmer:DDA")] = 0.03
        input_data[EXPECTED_FEATURES.index("MDVP:Jitter(%)")] = 0.03

    if "sleep_disturbances" in symptoms:
        input_data[EXPECTED_FEATURES.index("D2")] = 0.55
        input_data[EXPECTED_FEATURES.index("spread2")] = 0.6

    # Return the input data as a single-row 2D array, as required by the model
    return np.array(input_data).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index_updated.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize response variables
    message = ""
    
    # Get symptoms selected by user
    symptoms = request.form.getlist('symptoms')
    
    # Define minimum number of symptoms required for a valid prediction
    MIN_SYMPTOMS_REQUIRED = 3  # You can adjust this threshold as needed
    # Check if symptoms meet the minimum threshold
    if symptoms and len(symptoms) < MIN_SYMPTOMS_REQUIRED:
        message = "There is no indication of Parkinson's disease based on the inputs."
        preventions = [
                "Regular exercise",
                "Healthy diet rich in antioxidants",
                "Avoid exposure to pesticides or toxins",
                "Stay mentally active"]
        return render_template('index_updated.html', message=message,preventions=preventions)
    
    # Get frequencies inputted manually or from uploaded file
    frequencies = request.form.get('frequencies')
    uploaded_file = request.files.get('file-upload')

    # Prepare data array with default values
    data = get_default_data()
    
    # Case 1: Manually entered frequencies
    if frequencies:
        # Parse the manually entered frequencies
        freq_data = np.array([float(f) for f in frequencies.split(',')])
        if len(freq_data) == len(EXPECTED_FEATURES):
            data = freq_data.reshape(1, -1)
        else:
            message = f"Please enter exactly {len(EXPECTED_FEATURES)} frequencies."
            return render_template('index_updated.html', message=message)

    # Case 2: Uploaded file with frequencies
    elif uploaded_file:
        # Assume uploaded file has the necessary features in the correct order
        df = pd.read_csv(uploaded_file)
        if all(feature in df.columns for feature in EXPECTED_FEATURES):
            data = df[EXPECTED_FEATURES].iloc[0].to_numpy().reshape(1, -1)
        else:
            message = "Uploaded file is missing required features."
            return render_template('index_updated.html', message=message)

    # Case 3: Symptoms-based prediction
    elif symptoms:
        data = map_symptoms_to_input(symptoms)

    else:
        message = "Please provide at least one input method (symptoms, frequencies, or file upload)."
        return render_template('index_updated.html', message=message)

    # Make prediction
    try:
        prediction = model.predict(data)

        if prediction[0] == 1:
            # Parkinson's detected
            message = "Based on the inputs, there is a high likelihood of Parkinson's disease."
            doctors = ["Dr. Anoop Misra", "Dr. Vikram Shah", "Dr. Geeta Nair"]  # Example, replace with actual list
            treatments = [
                "Medication: Levodopa, Dopamine Agonists",
                "Physical therapy for mobility and balance",
                "Speech therapy for communication improvements",
                "Deep brain stimulation surgery (in severe cases)"
            ]
            return render_template('index_updated.html', message=message, doctors=doctors, treatments=treatments)
        else:
            # No Parkinson's detected
            message = "There is no indication of Parkinson's disease based on the inputs."
            preventions = [
                "Regular exercise",
                "Healthy diet rich in antioxidants",
                "Avoid exposure to pesticides or toxins",
                "Stay mentally active"
            ]
            return render_template('index_updated.html', message=message, preventions=preventions)

    except ValueError as e:
        message = f"Error in prediction: {str(e)}. Please ensure input format meets model requirements."
        return render_template('index_updated.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)