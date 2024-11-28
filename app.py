import numpy as np
from flask import  Flask, request,render_template
import pickle
import numpy
import pandas

app  = Flask(__name__)

model = pickle.load(open('C:/Users/KIIT/Documents/Parkinson-Desease-Prediction-Model-Using-Machine-Learning-main/Parkinson-Desease-Prediction-Model-Using-Machine-Learning-main/model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_text = request.form['text']
    input_text_sp = input_text.split(',')
    np_data = np.asarray(input_text_sp, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1,-1))

    if prediction == 1:
        output = "This person has a parkinson disease"
    else:
        output = "this person has no parkinson disease"

    return render_template("index.html", message= output)

if __name__ == "__main__":
    app.run(debug=True)