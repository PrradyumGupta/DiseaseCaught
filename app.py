from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and symptom order
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    symptom_order = data['symptom_order']

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    symptoms = symptom_order  # list of symptoms available for selection

    if request.method == 'POST':
        # Create input vector with 0s
        input_vector = np.zeros(len(symptom_order))

        # Iterate over symptom inputs from form
        for i, symptom in enumerate(symptom_order):
            # Check if user selected symptom i (form checkbox name = symptom name)
            if request.form.get(symptom) == 'on':
                input_vector[i] = 1

        # Predict disease
        prediction = model.predict([input_vector])[0]

    return render_template('predict.html', symptom_list=symptoms, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
