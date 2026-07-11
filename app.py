from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/xgboost_model.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    gender = request.form['gender']
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    credit_history = float(request.form['credit_history'])
    property_area = request.form['property_area']

    # Encoding categorical variables

    gender = 0 if gender == "Male" else 1
    married = 1 if married == "Yes" else 0

    if dependents == "3+":
        dependents = 3
    else:
        dependents = int(dependents)

    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    property_area = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    features = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}])

    features = scaler.transform(features)
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
