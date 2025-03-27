from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
print("Current Working Directory:", os.getcwd())



app = Flask(__name__)
# df = pd.read_csv("data/loan.csv", index_col=0)


# Define columns and model weights (bias included)
columns = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

# Updated weights with bias at index 0
weights = np.array([-4.34736745, -0.08902098,  0.1153322 ,  0.06739592, -5.94856442,
        5.6711723 , -2.72949655, 14.89732424,  0.08514448,  0.36894564,
        1.21539747,  0.74545384])

# Statistical data for feedback reference
stats = {
    "income_annum": {"25%": 200000, "50%": 5000000, "75%": 7500000},
    "loan_amount": {"25%": 7500000, "50%": 14600000, "75%": 22100000},
    "cibil_score": {"25%": 750, "50%": 790, "75%": 803},
    "loan_term": {"25%": 4, "50%": 10, "75%": 16}
}
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load dataset
        df = pd.read_csv("data/loan.csv", index_col=0)

        # Get form data
        form_data = {col: request.form[col] for col in columns}

        # Convert numerical fields
        numerical_columns = [
            'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]
        for key in numerical_columns:
            form_data[key] = float(form_data[key])

        # Convert categorical fields
        form_data['education'] = 1 if form_data['education'] == 'Graduate' else 0
        form_data['self_employed'] = 1 if form_data['self_employed'] == 'Yes' else 0

        # Ensure DataFrame has correct column order
        df = df[columns]

        # Append new data
        new_row = pd.DataFrame([form_data])
        df = pd.concat([df, new_row], ignore_index=True)

        # Scale numerical columns **without refitting** the scaler
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        # Extract last row and add bias term
        last_row = df.iloc[-1].values
        input_data = np.hstack(([1], last_row))  # Adding bias term (1) at the start

        # Calculate prediction
        prediction_value = np.dot(input_data, weights)
        probability = 1 / (1 + np.exp(-prediction_value))  # Sigmoid activation

        # Determine loan approval
        probability=probability*(3/4)
        prediction = "Approved" if probability >= 0.6 else "Rejected"
        feedback = []

        # --- Generate Feedback ---
        if form_data['income_annum'] < stats['income_annum']['25%']:
            feedback.append(f"❌ Your annual income ({form_data['income_annum']}) is below the recommended {stats['income_annum']['25%']}.")

        if form_data['loan_amount'] > stats['loan_amount']['75%']:
            feedback.append(f"❌ The loan amount requested ({form_data['loan_amount']}) is higher than the preferred {stats['loan_amount']['75%']}. Consider reducing it.")

        if form_data['cibil_score'] < stats['cibil_score']['25%']:
            feedback.append(f"❌ Your CIBIL score ({form_data['cibil_score']}) is below the recommended {stats['cibil_score']['25%']}. Try improving your credit score.")

        if form_data['loan_term'] < stats['loan_term']['25%']:
            feedback.append(f"❌ Your loan term ({form_data['loan_term']}) is shorter than the suggested {stats['loan_term']['25%']}. Consider a longer term.")

        # Compute total assets
        total_assets = (
            form_data['residential_assets_value'] +
            form_data['commercial_assets_value'] +
            form_data['luxury_assets_value'] +
            form_data['bank_asset_value']
        )

        # Compute loan-to-asset ratio
        loan_to_asset_ratio = form_data['loan_amount'] / total_assets if total_assets > 0 else float('inf')

        # Feedback based on loan-to-asset ratio
        if loan_to_asset_ratio > 0.7:
            feedback.append(f"❌ Your loan amount ({form_data['loan_amount']}) is high compared to your total assets ({total_assets}). A lower loan-to-asset ratio is preferred.")
        elif loan_to_asset_ratio > 0.5:
            feedback.append(f"⚠️ Your loan-to-asset ratio ({loan_to_asset_ratio:.2f}) is slightly high. Reducing your loan amount or increasing assets may help.")

        if not feedback:
            feedback.append("✅ Your application is balanced. Approval depends on internal risk factors.")

        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2),
            feedback=feedback
        )

    except Exception as e:
        return f"Error in processing: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
