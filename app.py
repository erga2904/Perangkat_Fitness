# app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model & encoders
model = joblib.load('best_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Brand Name': request.form['brand'],
            'Device Type': request.form['device_type'],
            'Selling Price': float(request.form['selling_price']),
            'Original Price': float(request.form['original_price']),
            'Average Battery Life (in days)': float(request.form['battery_life']),
            'Reviews': float(request.form['reviews']),
            'Display': request.form['display'],
            'Strap Material': request.form['strap']
        }

        df_input = pd.DataFrame([data])

        # Encode categorical features
        for col in ['Brand Name', 'Device Type', 'Display', 'Strap Material']:
            if df_input[col].iloc[0] in label_encoders[col].classes_:
                df_input[col] = label_encoders[col].transform(df_input[col])
            else:
                df_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0].max()
        result = "Kualitas Tinggi (Rating ≥ 4.0)" if pred == 1 else "Kualitas Rendah (Rating < 4.0)"
        confidence = f"{prob * 100:.1f}%"

        return render_template('result.html', result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)