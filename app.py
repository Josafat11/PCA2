import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar modelo
with open('titanic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    pca = model_data['pca']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    feature_order = model_data['feature_order']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Obtener datos del formulario
        form_data = request.form
        
        # 2. Preprocesamiento idéntico al entrenamiento
        sex_encoded = encoders['sex'].transform([[form_data['Sex']]])[0][0]
        deck_encoded = encoders['deck'].transform([[form_data['Deck'].upper()]])[0][0]
        
        # 3. Crear array de entrada en el orden correcto
        input_values = [
            float(form_data['Pclass']),
            float(form_data['Age']),
            float(form_data['Fare']),
            sex_encoded,
            deck_encoded
        ]
        
        # 4. Convertir a DataFrame manteniendo el orden de features
        input_df = pd.DataFrame([input_values], columns=feature_order)
        
        # 5. Aplicar transformaciones
        scaled_data = scaler.transform(input_df)
        pca_data = pca.transform(scaled_data)
        
        # 6. Predecir
        prediction = model.predict(pca_data)[0]
        result = "Sobrevivió" if prediction == 1 else "No sobrevivió"
        
        return render_template('form.html', prediction=result)
    
    except Exception as e:
        return render_template('form.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)