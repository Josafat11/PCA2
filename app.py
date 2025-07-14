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
        # 1. Obtener y validar datos del formulario
        form_data = request.form
        
        # 2. Preprocesamiento consistente con el entrenamiento
        sex_encoded = encoders['sex'].transform([[form_data['Sex']]])[0][0]
        deck_encoded = encoders['deck'].transform([[form_data['Deck'].upper()]])[0][0]
        
        # 3. Crear array con EXACTAMENTE las mismas features usadas en entrenamiento
        input_values = [
            float(form_data['Pclass']),
            float(form_data['Age']),
            float(form_data['Fare']),
            sex_encoded,
            deck_encoded
        ]
        
        # 4. Crear DataFrame con las columnas en el ORDEN CORRECTO
        input_df = pd.DataFrame([input_values], columns=feature_order)
        
        # 5. Verificar que tenemos todas las features necesarias
        missing_features = set(feature_order) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Faltan features: {missing_features}")
        
        # 6. Aplicar transformaciones en el mismo orden que en entrenamiento
        X_scaled = scaler.transform(input_df[feature_order])  # Asegurar orden
        X_pca = pca.transform(X_scaled)
        
        # 7. Predecir
        prediction = model.predict(X_pca)[0]
        result = "Sobrevivió" if prediction == 1 else "No sobrevivió"
        
        return render_template('form.html', prediction=result)
    
    except Exception as e:
        error_msg = f"Error: {str(e)}. Por favor verifica los datos e intenta nuevamente."
        return render_template('form.html', error=error_msg)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)