import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar modelo
with open('titanic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    pca = model_data['pca']
    scaler = model_data['scaler']
    sex_encoder = model_data['sex_encoder']
    deck_encoder = model_data['deck_encoder']
    required_features = ['Pclass', 'Age', 'Fare', 'Sex_male', 'Deck']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Obtener datos del formulario
        form_data = request.form
        
        # 2. Preprocesamiento idéntico al entrenamiento
        sex_encoded = sex_encoder.transform([[form_data['Sex']]])[0][0]
        deck_encoded = deck_encoder.transform([[form_data['Deck'].upper()]])[0][0]
        
        # 3. Crear DataFrame solo con las features usadas en el modelo final
        input_data = pd.DataFrame([[
            float(form_data['Pclass']),
            float(form_data['Age']),
            float(form_data['Fare']),
            sex_encoded,
            deck_encoded
        ]], columns=required_features)
        
        # 4. Escalar solo estas 5 features (no todas las columnas como en entrenamiento)
        scaler = StandardScaler()
        scaler.mean_ = model_data['scaler'].mean_[:5]  # Solo las 5 features usadas
        scaler.scale_ = model_data['scaler'].scale_[:5]  # Solo las 5 features usadas
        
        X_scaled = scaler.transform(input_data)
        X_pca = pca.transform(X_scaled)
        
        # 5. Predecir
        prediction = model.predict(X_pca)[0]
        result = "Sobrevivió" if prediction == 1 else "No sobrevivió"
        
        return render_template('form.html', prediction=result)
    
    except Exception as e:
        error_msg = f"Error: {str(e)}. Verifica que todos los campos estén completos y sean válidos."
        return render_template('form.html', error=error_msg)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)