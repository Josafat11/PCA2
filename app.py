import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelo
with open('titanic_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    pca = data['pca']
    scaler = data['scaler']
    sex_encoder = data['sex_encoder']
    features = data['features']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        input_data = {
            'Pclass': int(request.form['Pclass']),
            'Age': float(request.form['Age']),
            'Fare': float(request.form['Fare']),
            'Sex': request.form['Sex'],
            'Deck': request.form['Deck']
        }

        # Preprocesamiento
        sex_encoded = sex_encoder.transform([[input_data['Sex']]])[0]
        
        # Construir array de características
        X = np.array([
            input_data['Pclass'],
            input_data['Age'],
            input_data['Fare'],
            sex_encoded,
            input_data['Deck']
        ]).reshape(1, -1)
        
        # Escalar y aplicar PCA
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        
        # Predecir
        prediction = model.predict(X_pca)[0]
        result = "Sobrevivió" if prediction == 1 else "No sobrevivió"
        
        return render_template('form.html', prediction=result)
    
    except Exception as e:
        return render_template('form.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)