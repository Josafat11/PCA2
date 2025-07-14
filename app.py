from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar el modelo y preprocesadores
with open('titanic_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
pca = data['pca']
scaler = data['scaler']
sex_encoder = data['sex_encoder']
deck_encoder = data['deck_encoder']
features = data['features']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            pclass = int(request.form['pclass'])
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            sex = request.form['sex']  # 'male' or 'female'
            deck = request.form['deck']  # A, B, C, ...

            # Codificar sexo
            sex_male = sex_encoder.transform([[sex]])[:, 0]

            # Codificar cubierta
            deck_transformed = deck_encoder.transform([[deck]])

            # Armar el input
            input_data = pd.DataFrame([{
                'Pclass': pclass,
                'Age': age,
                'Fare': fare,
                'Sex_male': sex_male[0],
                'Deck': deck_transformed[0][0]
            }])

            # Escalar
            input_scaled = scaler.transform(input_data)

            # Aplicar PCA
            input_pca = pca.transform(input_scaled)

            # Predecir
            prediction = model.predict(input_pca)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
