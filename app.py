from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract ALL 22 features
        features = [
            float(request.form['symboling']),
            float(request.form['norm_losses']),
            request.form['make'],  # String
            request.form['fuel_type'],  # String
            request.form['aspiration'],  # String
            float(request.form['no_of_doors']),
            request.form['body_style'],  # String
            request.form['drive_wheels'],  # String
            float(request.form['wheel_base']),
            float(request.form['length']),
            float(request.form['width']),
            float(request.form['height']),
            float(request.form['curb_weight']),
            request.form['eng_type'],  # String
            float(request.form['no_of_cyl']),
            float(request.form['eng_size']),
            request.form['fuel_sys'],  # String
            float(request.form['bore']),
            float(request.form['stroke']),
            float(request.form['horsepower']),
            float(request.form['peak_rpm']),
            float(request.form['city_mpg'])
        ]

        # Convert categorical features to numerical (if required)
        categorical_features = ['make', 'fuel_type', 'aspiration', 'body_style', 'drive_wheels', 'eng_type', 'fuel_sys']
        encoded_features = [feature_mapping[f] if f in feature_mapping else f for f in features]

        # Convert to NumPy array
        input_features = np.array(encoded_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f'Estimated Price: ${prediction:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
