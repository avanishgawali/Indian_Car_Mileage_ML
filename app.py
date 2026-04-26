from flask import Flask, render_template, request, jsonify
import joblib
import os

# We tell Flask to look for templates and static files in the CURRENT directory ('.')
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# Load the saved model
model = joblib.load('mileage_model.pkl')

@app.route('/')
def home():
    # Now it will find index.html in your main folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Conversions
        engine_size = float(data['cc']) / 1000
        cylinders = float(data['cylinders'])
        hp = float(data['hp'])
        weight = float(data['weight']) * 2.204
        wheelbase = float(data['wheelbase'])
        length = float(data['length'])

        features = [[engine_size, cylinders, hp, weight, wheelbase, length]]
        prediction = model.predict(features)
        
        return jsonify({'mileage': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)