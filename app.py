from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import os

# Configure Flask for flat directory structure
app = Flask(__name__, template_folder='.', static_folder='.')

# Load the model
MODEL_PATH = 'mileage_model.pkl'

if os.path.exists(MODEL_PATH):
    try:
        ml_model = joblib.load(MODEL_PATH)
        print("✅ Success: mileage_model.pkl loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Critical Error: {MODEL_PATH} not found!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Order: Engine CC, Cylinders, Horsepower, Weight, Wheelbase, Length
        features = [[
            float(data['cc']),
            float(data['cylinders']),
            float(data['hp']),
            float(data['weight']),
            float(data['wheelbase']),
            float(data['length'])
        ]]
        prediction = ml_model.predict(features)[0]
        return jsonify({'mileage': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
