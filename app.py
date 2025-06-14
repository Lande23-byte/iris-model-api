from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure this filename matches your .pkl file)
model = joblib.load("iris_rf_model.pkl")

# HTML Form Template (embedded)
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            max-width: 500px;
            margin: auto;
            background: #f4f4f4;
        }
        h2 {
            text-align: center;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 10px;
        }
        label {
            display: block;
            margin-top: 15px;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
        }
        button {
            margin-top: 20px;
            padding: 10px;
            width: 100%;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h2>Iris Flower Prediction</h2>
    <form method="post" action="/">
        <label>Sepal Length (cm):</label>
        <input type="number" name="sepal_length" step="0.1" required>
        
        <label>Sepal Width (cm):</label>
        <input type="number" name="sepal_width" step="0.1" required>
        
        <label>Petal Length (cm):</label>
        <input type="number" name="petal_length" step="0.1" required>
        
        <label>Petal Width (cm):</label>
        <input type="number" name="petal_width" step="0.1" required>
        
        <button type="submit">Submit & Predict</button>
    </form>
    {% if prediction is not none %}
        <div class="result">Prediction: {{ prediction }}</div>
    {% endif %}
</body>
</html>
"""

# Home route handles both GET (show form) and POST (form submit)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Gather features from form
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            # Predict and map to species name
            pred_idx = model.predict([features])[0]
            class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            prediction = class_names[int(pred_idx)]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(html_form, prediction=prediction)

# API endpoint for JSON requests
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "JSON with a 'features' array is required"}), 400
    try:
        arr = np.array(data["features"], dtype=float).reshape(1, -1)
        pred_idx = model.predict(arr)[0]
        return jsonify({"prediction": int(pred_idx)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)