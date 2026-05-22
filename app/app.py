from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

# Bulletproof absolute paths to prevent PyTest and Docker pathing issues
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))
TEMPLATE_DIR = os.path.join(APP_DIR, 'templates')
MODEL_PATH = os.path.join(ROOT_DIR, 'model.pkl')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        for key in data:
            data[key] = float(data[key])
        df = pd.DataFrame([data])
        
        if model is None:
            return render_template("result.html", prediction="Error: ML Model not found by Flask.")
            
        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]
            
        prediction = model.predict(df)
        result_text = "Good Quality Wine" if prediction[0] == 1 else "Poor Quality Wine"
        return render_template("result.html", prediction=result_text)
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
