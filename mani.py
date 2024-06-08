from flask import Flask, render_template, request, jsonify
import numpy as np
from models import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array(data['X'])
    model = LinearRegression()
    model.fit(X, data['y'])
    predictions = model.predict(X).tolist()
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
