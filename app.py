from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load the trained model
clf = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent by the client
    request_data = request.get_json()
    
    # Convert JSON data to DataFrame
    input_data = pd.DataFrame(request_data['data'], index=[0])
    
    # Make predictions using the loaded model
    predictions = clf.predict(input_data)
    
    # Prepare response
    response = {
        'predictions': predictions.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
