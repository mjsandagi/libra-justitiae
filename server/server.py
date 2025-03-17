from flask import Flask, request, jsonify
from flask_cors import CORS  
from model import predict

app = Flask(__name__)
CORS(app)  # Enables CORS (Cross-Origin Resource Sharing) for all routes by default

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()  
    param1 = data.get('name')
    param2 = data.get('conviction')
    totalNumberOfConvictions = data.get('totalNumOfConvictions')

    # If any parameters are missing, returns an error 
    if not param1 or not param2:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Calls the predict function with the parameters
    result = predict(param2, totalNumberOfConvictions)
    
    # Returns the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)