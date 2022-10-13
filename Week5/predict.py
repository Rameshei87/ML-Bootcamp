import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

modelFile = 'model_C=1.0.bin'

with open(modelFile, 'rb') as f:
    newModel = pickle.load(f)

app = Flask('churn')

@app.route('/predict', methods=['POST'])

def predict():

    user = request.get_json()
    
    u = pd.DataFrame(user, index=[1])
    yPred = newModel.predict_proba(u)[0][1]
    churn = yPred >= 0.5
    
    res = {
        'churn probability' : float(yPred),
        'churn' : bool(churn)
    }

    return jsonify(res)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)