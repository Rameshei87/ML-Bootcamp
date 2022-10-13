import pickle
from flask import Flask
from flask import request
from flask import jsonify


dvFile = 'dv.bin'
modelFile = 'model1.bin'


with open(modelFile, 'rb') as f:
    model = pickle.load(f)

with open(dvFile, 'rb') as f:
    dv = pickle.load(f)

app = Flask('churn')

@app.route('/predict', methods=['POST'])

def predict():
    
    user = request.get_json()
    
    X = dv.transform(user)
    yPred = model.predict_proba(X)[:,1][0]
    churn = yPred >= 0.5
    
    res = {
        'churn probability' : float(yPred),
        'churn' : bool(churn)
    }

    return jsonify(res)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7777)