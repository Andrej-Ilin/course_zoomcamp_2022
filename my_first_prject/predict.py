import pickle

from flask import Flask
from flask import request
from  flask import jsonify
input_file = 'best_model.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('price')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform(customer)
    y_pred_proba = float(round(max(model.predict_proba(X)[0]), 2))
    price = model.predict(X)[0]

    result = {
        'probability': y_pred_proba,
        'price': int(price)
    }
    return jsonify(result)


# y_pred = model.predict(X)[0]
# print('input', customer)
# print(f'price_range {y_pred} with probability {y_pred_proba}')
# def ping():
#     return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


