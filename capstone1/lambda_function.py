import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
import pickle
import numpy as np

import os
from sanic import Sanic
from sanic.response import text

app = Sanic(__name__)

preprocessor = create_preprocessor('xception', target_size=(224, 224))

interpreter = tflite.Interpreter(model_path='model_0.936.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

with open("classes", "rb") as fp:  # Unpickling
    classes = pickle.load(fp)

url = 'https://glorypets.ru/wp-content/uploads/2020/07/1-tsarstvennost.jpg'


# url = 'https://sun9-79.userapi.com/c11422/u1430261/148960630/x_b6d7669e.jpg'

def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    ind = np.argpartition(preds[0], -2)[-2:]
    if np.diff(preds[0][ind]) > 0.8:  # if clear the breed
        return {classes[np.argmax(preds[0])]: np.max(preds)}
    else:  # half breed
        h_breed = {}
        for i in ind:
            h_breed[classes[i]] = preds[0][i]
        return h_breed


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ['PORT'], motd=False, access_log=False)
