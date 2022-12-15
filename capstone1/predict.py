import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
import pickle
import numpy as np

preprocessor = create_preprocessor('xception', target_size=(224, 224))

interpreter = tflite.Interpreter(model_path='model_0.936.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

with open("classes", "rb") as fp:   # Unpickling
    classes = pickle.load(fp)


url = 'https://glorypets.ru/wp-content/uploads/2020/07/1-tsarstvennost.jpg'



def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    ind = np.argpartition(preds[0], -2)[-2:]
    print(ind)
    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



#%%
