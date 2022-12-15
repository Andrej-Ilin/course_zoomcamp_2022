FROM python:3.9-slim
RUN pip install keras_image_helper
RUN pip install tflite-runtime

COPY model_0.936.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]