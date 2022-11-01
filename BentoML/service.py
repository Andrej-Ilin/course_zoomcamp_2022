import bentoml
from bentoml.io import JSON

model_ref = bentoml.catboost.get("price_range:latest")
dv = model_ref.custom_objects["DictVectorizer"]
model_runner = model_ref.to_runner()

svc = bentoml.Service("predict_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    return {"status_range": "200"}

