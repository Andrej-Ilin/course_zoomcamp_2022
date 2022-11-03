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
    # return {"status_range": "200"}
    result = {
     # 'probability': y_pred_proba,
        'price': prediction[0][0]
            }
    return result

# bentoml serve service.py:svc --reload
# bentoml models list
# bentoml models get price_range:g3hqcvkz4sxiz2bk      описание признаков модели
# \GitHub\course_zoomcamp_2022\BentoML>bentoml build
# \course_zoomcamp_2022\BentoML>bentoml containerize predict_classifier:yqnybks2sgqof2bk
# docker run -it --rm -p 3000:3000 predict_classifier:yqnybks2sgqof2bk