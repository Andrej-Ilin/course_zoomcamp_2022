
<div id="header" align="center">
  <img src="https://glorypets.ru/wp-content/uploads/2020/07/1-tsarstvennost.jpg" width="500"/>
</div>

# [70 Dog Breeds-Image Data Set from Kaggle](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)

## I developed a model that can identify the breed of a dog from a photo. 

## Description of files:
- [archive](https://github.com/Andrej-Ilin/course_zoomcamp_2022/tree/main/capstone1/archive) contains any images the test, train and validation sets. Images have size to 224 X 224 X 3 jpg format.
- [train.ipynb](https://github.com/Andrej-Ilin/course_zoomcamp_2022/blob/main/capstone1/train.ipynb):
  * Data loading and preprocessing. 
  * Loading a pretrain model "Exception".
  * Train model and get wight in SaturnCloud
  * Convert Keras to TF-Lite 
- [lambda_function.py](https://github.com/Andrej-Ilin/course_zoomcamp_2022/blob/main/capstone1/lambda_function.py):
  * Import tflite, preprocessing, lambda_handler
- [Dockerfile](https://github.com/Andrej-Ilin/course_zoomcamp_2022/blob/main/capstone1/Dockerfile) taked from lessons [Alexey Grigorev](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/code/Dockerfile) 

## Instuction for run:
  - $ docker build -t dog-model .
  - $ sudo docker run -it --rm -p8080:8080 dog-model:latest
  - in another command line In another command line run python test.py
  
*Yoy can replace link images in test.py*

    
  
