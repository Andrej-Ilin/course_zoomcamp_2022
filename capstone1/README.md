
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

___
  In the [train.ipynb](https://github.com/Andrej-Ilin/course_zoomcamp_2022/blob/main/capstone1/train.ipynb) a code performs several operations related to training and deploying a deep learning model using TensorFlow and Keras for image classification. Here’s a breakdown of the different parts:

### 1. **Importing Libraries and Initial Setup**:
   - **TensorFlow** is imported for building and training deep learning models.
   - **NumPy** is used for numerical operations (e.g., array manipulation).
   - **Matplotlib** is used for plotting the training and validation accuracy over epochs.
   - **Keras and ImageDataGenerator** are used to preprocess images and create datasets.
   - **Xception** is used as the base model for transfer learning, leveraging pretrained weights from ImageNet.

### 2. **Loading and Preprocessing Data**:
   - The image file located at `./archive/train/Doberman/001.jpg` is loaded using the `load_img` function from Keras.
   - **ImageDataGenerator** is configured to rescale the image pixel values and split the data into training, validation, and test sets.
   - **flow_from_directory** method is used to load the training and validation datasets from specified directories, resizing the images to 224x224.

### 3. **Model Creation**:
   - **Xception** model is used as a base model with weights pretrained on ImageNet. This model does not include the top classification layer (`include_top=False`).
   - The base model is frozen (`base_model.trainable = False`), meaning its weights will not be updated during training.
   - A custom head is added to the base model: a **Dropout** layer (for regularization), a **GlobalAveragePooling2D** layer (to reduce spatial dimensions), and a **Dense** layer with 70 output units (for the 70 classes).

### 4. **Model Compilation and Training**:
   - The model is compiled with the **Adam optimizer** and **categorical cross-entropy** loss function, suitable for multi-class classification.
   - The model is then trained on the training dataset for 10 epochs with validation using `validation_dataset`.
   - **ModelCheckpoint** callback is used to save the best model based on validation accuracy.

### 5. **Predictions and Evaluation**:
   - After training, the model is used to predict the class for a specific image.
   - Predictions are output as probabilities for each of the 70 classes.
   - The class with the highest probability is selected as the predicted class, but the code also checks the top two predictions to potentially handle cases of ambiguous classification (e.g., if the top two predictions have similar probabilities).

### 6. **Saving the Model**:
   - The model weights are saved in the HDF5 format (`.h5`), so they can be reloaded later for inference.
   - The model is converted to TensorFlow Lite format (`.tflite`) using `TFLiteConverter` to make it suitable for deployment on mobile and embedded devices.

### 7. **Error Handling**:
   - There are warnings and errors related to TensorFlow and CUDA libraries, such as missing `libcudart.so.11.0` and `libnvinfer.so.7` files. These are related to GPU acceleration, which can be ignored if you are not using a GPU or if the appropriate libraries are not available.

### 8. **Model Deployment**:
   - The final model is saved as `model_v1_07_0.936.h5`, and the weights are saved in `.tflite` format for efficient inference on devices with lower computational resources.
     
### Key Points:
- **Transfer Learning**: The Xception model, pretrained on ImageNet, is used as a feature extractor.
- **Data Augmentation**: ImageDataGenerator is used to preprocess and augment the images.
- **Model Saving**: The model is saved in both `.h5` and `.tflite` formats for later use.
- **Error Handling**: Warnings regarding missing libraries (like CUDA libraries) are displayed but are typically ignorable if you don’t have the appropriate hardware.
  
