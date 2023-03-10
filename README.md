# Concrete Crack Classification Using Transfer Learning
 
The objective of this project is classify the condition of concrete either it was crack or not. The concrete image dataset have been separated into two label, 'positive' which is refering to crack concrete while 'negative' define uncrack concrete. Transfer learning of pretrained model (in Tensorflow.Keras (MobileNetV2) was used in this project as base model before the classification layer were developed.

## 1. Data loading 

The dataset was loaded using using one of the tensorflow.keras.utils method called '.image_dataset_from_directory'. This method will directly scan into the dataset folder and directly set the label for the images based on the folder name 'positive' and 'negative'. Some of the loaded image concrete dataset were displayed with its label below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/120147015/208714141-31abbab0-30bb-4f68-87a1-6e6433408b53.png" />
</p>


## 2. Data Preprocessing

In data loading, the datasets are split into 2, 'train_dataset' and 'validation_dataset'. The 'validation_dataset' were further split into 'validation_dataset' and 'test_dataset' with equal separation. The 'train_dataset' and 'test_dataset was converted into prefetch dataset. Therea are some data augmentation been done on the image which is displayed below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/120147015/208714126-f26cab0c-cb25-440c-9a2c-a184d13ebe21.png" />
</p>

## 3. Model Development

In the model development, the pretrained model (MobileNetV2), were set as a base model. The based model weights were set to use 'imagenet' and the layers of base model were frozen. This base model will act as feature extractor and will not affected when training the dataset. The base model summary can be seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/120147015/208714135-12e4e99e-0d8c-44b4-b10e-e3fbb4216137.png" />
</p>

After setting the base model. a classifier was developed to classify the concrete condition. This classifier will then combine with the base model to create a new model which will be used for training. The model summary is shown below.

<p align="center">
  <img src="https://github.com/acrimn123/Concrete_Crack_Classification/blob/main/model.png" />
</p>

## 4. Model compile and training

In this step, the model were compile using Adam as optimizer, Sparse categorical entropy as loss and mccuracy as metrics. The model were fit with 'train_dataset' and validate using the 'validation_dataset'. The graph of training were shown below using tensorboard.

<p align="center">
  <img src="https://user-images.githubusercontent.com/120147015/208714144-4a23c5d8-3349-48f3-bd3e-f1f78b6fbc42.png" />
</p>

## 5. Model Evaluation

After completing model training, the 'test_dataset' was used to perform prediction using the model. The classification report for the model are shown bwlow.

<p align="center">
  <img src="https://user-images.githubusercontent.com/120147015/208714139-8c8ac5b9-3474-4833-a1ff-1a04f36ce94b.png" />
</p>

## 6. Results and Discussion

From the model evaluation, we can see that the model can predict with almost 0.99 accuracy based on the 'test_dataset'. Although the model accuracy seem very good, the model training shows that the model is having an overfit graph. Further improvement can be made to this model to reduce overfit:
  - introducing dropout layer on the classifier
  - unfreeze some of the layer inside the pretrained model, so the weights are not entirely dependent on 'imagenet' weights.
  - change or schedule the learning rate when training

# Acknowledgment

  - 2018 ??? ??zgenel, ??.F., G??nen?? Sorgu??, A. ???Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings???, ISARC 2018, Berlin. 
The dataset can be obtained [here](https://data.mendeley.com/datasets/5y9wdsg2zt/2)
