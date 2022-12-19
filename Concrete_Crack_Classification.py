#%%
# 1. Import Packages
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from keras import layers,optimizers,losses,callbacks,applications
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import datetime

#%%
# 2. Data Loading
# Images Path
PATH=os.path.join(os.getcwd(),'dataset')

#%%
# 3.Data Preparation
BATCH_SIZE=16
IMG_SIZE=(160,160)

train_dataset=tf.keras.utils.image_dataset_from_directory(PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True,validation_split=0.2,subset='training',seed=123)
validation_dataset=tf.keras.utils.image_dataset_from_directory(PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True,validation_split=0.2,subset='validation',seed=123)


#%%
# 4.Display some examples
class_name= train_dataset.class_names

#Plot some examples
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_name[labels[i]])
        plt.axis('off')

#%%
# 5.Performing validation data
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

# %%
# 6. Convert the train, validation and test dataset into prefetch dataset
AUTOTUNE=tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
# 7. Image Processing
data_augmentation=keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(0.2)])

#Repeatedly apply data augmentation on one image and see the result
for image,labels in train_dataset.take(1):
    first_image=images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image=data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
# 8.Model Development
preprocess_input=applications.mobilenet_v2.preprocess_input
IMG_SHAPE=IMG_SIZE+(3,)
base_model=applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# %%
# Set the pretrained model as non-trainable(frozen)
base_model.trainable=False
base_model.summary()

# %%
#(C) Create the classifier
#Create the global average pooling layer
global_avg=layers.GlobalAveragePooling2D()
#Create an output layer
output_layer=layers.Dense(len(class_name),activation='softmax')

# %%
#11. Link the layers together to form a pipeline
inputs=keras.Input(shape=IMG_SHAPE)

x=data_augmentation(inputs)
x=preprocess_input(x)
x=base_model(x)
x=global_avg(x)
x=layers.Dropout(0.3)(x)
outputs=output_layer(x)

#Instantiate the full model pipeline
model=keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())
keras.utils.plot_model(model,show_shapes=True)
# %%
#12. Compile the model
optimizer=optimizers.Adam()
loss=losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#13. Evaluate the model before training 
loss0, acc0=model.evaluate(validation_dataset)

print("Evaluation Before Training")
print("Loss= ",loss0)
print("Accuracy= ",acc0)

# callbacks
log_path=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb=callbacks.TensorBoard(log_dir=log_path)

#%%
#14. Model training
history=model.fit(train_dataset,validation_data=validation_dataset,epochs=5,callbacks=[tb])

#%% 
#15. Classification report
# Use model to prform prediction

image_batch,label_batch = test_dataset.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

# Evaluate prediction
print('Classification Report:\n', classification_report(y_pred,label_batch,zero_division=0))

#%%
#15.Model save
save_path = os.path.join(os.getcwd(),'image_classification_model.h5')
model.save(save_path)

