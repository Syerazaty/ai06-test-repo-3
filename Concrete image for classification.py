# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:30:15 2022

@author: HP
"""
#Import necessary packages
import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz\bin'

#1. Load image files, then split into train-validation set
file_path = r"C:\Users\HP\Documents\TensorFlow\datasets\Concrete Crack Images for Classification"
data_dir = pathlib.Path(file_path)

##Define path and create tensorflow dataset with the special function
SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.1,subset='training',seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.1,subset='validation',seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)

#%%
#2. Further split validation set, so that we obtain validation and test data
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#Create prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data preparation is completed at this step

#%%
#3. Create model by applying transfer learning, we are using MobileNetV2 for this project.
#3.1. Define a layer that preprocess inputs for the transfer learning model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#3.2. Create base model with MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#3.3. Freeze the base model and view the model structure
base_model.trainable = False
base_model.summary()

#%%
#3.4. Create classification layers with global average pooling and dense layer
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
import pydot
import pydotplus
from pydotplus import graphviz

class_names = train_dataset.class_names
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
output_dense = tf.keras.layers.Dense(len(class_names),activation='softmax')

#3.5. Use functional API to build the entire model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg_pool(x)
outputs = output_dense(x)

model = tf.keras.Model(inputs,outputs)

#Print out the model structure
model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#%%
#3.6. Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam,loss=loss,metrics=['accuracy'])

#%%
#4. Perform training
EPOCHS = 10
base_log_path = r"C:\Users\HP\Documents\Program AI06\p3_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
history = model.fit(train_dataset_pf,validation_data=validation_dataset_pf,epochs=EPOCHS,callbacks=[tb_callback,es_callback])

#%%
#5. Evaluate with test dataset
test_loss,test_accuracy = model.evaluate(test_dataset_pf)

print('------------------------Test Result----------------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#6. Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)

#%%
#7. Show some prediction results
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"C:\Users\HP\Documents\GitHub\ai06-test-repo-3\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()