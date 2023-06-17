import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

training_set = tf.keras.utils.image_dataset_from_directory(
    'D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/Food and Vegetable recognition/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    'D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/Food and Vegetable recognition/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=36,activation='softmax'))

#compiling
cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
training_history= cnn.fit(x=training_set,validation_data=validation_set,epochs=30)
cnn.save('trained_model.h5')
training_history.history

#recording history
import json
with open('training_hist.json','w') as f:
   json.dump(training_history.history,f)
   
print(training_history.history.keys())
print("validation set Accuracy : {} %".format (training_history.history['val_accuracy'][-1]*100))

#Training Visualization
epochs =[i for i in range(1,31)]
plt.plot(epochs,training_history.history ['accuracy'],color='red')
plt.xlabel('Ephocs')
plt.ylabel('Training Accuracy')
plt.title ('Visualization of Training Accuracy Result')
plt.show()

#Validation Visualization
plt.plot(epochs,training_history.history['val_accuracy'],color='blue')
plt.xlabel('Epochs')
plt.ylabel('validation accuracy')
plt.title('visualization of validation accuracy result')
plt.show()