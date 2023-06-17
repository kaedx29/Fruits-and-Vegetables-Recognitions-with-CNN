import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
cnn= tf.keras.models.load_model("D:/coding Stress/tugas/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/trained_model.h5")

faceCascade = cv2.CascadeClassifier('D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = load_model("trained_model.h5")
print("[INFO] finish load model...")
video_capture= cv2.VideoCapture(0)

import cv2
image_path ="D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/Food and Vegetable recognition/test/pear/Image_1.jpg"
img = cv2.imread(image_path)
plt.imshow(img)
plt.title('a')
plt.xticks([])
plt.yticks([])
plt.show()

image= tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr= tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions =cnn.predict(input_arr)

print(predictions)
print(max(predictions[0]))

test_set = tf.keras.utils.image_dataset_from_directory(
    'D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/Food and Vegetable recognition/test',
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

test_set.class_names
result_index= np.where(predictions[0]== max(predictions[0]))
print(result_index)

#displayimage
plt.imshow(img)
plt.title('Buah')
plt.xticks([])
plt.yticks([])
plt.show()

print("it's a{}".format(test_set.class_names[result_index[0][0]]))