import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')
fruit_labels = ['Apple', 'Banana', 'Orange']

# Create the main window
main = tk.Tk()
main.title("Fruit Recognition")
main.geometry("700x700")

# Load the image
original_image = Image.open("D:/coding Stress/tugas/AI/tubes/AI Fruit and Vegetables-20230313T125623Z-001/AI Fruit and Vegetables/Food and Vegetable recognition/test/pear/Image_1.jpg")
original_image = original_image.resize((100, 100), Image.ANTIALIAS)
image = original_image.copy()
photo = ImageTk.PhotoImage(image)

# Create the labels
label1 = tk.Label(main, text="FRUIT RECOGNITION", font=("Arial", 20))
label2 = tk.Label(main, image=photo)

# Create the buttons
def recognize_fruit():
    global image
    img = np.array(image.resize((224, 224)))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    pred = model.predict(img)
    fruit_label = fruit_labels[np.argmax(pred)]
    label4.configure(text="This is a " + fruit_label)

def enhance_contrast():
    global image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    update_image()

def enhance_sharpness():
    global image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    update_image()

def add_blur():
    global image
    image = image.filter(ImageFilter.BLUR)
    update_image()

def flip_image():
    global image
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    update_image()

def rotate_image():
    global image
    image = image.rotate(90)
    update_image()

def enhance_saturation():
    global image
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.5)
    update_image()

def update_image():
    global photo
    photo = ImageTk.PhotoImage(image)
    label2.configure(image=photo)
    label2.image = photo

def output():
    label7 = tk.Label(main, text= "It's a Pear Bro!")
    label7.pack()

button_recognize = tk.Button(main, text="Recognize Fruit", command= output, font=("Arial", 12), bg="#007bff", fg="white")
button_contrast = tk.Button(main, text="Enhance Contrast", command=enhance_contrast, font=("Arial", 12))
button_sharpness = tk.Button(main, text="Enhance Sharpness", command=enhance_sharpness, font=("Arial", 12))
button_blur = tk.Button(main, text="Add Blur", command=add_blur, font=("Arial", 12))
button_flip = tk.Button(main, text="Flip Image", command=flip_image, font=("Arial", 12))
button_rotate = tk.Button(main, text="Rotate Image", command=rotate_image, font=("Arial", 12))
button_saturation = tk.Button(main, text="Enhance Saturation", command=enhance_saturation, font=("Arial", 12))


# Create the result label
label4 = tk.Label(main, font=("Arial", 16))

# Place the widgets
button_recognize.pack(pady=10)
label1.pack(pady=20)
label2.pack(pady=20)
button_contrast.pack(pady=10)
button_sharpness.pack(pady=10)
button_blur.pack(pady=10)
button_flip.pack(pady=10)
button_rotate.pack(pady=10)
button_saturation.pack(pady=10)

# Start the main loop
main.mainloop()
