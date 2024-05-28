import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models # type: ignore
import tensorflow as tf

model = models.load_model("btc_model.keras")

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

img_path = '../brain-tumor-class/brain.jpeg'
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (64, 64))

img = img / 255.0

plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()

img_array = np.expand_dims(img, axis=0)
prediction = model.predict(img_array)
index = np.argmax(prediction)
print(class_names[index])

