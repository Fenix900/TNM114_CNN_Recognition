import numpy as np
import urllib
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import requests
import tensorflowjs as tfjs


#The classes we want to fetch
instruments = ["piano", "cello", "drums", "clarinet", "guitar", "trombone", "trumpet", "violin"]
number_of_classes = len(instruments)
'''
#The ground pathway to the url to get the classes
groundURL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

print(number_of_classes)

for i in instruments:
  #print("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"+i+".npy")
  url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"+i+".npy"
  response = requests.get(url)
  print(response.status_code)
  save_path = os.path.join("python/data", i)+".npy"
  print(save_path)
  with open(save_path, 'wb') as file:
        file.write(response.content)

        '''

nmbr_of_samples = 100; #How many images we want from the database per class
x_images = np.empty([0,784])
y_label = np.empty(0)

for i in instruments:
  path = os.path.join("python/data", i)+".npy"
  data = np.load(path)
  data = data[0:nmbr_of_samples]
  label = i

  x_images = np.concatenate((x_images,data),axis=0)
  y_label = np.append(y_label, [i]*nmbr_of_samples)

x_images, y_label = shuffle(x_images, y_label, random_state=42)

from sklearn.model_selection import train_test_split

#Reshape the images to 28x28 and then normalize them to 0-1 values
x_reshaped = np.reshape(x_images,(len(x_images),28,28,1))
x_images = x_reshaped.astype('float32')/255

#Split the data to 80-20 ratio of training- to test-sets
training_images, test_images, training_labels, test_labels = train_test_split(
    x_images, y_label, test_size=0.2, random_state=42, stratify=y_label)

#Model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))  # Added larger filter
model.add(layers.BatchNormalization())  # Added batch normalization

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))  # Added a dense layer
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))  # Added dropout
model.add(layers.Dense(number_of_classes, activation="softmax"))  # Number of classes in the last layer
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
test_labels_encoded = label_encoder.transform(test_labels)
training_labels_one_hot = to_categorical(training_labels_encoded, num_classes=number_of_classes)
test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=number_of_classes)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Added early stopping
history = model.fit(training_images, training_labels_one_hot,
                    epochs=10, validation_data=(test_images, test_labels_one_hot),
                    callbacks=[early_stopping])

plt.plot(history.history['accuracy'], label='accuracy of training data')
plt.plot(history.history['val_accuracy'], label='accuracy of test data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels_one_hot, verbose=2)

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
correct_predictions = np.sum(predicted_classes == test_labels_encoded)
accuracy = correct_predictions / len(test_labels_encoded)
print(f"Manuell beräknad accuracy: {accuracy}")
print(tf.version.VERSION)
import random

for i in range(30):
  '''if(predicted_classes[i] != test_labels_encoded[i]):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {label_encoder.inverse_transform([predicted_classes[i]])[0]}, Actual: {label_encoder.inverse_transform([test_labels_encoded[i]])[0]}")
    plt.show()'''

  '''idx = random.randint(0, len(test_images) - 1)
  plt.imshow(-test_images[idx].reshape(28, 28), cmap='gray')
  plt.title(f"Predicted: {label_encoder.inverse_transform([predicted_classes[idx]])[0]}, Actual: {label_encoder.inverse_transform([test_labels_encoded[idx]])[0]}")
  plt.show()'''

model.save('python/myModel/pythonModel.h5')

tfjs.converters.save_keras_model(model, "doodle_model_js")


from PIL import Image
import numpy as np

# Load the image
image = Image.open("pianoTest.png")
# Convert the image to grayscale
gray_image = image.convert("L")

# Convert grayscale image to a numpy array
gray_array = np.array(gray_image)

# Invert the grayscale values
inverted_array = 255 - gray_array

# Convert the array back to an image
inverted_image = Image.fromarray(inverted_array)

print(inverted_image)

print(x_reshaped[1])