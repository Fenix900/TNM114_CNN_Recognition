from PIL import Image
import numpy as np
import keras
import os
import matplotlib.pyplot as plt

instruments = ["piano", "cello", "drums", "clarinet", "guitar", "trombone", "trumpet", "violin"]
loaded_model = keras.saving.load_model("python/NewSavedModel/model.keras")

# Load the image
image = Image.open("/Users/filipk/Downloads/import_img.png")
# Resize the image to 28x28
image_resized = image.resize((28, 28))
# Pick out the alpha channel to use
r, g, b, a = image_resized.split()
gray_array = np.array(a)


#gray_array = np.array(a).flatten()
gray_array = np.where(gray_array > 120, 255, gray_array)
a_scaled = Image.fromarray(gray_array.astype('uint8')) #For showing
gray_array_flatten = gray_array.flatten()


# Plot the image
#plt.imshow(a,"gray_r")
#plt.axis('off')  # Turn off axis numbers and ticks
#plt.title("Image being predicted")
#plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Alpha channel before scaling
ax[0].imshow(a, cmap="gray_r")
ax[0].axis('off')
ax[0].set_title('Before Scaling')

# Alpha channel after scaling
ax[1].imshow(a_scaled, cmap="gray_r")
ax[1].axis('off')
ax[1].set_title('After Scaling')
plt.show()

# Invert the grayscale values
normalized_array = np.reshape(gray_array_flatten,(28,28,1))/255
#print(normalized_array)

prediction_img = np.expand_dims(normalized_array,axis=0)
print(prediction_img)

prediction = loaded_model.predict(prediction_img)
predicted_class = np.argmax(prediction, axis=1)
print("\n\n Predicted instrument: " + instruments[predicted_class[0]] + "\n")

os.remove("/Users/filipk/Downloads/import_img.png")