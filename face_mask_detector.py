# #importing all the required libraries
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Load and generate training data
training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                    '/home/anshul/Desktop/Deep_Learning/Project/Project_New_Files/data',
                    validation_split = 0.2,
                    subset = "training",
                    seed = 42,
                    image_size = (200, 200),
                    batch_size = 40
                    )

# Load and generate testing data
testing_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                    '/home/anshul/Desktop/Deep_Learning/Project/Project_New_Files/data',
                    validation_split = 0.2,
                    subset = "validation",
                    seed = 42,
                    image_size = (200, 200),
                    batch_size = 40
                    )

# Add class names to training dataset
class_names = training_dataset.class_names

# Visulaization of images from dataset with  labels
plt.figure(figsize=(8, 8))
for images, labels in training_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Configure dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_dataset = training_dataset.cache().prefetch(buffer_size =AUTOTUNE)
testing_dataset = testing_dataset.cache().prefetch(buffer_size = AUTOTUNE)

# Face mask detection model with
# 4 convolutional layers with relu activation function and maxpooling layers
# 1 flatten layers and 1 fully connected layer and 1 output layer
model = keras.models.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255),
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
        ])

# Train the model
model.compile(optimizer=Adam(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
training_time_start = time.time()
hist = model.fit(training_dataset, validation_data = testing_dataset, epochs=2)
training_time_end = time.time()
elpased_training_time = training_time_end - training_time_start
print("Elaped Training Time: ", round(elpased_training_time, 3), "seconds")

# Inference on new dataset
prediction = model.predict(testing_dataset)

#plot for accuracy and loss
plt.figure(1)

plt.subplot(2, 1, 1)
epochs = range(1,3)
plt.plot(epochs, hist.history['accuracy'], label = 'Training Accuracy')
plt.plot(epochs, hist.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy of Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, hist.history['loss'], label = 'Training Loss')
plt.plot(epochs, hist.history['val_loss'], label = 'Validation Loss')
plt.title('Training and Validation Loss of Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

image,label = next(iter(testing_dataset))

# Predicted images with prediction result
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(True)
    plt.axis('off')
    plt.imshow(image[i].numpy().astype("uint8"))
    plt.xlabel("Actual: " + class_names[label[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
plt.show()

# Save model for real time face detection using open CV
# face_mask_classifier.h5
model.save("face_mask_classifier.h5", hist)
print("model saved to disk")

print("[INFO] Script Exiting")
