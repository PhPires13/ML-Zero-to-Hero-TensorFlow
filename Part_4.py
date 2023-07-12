# Download zip containing the data
# https://public.roboflow.com/classification/rock-paper-scissors/1
# http://laurencemoroney.com/rock-paper-scissors-dataset
# Train: https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
# Test: https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
import os.path
import zipfile

import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator

# Extract the training and testing files
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
TRAINING_DIR = os.path.join('Images', 'RPS-Train')
zip_ref.extractall(TRAINING_DIR)
zip_ref.close()

local_zip = 'rps-valid.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
VALIDATION_DIR = os.path.join('Images', 'RPS-Valid')
zip_ref.extractall(VALIDATION_DIR)
zip_ref.close()


# Generates images for the training
training_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical'
)


#
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='categorical'
)


# Neural Network Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile Neural Network
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Fit the data
history = model.fit_generator(train_generator, epochs=25,
                              validation_data=validation_generator,
                              verbose=1)

# With the model trained you can call
# classes = model.predict(images, batch_size=10)
