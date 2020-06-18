import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

np.random.seed(12345)

path = 'English/Bmp'
batch_size = 128
epochs = 70
IMG_HEIGHT = 50
IMG_WIDTH = 50

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=123,
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    path, # same directory as training data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=123,
    subset='validation') # set as validation data

images = []
iterations = 5
label =list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
num_classes = len(label)
sample_training_images, idx = next(train_generator)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# plotImages(sample_training_images[:5])


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.5),
    
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    # top layer
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

filepath="best70.hdf5"
checkpoint = [ModelCheckpoint(filepath, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='min')]

history = model.fit_generator(
        train_generator,
        steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size),
        validation_data = validation_generator, 
        validation_steps = np.ceil(train_generator.samples/train_generator.batch_size),
        epochs = epochs,
        callbacks=checkpoint)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model70.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Evaluate the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# test_loss, test_acc = model.evaluate(te_images, te_labels, verbose=2)
# print(test_acc)