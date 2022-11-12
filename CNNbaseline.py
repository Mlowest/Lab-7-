# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:00:11 2022
## https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
@author: Michael
"""
# In[0] prepartation and model implementation


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



## verify that the dataset looks correct
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# define cnn BaseLine model

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same'))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same'))
model.add(BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same' ))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same'))
model.add(BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding ='same' ))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.5))

# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))

# Here's the complete architecture of your model:
model.summary()
# In[1]
# create data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)#, vertical_flip = True)
# prepare iterator
it_train = datagen.flow(train_images, train_labels, batch_size=64)
steps = int(train_images.shape[0] / 64)


# In[3] improving model by running the model into augmented data
## Compile and train the model with augmented data
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train_images, train_labels
history = model.fit(it_train, steps_per_epoch = steps, epochs=20, batch_size = 64,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# In[4] re-run the model through normal data after its been trained with augmented data
## Compile and train the model final run through with unaugmented data

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train_images, train_labels
history = model.fit(train_images, train_labels, epochs = 20,
                    validation_data=(test_images, test_labels))


## Evaluate the model after training without data augmentation
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
plt.show()

print(test_acc)
print()
print()

# In[5] improving model by running the model into augmented data
## Compile and train the model with augmented data
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train_images, train_labels
history = model.fit(it_train, steps_per_epoch = steps, epochs=20 , batch_size = 64,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)


# In[6] re-run the model through normal data after its been trained with augmented data
## Compile and train the model final run through with unaugmented data

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train_images, train_labels
history = model.fit(train_images, train_labels, epochs = 20,
                    validation_data=(test_images, test_labels))


## Evaluate the model after training without data augmentation
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
plt.show()

print(test_acc)
print()
print()

# In[98]
# load the trained CIFAR10 model for debug
#model = load_model('MyGroup_CIFARmodel.h5')
# In[99] save the model
## save trained model in file "MyGroup_CIFARmodel.h5"
model.save('MyGroup_CIFARmodel.h5')
# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 

