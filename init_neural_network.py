from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import tensorflowjs

PATH = os.getcwd()

# Define data path
data_path = PATH + '/datasets'
data_dir_list = os.listdir(data_path)
classes = ['black_olive', 'green_olive']
root_dir = 'datasets/'  # data root path
classes_dir = data_dir_list  # total labels

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        root_dir + 'train/',  # This is the source directory for training images
        classes = ['black_olive', 'green_olive'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        root_dir + 'val/',  # This is the source directory for training images
        classes = ['black_olive', 'green_olive'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=19,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

model = tf.keras.models.Sequential([
# Note the input shape is the desired size of the image 200x200 with 3 bytes color
# This is the first convolution
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
# The second convolution
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The third convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The fourth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# # The fifth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# Flatten the results to feed into a DNN
tf.keras.layers.Flatten(),
# 512 neuron hidden layer
tf.keras.layers.Dense(512, activation='relu'),
# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy',
optimizer=tf.keras.optimizers.Nadam(),
metrics='accuracy')
print(model.summary())


# Start training with model and data
history = model.fit(train_generator,
steps_per_epoch=8,
epochs=15,
verbose=1,
validation_data = validation_generator,
validation_steps=8)

model.evaluate(validation_generator)
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)


def show_fit_plots():
        # Model accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Model loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


path = 'datasets/test/green_olive/green_olive_73.jpg'
img = image.load_img(path, target_size=(200, 200))
x = image.img_to_array(img)
plt.imshow(x / 255.)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
predict = model.predict(images, batch_size=10)
print(predict[0])
if predict[0] < 0.5:
        print(path + " is a black olive")
else:
        print(path + " is a green olive")


with open('class_names.txt', 'w') as file_handler:
    for item in classes:
        file_handler.write("{}\n".format(item))

model.save('keras.h5')
