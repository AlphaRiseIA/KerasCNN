import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#------------------------------------------------------------------
#This is a image generator that adapt the images and make them readable for the generator
datagen = ImageDataGenerator(rescale=1./255)

#This is where he will get the images, with the directory
img_dir = "/ruta/a/directorio/de/imagenes"
img_width, img_height = 150, 150
#We create the image "transformer"
train_generator = datagen.flow_from_directory(
        img_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
#------------------------------------------------------------------
#This is the CNN
conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 3))
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
#------------------------------------------------------------------
#This is the NN and where we build up the sequential model
dense1 = tf.keras.layers.Dense(units=3)
dense2 = tf.keras.layers.Dense(units=3)
out = tf.keras.layers.Dense(units=1, activation='relu')
model = tf.keras.Sequential([conv1, pool1, conv2, pool2, tf.keras.layers.Flatten(), dense1, dense2, out])
#------------------------------------------------------------------
#Now we compile
model.compile(optimizer='adam', loss='mean_squared_error')
#And now we train it with the image adapter, with out it, it will be impossible reading the images
history = model.fit(train_generator, epochs=1000, verbose=False)