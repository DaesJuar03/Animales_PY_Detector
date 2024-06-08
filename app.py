import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Directorios de datos
train_dir = 'data/train'
validation_dir = 'data/validation'
# Cargar una imagen para predecir
img_path = './data/test/img/nombre_de_imagen.jpg'  # Ruta de la imagen (Inserta las imagenes en esa ruta para que pueda predecirlas)

# Generador de datos para el entrenamiento con aumentación de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de datos para la validación (sin aumentación, solo reescalado)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Construye el modelo
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# Entrena el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Número de imágenes en el set de entrenamiento dividido por el tamaño del batch
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50   # Número de imágenes en el set de validación dividido por el tamaño del batch
)

# Guardar el modelo
model.save('cats_and_dogs_classifier.h5')

# Evaluar el modelo
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Precisión de entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()

plt.show()

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('cats_and_dogs_classifier.h5')


img = image.load_img(img_path, target_size=(150, 150))  # Cargar la imagen y la redimensiona
img_array = image.img_to_array(img)  # convierte la imagen en un array
img_array = np.expand_dims(img_array, axis=0)  

# Preprocesa la imagen (reescala los píxeles)
img_array /= 255.

# Predicción
prediction = model.predict(img_array)

# Muestra los resultados de la prediccion (Revisar terminal)
if prediction[0][0] > 0.5:
    print("Es un perro")
else:
    print("Es un gato")
