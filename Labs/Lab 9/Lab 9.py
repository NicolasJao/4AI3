import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 2. Visualizing the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

# 3. Normalizing the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. Building the CNN model (10 classes)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

# 5. Evaluating accuracy on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# 6a. Creating the binary dataset
# vehicle classes are 0 (airplane), 1 (automobile), 8 (ship), 9 (truck)
# vehicle = 1, non-vehicle = 0
vehicle_classes = [0, 1, 8, 9]
y_train_bin = np.isin(y_train, vehicle_classes).astype(int)
y_test_bin = np.isin(y_test, vehicle_classes).astype(int)

# 6b. Defining the binary classification model
binary_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6c. Training and evaluating the binary model
binary_model.fit(x_train, y_train_bin, epochs=10)
bin_loss, bin_acc = binary_model.evaluate(x_test, y_test_bin, verbose=2)
print('Binary test accuracy:', bin_acc)

# 7. What I changed for binary classification
# labels were changed to 0 or 1 (vehicle = 1, non-vehicle = 0)
# last Dense layer was changed from 10 units with softmax to 1 unit with sigmoid
# loss was changed from sparse_categorical_crossentropy to binary_crossentropy
