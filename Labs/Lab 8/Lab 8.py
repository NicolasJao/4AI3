# Theoretical Exercises

# 1. activation functions
# relu: gives max(0, x). used in hidden layers most of the time because it is fast
# sigmoid: squashes output between 0 and 1. used in the output layer for binary classification
# tanh: squashes output between -1 and 1. used in hidden layers
# softmax: makes the outputs into probabilities that add up to 1. used in the output layer for multi-class classification
# linear: no change to the output. used in the output layer for regression
# leaky_relu: like relu but lets a small negative value through. used when relu neurons die
# elu / selu: smooth versions of relu. used in deeper networks
# softplus / softsign: smooth versions of relu and tanh. not used as often
# gelu / swish: smooth relu like functions. used in big deep networks

# 2. compile options
# optimizer (how the model updates its weights)
# SGD: basic gradient descent, simple
# RMSprop: changes the learning rate for each weight, often used for RNNs
# Adam: most popular, works well in most cases
# Adagrad / Adadelta / Adamax / Nadam: other versions that change the learning rate in different ways
# loss function (what the model tries to make small)
# MeanSquaredError / MeanAbsoluteError: for regression
# BinaryCrossentropy: for binary classification (0 or 1)
# CategoricalCrossentropy: for multi-class classification with one hot labels
# SparseCategoricalCrossentropy: for multi-class classification with integer labels
# metrics (just to see how the model is doing, they do not change the training)
# accuracy: how many predictions are correct, for classification
# Precision / Recall / AUC / FalseNegatives: other classification scores
# MeanAbsoluteError / MeanSquaredError: for regression

# 3. fit parameters
# epochs: how many times the model goes through the whole training data. epochs=1000 means 1000 times
# verbose: how much is printed while training. 0 is nothing, 1 is a progress bar, 2 is one line per epoch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical

# Part 1: Neural Networks I (diabetes dataset)

# 1. loading dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 2. visualizing the data
sns.pairplot(df)
plt.show()

# 3. splitting dataset into train, validation and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=67)

# 4. normalizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5a. creating the model with 10 units in the first dense layer
model = Sequential([Dense(10, activation="relu", input_shape=[X_train_scaled.shape[1]]), Dense(1, activation="sigmoid")])

# 5b. training the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train_scaled, y_train, epochs=100, verbose=0, validation_data=(X_val_scaled, y_val))

# 5c. visualizing the training process (epoch vs loss)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train Error", "Val Error"], loc="upper right")
plt.show()

# 5d. evaluating the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# 6. hyperparameter tuning
units = [10, 50]
epochs = [100, 1000]
optimizers = ["adam", "SGD"]
best_accuracy = 0
best_units = 0
best_epochs = 0
best_optimizer = ""
for u in units:
    for e in epochs:
        for o in optimizers:
            # build the model using u
            model = Sequential([
                Dense(u, activation="relu", input_shape=[X_train_scaled.shape[1]]),
                Dense(1, activation="sigmoid")
            ])
            # compile the model using o
            model.compile(optimizer=o, loss="binary_crossentropy", metrics=["accuracy"])
            # train the model using e
            model.fit(X_train_scaled, y_train, epochs=e, verbose=0, validation_data=(X_val_scaled, y_val))
            # get the score of this model
            loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            print("Units:", u, "Epochs:", e, "Optimizer:", o, "Test Accuracy:", accuracy)
            # find best parameter which has best score
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_units = u
                best_epochs = e
                best_optimizer = o
print("Best units:", best_units)
print("Best epochs:", best_epochs)
print("Best optimizer:", best_optimizer)
print("Best test accuracy:", best_accuracy)

# Part 2: Neural Networks II (MNIST dataset)

# 1. loading dataset
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()
images = np.concatenate([images_train, images_test])
labels = np.concatenate([labels_train, labels_test])

# 2. visualizing the data
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title("Label: " + str(labels[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()

# 3. preprocessing the data
# 3a. reshape to have one color channel
images = images.reshape(-1, 28, 28, 1)

# 3b. normalize the pixel values to 0 - 1
images = images / 255.0

# 4. splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=67)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=67)
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# 5. building and training the neural network model
# 5a and 5b. build the model with relu, tanh and sigmoid in the hidden layer and compare
activation_functions = ["relu", "tanh", "sigmoid"]
for activation in activation_functions:
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    # 5c. compile and train
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64, verbose=0)
    # 5d. plot accuracy and loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy (" + activation + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Val"], loc="lower right")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss (" + activation + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Val"], loc="upper right")
    plt.show()

    # 6. evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Activation:", activation, "Test Loss:", score[0], "Test Accuracy:", score[1])

# 7. experimenting with activation functions and dropout rates
dropout_rates = [0.2, 0.5]
best_accuracy = 0
best_activation = ""
best_dropout = 0
for activation in activation_functions:
    for dropout_rate in dropout_rates:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(128, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64, verbose=0)
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f"Activation: {activation}, Dropout: {dropout_rate}, Test Accuracy: {score[1]}")
        if score[1] > best_accuracy:
            best_accuracy = score[1]
            best_activation = activation
            best_dropout = dropout_rate

# 8. analyzing and summarizing the results
print("Best activation:", best_activation)
print("Best dropout rate:", best_dropout)
print("Best test accuracy:", best_accuracy)

'''
This is one result after running the file:
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7273 - loss: 0.5783 
Test loss: 0.5783242583274841
Test accuracy: 0.7272727489471436
Units: 10 Epochs: 100 Optimizer: adam Test Accuracy: 0.7467532753944397
Units: 10 Epochs: 100 Optimizer: SGD Test Accuracy: 0.7207792401313782
Units: 10 Epochs: 1000 Optimizer: adam Test Accuracy: 0.7272727489471436
Units: 10 Epochs: 1000 Optimizer: SGD Test Accuracy: 0.7532467246055603
Units: 50 Epochs: 100 Optimizer: adam Test Accuracy: 0.7597402334213257
Units: 50 Epochs: 100 Optimizer: SGD Test Accuracy: 0.7272727489471436
Units: 50 Epochs: 1000 Optimizer: adam Test Accuracy: 0.701298713684082
Units: 50 Epochs: 1000 Optimizer: SGD Test Accuracy: 0.7467532753944397
Best units: 50
Best epochs: 100
Best optimizer: adam
Best test accuracy: 0.7597402334213257
Activation: relu Test Loss: 0.09120839834213257 Test Accuracy: 0.9766428470611572
Activation: tanh Test Loss: 0.09517212957143784 Test Accuracy: 0.9730714559555054
Activation: sigmoid Test Loss: 0.08573217689990997 Test Accuracy: 0.9732857346534729
Activation: relu, Dropout: 0.2, Test Accuracy: 0.9786428809165955
Activation: relu, Dropout: 0.5, Test Accuracy: 0.9754999876022339
Activation: tanh, Dropout: 0.2, Test Accuracy: 0.9732142686843872
Activation: tanh, Dropout: 0.5, Test Accuracy: 0.9635000228881836
Activation: sigmoid, Dropout: 0.2, Test Accuracy: 0.9727143049240112
Activation: sigmoid, Dropout: 0.5, Test Accuracy: 0.9698571562767029
Best activation: relu
Best dropout rate: 0.2
Best test accuracy: 0.9786428809165955

Process finished with exit code 0
'''

# the best combination was relu with dropout 0.2, test accuracy 0.9786 (97.86%)

# activation functions
# relu was the best. it got 97.86% with dropout 0.2 and 97.55% with dropout 0.5
# sigmoid was second on average. it got 97.27% with 0.2 and 96.99% with 0.5
# tanh was the worst on average. it got 97.32% with 0.2 and 96.35% with 0.5
# relu is probably the best because it does not squash the values, so the model
# learns faster in 20 epochs. sigmoid and tanh squash the values, so learning is slower

# dropout rates
# dropout 0.2 gave a higher accuracy than 0.5 for all three activation functions
# the drop was small for relu (about 0.3%) and sigmoid (about 0.3%) but bigger for tanh (about 1%)
# 0.5 probably hurt because it turns off half of the 128 neurons while training,
# and the model only has one hidden layer and only 20 epochs, so it learns less
# 0.2 keeps more neurons so the model learns better

# overall all the results are between 96.35% and 97.86%, so the differences are small
# and every combination works well on MNIST. the model also scored about the same in
# step 5 (relu 97.66%, tanh 97.31%, sigmoid 97.33%) which shows relu is the best there too
