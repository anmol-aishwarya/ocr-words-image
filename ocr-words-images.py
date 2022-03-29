# -*- coding: utf-8 -*-
"""
# Recognizing words in images
Optical character recognition has been the focus of much ML research. Solve the more general OCR task by recognizing the word from a given image on a small vocabulary of words. The first letter of each word was removed since these were capital letters that would make the task harder.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

"""## Data Loading & Visualization"""

data_lines = []
target_lines = []
folds = []
with open('/content/sample_data/letter.data') as f:
    data_line = []
    target_line = []
    for line in f:
        # print(line.split('\t')[1])
        l = line.strip().split('\t')
        pixels = np.array([float(digit) for digit in l[6:]]).reshape([16,8])
        letter = l[1]
        data_line.append(pixels)
        target_line.append(letter)
        if float(l[2]) == -1:
            # end of line
            # print(target_line)
            data_lines.append(data_line)
            target_lines.append(target_line)
            folds.append(int(l[5]))
            data_line = []
            target_line = []

len(data_lines)

sum([len(x) for x in data_lines])/len(data_lines)

np.unique(folds, return_counts=True)

sns.countplot(x=folds)
plt.xlabel('Fold')
plt.title('Number of Words in each Fold')
plt.show()

def show_line(data_line, target_line=None):
    # number of characters in the line
    n = len(data_line)
    if n < 1:
        print("Empty line")
        return
    
    fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
    for i in range(n):
        # read the character image
        im = data_line[i]
        ax = axes[i]
        ax.imshow(im)
        if target_line:
            ax.set_xlabel(target_line[i], fontsize=18)
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    fig.suptitle('Handwritten Image')
    #plt.tight_layout(pad=1, w_pad=-20, h_pad=0)
    plt.show()

idx = 4000
show_line(data_lines[idx], target_lines[idx])

idx = 2000
show_line(data_lines[idx], target_lines[idx])

# Find out how many unique characters are there in the dataset
Chars = set([c for line in target_lines for c in line])
len(Chars)

# Clearly only English alphabets

"""## Model 1: Recognizing Characters using SVM

Image/Character Recognition Using Support Vector Machine (SVM) model.

We perform 10-fold cross validation with the provided folds in the data itself.
"""

# Prepare training data & testing data using 9 fold to 1 fold ratio
test_fold = 9
X_train, y_train = [], []
X_test, y_test = [], []

for data_line, target_line, fold in zip(data_lines, target_lines, folds):
    for img, target in zip(data_line, target_line):
        if fold == test_fold:
            X_test.append(img.flatten())
            y_test.append(target)
        else:
            X_train.append(img.flatten())
            y_train.append(target)

# Train an SVM Classifier
svm_clf = SVC(random_state=0).fit(X_train, y_train)

svm_clf.score(X_test, y_test)

# kNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_clf.score(X_test, y_test)

# Logistic Regression
log_clf = LogisticRegression(random_state=0, solver='liblinear')
log_clf.fit(X_train, y_train)
log_clf.score(X_test, y_test)

# Neural Network
NN_clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=(128, 64))
NN_clf.fit(X_train, y_train)
NN_clf.score(X_test, y_test)

total_acc = 0
for test_fold in range(10):
    # Prepare training data & testing data using 9 fold to 1 fold ratio
    # test_fold = 9
    X_train, y_train = [], []
    X_test, y_test = [], []

    for data_line, target_line, fold in zip(data_lines, target_lines, folds):
        for img, target in zip(data_line, target_line):
            if fold == test_fold:
                X_test.append(img.flatten())
                y_test.append(target)
            else:
                X_train.append(img.flatten())
                y_train.append(target)

    print("Cross Validation Step %d" %(test_fold+1))
    # Train an SVM Classifier
    svm_clf = SVC(random_state=0).fit(X_train, y_train)
    acc = svm_clf.score(X_test, y_test)
    print('Accuracy = %.3f' %(acc))
    total_acc += acc
    
print('Average Accuracy = %.3f' %(total_acc/10))

# Predictions
idx = 2400
predictions = [svm_clf.predict(x.reshape([1,-1])) for x in data_lines[idx]] 
show_line(data_lines[idx], predictions)

"""## Model 2: Using CNN
Using convolutional Neural Network (CNN)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

# Model / data parameters
num_classes = 26
input_shape = (16, 8, 1)
a = ord('a')

# Prepare training data & testing data using 9 fold to 1 fold ratio
test_fold = 9
x_train, y_train = [], []
x_test, y_test = [], []

for data_line, target_line, fold in zip(data_lines, target_lines, folds):
    for img, target in zip(data_line, target_line):
        if fold == test_fold:
            x_test.append(img)
            y_test.append(ord(target)-a)
        else:
            x_train.append(img)
            y_train.append(ord(target)-a)

# Convert into numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate performance
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""#### Perform Cross Validation"""

# Prepare training data & testing data using 9 fold to 1 fold ratio
total_acc = 0
for test_fold in range(10):
    # Prepare training data & testing data using 9 fold to 1 fold ratio
    x_train, y_train = [], []
    x_test, y_test = [], []

    for data_line, target_line, fold in zip(data_lines, target_lines, folds):
        for img, target in zip(data_line, target_line):
            if fold == test_fold:
                x_test.append(img)
                y_test.append(ord(target)-a)
            else:
                x_train.append(img)
                y_train.append(ord(target)-a)

    print("Cross Validation Step %d" %(test_fold+1))

    # Convert into numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Train a CNN model
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    # Evaluate performance
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    total_acc += score[1]

print('Average Accuracy = %.4f' %(total_acc/10))

"""## Model 3: CNN + RNN
Use a recurrent neural networks (RNN) for language modeling
"""

imgs = data_lines[0]
labels = target_lines[0]
labels

x = np.array(imgs)
x = np.expand_dims(x, -1)
x.shape

np.argmax(model.predict(x), axis=1)

[ord(y)-a for y in labels]

rnn = keras.Sequential([
    keras.Input(shape=(None,num_classes)),
    layers.Masking(mask_value=0.),
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 64)
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

rnn.summary()

x_in = np.expand_dims(model.predict(x), 0)
x_in.shape

rnn(x_in).shape

plot_model(rnn, to_file='rnn.png', show_shapes=True, show_layer_names=True)

lengths = [len(x) for x in target_lines]
max_length = max(lengths)
print("maximum sequence length = ", max_length)

# prepare training data
x_train, y_train = [], []
x_test, y_test = [], []
for data_line, target_line, fold in zip(data_lines, target_lines, folds):
    x = model.predict(np.expand_dims(np.array(data_line), -1))
    pad_size = max_length-x.shape[0]
    padding = np.zeros([pad_size, num_classes])
    x = np.vstack([x, padding])
    # labels
    y = [ord(letter)-a for letter in target_line]
    y = keras.utils.to_categorical(y, 26)
    y = np.vstack([y, np.zeros([pad_size, 26])])
    
    if fold == 0:
        x_test.append(x)
        y_test.append(y)
    else:
        x_train.append(x)
        y_train.append(y)

# Convert into numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train.shape

batch_size = 128
epochs = 15

rnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

rnn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate performance
score = rnn.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

len(data_lines)

"""### Explore Mis-Classifications"""

# Searching for mis-classification
pred = rnn.predict(x_test)
pred = np.argmax(pred, axis=-1)
pred.shape

actual = np.argmax(y_test, axis=-1)
actual.shape

test_data = []
test_labels = []
for data_line, target_line, fold in zip(data_lines, target_lines, folds):
    if fold == 0:
        test_data.append(data_line)
        test_labels.append(target_line)

len(test_labels)

line_idx = []
for i, test_label in enumerate(test_labels):
    for j, c in enumerate(test_label):
        if pred[i][j] != actual[i][j]:
            line_idx.append(i)

word_acc = (626 - len(np.unique(line_idx)))/626*100
print("word level accuracy = %.3f%%" % (word_acc))

test_labels[45]
l = len(test_labels[45])

show_line(test_data[45], test_labels[45])

[chr(o+a) for o in pred[45][:l]]

idx = 607
show_line(test_data[idx], test_labels[idx])

l = len(test_labels[idx])
[chr(o+a) for o in pred[idx][:l]]
