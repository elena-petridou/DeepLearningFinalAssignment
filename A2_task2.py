# %% [markdown]
# # Introduction to Deep Learning, Assignment 2, Task 2
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 
# When using the Lab Computers, download the Jupyter Notebook to one of the machines first.
# 
# If you want to use Google Colab, you should first copy this notebook and enable GPU runtime in 'Runtime -> Change runtime type -> Hardware acceleration -> GPU **OR** TPU'.
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, Input, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose

# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %% [markdown]
# ## Helper functions
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
    n = len(labels)
    length = len(labels[0])
    char_map = dict(zip(unique_characters, range(len(unique_characters))))
    one_hot = np.zeros([n, length, len(unique_characters)])
    for i, label in enumerate(labels):
        m = np.zeros([length, len(unique_characters)])
        for j, char in enumerate(label):
            m[j, char_map[char]] = 1
        one_hot[i] = m

    return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    predicted = [''.join([unique_characters[i] for i in j]) for j in pred]

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

splits_train_test = np.arange(0.05, 1, 0.05)

# %% [markdown]
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %%
def build_text2text_model(additional_lstm = False):

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    if additional_lstm:
        text2text.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

accuracies_txt1 = []
losses_txt1 = []

for split in splits_train_test:
    # Splitting train and test sets
    X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text, y_text, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    X_text_train_onehot = encode_labels(X_text_train)
    y_text_train_onehot = encode_labels(y_text_train)
    X_text_test_onehot = encode_labels(X_text_test)
    y_text_test_onehot = encode_labels(y_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, start_from_epoch = 30)

    # Building and fitting the model on the train set
    text2text_model1 = build_text2text_model()
    text2text_model1.fit(X_text_train_onehot, y_text_train_onehot, epochs=50, batch_size=128, validation_split=0.3,callbacks=[early_stop])

    # Predicting the test set and accuracy
    test_metrics = text2text_model1.evaluate(X_text_test_onehot,y_text_test_onehot)
    losses_txt1.append(test_metrics[0])
    accuracies_txt1.append(test_metrics[1])

    raw_preds = text2text_model1.predict(X_text_test_onehot)
    predictions = decode_labels(raw_preds)
    for i in range(20):
        print(f"{X_text_test[i]} = {predictions[i]}")

# %%
for i, split in enumerate(splits_train_test):
    print(f"Accuracy and loss for test split {splits_train_test[i]:.2f}: {accuracies_txt1[i]:.2f}, {losses_txt1[i]:.2f}")

print(f"Best-performing test split: {splits_train_test[accuracies_txt1.index(max(accuracies_txt1))]}")

plt.plot(splits_train_test,accuracies_txt1, label="Accuracy")
plt.plot(splits_train_test, losses_txt1, label="Losses")
plt.legend()
plt.title("Accuracies and losses by test split")
plt.xlabel("Test split")
plt.ylabel("Metrics")
plt.show()

# Getting wrongs predictions from the worst split (likely the last one)
predictions_txt1 = decode_labels(text2text_model1.predict(X_text_test_onehot))
predictions_txt1 = np.array(predictions_txt1)
mask_txt1 = predictions_txt1 != y_text_test
wrongs_expr_txt1 = X_text_test[mask_txt1]
wrong_preds_txt1 = predictions_txt1[mask_txt1]
wrong_true_txt1 = y_text_test[mask_txt1]
for i in range(30):
    print(f"Expression: {wrongs_expr_txt1[i]} = True: {wrong_true_txt1[i]}, Predicted: {wrong_preds_txt1[i]}")

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

accuracies_txt2 = []
losses_txt2 = []

for split in splits_train_test:
    # Splitting train and test sets
    X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text, y_text, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    X_text_train_onehot = encode_labels(X_text_train)
    y_text_train_onehot = encode_labels(y_text_train)
    X_text_test_onehot = encode_labels(X_text_test)
    y_text_test_onehot = encode_labels(y_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, start_from_epoch = 30)

    # Building and fitting the model on the train set
    text2text_model2 = build_text2text_model(additional_lstm=True)
    text2text_model2.fit(X_text_train_onehot, y_text_train_onehot, epochs=50, batch_size=128, validation_split=0.3,callbacks=[early_stop])

    # Predicting the test set and accuracy
    test_metrics = text2text_model2.evaluate(X_text_test_onehot,y_text_test_onehot)
    losses_txt2.append(test_metrics[0])
    accuracies_txt2.append(test_metrics[1])
    
    raw_preds = text2text_model2.predict(X_text_test_onehot)
    predictions = decode_labels(raw_preds)
    for i in range(20):
        print(f"{X_text_test[i]} = {predictions[i]}")

# %%

for i, split in enumerate(splits_train_test):
    print(f"Accuracy and loss for test split {splits_train_test[i]:.2f}: {accuracies_txt2[i]:.2f}, {losses_txt2[i]:.2f}")

print(f"Best-performing test split: {splits_train_test[accuracies_txt2.index(max(accuracies_txt2))]}")

plt.plot(splits_train_test,accuracies_txt2, label="Accuracy")
plt.plot(splits_train_test, losses_txt2, label="Losses")
plt.legend()
plt.title("Accuracies and losses by test split")
plt.xlabel("Test split")
plt.ylabel("Metrics")
plt.show()

# Getting wrongs predictions from the worst split (likely the last one)
predictions_txt2 = decode_labels(text2text_model2.predict(X_text_test_onehot))
predictions_txt2 = np.array(predictions_txt2)
mask_txt2 = predictions_txt2 != y_text_test
wrongs_expr_txt2 = X_text_test[mask_txt2]
wrong_preds_txt2 = predictions_txt2[mask_txt2]
wrong_true_txt2 = y_text_test[mask_txt2]
for i in range(30):
    print(f"Expression: {wrongs_expr_txt2[i]} = True: {wrong_true_txt2[i]}, Predicted: {wrong_preds_txt2[i]}")

# %%


# %% [markdown]
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %%
from keras.src import Sequential
## Your code
## Your code
X_img = X_img.reshape(X_img.shape[0],5,28,28,1)
#y_img = y_img.reshape(X_img.shape[0],3,28,28,1)

def build_image2text_model(
    image_shape=(28, 28, 1),
    sequence_length=5,
    output_length=3,
    num_chars=len(unique_characters),
    additional_lstm = False
):

    # Encoding images with a convolutional network
    cnn = Sequential([
        Conv2D(32, (2,2), activation='relu', padding='same', input_shape=image_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (2,2), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu')
    ])

    image2text = Sequential()
    # Apply the convolutional network on each of the 5 images per observation (the shape is (N,5,28,28))
    image2text.add(Input(shape=(sequence_length, *image_shape)))
    image2text.add(TimeDistributed(cnn))

    # Now the input has shape: (batch, 5, 128)

    # We use th same structure as before
    if additional_lstm:
        image2text.add(LSTM(256, return_sequences=True))
    image2text.add(LSTM(256))
    image2text.add(RepeatVector(output_length))
    image2text.add(LSTM(256, return_sequences=True))
    image2text.add(TimeDistributed(
        Dense(num_chars, activation='softmax')
    ))
    image2text.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    image2text.summary()

    return image2text

# %%
# Splitting train and test sets
accuracies_ixt1 = []
losses_ixt1 = []

for split in splits_train_test:
    X_img_train, X_img_test, y_text_train, y_text_test = train_test_split(
        X_img, y_text, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    y_text_train_onehot = encode_labels(y_text_train)
    y_text_test_onehot = encode_labels(y_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=5, start_from_epoch = 30)

    # Building and fitting the model on the train set
    image2text_model1 = build_image2text_model()
    image2text_model1.fit(X_img_train, y_text_train_onehot, epochs=100, batch_size=128, validation_split=0.2,callbacks=[early_stop])

    # Predicting the test set and accuracy
    metrics = image2text_model1.evaluate(X_img_test,y_text_test_onehot)
    accuracies_ixt1.append(metrics[1])
    losses_ixt1.append(metrics[0])
    raw_preds = image2text_model1.predict(X_img_test)
    predictions = decode_labels(raw_preds)
    for i in range(20):
        plt.figure(figsize=(10,2))
        sequence_images = X_img_test[i]

        # Remove channel dimension if present
        sequence_images = sequence_images.squeeze()  # shape: (5,28,28)

        # Stack images horizontally
        display_image = np.hstack(sequence_images)  # shape: (28, 28*5)

        plt.imshow(display_image, cmap='gray')
        plt.axis('off')
        plt.title(f"Predicted: {predictions[i]}")
        plt.show()

# %%
for i, split in enumerate(splits_train_test):
    print(f"Accuracy and loss for test split {splits_train_test[i]:.2f}: {accuracies_ixt1[i]:.2f}, {losses_ixt1[i]:.2f}")

print(f"Best-performing test split: {splits_train_test[accuracies_ixt1.index(max(accuracies_ixt1))]}")

plt.plot(splits_train_test,accuracies_ixt1, label="Accuracy")
plt.plot(splits_train_test, losses_ixt1, label="Losses")
plt.legend()
plt.title("Accuracies and losses by test split")
plt.xlabel("Test split")
plt.ylabel("Metrics")
plt.show()

# Getting wrongs predictions from the worst split (likely the last one)
predictions_ixt1 = decode_labels(image2text_model1.predict(X_img_test))
predictions_ixt1 = np.array(predictions_ixt1)
mask_ixt1 = predictions_ixt1 != y_text_test
wrongs_expr_ixt1 = X_img_test[mask_ixt1]
wrong_preds_ixt1 = predictions_ixt1[mask_ixt1]
wrong_true_ixt1 = y_text_test[mask_ixt1]
for i in range(30):
    plt.figure(figsize=(10,2))
    sequence_images = X_img_test[i]
    # Remove channel dimension if present
    sequence_images = sequence_images.squeeze()  # shape: (5,28,28)
    # Stack images horizontally
    display_image = np.hstack(sequence_images)  # shape: (28, 28*5)
    plt.imshow(display_image, cmap='gray')
    plt.axis('off')
    plt.title(f"True: {wrong_true_ixt1[i]}, Predicted: {wrong_preds_ixt1[i]}")
    plt.show()

# %%
# Splitting train and test sets
accuracies_ixt2 = []
losses_ixt2 = []

for split in splits_train_test:
    X_img_train, X_img_test, y_text_train, y_text_test = train_test_split(
        X_img, y_text, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    y_text_train_onehot = encode_labels(y_text_train)
    y_text_test_onehot = encode_labels(y_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=5, start_from_epoch = 30)

    # Building and fitting the model on the train set
    image2text_model2 = build_image2text_model(additional_lstm=True)
    image2text_model2.fit(X_img_train, y_text_train_onehot, epochs=100, batch_size=128, validation_split=0.2,callbacks=[early_stop])

    # Predicting the test set and accuracy
    metrics = image2text_model2.evaluate(X_img_test,y_text_test_onehot)
    accuracies_ixt2.append(metrics[1])
    losses_ixt2.append(metrics[0])
    raw_preds = image2text_model2.predict(X_img_test)
    predictions = decode_labels(raw_preds)
    for i in range(20):
        plt.figure(figsize=(10,2))
        sequence_images = X_img_test[i]

        # Remove channel dimension if present
        sequence_images = sequence_images.squeeze()  # shape: (5,28,28)

        # Stack images horizontally
        display_image = np.hstack(sequence_images)  # shape: (28, 28*5)

        plt.imshow(display_image, cmap='gray')
        plt.axis('off')
        plt.title(f"Predicted: {predictions[i]}")
        plt.show()

# %%
for i, split in enumerate(splits_train_test):
    print(f"Accuracy and loss for test split {splits_train_test[i]:.2f}: {accuracies_ixt2[i]:.2f}, {losses_ixt2[i]:.2f}")

print(f"Best-performing test split: {splits_train_test[accuracies_ixt2.index(max(accuracies_ixt2))]}")

plt.plot(splits_train_test,accuracies_ixt2, label="Accuracy")
plt.plot(splits_train_test, losses_ixt2, label="Losses")
plt.legend()
plt.title("Accuracies and losses by test split")
plt.xlabel("Test split")
plt.ylabel("Metrics")
plt.show()

# Getting wrongs predictions from the worst split (likely the last one)
predictions_ixt2 = decode_labels(image2text_model2.predict(X_img_test))
predictions_ixt2 = np.array(predictions_ixt2)
mask_ixt2 = predictions_ixt2 != y_text_test
wrongs_expr_ixt2 = X_img_test[mask_ixt2]
wrong_preds_ixt2 = predictions_ixt2[mask_ixt2]
wrong_true_ixt2 = y_text_test[mask_ixt2]
for i in range(30):
    plt.figure(figsize=(10,2))
    sequence_images = X_img_test[i]
    # Remove channel dimension if present
    sequence_images = sequence_images.squeeze()  # shape: (5,28,28)
    # Stack images horizontally
    display_image = np.hstack(sequence_images)  # shape: (28, 28*5)
    plt.imshow(display_image, cmap='gray')
    plt.axis('off')
    plt.title(f"True: {wrong_true_ixt2[i]}, Predicted: {wrong_preds_ixt2[i]}")
    plt.show()

# %% [markdown]
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %%
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2DTranspose, Reshape, LSTM, Input

# Your code
def build_text2image_model(
    sequence_length=5,
    output_length=3,
    num_chars=len(unique_characters),
    additional_lstm=False
    ):
    decoder_input = Input(shape=(256,))
    x = Dense(7*7*64, activation='relu')(decoder_input)
    #x = Dense(7*7*256, activation='relu')(x)
    #x = Dense(7*7*256, activation='relu')(x)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
    decoder_output = Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
        
    decoder = Model(decoder_input, decoder_output)

    text2image = Sequential()
    if additional_lstm:
        text2image.add(LSTM(256, input_shape=(5, num_chars), return_sequences=True))
    text2image.add(LSTM(256, input_shape=(5, num_chars)))
    text2image.add(RepeatVector(output_length))
    text2image.add(LSTM(256, return_sequences=True))
    text2image.add(TimeDistributed(decoder))

    text2image.compile(optimizer='adam', loss='binary_crossentropy')
    text2image.summary()

    return(text2image)

# %%
losses_txi1 = []

for split in splits_train_test:
    # Splitting train and test sets
    X_text_train, X_text_test, y_img_train, y_img_test = train_test_split(
        X_text, y_img, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    X_text_train_onehot = encode_labels(X_text_train)
    X_text_test_onehot = encode_labels(X_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5, start_from_epoch = 30)

    # Building and fitting the model on the train set
    text2image_model1 = build_text2image_model()
    text2image_model1.fit(X_text_train_onehot, y_img_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    # Predicting the test set and accuracy
    metrics = text2image_model1.evaluate(X_text_test_onehot,y_img_test)
    losses_txi1.append(metrics)
    raw_preds = text2image_model1.predict(X_text_test_onehot)
    #predictions = decode_labels(raw_preds)

    for i in range(10):
        plt.figure(figsize=(6,2))
        sequence_images = raw_preds[i]

        # Remove channel dimension if present
        sequence_images = sequence_images.squeeze()  # shape: (5,28,28)

        # Stack images horizontally
        display_image = np.hstack(sequence_images)  # shape: (28, 28*5)

        plt.imshow(display_image, cmap='gray')
        plt.axis('off')
        plt.title(f"{X_text_test[i]} =")
        plt.show()

# %%
for i, split in enumerate(splits_train_test):
    print(f"Loss for test split {split:.2f}: {losses_txi1[i]:.3f}")

print(f"Best-performing test split: {splits_train_test[losses_txi1.index(min(losses_txi1))]:.2f}")

plt.plot(splits_train_test, losses_txi1)
plt.title("Losses by test split")
plt.xlabel("Test split")
plt.ylabel("Loss")
plt.show()

# %%
losses_txi2 = []

for split in splits_train_test:
    # Splitting train and test sets
    X_text_train, X_text_test, y_img_train, y_img_test = train_test_split(
        X_text, y_img, test_size=split, random_state=42)

    # One-hot encoding the inputs and labels
    X_text_train_onehot = encode_labels(X_text_train)
    X_text_test_onehot = encode_labels(X_text_test)

    # Setting an early stop callback
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5, start_from_epoch = 30)

    # Building and fitting the model on the train set
    text2image_model2 = build_text2image_model(additional_lstm=True)
    text2image_model2.fit(X_text_train_onehot, y_img_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stop])

    # Predicting the test set and accuracy
    metrics = text2image_model2.evaluate(X_text_test_onehot,y_img_test)
    losses_txi2.append(metrics)
    raw_preds = text2image_model2.predict(X_text_test_onehot)
    #predictions = decode_labels(raw_preds)

    for i in range(10):
        plt.figure(figsize=(6,2))
        sequence_images = raw_preds[i]

        # Remove channel dimension if present
        sequence_images = sequence_images.squeeze()  # shape: (5,28,28)

        # Stack images horizontally
        display_image = np.hstack(sequence_images)  # shape: (28, 28*5)

        plt.imshow(display_image, cmap='gray')
        plt.axis('off')
        plt.title(f"{X_text_test[i]} =")
        plt.show()

# %%
for i, split in enumerate(splits_train_test):
    print(f"Loss for test split {split:.2f}: {losses_txi2[i]:.2f}")

print(f"Best-performing test split: {splits_train_test[losses_txi2.index(min(losses_txi2))]}")

plt.plot(splits_train_test, losses_txi2)
plt.title("Losses by test split")
plt.xlabel("Test split")
plt.ylabel("Loss")
plt.show()


