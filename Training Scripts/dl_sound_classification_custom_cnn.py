# -*- coding: utf-8 -*-
"""
CNN_Based_Sound_Classification_for_Multi_Class_Recognition_Custom_CNN

Author: Ojasva Goyal

Description:
This script implements a Convolutional Neural Network (CNN) for the classification 
of sound files into multiple categories using Mel spectrograms as input features. 
The model is built using TensorFlow and Keras and is trained on a custom dataset 
containing audio samples from various classes.

Usage:
- Ensure the dataset is available in the specified directory structure.
- Adjust the `dataset_path`, `train_path`, and `val_path` variables as needed.
- Run the script to preprocess data, build the model, train the model, and evaluate it.

Dependencies:
- librosa
- numpy
- tensorflow
- sklearn
- matplotlib

License:
Apache License 2.0

"""

import librosa
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

# Set seed for NumPy
np.random.seed(42)

# Set seed for TensorFlow
tf.random.set_seed(42)

dataset_path = '/kaggle/input/sound-classification-13-classes/audio_dataset'

def load_and_preprocess_audio(file_path, duration=2.5, sr=22050):
    
    """
    Load and preprocess an audio file.

    Args:
    file_path (str): Path to the audio file.
    duration (float): Duration of the audio to load (in seconds). It is set as 2.5 seconds.
    sr (int): Sample rate for loading the audio.

    Returns:
    np.ndarray: Mel spectrogram of the audio.
    """
    
    # Load audio file with librosa
    audio, sr = librosa.load(file_path, duration=duration, sr=sr)
    # Ensure audio is 2.5 seconds long
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, max(0, int(sr * duration) - len(audio))), "constant")
    # Generate Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

def load_dataset(dataset_path):
    
    """
    Load and preprocess the entire dataset.

    Args:
    dataset_path (str): Path to the dataset.

    Returns:
    tuple: Arrays of spectrograms and their corresponding labels.
    """
    
    labels = []
    spectrograms = []
    for folder_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                spectrogram = load_and_preprocess_audio(file_path)
                spectrograms.append(spectrogram)
                labels.append(folder_name)
    return np.array(spectrograms), np.array(labels)

def prepare_data(train_path, val_path):
    
    """
    Prepare training and validation data.

    Args:
    train_path (str): Path to the training dataset.
    val_path (str): Path to the validation dataset.

    Returns:
    tuple: Preprocessed training and validation data along with their labels.
    """
    
    # Load and preprocess data
    X_train, y_train = load_dataset(train_path)
    X_val, y_val = load_dataset(val_path)

    # Reshape for CNN input
    X_train = X_train[..., np.newaxis] # Adding a channel dimension
    X_val = X_val[..., np.newaxis]

    # Encode labels
    # Define custom order of labels
    custom_label_order = ['car_horn', 'dog_barking', 'drilling', 'Fart', 'Guitar', 'Gunshot_and_gunfire', 'Hi-hat', 'Knock', 'Laughter', 'Shatter', 'siren', 'Snare_drum', 'Splash_and_splatter']

    def custom_encode_labels(labels, custom_label_order):
    # Map each label to its corresponding index in custom_label_order
        encoded_labels = np.array([custom_label_order.index(label) for label in labels])
        return encoded_labels

    # Use the custom function to encode y_train and y_val
    y_train_encoded = to_categorical(custom_encode_labels(y_train, custom_label_order), num_classes=len(custom_label_order))
    y_val_encoded = to_categorical(custom_encode_labels(y_val, custom_label_order), num_classes=len(custom_label_order))

    #print("Order of labels for training:", encoded_labels)

    return X_train, X_val, y_train_encoded, y_val_encoded

def build_advanced_cnn(input_shape, num_classes):
    
    """
    Build an advanced CNN model for audio classification.

    Args:
    input_shape (tuple): Shape of the input data.
    num_classes (int): Number of output classes.

    Returns:
    Model: A compiled CNN model.
    """
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = build_advanced_cnn(input_shape=(128, 108, 1), num_classes=13)

cnn_model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.62), loss='categorical_crossentropy', metrics=['accuracy'])


# Print the model's architecture
cnn_model.summary()

# Load and preprocess the dataset
X_train, X_val, y_train, y_val = prepare_data('/kaggle/input/sound-classification-13-classes/audio_dataset/train', '/kaggle/input/sound-classification-13-classes/audio_dataset/val')

# Train the model
history = cnn_model.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32
)

# Evaluate the model on the validation set
val_loss, val_acc = cnn_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

import matplotlib.pyplot as plt

def plot_training_history(history):
    
    """
    Plot training and validation accuracy and loss.

    Args:
    history (History): Training history of the model.
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Plot the training history
plot_training_history(history)

# Save the trained model
cnn_model.save('audio_classification_custon_cnn_5.keras')

from sklearn.metrics import classification_report
import numpy as np


# To make predictions on X_val:
predictions = cnn_model.predict(X_val)

# Convert one-hot encoded predictions and true labels back to class labels
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

# Using sklearn's classification report to view per-class accuracy
print(classification_report(true_labels, predictions_labels))

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.models import load_model  # If your model is saved and needs to be loaded

#To load the model: cnn_model = load_model('path_to_your_model.h5')


# Make predictions on the validation set
predictions = cnn_model.predict(X_val)

# Convert predictions from one-hot encoded to class indices
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predictions_labels)

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)


class_names = custom_label_order = ['car_horn', 'dog_barking', 'drilling', 'Fart', 'Guitar', 'Gunshot_and_gunfire', 'Hi-hat', 'Knock', 'Laughter', 'Shatter', 'siren', 'Snare_drum', 'Splash_and_splatter']  # Extract class names in the correct order

# Printing per-class accuracy with class names
for idx, class_name in enumerate(class_names):
    print(f"Accuracy for class {class_name}: {per_class_accuracy[idx]:.2f}")

# Print a classification report for additional metrics
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions_labels, target_names=class_names))
