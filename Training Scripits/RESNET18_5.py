''' 
CNN_Based_Sound_Classification_for_Multi_Class_Recognition_ResNet18

Author: Ojasva Goyal
Description:
This script implements a ResNet-18 Convolutional Neural Network (CNN) for the classification 
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
'''

import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Set seed for NumPy and TensorFlow to ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the path to the dataset
dataset_path = r'/home/Goyal/audio_dataset_final/audio_dataset'


def load_and_preprocess_audio(file_path, duration=2.5, sr=22050):
    
    """
    Load and preprocess an audio file.

    Args:
    file_path (str): Path to the audio file.
    duration (float): Duration of the audio to load (in seconds).
    sr (int): Sample rate for loading the audio.

    Returns:
    np.ndarray: Mel spectrogram of the audio.
    """
    try:
        # Loading audio file with librosa
        audio, sr = librosa.load(file_path, duration=duration, sr=sr)
    except Exception as e:
        print(f"Error loading audio file '{file_path}': {e}")
        return None

    if audio is None:
        print(f"Audio file '{file_path}' is empty or could not be loaded.")
        return None

    # Ensure audio is 2.5 seconds long
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, max(0, int(sr * duration) - len(audio))), "constant")

    # Generate Mel Spectrogram
    try:
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    except Exception as e:
        print(f"Error generating spectrogram for audio file '{file_path}': {e}")
        return None

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
                if spectrogram is not None:
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
    
    # Loading and preprocessing data
    X_train, y_train = load_dataset(train_path)
    X_val, y_val = load_dataset(val_path)

    # Reshaping for CNN input
    X_train = X_train[..., np.newaxis] # Adding a channel dimension
    X_val = X_val[..., np.newaxis]

    # Encoding labels
    custom_label_order = ['car_horn', 'dog_barking', 'drilling', 'Fart', 'Guitar', 'Gunshot_and_gunfire', 'Hi-hat', 'Knock', 'Laughter', 'Shatter', 'siren', 'Snare_drum', 'Splash_and_splatter']

    def custom_encode_labels(labels, custom_label_order):
    # Maping each label to its corresponding index in custom_label_order
        encoded_labels = np.array([custom_label_order.index(label) for label in labels])
        return encoded_labels

    # Using the custom function to encode y_train and y_val
    y_train_encoded = to_categorical(custom_encode_labels(y_train, custom_label_order), num_classes=len(custom_label_order))
    y_val_encoded = to_categorical(custom_encode_labels(y_val, custom_label_order), num_classes=len(custom_label_order))

    return X_train, X_val, y_train_encoded, y_val_encoded


def basic_block(inputs, num_filters, strides=(1, 1)):
    
    """
    Basic residual block for ResNet.

    Args:
    inputs (tensor): Input tensor.
    num_filters (int): Number of filters for the convolutional layers.
    strides (tuple): Strides for the convolutional layers.

    Returns:
    tensor: Output tensor of the block.
    """
    
    x = Conv2D(num_filters, (3, 3), strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if strides != (1, 1) or inputs.shape[-1] != num_filters:
        shortcut = Conv2D(num_filters, (1, 1), strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_resnet18(input_shape, num_classes):
    
    """
    Build a ResNet-18 model.

    Args:
    input_shape (tuple): Shape of the input data.
    num_classes (int): Number of output classes.

    Returns:
    Model: A compiled ResNet 18 model
    """
    
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Configurations for the 4 stages of ResNet-18
    blocks = [2, 2, 2, 2]
    num_filters = [64, 128, 256, 512]

    for i, num_blocks in enumerate(blocks):
        for j in range(num_blocks):
            if j == 0:
                if i != 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)
            else:
                strides = (1, 1)
            x = basic_block(x, num_filters[i], strides)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


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

# Building ResNet-18 model
resnet18_model = build_resnet18(input_shape=(128, 108, 1), num_classes=13)

# Compiling the model
resnet18_model.compile(optimizer=SGD(learning_rate = 0.00065), loss='categorical_crossentropy', metrics=['accuracy'])


# Printing model summary
resnet18_model.summary()


X_train, X_val, y_train, y_val = prepare_data(
    r'C:/home/Goyal/audio_dataset_final/audio_dataset/train',
    r'C:/home/Goyal/audio_dataset_final/audio_dataset/val')


history = resnet18_model.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)


# RESNET 18 for SGD Optimiser
val_loss, val_acc = resnet18_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")



plot_training_history(history)

resnet18_model.save('/home/Goyal/DL/audio_classification_resnet18_5.keras')

saved_model_path = '/home/Goyal/DL/audio_classification_resnet18_5.keras'


model = load_model("/home/Goyal/DL/audio_classification_resnet18_5.keras")  

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.models import load_model  

predictions = model.predict(X_val)

predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)  


cm = confusion_matrix(true_labels, predictions_labels)


per_class_accuracy = cm.diagonal() / cm.sum(axis=1)


class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']  
    

# Print per-class accuracy with class names
for idx, class_name in enumerate(class_names):
    print(f"Accuracy for class {class_name}: {per_class_accuracy[idx]:.2f}")


# Calculate and print overall precision and recall
report = classification_report(true_labels, predictions_labels, target_names=class_names, output_dict=True)
overall_precision = report['macro avg']['precision']
overall_recall = report['macro avg']['recall']

print(f"\nOverall Precision: {overall_precision:.2f}")
print(f"Overall Recall: {overall_recall:.2f}")

# Print a classification report for additional metrics
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions_labels, target_names=class_names))

