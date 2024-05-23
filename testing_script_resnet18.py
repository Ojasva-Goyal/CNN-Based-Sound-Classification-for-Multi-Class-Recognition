# Import all necessary python libraries here

import os
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import argparse


# Assuming the model is in a 'model' directory at the same level as this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'audio_classification_resnet18_final.keras')

# Constants
TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "C:/Users/admin/Desktop/DL_Project_Task-1/test_data"  # Change this to the actual path in your environment
OUTPUT_CSV_ABSOLUTE_PATH = "C:/Users/admin/Desktop/DL_Project_Task-1/output1.csv"  # Change this to the desired output path
#/path/to/your/model.h5

#Parse command-line arguments for test data directory and output CSV path
parser = argparse.ArgumentParser(description='Run the sound classification test script with optional batch processing.')
parser.add_argument('--test_data_directory', type=str, required=True,
                    help='Relative or absolute path to the directory containing test audio files.')
parser.add_argument('--output_csv', type=str, required=True,
                    help='Relative or absolute path where the output CSV file will be saved.')
parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                    help='Execution mode: "single" for processing files one by one, "batch" for batch processing. Default is "single".')
args = parser.parse_args()

# Load the trained model
model = load_model(MODEL_PATH)
LABEL_ORDER = ['car_horn', 'dog_barking', 'drilling', 'Fart', 'Guitar', 'Gunshot_and_gunfire', 'Hi-hat', 'Knock', 'Laughter', 'Shatter', 'siren', 'Snare_drum', 'Splash_and_splatter']

# Function to preprocess audio files, adapted from the training script
def load_and_preprocess_audio(file_path, duration=2.5, sr=22050):
    audio, sr = librosa.load(file_path, duration=duration, sr=sr)
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, max(0, int(sr * duration) - len(audio))), "constant")
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram[..., np.newaxis]  # Adding a channel dimension
    return np.expand_dims(spectrogram, axis=0)  # Adding a batch dimension

def evaluate(file_path):
    processed_audio = load_and_preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    predicted_index = np.argmax(prediction)  # Get the index of the highest probability
    predicted_class = LABEL_ORDER[predicted_index]  # Map index to label name
    return predicted_index + 1


def evaluate_batch(file_path_batch, batch_size=32):
    batch_predictions = []
    for file_path in file_path_batch:
        processed_audio = load_and_preprocess_audio(file_path)
        prediction = model.predict(processed_audio)
        predicted_index = np.argmax(prediction)  # Adjust based on class numbering
        predicted_class = LABEL_ORDER[predicted_index]  # Map index to label name
        batch_predictions.append(predicted_class)
    return predicted_index + 1


def test():
    filenames = []
    predictions = []
    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


def test_batch(batch_size=32):
    filenames = []
    predictions = []
    all_files = [os.path.join(args.test_data_directory, f) for f in os.listdir(args.test_data_directory)]
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        batch_predictions = evaluate_batch(batch_files, batch_size)
        filenames.extend([os.path.basename(f) for f in batch_files])
        predictions.extend(batch_predictions)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(args.output_csv, index=False)


# Execute the testing function
if __name__ == '__main__':
    if args.mode == 'single':
        test()
    elif args.mode == 'batch':
        test_batch()