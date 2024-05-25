# Sound Classification for Multi-Class Recognition
# Overview
This project involves building and evaluating two different CNN-based classifiers for classifying audio samples into 13 distinct categories, such as laughter, car horn, dog bark, etc. The models are trained on provided training and validation datasets. 

This testing script is designed to evaluate audio files using a pre-trained deep learning model for sound classification. It supports processing files individually or in batches for efficiency. The script outputs predictions in a CSV file, associating audio filenames with their predicted classes.

# Prerequisites
Python Version: Ensure Python 3.6 or newer is installed on your system.
Dependencies: The script requires tensorflow, keras, librosa, pandas, and numpy. These can be installed via pip.

# Setup Instructions
#### Install Python Dependencies:

Open a terminal or command prompt and execute the following command to install the required Python libraries:
```bash
  pip install tensorflow keras librosa pandas numpy
```

#### Prepare Your Environment:
Ensure your audio files for testing are organized in a directory accessible to the script.

#### Script Parameters:
The script accepts several parameters to control its execution:

--test_data_directory: Path to the directory containing test audio files.

--output_csv: Path for the output CSV file where predictions will be saved.

--mode: Execution mode (single for processing files one by one or batch for processing files in batches). Default is single.

#### Running the Script
Navigate to the directory containing the script and execute one of the following commands in your terminal or command prompt, depending on your chosen mode of execution.

###### For Single-file Processing:

```bash
python test_script.py --mode single --test_data_directory "./path/to/test/data" --output_csv "./path/to/output.csv"
```
###### For Batch Processing:

```bash
python test_script.py --mode batch --test_data_directory "./path/to/test/data" --output_csv "./path/to/output.csv"
```
Replace `./path/to/test/data` with the path to your test audio files and `./path/to/output.csv` with the desired location and name of the output CSV file. These paths can be relative to the script location or absolute paths on your system.

#### Output Format
The script generates a CSV file containing two columns:

filename: The name of each audio file processed.

pred: The predicted class for each audio file, starting from 1.

#### Troubleshooting
Permission Errors: If you encounter permission errors when writing the output CSV, ensure the script has write access to the output directory or try running the script with elevated permissions.

Pandas Installation Issues: Should there be any issues with pandas, reinstall it using `pip install --upgrade --force-reinstall pandas.`

Model Not Found: Verify the model file `your_model.h5` is placed correctly in the model directory adjacent to the script.

# Conclusion
This project demonstrates the use of CNN-based architectures for sound classification. The two different models trained and evaluated as part of this project showcase the versatility of CNNs in handling audio data transformed for image-like processing. The testing script provides an easy-to-use interface for generating predictions on new audio samples.

For more details on the implementation and model architectures, please refer to the individual scripts and the documentation within the code.
