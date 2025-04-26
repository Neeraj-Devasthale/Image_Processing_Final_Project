# Image_Processing_Final_Project
Code for image processing final project

## Title: Using deep learning to classify the notes of Budgerigars (Melopsittacus undulatus) 

One of the key steps in analysis of bioacoustics signals is categorizing sounds in classes or note types. These categories are determined by different factors such as note duration, entropy of the note, harmonic nature, frequency modulation and amplitude modulation, etc. Categorization into these note types allows complex analysis and helps in understanding and discovery of patterns present in sequences and songs of the organism. Budgerigars (Melopsittacus undulatus) have a complex vocal repertoire and vocalize frequently. Manually annotating the data consumes a lot of time and effort. Along with this, there are discrepancies and disagreements between researchers when they annotate same data due to subjectivity of class boundaries. The manual classification of these notes is done by visualizing them in spectrogram form and observing features like frequency modulation, harshness, entropy, duration, etc. Since this task is performed by identifying patterns, I believe a convolution neural network architecture might be a good solution to automate the process. The dataset is a private dataset from a research lab. The data consists of raw audio files and corresponding text files containing information about the begin time, end time and category of the note from which JSONs have been created. This preprocessing code uses libraries like librosa, numpy, os, json, pandas, etc. 

Problem can be formulated as an input-output statement as follows: 
• Input- Numpy arrays structured from JSON files which carry 2D information of 
MFCCs (MFCCs indicate tonality). Hence, these can be thought of as 
grayscale images 
• Output- Class of the note. We primarily have a,b,c,d,e,f and m classes 

The model will be built using strategies like early stopping and dropout regularization, convolution-max pooling layers, cross-entropy loss function, ADAM optimizer, etc. Since CNNs are good at pattern recognition, they may be able to detect patterns which humans detect when examining these spectrograms and I will try different filter dimensions to capture these patterns. 

Human annotations achieve 85 percent inter-observer agreement. This provides a benchmark for the model performance.

Apart from required files mentioned in the project structure guidelines, I have included following files to aid evaluation, understanding and implementation of code and some utility codes I used along the way. These are provided in Utils folder and include:
1. A folder called "Complete model run". This has directories 'checkpoints', 'data', a .docx file containing entire terminal output and a .png file with plots of accuracy and loss for validation and training with epoch number on x-axis
2. A .py file called 'double_loop_train.py'. As per the exchange we had over email I sent you titled "Urgent : Some doubts about image processing project", I have attached this code which implements train.py with a double for loop for its training.
3. A .py file called 'test_data_creator.py'. This file moves 10 samples per class randomly from directory 'JSONS', my training directory to the testing directory 'data'
4. A .py file called 'Example main script.py' which contains an implementation of the entire code to process raw JSONs, train, and test.

## Possibly important for evaluation : Slight deviations made for required to handle the JSON data and data directory arrangements
1. Since I had to make different data loaders for train, test and validation my train function takes in val_loader as an additional input.
2. Use of model.fit function of tensorflow instead of double for loop in train. A double for loop code is provided as mentioned above but using inbuilt functions gives a better UI and is much faster to train. However, results of both the implimentations are very similar.
3. 


## File Descriptions

### 1. config.py
Contains all configuration parameters:
- Model architecture parameters (input dimensions, classes)
- Training hyperparameters (batch size, epochs, learning rate)
- Loss function and optimizer configuration

**Key Variables:**
- `resize_x`, `resize_y`: Input dimensions (39x120)
- `batch_size`: 512
- `loss_fn`: Sparse Categorical Crossentropy
- `optimizer`: Adam with learning rate 0.001

### 2. dataset.py
Handles data loading and processing:

**Dataset Class:**
- `load_data()`: Loads and preprocesses training data
- `load_test_data()`: Loads separate test dataset
- `split_data()`: 80-20 train-validation split
- `save_processed_data()`: Saves processed numpy arrays

**Data Loader:**
- `data_loader()`: Creates TensorFlow datasets
- data_loader(data_dir, batch_size=256, mode='train'). Program indicates default batch_size and mode is train
  - Modes: 'train', 'test', 'inference'

### 3. model.py
CNN architecture with residual blocks:
- 4 convolutional blocks with max pooling
- Global average pooling
- Dense layers with dropout
- Output layer with softmax activation

### 4. train.py
Training workflow:
- Compiles model with specified loss and optimizer
- Implements early stopping
- Saves training history plots
- Returns training metrics
- train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer). This function takes these inputs in this particular order.

### 5. predict.py
Evaluation and inference:
- Generates classification reports
- Creates confusion matrix
- Saves predictions to CSV
- Handles test data loading
- test(model, test_loader=None, test_dir='data'). The function takes in inputs model, test_loader instance which loads the data from test_dir. test_dir is set to 'data' by default

### 6. interface.py
Main execution script:
- Initializes dataset and model
- Coordinates training and evaluation
- Uses standardized imports as per project requirements along with some key imports from config.py

## Installation
1. Model has been programmed on python version 3.9.13 It is possible higher versions might show tensorflow compatibility errors with 'distutils' module of python which is why I eventually switched to this python version. This one works very nicely.
2. Some installation requirements:
```bash
pip install tensorflow==2.16 pandas scikit-learn tabulate matplotlib numpy json os 
