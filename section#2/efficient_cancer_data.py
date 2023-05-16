# Copyright 2013 Philip N. Klein
# Import necessary libraries
from vec import Vec
from vecutil import vec2list
from sympy import Matrix
import numpy as np

# Define function to read in training data
import numpy as np

# Define function to read in training data
def read_training_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [] # A
    labels = [] # y
    for line in lines:
        # Split the line into its components
        line = line.strip().split(',')
        # Convert the label to a numerical value (-1 for 'B', 1 for 'M')
        labels.append(1 if line[1] == 'M' else -1)
        # Convert the features to floats and add to the data array
        features = [float(x) for x in line[2:]]
        data.append(features)
    # Convert data and labels to numpy arrays for easier manipulation
    return np.array(data), np.array(labels, dtype=np.float64)


# Define function to perform Gram-Schmidt orthogonalization and QR decomposition

def gram_schmidt_qr(A, b):
    Q, R = np.linalg.qr(A)
    x = np.linalg.solve(R, Q.T.dot(b))
    return x



# Define function to read in validation data
def read_validation_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    labels = []
    for line in lines:
        # Split the line into its components
        line = line.strip().split(',')
        # Convert the label to a numerical value (-1 for 'B', 1 for 'M')
        labels.append(1 if line[1] == 'M' else -1)
        # Convert the features to floats and add to the data array
        features = [float(x) for x in line[2:]]
        data.append(features)
    # Convert data and labels to numpy arrays for easier manipulation
    return np.array(data), np.array(labels)

# Define function to classify data based on a threshold
def classify(predictions, threshold=0):
    """
    Given a set of predicted labels and a threshold, classify the data
    as either 1 or -1 depending on whether the prediction is above or
    below the threshold.

    Args:
        predictions: A 1D numpy array of predicted labels.
        threshold: The threshold value to use for classification.

    Returns:
        A 1D numpy array of classified labels, where each label is either
        1 or -1.
    """
    return np.where(predictions > threshold, 1, -1)
