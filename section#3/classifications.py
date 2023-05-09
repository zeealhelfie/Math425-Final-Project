import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np # linear algebra
import pandas as pd # data processing,
import math
import datetime
import platform


# data processing
X_train = np.loadtxt("data/handwriting_training_set.txt", delimiter=",")
y_train = np.loadtxt("data/handwriting_training_set_labels.txt")
X_test = np.loadtxt("data/handwriting_test_set.txt", delimiter=",")
y_test = np.loadtxt("data/handwriting_test_set_labels.txt")

# Replace label values of 10 with 0
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

# Divide training set into 10 classes based on labels
classes = []
for i in range(10):
    class_i = X_train[y_train == i]
    classes.append(class_i)

# Compute SVD of each class matrix
k_values = [5, 10, 15, 20]  # values of k to use
class_bases = []
for class_i in classes:
    # Compute mean image
    mean_image = np.mean(class_i, axis=0)
    # Subtract mean image from each training example
    diff = class_i - mean_image
    # Compute SVD
    #The U and S matrices obtained from the SVD are not being used: the primary goal of the code is to find a low-dimensional representation of the data using the V matrix.
    _, _, vT = np.linalg.svd(diff)
    # Store first k singular vectors as basis for class
    bases = vT[:k_values[-1]].T
    class_bases.append(bases[:, :k_values[-1]])

# First-stage classification using only first singular vector
correct_first_stage = 0
for i in range(len(X_test)):
    x = X_test[i]
    errors = []
    for j in range(10):
        diff = x - np.mean(classes[j], axis=0)
        proj = np.dot(diff, class_bases[j][:, 0])
        recon = np.dot(proj, class_bases[j][:, 0].T) + np.mean(classes[j], axis=0)
        error = np.linalg.norm(x - recon)
        errors.append(error)
    pred = np.argmin(errors)
    if pred == int(y_test[i]):
        correct_first_stage += 1
print(f'Accuracy using first singular vector only: {correct_first_stage/len(X_test):.2%}')

# Two-stage classification algorithm
for k in k_values:
    correct_second_stage = 0
    for i in range(len(X_test)):
        x = X_test[i]
        # Compute residual error for each class
        errors = []
        for j in range(10):
            diff = x - np.mean(classes[j], axis=0)
            
            # proj: is a variable that stores the projection of the difference between the current test example x and the mean image of the current class j onto the first k singular vectors of the SVD basis for the class. 
            proj = np.dot(diff, class_bases[j][:, :k])
            
            # recon: is a variable that stores the reconstructed image of the current test example x using the basis vectors and mean image for a given class
            recon = np.dot(proj, class_bases[j][:, :k].T) + np.mean(classes[j], axis=0)
            error = np.linalg.norm(x - recon)
            errors.append(error)
        # Classify test example as digit with smallest residual error
        pred = np.argmin(errors)
        if pred == int(y_test[i]):
            correct_second_stage += 1
    print(f'Accuracy using {k} singular vectors: {correct_second_stage/len(X_test):.2%}')
    
    
# Classify test examples
digit_correct = np.zeros(10)
digit_total = np.zeros(10)
for k in k_values:
    correct = np.zeros(10)
    total = np.zeros(10)
    for i in range(len(X_test)):
        x = X_test[i]
        # Compute residual error for each class
        errors = []
        for j in range(10):
            diff = x - np.mean(classes[j], axis=0)
            proj = np.dot(diff, class_bases[j][:, :k])
            recon = np.dot(proj, class_bases[j][:, :k].T) + np.mean(classes[j], axis=0)
            error = np.linalg.norm(x - recon)
            errors.append(error)
        # Classify test example as digit with smallest residual error
        pred = np.argmin(errors)
        # Check if classification is correct and update counts
        if pred == int(y_test[i]):
            correct[int(y_test[i])] += 1
            digit_correct[int(y_test[i])] += 1
        total[int(y_test[i])] += 1
        digit_total[int(y_test[i])] += 1

# Print accuracy for each digit
print('Accuracy for each digit:')
for i in range(10):
    acc = digit_correct[i] / digit_total[i]
    print(f'Digit {i}: {acc:.2%}')
print()