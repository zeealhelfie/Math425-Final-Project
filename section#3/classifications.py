import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np # linear algebra
import pandas as pd # data processing,
import math
import datetime
import platform


# data processing
trainImages = pd.DataFrame(pd.read_csv("data/handwriting_training_set.txt"))
trainImagesLabels = pd.DataFrame(pd.read_csv("data/handwriting_training_set_labels.txt"))
testImages = pd.DataFrame(pd.read_csv("data/handwriting_test_set.txt"))
testImagesLabels = pd.DataFrame(pd.read_csv("data/handwriting_test_set_labels.txt"))


print('X_train:', trainImages.shape)
print('y_train:', trainImagesLabels.shape)
print('X_test:', testImages.shape)
print('y_test:', testImagesLabels.shape)


# Data Overview 
trainImages.head()

trainImages.info() 

trainImages.describe()

trainImages.isna().any().any()


X = trainImages.iloc[:, 0:400]
y = trainImagesLabels.iloc[:, 0]
X_test = testImages.iloc[:, 0:400]


trainImages_ar = trainImages.to_numpy().reshape(3999, 20, 20)
trainImagesLabels_ar = trainImagesLabels.values
testImages_ar = testImages.to_numpy().reshape(999, 20, 20)
testImagesLabels_ar = testImagesLabels.values


print('X_train:', trainImages_ar.shape)
print('y_train:', trainImagesLabels_ar.shape)
print('X_test:', testImages_ar.shape)
print('y_test:', testImagesLabels_ar.shape)



# Save image parameters to the constants that we will use later for data re-shaping and for model traning.
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = trainImages_ar.shape
IMAGE_CHANNELS = 1

print('IMAGE_WIDTH:', IMAGE_WIDTH);
print('IMAGE_HEIGHT:', IMAGE_HEIGHT);
print('IMAGE_CHANNELS:', IMAGE_CHANNELS);


pd.DataFrame(trainImages_ar[0])



# Split the training data into individual digit matrices
digit_matrices = []
for i in range(10):
    start_index = i * 400
    end_index = start_index + 400
    digit_matrix = trainImages[start_index:end_index]
    digit_matrices.append(digit_matrix)
trainImages.describe()
trainImages.info() 

# Classify each test digit using SVD
num_basis_vectors = [5, 10, 15, 20]
num_correct = np.zeros(len(num_basis_vectors))
for i in range(len(testImages)):
    test_digit = testImages[i].reshape(400, 20, 20)
    test_u, test_s, test_vt = np.linalg.svd(test_digit)
    distances = []
    for j in range(10):
        class_u, class_s, class_vt = digit_svds[j]
        for k in num_basis_vectors:
            test_reconstructed = (test_u[:, :k] @ np.diag(test_s[:k]) @ test_vt[:k, :]).flatten()
            class_reconstructed = (class_u[:, :k] @ np.diag(class_s[:k]) @ class_vt[:k, :]).flatten()
            distance = np.linalg.norm(test_reconstructed - class_reconstructed)
            distances.append((distance, j))
    distances.sort()
    for j, nbv in enumerate(num_basis_vectors):
        num_correct[j] += int(testImagesLabels[i] == distances[0][1] and nbv == distances[0][1])

# Compute percentage of correctly classified digits
percent_correct = num_correct / len(trainImages) * 100

# Print results
print("Percentage Correctly Classified:")
for i, nbv in enumerate(num_basis_vectors):
    print(f"{nbv} basis vectors: {percent_correct[i]:.2f}%")