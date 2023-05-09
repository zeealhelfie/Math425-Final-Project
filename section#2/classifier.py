import numpy as np
import efficient_cancer_data as ecd

#During the data reading process "b" or "y" values are converted binary (or 1,-1) so we can use these values to test the Accuracy of the model at later stages. Testing is done by comparing if the "y" or "b" values are equal or not equal to the predictions values that are later computed using gram schmidt process.

# read training data
A, b = ecd.read_training_data('train.data')

# Calculate the coefficients of the least squares solution using the Gram-Schmidt QR algorithm
x = ecd.gram_schmidt_qr(np.copy(A), np.copy(b))

# print coefficients of the linear model
for i, coef in enumerate(x):
    print(f"{i+1}. {coef:.4f}")

# read validation data
A_val, b_val = ecd.read_validation_data('validate.data')

# apply linear model to validation data
predictions = A_val @ x

# classify predictions using threshold of 0
classifications = ecd.classify(predictions, threshold=0)

# compute percentage of misclassifications on validation data
misclassifications_val = sum(1 for p, q in zip(classifications, b_val) if p != q)
percentage_misclassified_val = 100 * misclassifications_val / len(b_val)
accuracy_val = 100 - percentage_misclassified_val

# compute percentage of misclassifications on training data
predictions_train = A @ x
classifications_train = ecd.classify(predictions_train, threshold=0)
misclassifications_train = sum(1 for p, q in zip(classifications_train, b) if p != q)
percentage_misclassified_train = 100 * misclassifications_train / len(b)
accuracy_train = 100 - percentage_misclassified_train

# print percentage of misclassifications and accuracy rate on validation and training data

print(f"Percentage of misclassifications on training data: {percentage_misclassified_train:.2f}%, Accuracy rate: {accuracy_train:.2f}%")

print(f"Percentage of misclassifications on validation data: {percentage_misclassified_val:.2f}%, Accuracy rate: {accuracy_val:.2f}%")

# compare with success rate on training data
if percentage_misclassified_val > percentage_misclassified_train:
    print("Percentage of misclassifications on validation data is greater than success rate on training data.")
elif percentage_misclassified_val < percentage_misclassified_train:
    print("Percentage of misclassifications on validation data is smaller than success rate on training data.")
else:
    print("Percentage of misclassifications on validation data is equal to success rate on training data.")
