import efficient_cancer_data as ecd
import numpy as np

# Step 1: Import efficient_cancer_data module

# Step 2: Read training data
A, b = ecd.read_training_data("train.data")

# Step 3: Use QR algorithm to find the least-squares linear model
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T.dot(b))

# Step 4: Predict the malignancy of the tissues in the validate.data file
validate_A, validate_b = ecd.read_training_data('validate.data')
predictions = np.sign(validate_A.dot(x))

# Step 5: Compute the percentage of samples that are incorrectly classified
incorrectly_classified = np.sum(predictions != validate_b)
total_samples = len(validate_b)
accuracy = (total_samples - incorrectly_classified) / total_samples
print("Accuracy:", accuracy)
