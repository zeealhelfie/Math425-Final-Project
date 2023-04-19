import efficient_cancer_data as ecd
import numpy as np


# Read training data
A, b = ecd.read_training_data('train.data', )

Q, R = np.linalg.qr(A)
coefficients = np.linalg.inv(R) @ Q.T @ b
print(coefficients)

def classifier(y):
    if y >= 0:
        return 1
    else:
        return -1
from efficient_cancer_data import read_validation_data

A_val, b_val = read_validation_data('validate.data', )
predictions = A_val @ coefficients
num_errors = 0
for i in range(len(predictions)):
    if classifier(predictions[i]) != b_val[i]:
        num_errors += 1
error_rate = num_errors / len(predictions)
print("Validation error rate: {:.2f}%".format(error_rate * 100))
