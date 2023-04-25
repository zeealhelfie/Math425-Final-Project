import efficient_cancer_data as ecd
import numpy as np
from efficient_cancer_data import read_validation_data
from efficient_cancer_data import read_validation_data

# a.
# Read training data
A, b = ecd.read_training_data('train.data', )

Q, R = ecd.gram_schmidt_qr(A)  # use Gram-Schmidt QR factorization
coefficients = np.linalg.inv(R).dot(Q.T).dot(b)

print("coefficients:", coefficients) # hat(x) or hat(beta)

# b. 
def classifier(y):
    if y >= 0:
        return 1
    else:
        return -1

# c.
A_val, b_val = read_validation_data('validate.data', )
predictions = np.dot(A_val, coefficients)
incorrect = 0
predicted_labels = [classifier(y) for y in predictions]
for i in range(len(predicted_labels)):
    if predicted_labels[i] != b_val[i]:
        incorrect += 1
error_rate = incorrect / len(predictions)
print("Validation error rate: {:.2f}%".format(error_rate * 100))
