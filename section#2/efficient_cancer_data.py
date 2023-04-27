# Copyright 2013 Philip N. Klein
from vec import Vec
from vecutil import vec2list
from sympy import Matrix
import numpy as np


def read_training_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    labels = []
    for line in lines:
        line = line.strip().split(',')
        labels.append(1 if line[1] == 'M' else -1)
        features = [float(x) for x in line[2:]]
        data.append(features)
    return np.array(data), np.array(labels, dtype=np.float64)

def gram_schmidt_qr(A, b):
    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]))
    for j in range(A.shape[1]):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    x = np.linalg.solve(R, Q.T.dot(b))
    return x


def read_validation_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    labels = []
    for line in lines:
        line = line.strip().split(',')
        labels.append(1 if line[1] == 'M' else -1)
        features = [float(x) for x in line[2:]]
        data.append(features)
    return np.array(data), np.array(labels)

def classify(predictions, threshold=0):
    return np.where(predictions > threshold, 1, -1)
