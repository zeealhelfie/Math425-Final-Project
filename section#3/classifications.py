import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import datetime
import platform

dig_train = pd.DataFrame(pd.read_csv("/data/handwriting_training_set.txt", sep=" "))
dig_test = pd.DataFrame(pd.read_csv("/data/handwriting_test_set.txt", sep=" "))

# Data Overview 
train.head()