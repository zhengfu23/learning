import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding seed variable so random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

# 1/5 of the training data will be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
data_path = 'training_data'

data = dataset.read_training_dataset(data_path, img_size, classes, validation_size)

print("Finished reading input data. Printing a snippet")
print("Number of files in Training set: \t\t{}".format(len(data.train.labels)))
print("Number of files in Validation set: \t\t{}".format(len(data.valid.labels)))

