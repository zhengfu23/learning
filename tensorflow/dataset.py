import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_training_data(data_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Reading training images')
    for c in classes:
        index = classes.index(c)
        print('Reading {} files (Index: {})'.format(c, index))
        path = os.path.join(data_path, c, '*g')
        files = glob.glob(path)
        for file in files:
            img = cv2.imread(file)
            img = cv2.resize(img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)
            images.append(img)
            # one-hot encoding the label
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            file_base = os.path.basename(file)
            img_names.append(file_base)
            cls.append(c)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        cls = np.array(cls)

        return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self.cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
