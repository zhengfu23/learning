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
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Updates after each epoch
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_training_dataset(data_path, image_size, classes, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, img_names, cls = load_training_data(data_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    training_images = images[validation_size:]
    training_labels = labels[validation_size:]
    training_img_names = img_names[validation_size:]
    training_cls = cls[validation_size:]

    data_sets.train = DataSet(training_images, training_labels, training_img_names, training_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets
