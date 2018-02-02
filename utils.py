from __future__ import division
import numpy as np


def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


def random_classifier_accuracy(labels):
    labels_copy = np.array(labels).copy()
    np.random.shuffle(labels_copy)
    hits = np.array(labels) == labels_copy
    return sum(hits) / len(hits)
